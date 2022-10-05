"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Training and evaluation codes for
3D hand mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
from os.path import dirname
import os.path as op
os.chdir('/userhome/alanjjp/Project/MeshGraphormer')
# os.chdir(dirname(os.getcwd()))
# import sys
# sys.path.append(dirname(os.getcwd()))
import code
import json
import time
import datetime
import warnings
import torch
import tqdm
import torchvision.models as models
from torchvision.utils import make_grid
import gc
import numpy as np
import cv2
import copy
import imageio
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from src.modeling.bert import BertConfig, Graphormer
from src.modeling.bert import Graphormer_Hand_Network as Graphormer_Network
from src.modeling.EvRGBStereo import EvRGBStereo
from src.modeling.Loss import Loss
from src.modeling._mano import MANO, Mesh
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
import src.modeling.data.config as cfg
from src.datasets.build import make_hand_data_loader

from src.utils.joint_indices import indices_change
from src.utils.logger import setup_logger
from src.utils.comm import synchronize, is_main_process, get_rank, get_world_size, all_gather, to_device
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.metric_logger import AverageMeter
from src.utils.renderer import Render
from src.utils.metric_pampjpe import reconstruction_error
from src.utils.geometric_layers import orthographic_projection
from src.configs.config_parser import ConfigParser


# from azureml.core.run import Run
# aml_run = Run.get_context()

def save_checkpoint(model, config, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(config['exper']['output_dir'], 'checkpoint-{}-{}'.format(
        epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save(model_to_save, op.join(checkpoint_dir, 'model.bin'))
            torch.save(model_to_save.state_dict(), op.join(checkpoint_dir, 'state_dict.bin'))
            print("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        print("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir


def adjust_learning_rate(optimizer, epoch, config):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = config['exper']['lr'] * (0.1 ** (epoch // (config['exper']['num_train_epochs'] / 2.0)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def run(config, train_dataloader, EvRGBStereo_model, Loss):
    save_config = copy.deepcopy(config)
    save_config['exper']['device'] = 'cuda'
    ConfigParser.save_config_dict(save_config, config['exper']['output_dir'], 'train.yaml')
    render = Render(config)
    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // config['exper']['num_train_epochs']

    optimizer = torch.optim.Adam(params=list(EvRGBStereo_model.parameters()),
                                 lr=config['exper']['lr'],
                                 betas=(0.9, 0.999),
                                 weight_decay=0)

    if config['exper']['distributed']:
        EvRGBStereo_model = torch.nn.parallel.DistributedDataParallel(
            EvRGBStereo_model, device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            find_unused_parameters=True,
        )
        Loss = torch.nn.parallel.DistributedDataParallel(
            Loss, device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            find_unused_parameters=True,
        )

    start_training_time = time.time()
    end = time.time()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    for iteration, (frames, meta_data) in enumerate(train_dataloader):
        EvRGBStereo_model.train()
        iteration += 1
        epoch = iteration // iters_per_epoch
        adjust_learning_rate(optimizer, epoch, config)
        data_time.update(time.time() - end)

        device = 'cuda'
        rgb, l_ev_frame, r_ev_frame = to_device(frames['rgb'], device), \
                                      to_device(frames['l_ev_frame'], device), \
                                      to_device(frames['r_ev_frame'], device)
        meta_data = to_device(meta_data, device)

        preds = EvRGBStereo_model(rgb.permute(0, 3, 1, 2), l_ev_frame.permute(0, 3, 1, 2), r_ev_frame.permute(0, 3, 1, 2), meta_data)

        loss_sum, loss_items = Loss(meta_data, preds)

        # back prop
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        if is_main_process():
            for key in loss_items.keys():
                tf_logger.add_scalar(key, loss_items[key], iteration)

        batch_time.update(time.time() - end)
        end = time.time()

        if (iteration-1) % config['utils']['logging_steps'] == 0 or iteration == max_iter:
            eta_seconds = batch_time.avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            if is_main_process():
                print(
                    ' '.join(
                        ['eta: {eta}', 'epoch: {ep}', 'iter: {iter}', 'max mem : {memory:.0f}', ]
                    ).format(eta=eta_string, ep=epoch, iter=iteration,
                             memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
                    + '   compute: {:.4f}, data: {:.4f}, lr: {:.6f}'.format(
                        batch_time.avg,
                        data_time.avg,
                        optimizer.param_groups[0]['lr'])
                )
        # if (iteration - 1) % (10*config['utils']['logging_steps']) == 0 or iteration == max_iter:
        #     #todo this only for er training
        #     kps_3d_abs = preds['pred_3d_joints'].reshape(-1, 21, 3) + meta_data['3d_joints'][:, :1, :]
        #     kps_3d = torch.bmm(meta_data['R_rgb'].reshape(-1, 3, 3),
        #                        kps_3d_abs.transpose(1, 2)).transpose(1, 2) + \
        #             meta_data['t_rgb'].reshape(-1, 1, 3)
        #     kps_2d = torch.bmm(meta_data['K_rgb'].reshape(-1, 3, 3), kps_3d.transpose(1, 2)).transpose(1, 2)
        #     kps_2d = kps_2d[:, :, :2] / kps_2d[:, :, 2:]
        #     img_render, kps_vis = render.visualize(
        #         K=meta_data['K_rgb'].detach(),
        #         R=meta_data['R_rgb'].detach(),
        #         t=meta_data['t_rgb'].detach(),
        #         hw=config['data']['rgb_hw'],
        #         img_bg=frames['rgb_ori']/255.,
        #         vertices=(preds['pred_vertices'] + meta_data['3d_joints'][:, :1, :]).detach(),
        #         kps_2d=kps_2d[:, indices_change(1, 2)].detach(),
        #     )
        #     img_cat = np.concatenate([img_render, kps_vis], axis=2)
        #     if is_main_process():
        #         tf_logger.add_images(
        #             'rendered_images', img_cat, iteration, dataformats='NHWC'
        #         )

        if iteration % iters_per_epoch == 0:
            if epoch % 10 == 0:
                checkpoint_dir = save_checkpoint(EvRGBStereo_model, config, epoch, iteration)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    print('Total training time: {} ({:.4f} s / iter)'.format(
        total_time_str, total_training_time / max_iter)
    )
    checkpoint_dir = save_checkpoint(EvRGBStereo_model, config, epoch, iteration)


def update_errors_list(errors_list, mpjpe_eachjoint_eachitem, index):
    if len(mpjpe_eachjoint_eachitem) != 0:
        for i, id in enumerate(index):
            errors_list[id].append(mpjpe_eachjoint_eachitem[i].detach().cpu())


def print_metrics(errors, metric='MPJPE', f=None):
    errors_all = 0
    joints_all = 0
    # remove the fast motion results
    for key, value in errors.items():
        if value is None:
            if f is None:
                print('{} is {}'.format(key, None))
            else:
                f.write('{} is {}'.format(key, None) + '\n')
        else:
            valid_joint = value != 0.
            mpjpe_tmp = torch.sum(value) / (torch.sum(valid_joint)+1e-6)
            errors_all += torch.sum(value)
            joints_all += torch.sum(valid_joint)
            if f is None:
                print('{} {} is {}'.format(metric, key, mpjpe_tmp))
            else:
                f.write('{} {} is {}'.format(metric, key, mpjpe_tmp) + '\n')
    mpjpe_all = errors_all / (joints_all+1e-6)
    if f is None:
        print('{} all is {}'.format(metric, mpjpe_all))
    else:
        f.write('{} all is {}'.format(metric, mpjpe_all) + '\n')


def run_eval_and_show(config, val_dataloader, EvRGBStereo_model, _loss):
    mano_layer = MANO(config['data']['smplx_path'], use_pca=False, is_rhand=True).cuda()
    render = Render(config)

    if config['exper']['distributed']:
        EvRGBStereo_model = torch.nn.parallel.DistributedDataParallel(
            EvRGBStereo_model, device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            find_unused_parameters=True,
        )
        mano_layer = torch.nn.parallel.DistributedDataParallel(
            mano_layer, device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            find_unused_parameters=True,
        )

    EvRGBStereo_model.eval()

    start_eval_time = time.time()
    end = time.time()

    batch_time = AverageMeter()

    labels_list = ['n_f', 'n_r', 'h_f', 'h_r', 'f_f', 'f_r', 'fast']
    colors_list = ['r', 'g', 'b', 'gold', 'purple', 'cyan', 'm']
    mpjpe_errors_list = [[], [], [], [], [], [], []]
    vertices_errors_list = [[], [], [], [], [], []]

    mkdir(config['exper']['output_dir'])

    file = open(os.path.join(config['exper']['output_dir'], 'error_joints.txt'), 'a')

    last_seq = 0

    with torch.no_grad():
        for iteration, (frames, meta_data) in enumerate(val_dataloader):

            if last_seq != str(meta_data['seq_id'][0].item()):
                last_seq = str(meta_data['seq_id'][0].item())
                print('Now for seq id: ', last_seq)

            device = 'cuda'
            rgb, l_ev_frame, r_ev_frame = to_device(frames['rgb'], device), \
                                          to_device(frames['l_ev_frame'], device), \
                                          to_device(frames['r_ev_frame'], device)
            meta_data = to_device(meta_data, device)

            preds = EvRGBStereo_model(rgb.permute(0, 3, 1, 2), l_ev_frame.permute(0, 3, 1, 2),
                                      r_ev_frame.permute(0, 3, 1, 2), meta_data)
            batch_time.update(time.time() - end)
            end = time.time()

            fast_index = meta_data['seq_type'] == 6
            no_fast_index = torch.logical_not(fast_index)
            if torch.sum(fast_index) != 0:
                pred_3d_joints = preds['pred_3d_joints'][fast_index]
                pred_3d_joints_abs = pred_3d_joints + meta_data['3d_joints'][fast_index][:, :1]
                pred_2d_joints = torch.bmm(meta_data['K_event_r'][fast_index], pred_3d_joints_abs.permute(0, 2, 1)).permute(0, 2, 1)
                pred_2d_joints = pred_2d_joints[:, :, :2] / pred_2d_joints[:, :, 2:]
                gt_2d_joints = meta_data['2d_joints_event'][fast_index]
                pred_2d_joints_aligned = pred_2d_joints - pred_2d_joints[:, :1] + gt_2d_joints[:, :1]
                # print('fast index', fast_index)
                mpjpe_eachjoint_eachitem = torch.sqrt(torch.sum((pred_2d_joints_aligned - gt_2d_joints) ** 2, dim=-1))[meta_data['joints_2d_valid_ev'][fast_index]]
                # print('mpjpe shape', mpjpe_eachjoint_eachitem.shape)
                # print('seq type shape', meta_data['seq_type'][fast_index])
                update_errors_list(mpjpe_errors_list, mpjpe_eachjoint_eachitem, meta_data['seq_type'][fast_index][meta_data['joints_2d_valid_ev'][fast_index]])
            if torch.sum(no_fast_index) != 0:
                gt_3d_joints = meta_data['3d_joints'][no_fast_index]
                gt_3d_joints_sub = gt_3d_joints - gt_3d_joints[:, :1]
                pred_3d_joints = preds['pred_3d_joints'][no_fast_index]
                pred_3d_joints_sub = pred_3d_joints - pred_3d_joints[:, :1]
                mpjpe_eachjoint_eachitem = torch.sqrt(
                    torch.sum((pred_3d_joints_sub - gt_3d_joints_sub) ** 2, dim=-1)) * 1000.
                update_errors_list(mpjpe_errors_list, mpjpe_eachjoint_eachitem, meta_data['seq_type'][no_fast_index])

                manos = meta_data['mano']
                gt_dest_mano_output = mano_layer(
                    global_orient=manos['rot_pose'][no_fast_index].reshape(-1, 3),
                    hand_pose=manos['hand_pose'][no_fast_index].reshape(-1, 45),
                    betas=manos['shape'][no_fast_index].reshape(-1, 10),
                    transl=manos['trans'][no_fast_index].reshape(-1, 3)
                )
                gt_vertices_sub = gt_dest_mano_output.vertices - gt_dest_mano_output.joints[:, :1, :]
                error_vertices = torch.mean(torch.abs(preds['pred_vertices'][no_fast_index] - gt_vertices_sub), dim=0)
                update_errors_list(vertices_errors_list, error_vertices, meta_data['seq_type'][no_fast_index])
            if iteration*config['exper']['per_gpu_batch_size'] % 20 != 0:
                continue
            if config['eval']['output']['save']:
                predicted_meshes = preds['pred_vertices'] + meta_data['3d_joints'][:, :1]
                for i in range(rgb.shape[0]):
                    seq_dir = op.join(config['exper']['output_dir'], str(meta_data['seq_id'][i].item()))
                    mkdir(seq_dir)
                    if config['eval']['output']['mesh']:
                        mesh_dir = op.join(seq_dir, 'mesh')
                        mkdir(mesh_dir)
                        with open(os.path.join(mesh_dir, '{}.obj'.format(meta_data['annot_id'][i].item())), 'w') as file_object:
                            for ver in predicted_meshes[i].detach().cpu().numpy():
                                print('v %f %f %f'%(ver[0], ver[1], ver[2]), file=file_object)
                            faces = mano_layer.faces
                            for face in faces:
                                print('f %d %d %d'%(face[0]+1, face[1]+1, face[2]+1), file=file_object)

                if config['eval']['output']['attention_map']:
                    id_tmp = 0
                    annot_dir = op.join(config['exper']['output_dir'], str(meta_data['seq_id'][id_tmp].item()), 'attention', str(meta_data['annot_id'][id_tmp].item()))
                    mkdir(annot_dir)
                    encoder_ids = range(len(config['model']['tfm']['input_feat_dim']))
                    layer_ids = range(config['model']['tfm']['num_hidden_layers'])
                    for encoder_id in encoder_ids:
                        for layer_id in layer_ids:
                            if 'att' in preds.keys():

                                att_map = preds['att'][encoder_id][layer_id][id_tmp]

                                attention_all = np.sum(att_map.detach().cpu().numpy(), axis=0)
                                max_j_all = np.max(attention_all, axis=1, keepdims=True)
                                min_j_all = np.min(attention_all, axis=1, keepdims=True)
                                attention_all_normal = (attention_all - min_j_all) / (max_j_all - min_j_all)
                                plt.title('Attention Map All')
                                plt.imshow(attention_all_normal, cmap="inferno")
                                fig_ = plt.gcf()
                                fig_.savefig(op.join(annot_dir, 'attention_all_encoder{}_layer{}.png'.format(encoder_id, layer_id)))
                                fig_.clear()

                                shapes = np.array([16, 36, 16], dtype=np.int32)
                                cnn_shape = shapes * np.array(config['model']['method']['ere_usage'])
                                att_map_all = np.zeros((21, cnn_shape.sum()), dtype=np.float32)
                                for i in range(config['model']['tfm']['num_attention_heads']):
                                    att_map_all += att_map[i][:21, -cnn_shape.sum():].detach().cpu().numpy()
                                max_j = np.max(att_map_all, axis=1, keepdims=True)
                                min_j = np.min(att_map_all, axis=1, keepdims=True)
                                att_map_joints = (att_map_all - min_j) / (max_j - min_j)
                                att_map_joints = att_map_joints[indices_change(1, 2)]
                                fig, axes = plt.subplots(nrows=3, ncols=7*len(cnn_shape), figsize=(12*len(cnn_shape), 6))
                                for i in range(3):
                                    for j in range(7):
                                        joint_id = 7 * i + j
                                        col = j * sum(config['model']['method']['ere_usage'])
                                        if config['model']['method']['ere_usage'][0]:
                                            axes[i, col].imshow(l_ev_frame[id_tmp].detach().cpu().numpy())
                                            att_map_ = cv2.resize(att_map_joints[joint_id, :cnn_shape[0]].reshape(4, 4),
                                                                  (l_ev_frame.shape[1:3]),
                                                                  interpolation=cv2.INTER_NEAREST)
                                            axes[i, col].imshow(att_map_, cmap="inferno", alpha=0.6)
                                            axes[i, col].title.set_text(f"Event {cfg.J_NAME[joint_id]}")
                                            axes[i, col].axis("off")
                                            col += 1
                                        if config['model']['method']['ere_usage'][1]:
                                            axes[i, col].imshow(rgb[id_tmp].detach().cpu().numpy())
                                            att_map_ = cv2.resize(att_map_joints[joint_id, cnn_shape[0]:sum(cnn_shape[:2])].reshape(6, 6), (rgb.shape[1:3]), interpolation=cv2.INTER_NEAREST)
                                            axes[i, col].imshow(att_map_, cmap="inferno", alpha=0.6)
                                            axes[i, col].title.set_text(f"RGB {cfg.J_NAME[joint_id]}")
                                            axes[i, col].axis("off")
                                            col += 1

                                # fig.show()
                                fig.savefig(op.join(annot_dir, 'encoder{}_layer{}.png'.format(encoder_id, layer_id)))
                                fig.clear()

                        # todo draw attention map
                        pass

                if config['eval']['output']['rendered']:
                    key = config['eval']['output']['vis_rendered']
                    if key == 'rgb':
                        img_bg = frames[key+'_ori']/255.
                        hw = config['data']['rgb_hw']
                    else:
                        img_bg = frames[key + '_ori']
                        hw = config['data']['event_hw']
                    img_render = render.visualize(
                        K=meta_data['K_'+key].detach(),
                        R=meta_data['R_'+key].detach(),
                        t=meta_data['t_'+key].detach(),
                        hw=hw,
                        img_bg=img_bg,
                        vertices=predicted_meshes.detach(),
                    )
                    for i in range(img_render.shape[0]):
                        img_dir = op.join(config['exper']['output_dir'], str(meta_data['seq_id'][i].item()), 'rendered')
                        mkdir(img_dir)
                        imageio.imwrite(op.join(img_dir, '{}.jpg'.format(meta_data['annot_id'][i].item())),
                                        (img_render[i].detach().cpu().numpy() * 255).astype(np.uint8))

            if (iteration - 1) % config['utils']['logging_steps'] == 0:
                eta_seconds = batch_time.avg * (len(val_dataloader) / config['exper']['per_gpu_batch_size'] - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if is_main_process():
                    print(
                        ' '.join(
                            ['eta: {eta}', 'iter: {iter}', 'max mem : {memory:.0f}', ]
                        ).format(eta=eta_string, iter=iteration,
                                 memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
                        + '   compute: {:.4f}'.format(batch_time.avg)
                    )

    mpjpe_errors, vertices_errors = {}, {}
    for i, key in enumerate(labels_list):
        if len(mpjpe_errors_list[i]) != 0:
            mpjpe_errors.update(
                {key: torch.stack(mpjpe_errors_list[i])}
            )
    for i, key in enumerate(labels_list[:-1]):
        if len(vertices_errors_list[i]) != 0:
            vertices_errors.update(
                {key: torch.stack(vertices_errors_list[i])}
            )

    print_metrics(mpjpe_errors, metric='MPJPE', f=file)
    print_metrics(vertices_errors, metric='Vertices', f=file)

    if config['eval']['output']['save']:
        if config['eval']['output']['errors']:
            error_dir = op.join(config['exper']['output_dir'], 'error')
            mkdir(error_dir)
            torch.save(mpjpe_errors, os.path.join(error_dir, 'mpjpe_errors.pt'))
            torch.save(vertices_errors, os.path.join(error_dir, 'vertices_errors.pt'))

    file.close()

    total_eval_time = time.time() - start_eval_time
    total_time_str = str(datetime.timedelta(seconds=total_eval_time))
    print('Total eval time: {} ({:.4f} s / iter)'.format(
        total_time_str, total_eval_time / (len(val_dataloader) / config['exper']['per_gpu_batch_size']))
    )

    return

def run_eval_and_save(config, val_dataloader, EvRGBStereo_model):
    mano_layer = MANO(config['data']['smplx_path'], use_pca=False, is_rhand=True).cuda()
    if config['exper']['distributed']:
        EvRGBStereo_model = torch.nn.parallel.DistributedDataParallel(
            EvRGBStereo_model, device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            find_unused_parameters=True,
        )
        mano_layer = torch.nn.parallel.DistributedDataParallel(
            mano_layer, device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            find_unused_parameters=True,
        )

    EvRGBStereo_model.eval()

    start_eval_time = time.time()

    pred_vertices_save = {}
    pred_joints_save = {}

    mkdir(config['exper']['output_dir'])

    last_seq = 0

    with torch.no_grad():
        for iteration, (frames, meta_data) in enumerate(val_dataloader):
            device = 'cuda'
            rgb, l_ev_frame, r_ev_frame = to_device(frames['rgb'], device), \
                                          to_device(frames['l_ev_frame'], device), \
                                          to_device(frames['r_ev_frame'], device)
            meta_data = to_device(meta_data, device)

            preds = EvRGBStereo_model(rgb.permute(0, 3, 1, 2), l_ev_frame.permute(0, 3, 1, 2),
                                      r_ev_frame.permute(0, 3, 1, 2), meta_data)

            manos = meta_data['mano']
            gt_dest_mano_output = mano_layer(
                global_orient=manos['rot_pose'].reshape(-1, 3),
                hand_pose=manos['hand_pose'].reshape(-1, 45),
                betas=manos['shape'].reshape(-1, 10),
                transl=manos['trans'].reshape(-1, 3)
            )

            for i in range(frames['rgb'].shape[0]):
                seq_id = meta_data['seq_id'][i].item()
                annot_id = meta_data['annot_id'][i].item()
                if seq_id not in pred_joints_save.keys():
                    pred_vertices_save[seq_id] = {}
                    pred_joints_save[seq_id] = {}
                pred_joints_tmp = preds['pred_3d_joints'][i] - preds['pred_3d_joints'][i, :1] + meta_data['3d_joints'][i, :1]
                pred_vertices_tmp = preds['pred_vertices'][i] + gt_dest_mano_output.joints[i, :1]
                pred_joints_w_tmp = ((meta_data['tf_w_c'][i, :3, :3] @ pred_joints_tmp.transpose(0, 1)) + meta_data['tf_w_c'][i, :3, 3:]).transpose(0, 1)
                pred_vertices_w_tmp = ((meta_data['tf_w_c'][i, :3, :3] @ pred_vertices_tmp.transpose(0, 1)) + meta_data['tf_w_c'][i, :3, 3:]).transpose(0, 1)
                pred_joints_save[seq_id][annot_id] = pred_joints_w_tmp.detach().cpu()
                pred_vertices_save[seq_id][annot_id] = pred_vertices_w_tmp.detach().cpu()


    file_name = 'pred_scale{:.0f}_rot{:.0f}_.pt'.format(config['eval']['augment']['scale'] * 100, config['eval']['augment']['rot'] * 100)
    torch.save([pred_joints_save, pred_vertices_save], os.path.join(config['exper']['output_dir'], file_name))
    print('Save results to {}'.format(os.path.join(config['exper']['output_dir'], file_name)))
    total_eval_time = time.time() - start_eval_time
    total_time_str = str(datetime.timedelta(seconds=total_eval_time))
    print('Total training time: {} for scale {} rot {}'.format(
        total_time_str, str(config['eval']['augment']['scale']), str(config['eval']['augment']['rot']))
    )
#
# def run_aml_inference_hand_mesh(args, val_loader, Graphormer_model, criterion, criterion_vertices, epoch, mano_model,
#                                 mesh_sampler, renderer, split):
#     # switch to evaluate mode
#     Graphormer_model.eval()
#     fname_output_save = []
#     mesh_output_save = []
#     joint_output_save = []
#     world_size = get_world_size()
#     with torch.no_grad():
#         for i, (img_keys, images, annotations) in enumerate(val_loader):
#             batch_size = images.size(0)
#             # compute output
#             images = images.cuda()
#
#             # forward-pass
#             pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices = Graphormer_model(images, mano_model,
#                                                                                              mesh_sampler)
#             # obtain 3d joints from full mesh
#             pred_3d_joints_from_mesh = mano_model.get_3d_joints(pred_vertices)
#
#             for j in range(batch_size):
#                 fname_output_save.append(img_keys[j])
#                 pred_vertices_list = pred_vertices[j].tolist()
#                 mesh_output_save.append(pred_vertices_list)
#                 pred_3d_joints_from_mesh_list = pred_3d_joints_from_mesh[j].tolist()
#                 joint_output_save.append(pred_3d_joints_from_mesh_list)
#
#     if world_size > 1:
#         torch.distributed.barrier()
#     print('save results to pred.json')
#     output_json_file = 'pred.json'
#     print('save results to ', output_json_file)
#     with open(output_json_file, 'w') as f:
#         json.dump([joint_output_save, mesh_output_save], f)
#
#     azure_ckpt_name = '200'  # args.resume_checkpoint.split('/')[-2].split('-')[1]
#     inference_setting = 'sc%02d_rot%s' % (int(args.sc * 10), str(int(args.rot)))
#     output_zip_file = args.output_dir + 'ckpt' + azure_ckpt_name + '-' + inference_setting + '-pred.zip'
#
#     resolved_submit_cmd = 'zip ' + output_zip_file + ' ' + output_json_file
#     print(resolved_submit_cmd)
#     os.system(resolved_submit_cmd)
#     resolved_submit_cmd = 'rm %s' % (output_json_file)
#     print(resolved_submit_cmd)
#     os.system(resolved_submit_cmd)
#     if world_size > 1:
#         torch.distributed.barrier()
#
#     return
#
#
# def run_inference_hand_mesh(args, val_loader, Graphormer_model, criterion, criterion_vertices, epoch, mano_model,
#                             mesh_sampler, renderer, split):
#     # switch to evaluate mode
#     Graphormer_model.eval()
#     fname_output_save = []
#     mesh_output_save = []
#     joint_output_save = []
#     with torch.no_grad():
#         for i, (img_keys, images, annotations) in enumerate(val_loader):
#             batch_size = images.size(0)
#             # compute output
#             images = images.cuda()
#
#             # forward-pass
#             pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices = Graphormer_model(images, mano_model,
#                                                                                              mesh_sampler)
#
#             # obtain 3d joints from full mesh
#             pred_3d_joints_from_mesh = mano_model.get_3d_joints(pred_vertices)
#             pred_3d_pelvis = pred_3d_joints_from_mesh[:, cfg.J_NAME.index('Wrist'), :]
#             pred_3d_joints_from_mesh = pred_3d_joints_from_mesh - pred_3d_pelvis[:, None, :]
#             pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]
#
#             for j in range(batch_size):
#                 fname_output_save.append(img_keys[j])
#                 pred_vertices_list = pred_vertices[j].tolist()
#                 mesh_output_save.append(pred_vertices_list)
#                 pred_3d_joints_from_mesh_list = pred_3d_joints_from_mesh[j].tolist()
#                 joint_output_save.append(pred_3d_joints_from_mesh_list)
#
#             if i % 20 == 0:
#                 # obtain 3d joints, which are regressed from the full mesh
#                 pred_3d_joints_from_mesh = mano_model.get_3d_joints(pred_vertices)
#                 # obtain 2d joints, which are projected from 3d joints of mesh
#                 pred_2d_joints_from_mesh = orthographic_projection(pred_3d_joints_from_mesh.contiguous(),
#                                                                    pred_camera.contiguous())
#                 visual_imgs = visualize_mesh(renderer,
#                                              annotations['ori_img'].detach(),
#                                              annotations['joints_2d'].detach(),
#                                              pred_vertices.detach(),
#                                              pred_camera.detach(),
#                                              pred_2d_joints_from_mesh.detach())
#
#                 visual_imgs = visual_imgs.transpose(0, 1)
#                 visual_imgs = visual_imgs.transpose(1, 2)
#                 visual_imgs = np.asarray(visual_imgs)
#
#                 inference_setting = 'sc%02d_rot%s' % (int(args.sc * 10), str(int(args.rot)))
#                 temp_fname = args.output_dir + args.resume_checkpoint[
#                                                0:-9] + 'freihand_results_' + inference_setting + '_batch' + str(
#                     i) + '.jpg'
#                 cv2.imwrite(temp_fname, np.asarray(visual_imgs[:, :, ::-1] * 255))
#
#     print('save results to pred.json')
#     with open('pred.json', 'w') as f:
#         json.dump([joint_output_save, mesh_output_save], f)
#
#     run_exp_name = args.resume_checkpoint.split('/')[-3]
#     run_ckpt_name = args.resume_checkpoint.split('/')[-2].split('-')[1]
#     inference_setting = 'sc%02d_rot%s' % (int(args.sc * 10), str(int(args.rot)))
#     resolved_submit_cmd = 'zip ' + args.output_dir + run_exp_name + '-ckpt' + run_ckpt_name + '-' + inference_setting + '-pred.zip  ' + 'pred.json'
#     print(resolved_submit_cmd)
#     os.system(resolved_submit_cmd)
#     resolved_submit_cmd = 'rm pred.json'
#     print(resolved_submit_cmd)
#     os.system(resolved_submit_cmd)
#     return
#
#
# def visualize_mesh(renderer,
#                    images,
#                    gt_keypoints_2d,
#                    pred_vertices,
#                    pred_camera,
#                    pred_keypoints_2d):
#     """Tensorboard logging."""
#     gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
#     to_lsp = list(range(21))
#     rend_imgs = []
#     batch_size = pred_vertices.shape[0]
#     # Do visualization for the first 6 images of the batch
#     for i in range(min(batch_size, 10)):
#         img = images[i].cpu().numpy().transpose(1, 2, 0)
#         # Get LSP keypoints from the full list of keypoints
#         gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
#         pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
#         # Get predict vertices for the particular example
#         vertices = pred_vertices[i].cpu().numpy()
#         cam = pred_camera[i].cpu().numpy()
#         # Visualize reconstruction and detected pose
#         rend_img = visualize_reconstruction(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer)
#         rend_img = rend_img.transpose(2, 0, 1)
#         rend_imgs.append(torch.from_numpy(rend_img))
#     rend_imgs = make_grid(rend_imgs, nrow=1)
#     return rend_imgs
#
#
# def visualize_mesh_test(renderer,
#                         images,
#                         gt_keypoints_2d,
#                         pred_vertices,
#                         pred_camera,
#                         pred_keypoints_2d,
#                         PAmPJPE):
#     """Tensorboard logging."""
#     gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
#     to_lsp = list(range(21))
#     rend_imgs = []
#     batch_size = pred_vertices.shape[0]
#     # Do visualization for the first 6 images of the batch
#     for i in range(min(batch_size, 10)):
#         img = images[i].cpu().numpy().transpose(1, 2, 0)
#         # Get LSP keypoints from the full list of keypoints
#         gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
#         pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
#         # Get predict vertices for the particular example
#         vertices = pred_vertices[i].cpu().numpy()
#         cam = pred_camera[i].cpu().numpy()
#         score = PAmPJPE[i]
#         # Visualize reconstruction and detected pose
#         rend_img = visualize_reconstruction_test(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam,
#                                                  renderer, score)
#         rend_img = rend_img.transpose(2, 0, 1)
#         rend_imgs.append(torch.from_numpy(rend_img))
#     rend_imgs = make_grid(rend_imgs, nrow=1)
#     return rend_imgs
#
#
# def visualize_mesh_no_text(renderer,
#                            images,
#                            pred_vertices,
#                            pred_camera):
#     """Tensorboard logging."""
#     rend_imgs = []
#     batch_size = pred_vertices.shape[0]
#     # Do visualization for the first 6 images of the batch
#     for i in range(min(batch_size, 1)):
#         img = images[i].cpu().numpy().transpose(1, 2, 0)
#         # Get predict vertices for the particular example
#         vertices = pred_vertices[i].cpu().numpy()
#         cam = pred_camera[i].cpu().numpy()
#         # Visualize reconstruction only
#         rend_img = visualize_reconstruction_no_text(img, 224, vertices, cam, renderer, color='hand')
#         rend_img = rend_img.transpose(2, 0, 1)
#         rend_imgs.append(torch.from_numpy(rend_img))
#     rend_imgs = make_grid(rend_imgs, nrow=1)
#     return rend_imgs


def get_config():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--config', type=str, default='./src/configs/train_evrealhands.yaml')
    parser.add_argument('--resume_checkpoint', type=str, default='')
    parser.add_argument('--config_merge', type=str, default='')
    parser.add_argument('--run_eval_only', action='store_true')
    parser.add_argument('--s', default=1.0, type=float, help='scale')
    parser.add_argument('--r', default=0., type=float, help='rotate')
    parser.add_argument('--output_dir', type=str,
                        default='./output/test0')
    args = parser.parse_args()
    config = ConfigParser(args.config)


    if args.config_merge != '':
        config.merge_configs(args.config_merge)
    config = config.config
    if args.output_dir != '':
        config['exper']['output_dir'] = args.output_dir
    if args.resume_checkpoint != '':
        config['exper']['resume_checkpoint'] = args.resume_checkpoint
    config['exper']['run_eval_only'] = args.run_eval_only
    dataset_config = ConfigParser(config['data']['dataset_yaml']).config
    config['data']['dataset_info'] = dataset_config
    if 'augment' not in config['eval'].keys():
        config['eval']['augment'] = {}
    config['eval']['augment']['scale'] = args.s
    config['eval']['augment']['rot'] = args.r
    return config


def main(config):
    global tf_logger
    # Setup CUDA, GPU & distributed training
    num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    os.environ['OMP_NUM_THREADS'] = str(config['exper']['num_workers'])
    print('set os.environ[OMP_NUM_THREADS] to {}'.format(os.environ['OMP_NUM_THREADS']))

    config['exper']['distributed'] = num_gpus > 1
    config['exper']['device'] = torch.device(config['exper']['device'])
    if config['exper']['distributed']:
        print("Init distributed training on local rank {}".format(int(os.environ["LOCAL_RANK"])))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(
            backend='nccl'  # , init_method='env://'
        )
        synchronize()

    mkdir(config['exper']['output_dir'])
    # mkdir(os.path.join(config['exper']['output_dir'], 'logger'))
    mkdir(os.path.join(config['exper']['output_dir'], 'tf_logs'))
    # logger = setup_logger("EvRGBStereo", config['exper']['output_dir'], get_rank())
    if is_main_process() and not config['exper']['run_eval_only']:
        tf_logger = SummaryWriter(os.path.join(config['exper']['output_dir'], 'tf_logs'))
    else:
        tf_logger = None

    set_seed(config['exper']['seed'], num_gpus)
    print("Using {} GPUs".format(num_gpus))


    if config['exper']['run_eval_only'] == True and config['exper']['resume_checkpoint'] != None and\
            config['exper']['resume_checkpoint'] != 'None' and 'state_dict' not in config['exper']['resume_checkpoint']:
        # if only run eval, load checkpoint
        print("Evaluation: Loading from checkpoint {}".format(config['exper']['resume_checkpoint']))
        _model = torch.load(config['exper']['resume_checkpoint'])

    else:

        # build network
        _model = EvRGBStereo(config=config)

        if config['exper']['resume_checkpoint'] != None and config['exper']['resume_checkpoint'] != 'None':
            # for fine-tuning or resume training or inference, load weights from checkpoint
            print("Loading state dict from checkpoint {}".format(config['exper']['resume_checkpoint']))
            # workaround approach to load sparse tensor in graph conv.
            state_dict = torch.load(config['exper']['resume_checkpoint'])
            _model.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()

    _model.to(config['exper']['device'])
    print("Training parameters %s", str(config))

    _loss = Loss(config)
    _loss.to(config['exper']['device'])

    if config['exper']['run_eval_only'] == True:
        val_dataloader = make_hand_data_loader(config)
        if config['eval']['multiscale']:
            run_eval_and_save(config, val_dataloader, _model)
        else:
            run_eval_and_show(config, val_dataloader, _model, _loss)

    else:
        train_dataloader = make_hand_data_loader(config)
        run(config, train_dataloader, _model, _loss)

    if is_main_process() and not config['exper']['run_eval_only']:
        tf_logger.close()


if __name__ == "__main__":
    config = get_config()
    main(config)
