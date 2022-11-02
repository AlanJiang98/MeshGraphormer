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
# os.chdir('/userhome/wangbingxuan/code/MeshGraphormer')
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
from src.utils.metric_pampjpe import get_alignMesh, compute_similarity_transform_batch
from fvcore.nn import FlopCountAnalysis
import tarfile
import shutil
import hfai.checkpoint
from ffrecord import PackedFolder
import io


import hfai_env
hfai_env.set_env('evrgb_hfai')

# from azureml.core.run import Run
# aml_run = Run.get_context()

def save_latest(model, optimizer,scheduler, config, epoch,iteration,num_trial=10):
    checkpoint_path = op.join(config['exper']['output_dir'], 'latest.ckpt')
    checkpoint_path_else = op.join(config['exper']['output_dir'], 'latest_else.ckpt')
    if not is_main_process():
        return checkpoint_path
    if not os.path.exists(config['exper']['output_dir']):
        os.makedirs(config['exper']['output_dir'])
    model_to_save = model.module if hasattr(model, 'module') else model
    optimizer_to_save = optimizer.module if hasattr(optimizer, 'module') else optimizer
    scheduler_to_save = scheduler.module if hasattr(scheduler, 'module') else scheduler
    output_dict_model = {
        'model': model_to_save.state_dict(),
        # 'optimizer': optimizer_to_save.state_dict(),
        # 'scheduler': scheduler_to_save.state_dict(),
        'epoch': epoch,
        'iteration': iteration,
    }
    output_dict_else = {
        # 'model': model_to_save.state_dict(),
        'optimizer': optimizer_to_save.state_dict(),
        'scheduler': scheduler_to_save.state_dict(),
        'epoch': epoch,
        'iteration': iteration,
    }
    for i in range(num_trial):
        try:
            torch.save(output_dict_model, checkpoint_path)
            torch.save(output_dict_else, checkpoint_path_else)
            print("Save latest checkpoint to {}".format(checkpoint_path))
            break
        except:
            pass
    else:
        print("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_path


def save_checkpoint(model, config, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(config['exper']['output_dir'], 'checkpoint-{}-{}'.format(
        epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            # torch.save(model_to_save, op.join(checkpoint_dir, 'model.bin'))
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
    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // config['exper']['num_train_epochs']

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
    
    # todo change the pos of the optimizer
    optimizer = torch.optim.Adam(params=list(EvRGBStereo_model.parameters()),
                                 lr=config['exper']['lr'],
                                 betas=(0.9, 0.999),
                                 weight_decay=0.0001)

    # todo add scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['exper']['num_train_epochs'])

    start_training_time = time.time()
    end = time.time()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()
    last_epoch = 0
    last_step = 0
    if os.path.exists(os.path.join(config['exper']['output_dir'], 'latest_else.ckpt')):
            print("Loading from recent...")
            # device_name = torch.cuda.current_device()
            
            latest_dict = torch.load(os.path.join(config['exper']['output_dir'], 'latest_else.ckpt'),map_location=torch.device('cuda:%d' % int(os.environ["LOCAL_RANK"])))
            optimizer.load_state_dict(latest_dict["optimizer"])
            scheduler.load_state_dict(latest_dict["scheduler"])
            last_epoch = latest_dict["epoch"]
            last_step = latest_dict["iteration"]
            del latest_dict
            gc.collect()
            torch.cuda.empty_cache()
            
    for iteration, (frames, meta_data) in enumerate(train_dataloader):
        # if iteration < last_step:
        #     continue
        EvRGBStereo_model.train()
        iteration += max(last_step,1)
        epoch = iteration // iters_per_epoch
        # adjust_learning_rate(optimizer, epoch, config)
        data_time.update(time.time() - end)

        device = 'cuda'
        batch_size = frames[0]['rgb'].shape[0]
        frames = to_device(frames, device)
        meta_data = to_device(meta_data, device)
        preds, atts = EvRGBStereo_model(frames, return_att=True, decode_all=False)
        loss_sum, loss_items = Loss(meta_data, preds)

        log_losses.update(loss_sum.item(), batch_size)
        # back prop
        optimizer.zero_grad()
        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(EvRGBStereo_model.parameters(), 0.3)
        optimizer.step()
        if last_epoch != epoch:
            scheduler.step()

        last_epoch = epoch
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
                    + '   compute: {:.4f}, data: {:.4f}, lr: {:.6f} loss: {:.6f}'.format(
                        batch_time.avg,
                        data_time.avg,
                        optimizer.param_groups[0]['lr'],
                        log_losses.avg
                    )
                )

        if iteration % iters_per_epoch == 0:
            save_latest(EvRGBStereo_model, optimizer, scheduler, config, epoch, iteration)
            if epoch % 20 == 0 and epoch > 180:
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
    count = 0
    scene_errors = 0
    scene_items = 0

    for key, value in errors.items():
        count += 1
        if value is None:
            if f is None:
                print('{} is {}'.format(key, None))
            else:
                f.write('{} is {}'.format(key, None) + '\n')
        else:
            valid_items = value != 0.
            mpjpe_tmp = torch.sum(value) / (torch.sum(valid_items)+1e-6)
            errors_all += torch.sum(value)
            joints_all += torch.sum(valid_items)
            scene_errors += torch.sum(value)
            scene_items += torch.sum(valid_items)
            if f is None:
                print('{} {} is {}'.format(metric, key, mpjpe_tmp))
            else:
                f.write('{} {} is {}'.format(metric, key, mpjpe_tmp) + '\n')
            if count % 2 == 0:
                if f is None:
                    print('Scene Average is {}'.format(scene_errors / (scene_items + 1e-6)))
                else:
                    f.write('Scene Average is {}'.format(scene_errors / (scene_items + 1e-6)) + '\n')
                scene_errors = 0
                scene_items = 0
    mpjpe_all = errors_all / (joints_all+1e-6)
    if f is None:
        print('{} all is {}'.format(metric, mpjpe_all))
    else:
        f.write('{} all is {}'.format(metric, mpjpe_all) + '\n')

def print_sequence_error(mpjpe_eachjoint_eachitem, metric, seq_type, dir):
    x = np.arange(0, mpjpe_eachjoint_eachitem.shape[0])
    mpjpe_pre = torch.mean(mpjpe_eachjoint_eachitem, dim=1).detach().cpu().numpy() / 1000.
    plt.plot(x, mpjpe_pre, label=metric)
    plt.plot(x, mpjpe_pre * 0., label='zero')
    plt.xlabel('item')
    plt.ylabel('m')
    plt.title(seq_type)
    plt.legend()
    f_t = plt.gcf()
    mkdir(dir)
    f_t.savefig(os.path.join(dir, '{}_{}.png'.format(seq_type, metric)))
    f_t.clear()


def print_items(mpjpe_errors_list, labels_list, metric, file):
    mpjpe_errors = {}
    for i, key in enumerate(labels_list):
        if len(mpjpe_errors_list[i]) != 0:
            mpjpe_errors.update(
                {key: torch.stack(mpjpe_errors_list[i])}
            )
    print_metrics(mpjpe_errors, metric=metric, f=file)


def run_eval_and_show(config, val_dataloader_normal, val_dataloader_fast, EvRGBStereo_model, _loss):
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
    infer_time = AverageMeter()

    labels_list = ['n_f', 'n_r', 'h_f', 'h_r', 'f_f', 'f_r', 'fast']
    colors_list = ['r', 'g', 'b', 'gold', 'purple', 'cyan', 'm']
    steps = config['exper']['preprocess']['steps']

    # mpjpe_errors_list = [[], [], [], [], [], [], []]
    # mpvpe_errors_list = [[], [], [], [], [], [], []]
    # pa_mpjpe_errors_list = [[], [], [], [], [], [], []]
    # pa_mpvpe_errors_list = [[], [], [], [], [], [], []]
    errors_list = []
    for i in range(4):
        errors_list.append([])
        for j in range(steps):
            errors_list[i].append([])
            for k in range(len(labels_list)):
                errors_list[i][j].append([])

    metrics = ['MPJPE', 'PA_MPJPE', 'MPVPE', 'PA_MPVPE']

    mkdir(config['exper']['output_dir'])

    file = open(os.path.join(config['exper']['output_dir'], 'error_joints.txt'), 'a')

    last_seq = 0

    with torch.no_grad():
        for iteration, (frames, meta_data) in enumerate(val_dataloader_normal):


            if last_seq != str(meta_data[0]['seq_id'][0].item()):
                last_seq = str(meta_data[0]['seq_id'][0].item())
                print('Now for seq id: ', last_seq)

            device = 'cuda'
            batch_size = frames[0]['rgb'].shape[0]
            frames = to_device(frames, device)
            meta_data = to_device(meta_data, device)
            t_start_infer = time.time()
            preds, atts = EvRGBStereo_model(frames, return_att=True, decode_all=False)
            infer_time.update(time.time() - t_start_infer, batch_size)
            batch_time.update(time.time() - end)
            end = time.time()

            if iteration == 0:
                flops = FlopCountAnalysis(EvRGBStereo_model, frames)
                print('FLOPs: {} G FLOPs'.format(flops.total() / batch_size / 1024**3))
                file.write('FLOPs: {} G FLOPs\n'.format(flops.total() / batch_size / 1024**3))
                # for parameters
                all_params = sum(p.numel() for p in EvRGBStereo_model.parameters()) / 1024.**2
                print('Params: {} M'.format(all_params))
                file.write('Params: {} M\n'.format(all_params))
                if config['model']['method']['framework'] != 'eventhands':
                    cnn_params = sum(p.numel() for p in EvRGBStereo_model.ev_backbone.parameters()) / 1024.**2
                    print('each CNN params: {} M'.format(cnn_params))
                    file.write('each CNN params: {} M\n'.format(cnn_params))
            for step in range(steps):

                bbox_valid = meta_data[step]['bbox_valid']
                mano_valid = meta_data[step]['mano_valid'] * bbox_valid
                joints_3d_valid = meta_data[step]['joints_3d_valid'] * bbox_valid

                gt_3d_joints = meta_data[step]['3d_joints']
                gt_3d_joints_sub = gt_3d_joints - gt_3d_joints[:, :1]
                pred_3d_joints = preds[step][-1]['pred_3d_joints']
                pred_3d_joints_sub = pred_3d_joints - pred_3d_joints[:, :1]
                mpjpe_eachjoint_eachitem = torch.sqrt(
                    torch.sum((pred_3d_joints_sub - gt_3d_joints_sub) ** 2, dim=-1)) * 1000.
                update_errors_list(errors_list[0][step], mpjpe_eachjoint_eachitem[joints_3d_valid], meta_data[step]['seq_type'][joints_3d_valid])
                align_pred_3d_joints_sub = compute_similarity_transform_batch(pred_3d_joints_sub, gt_3d_joints_sub)
                pa_mpjpe_eachjoint_eachitem = torch.sqrt(torch.sum((align_pred_3d_joints_sub - gt_3d_joints_sub.detach().cpu()) ** 2, dim=-1)) * 1000.
                update_errors_list(errors_list[1][step], pa_mpjpe_eachjoint_eachitem[joints_3d_valid], meta_data[step]['seq_type'][joints_3d_valid])

                manos = meta_data[step]['mano']
                gt_dest_mano_output = mano_layer(
                    global_orient=manos['rot_pose'].reshape(-1, 3),
                    hand_pose=manos['hand_pose'].reshape(-1, 45),
                    betas=manos['shape'].reshape(-1, 10),
                    transl=manos['trans'].reshape(-1, 3)
                )
                gt_vertices_sub = gt_dest_mano_output.vertices - gt_dest_mano_output.joints[:, :1, :]
                error_vertices = torch.mean(torch.abs(preds[step][-1]['pred_vertices'] - gt_vertices_sub), dim=(1, 2))
                update_errors_list(errors_list[2][step], error_vertices[mano_valid], meta_data[step]['seq_type'][mano_valid])

                aligned_pred_vertices = compute_similarity_transform_batch(preds[step][-1]['pred_vertices'], gt_vertices_sub)
                pa_error_vertices = torch.mean(torch.abs(aligned_pred_vertices - gt_vertices_sub.detach().cpu()), dim=(1, 2))
                update_errors_list(errors_list[3][step], pa_error_vertices[mano_valid], meta_data[step]['seq_type'][mano_valid])

            # if iteration*config['exper']['per_gpu_batch_size'] % 20 != 0:
            #     continue
            if config['eval']['output']['save']:
                predicted_meshes = preds[0][-1]['pred_vertices'] + meta_data[0]['3d_joints'][:, :1]
                # for i in range(rgb.shape[0]):
                #     seq_dir = op.join(config['exper']['output_dir'], str(meta_data['seq_id'][i].item()))
                #     mkdir(seq_dir)
                #     if config['eval']['output']['mesh']:
                #         mesh_dir = op.join(seq_dir, 'mesh')
                #         mkdir(mesh_dir)
                #         with open(os.path.join(mesh_dir, '{}.obj'.format(meta_data['annot_id'][i].item())), 'w') as file_object:
                #             for ver in predicted_meshes[i].detach().cpu().numpy():
                #                 print('v %f %f %f'%(ver[0], ver[1], ver[2]), file=file_object)
                #             faces = mano_layer.faces
                #             for face in faces:
                #                 print('f %d %d %d'%(face[0]+1, face[1]+1, face[2]+1), file=file_object)

                # if config['eval']['output']['attention_map']:
                #     id_tmp = 0
                #     annot_dir = op.join(config['exper']['output_dir'], str(meta_data['seq_id'][id_tmp].item()), 'attention', str(meta_data['annot_id'][id_tmp].item()))
                #     mkdir(annot_dir)
                #     encoder_ids = range(len(config['model']['tfm']['input_feat_dim']))
                #     layer_ids = range(config['model']['tfm']['num_hidden_layers'])
                #     for encoder_id in encoder_ids:
                #         for layer_id in layer_ids:
                #             if 'att' in preds.keys():
                #
                #                 att_map = preds['att'][encoder_id][layer_id][id_tmp]
                #
                #                 attention_all = np.sum(att_map.detach().cpu().numpy(), axis=0)
                #                 max_j_all = np.max(attention_all, axis=1, keepdims=True)
                #                 min_j_all = np.min(attention_all, axis=1, keepdims=True)
                #                 attention_all_normal = (attention_all - min_j_all) / (max_j_all - min_j_all)
                #                 fig_, axes_ = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
                #                 axes_.title.set_text('Attention Map All')
                #                 axes_.imshow(attention_all_normal, cmap="inferno")
                #                 fig_.savefig(op.join(annot_dir, 'attention_all_encoder{}_layer{}.png'.format(encoder_id, layer_id)))
                #                 fig_.clear()
                #
                #                 shapes = np.array([16, 36, 16], dtype=np.int32)
                #                 cnn_shape = shapes * np.array(config['model']['method']['ere_usage'])
                #                 att_map_all = np.zeros((21, cnn_shape.sum()), dtype=np.float32)
                #                 for i in range(config['model']['tfm']['num_attention_heads']):
                #                     att_map_all += att_map[i][:21, -cnn_shape.sum():].detach().cpu().numpy()
                #                 max_j = np.max(att_map_all, axis=1, keepdims=True)
                #                 min_j = np.min(att_map_all, axis=1, keepdims=True)
                #                 att_map_joints = (att_map_all - min_j) / (max_j - min_j)
                #                 att_map_joints = att_map_joints[indices_change(1, 2)]
                #                 fig, axes = plt.subplots(nrows=3, ncols=7*len(cnn_shape), figsize=(12*len(cnn_shape), 6))
                #                 for i in range(3):
                #                     for j in range(7):
                #                         joint_id = 7 * i + j
                #                         col = j * sum(config['model']['method']['ere_usage'])
                #                         if config['model']['method']['ere_usage'][0]:
                #                             axes[i, col].imshow(l_ev_frame[id_tmp].detach().cpu().numpy())
                #                             att_map_ = cv2.resize(att_map_joints[joint_id, :cnn_shape[0]].reshape(4, 4),
                #                                                   (l_ev_frame.shape[1:3]),
                #                                                   interpolation=cv2.INTER_NEAREST)
                #                             axes[i, col].imshow(att_map_, cmap="inferno", alpha=0.6)
                #                             axes[i, col].title.set_text(f"Event {cfg.J_NAME[joint_id]}")
                #                             axes[i, col].axis("off")
                #                             col += 1
                #                         if config['model']['method']['ere_usage'][1]:
                #                             axes[i, col].imshow(rgb[id_tmp].detach().cpu().numpy())
                #                             att_map_ = cv2.resize(att_map_joints[joint_id, cnn_shape[0]:sum(cnn_shape[:2])].reshape(6, 6), (rgb.shape[1:3]), interpolation=cv2.INTER_NEAREST)
                #                             axes[i, col].imshow(att_map_, cmap="inferno", alpha=0.6)
                #                             axes[i, col].title.set_text(f"RGB {cfg.J_NAME[joint_id]}")
                #                             axes[i, col].axis("off")
                #                             col += 1
                #
                #                 # fig.show()
                #                 fig.savefig(op.join(annot_dir, 'encoder{}_layer{}.png'.format(encoder_id, layer_id)))
                #                 fig.clear()
                #
                #         # todo draw attention map
                #         pass
                #
                if config['eval']['output']['rendered']:
                    key = config['eval']['output']['vis_rendered']
                    if key == 'rgb':
                        img_bg = frames[0][key+'_ori']
                        _, h, w, _ = img_bg.shape
                        hw = [h, w]
                    else:
                        img_bg = frames[0][key + '_ori'][-1]
                        _, h, w, _ = img_bg.shape
                        hw = [h, w]
                    # print(meta_data[0]['K_'+key].device)
                    # print(img_bg.device)
                    # print(predicted_meshes.device)
                    img_render = render.visualize(
                        K=meta_data[0]['K_'+key].detach(),
                        R=meta_data[0]['R_'+key].detach(),
                        t=meta_data[0]['t_'+key].detach(),
                        hw=hw,
                        img_bg=img_bg.cpu(),
                        vertices=predicted_meshes.detach(),
                    )
                    for i in range(img_render.shape[0]):
                        img_dir = op.join(config['exper']['output_dir'], str(meta_data[0]['seq_id'][i].item()), 'rendered')
                        mkdir(img_dir)
                        imageio.imwrite(op.join(img_dir, '{}.jpg'.format(meta_data[0]['annot_id'][i].item())),
                                        (img_render[i].detach().cpu().numpy() * 255).astype(np.uint8))

            if (iteration - 1) % config['utils']['logging_steps'] == 0:
                eta_seconds = batch_time.avg * (len(val_dataloader_normal) / config['exper']['per_gpu_batch_size'] - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if is_main_process():
                    print(
                        ' '.join(
                            ['eta: {eta}', 'iter: {iter}', 'max mem : {memory:.0f}', ]
                        ).format(eta=eta_string, iter=iteration,
                                 memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
                        + '   compute: {:.4f}'.format(batch_time.avg)
                    )

        '''
        for fast sequences
        '''
        for iteration, (frames, meta_data) in enumerate(val_dataloader_fast):

            if last_seq != str(meta_data[0]['seq_id'][0].item()):
                last_seq = str(meta_data[0]['seq_id'][0].item())
                print('Now for seq id: ', last_seq)

            device = 'cuda'
            batch_size = frames[0]['rgb'].shape[0]
            frames = to_device(frames, device)
            meta_data = to_device(meta_data, device)

            preds, atts = EvRGBStereo_model(frames, return_att=True, decode_all=False)
            batch_time.update(time.time() - end)
            end = time.time()
            steps = len(preds)
            for step in range(steps):
                joints_2d_valid = meta_data[step]['joints_2d_valid_ev'] * meta_data[step]['bbox_valid']

                pred_3d_joints = preds[step][-1]['pred_3d_joints']
                pred_3d_joints_abs = pred_3d_joints + meta_data[step]['3d_joints'][:, :1]
                pred_3d_joints_abs = torch.bmm(meta_data[step]['R_event'].reshape(-1, 3, 3), pred_3d_joints_abs.transpose(2, 1)).transpose(2, 1) + meta_data[step]['t_event'].reshape(-1, 1, 3)
                # todo check here!!
                pred_2d_joints = torch.bmm(meta_data[step]['K_event'], pred_3d_joints_abs.permute(0, 2, 1)).permute(0, 2, 1)
                pred_2d_joints = pred_2d_joints[:, :, :2] / pred_2d_joints[:, :, 2:]
                gt_2d_joints = meta_data[step]['2d_joints_event']
                pred_2d_joints_aligned = pred_2d_joints - pred_2d_joints[:, :1] + gt_2d_joints[:, :1]
                # print('fast index', fast_index)
                mpjpe_eachjoint_eachitem = torch.sqrt(torch.sum((pred_2d_joints_aligned - gt_2d_joints) ** 2, dim=-1))
                # print('mpjpe shape', mpjpe_eachjoint_eachitem.shape)
                # print('seq type shape', meta_data['seq_type'][fast_index])
                update_errors_list(errors_list[0][step], mpjpe_eachjoint_eachitem[joints_2d_valid], meta_data[step]['seq_type'][joints_2d_valid])

    for step in range(steps):
        file.write('Step: {}'.format(step) + '\n')
        for i, metric in enumerate(metrics):
            print_items(errors_list[i][step], labels_list, metric, file)
            file.write('\n')
        file.write('\n\n')

    for step in range(steps):
        for i, key in enumerate(labels_list):
            if len(errors_list[0][step][i]) != 0:
                errors_seq = torch.stack(errors_list[0][step][i])
                print_sequence_error(errors_seq, metrics[0], labels_list[i]+'_step'+str(step), os.path.join(config['exper']['output_dir'], 'seq_errors'))

    # if config['eval']['output']['save']:
    #     if config['eval']['output']['errors']:
    #         error_dir = op.join(config['exper']['output_dir'], 'error')
    #         mkdir(error_dir)
    #         torch.save(l_mpjpe_errors, os.path.join(error_dir, 'l_mpjpe_errors.pt'))
    #         torch.save(l_vertices_errors, os.path.join(error_dir, 'l_vertices_errors.pt'))
    #         torch.save(l_pa_mpjpe_errors, os.path.join(error_dir, 'l_pa_mpjpe_errors.pt'))
    #         torch.save(l_pa_vertices_errors, os.path.join(error_dir, 'l_pa_vertices_errors.pt'))


    print('Inference time each item: {}'.format(infer_time.avg))
    file.write('Inference time each item: {}\n'.format(infer_time.avg))
    file.close()

    total_eval_time = time.time() - start_eval_time
    total_time_str = str(datetime.timedelta(seconds=total_eval_time))
    print('Total eval time: {} ({:.4f} s / iter)'.format(
        total_time_str, total_eval_time / ((len(val_dataloader_normal) + len(val_dataloader_fast)) / config['exper']['per_gpu_batch_size']))
    )

    return

def run_eval_and_save(config, val_dataloader, EvRGBStereo_model):
    mano_layer = MANO(config['data']['smplx_path'], use_pca=False, is_rhand=True).cuda()
    if config['exper']['distributed']:
        if os.environ["LOCAL_RANK"] is None:
            EvRGBStereo_model = torch.nn.parallel.DistributedDataParallel(
                EvRGBStereo_model, device_ids=[int(config["LOCAL_RANK"])],
                output_device=int(config["LOCAL_RANK"]),
                find_unused_parameters=True,
            )
            mano_layer = torch.nn.parallel.DistributedDataParallel(
                mano_layer, device_ids=[int(config["LOCAL_RANK"])],
                output_device=int(config["LOCAL_RANK"]),
                find_unused_parameters=True,
            )
        else:
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


def get_config():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--local_rank', type=int, default=0)
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
    config["LOCAL_RANK"] = args.local_rank
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
        if is_main_process():
            temp_path = os.path.join(os.getcwd(), 'temp')
            folder = PackedFolder(config['data']['dataset_info']['evrealhands']['ffr_dir'])
            if os.path.exists(temp_path) and os.path.exists(os.path.join(temp_path, "EvRealHands",'0',"event.aedat4")):
                pass
            else:
                os.makedirs(temp_path)
                # tar = tarfile.open(config['data']['dataset_info']['evrealhands']['data_dir'],"r")
                # name_list =[i for i in tar.getmembers() if i.name.endswith(".aedat4")]
                # tar.extractall(path=temp_path, members=name_list)
                seq_ids = folder.list("")
                for seq_id in seq_ids:
                    if folder.is_dir(seq_id):
                        if not os.path.exists(os.path.join(temp_path, seq_id)):
                            os.makedirs(os.path.join(temp_path, "EvRealHands",seq_id))
                        event_path = os.path.join(seq_id, "event.aedat4")
                        fp = io.BytesIO(folder.read_one(event_path))
                        with open(os.path.join(temp_path, "EvRealHands",seq_id,"event.aedat4"), 'wb') as f:
                            f.write(fp.read())

        synchronize()

    mkdir(config['exper']['output_dir'])
    # mkdir(os.path.join(config['exper']['output_dir'], 'logger'))

    # logger = setup_logger("EvRGBStereo", config['exper']['output_dir'], get_rank())
    if is_main_process() and not config['exper']['run_eval_only']:
        mkdir(os.path.join(config['exper']['output_dir'], 'tf_logs'))
        tf_logger = SummaryWriter(os.path.join(config['exper']['output_dir'], 'tf_logs'))
    else:
        tf_logger = None

    set_seed(config['exper']['seed'], num_gpus)
    print("Using {} GPUs".format(num_gpus))

    start_iter = 0
    if config['exper']['run_eval_only'] == True and config['exper']['resume_checkpoint'] != None and\
            config['exper']['resume_checkpoint'] != 'None' and 'state_dict' not in config['exper']['resume_checkpoint']:
        # if only run eval, load checkpoint
        print("Evaluation: Loading from checkpoint {}".format(config['exper']['resume_checkpoint']))
        _model = torch.load(config['exper']['resume_checkpoint'])

    else:

        # build network
        _model = EvRGBStereo(config=config)
        
        if os.path.exists(os.path.join(config['exper']['output_dir'], 'latest.ckpt')):
            print("Loading from recent...")
            latest_dict = torch.load(os.path.join(config['exper']['output_dir'], 'latest.ckpt'),map_location=torch.device('cpu'))
            _model.load_state_dict(latest_dict["model"])
            start_iter = latest_dict['iteration']
            del latest_dict
            gc.collect()
            torch.cuda.empty_cache()
            print("finish loading model")



        if config['exper']['resume_checkpoint'] != None and config['exper']['resume_checkpoint'] != 'None':
            # for fine-tuning or resume training or inference, load weights from checkpoint
            print("Loading state dict from checkpoint {}".format(config['exper']['resume_checkpoint']))
            # workaround approach to load sparse tensor in graph conv.
            state_dict = torch.load(config['exper']['resume_checkpoint'], map_location=torch.device('cpu'))
            _model.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()

    _model.to(config['exper']['device'])
    if is_main_process():
        print("Training parameters %s", str(config))

    _loss = Loss(config)
    _loss.to(config['exper']['device'])

    if config['exper']['run_eval_only'] == True:
        val_dataloader_normal = make_hand_data_loader(config)
        config_fast = copy.deepcopy(config)
        config_fast['eval']['fast'] = True
        val_dataloader_fast = make_hand_data_loader(config_fast)
        if config['eval']['multiscale']:
            run_eval_and_save(config, val_dataloader_normal, val_dataloader_fast, _model)
        else:
            run_eval_and_show(config, val_dataloader_normal, val_dataloader_fast, _model, _loss)

    else:
        train_dataloader = make_hand_data_loader(config,start_iter)
        run(config, train_dataloader, _model, _loss)

    if is_main_process() and not config['exper']['run_eval_only']:
        tf_logger.close()


if __name__ == "__main__":
    config = get_config()
    main(config)
