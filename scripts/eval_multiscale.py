import argparse
import os
import os.path as op
from os.path import dirname
os.chdir(dirname(os.getcwd()))
import os.path as op
import torch
import numpy as np
import warnings
from src.configs.config_parser import ConfigParser
import cv2
import copy
import imageio
import matplotlib.pyplot as plt
from src.utils.metric_logger import AverageMeter
from src.utils.comm import synchronize, is_main_process, get_rank, get_world_size, all_gather, to_device
import src.modeling.data.config as cfg
from src.datasets.build import make_hand_data_loader
from src.utils.joint_indices import indices_change
from src.modeling._mano import MANO, Mesh
from src.utils.renderer import Render
from src.utils.miscellaneous import mkdir, set_seed
import time, datetime

global rotations, scale
rotations = [0.0]
for i in range(1, 10):
    rotations.append(i * 0.02)
    rotations.append(i * -0.02)
scale = [0.9, 1.0, 1.1, 0.8, 1.2]

def multiscale_fusion(args):
    output_dir = args.output_dir
    filepath = op.join(output_dir, 'pred_scale100_rot0_.pt')
    preds_joints_, preds_vertices_ = torch.load(filepath)
    preds_count = {}
    for seq_id in preds_joints_.keys():
        if seq_id not in preds_count.keys():
            preds_count[seq_id] = {}
        for annot_id in preds_joints_[seq_id].keys():
            preds_count[seq_id][annot_id] = 1

    for s in scale:
        for r in rotations:
            if s == 1. and r == 0.:
                continue
            filename = 'pred_scale{:.0f}_rot{:.0f}_.pt'.format(s * 100, r * 100)
            path_tmp = op.join(output_dir, filename)
            preds_joints_tmp, preds_vertices_tmp = torch.load(path_tmp)
            for seq_id in preds_joints_.keys():
                if seq_id not in preds_joints_tmp.keys():
                    continue
                for annot_id in preds_joints_[seq_id].keys():
                    if annot_id not in preds_joints_tmp[seq_id].keys():
                        continue
                    else:
                        preds_joints_[seq_id][annot_id] += preds_joints_tmp[seq_id][annot_id]
                        preds_vertices_[seq_id][annot_id] += preds_vertices_tmp[seq_id][annot_id]
                        preds_count[seq_id][annot_id] += 1

    for seq_id in preds_joints_.keys():
        for annot_id in preds_joints_[seq_id].keys():
            preds_joints_[seq_id][annot_id] /= preds_count[seq_id][annot_id]
            preds_vertices_[seq_id][annot_id] /= preds_count[seq_id][annot_id]

    torch.save([preds_joints_, preds_vertices_], op.join(output_dir, 'pred_.pt'))

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


def run_multiscale_inference(args):
    job_cmd = "CUDA_VISIBLE_DEVICES=%s " \
              "/userhome/alanjjp/tools/miniconda3/envs/evrgb/bin/python scripts/train.py " \
              "--config %s " \
              "--resume_checkpoint %s " \
              "--config_merge %s "\
              "--run_eval_only " \
              "--r %f " \
              "--s %f " \
              "--output_dir %s"
    print(rotations)
    print(scale)
    for s in scale:
        for r in rotations:
            resolved_submit_cmd = job_cmd % (args.gpu, args.config, args.resume_checkpoint,
                                             args.config_merge, r, s, args.output_dir)
            print(resolved_submit_cmd)
            os.system(resolved_submit_cmd)

def update_errors_list(errors_list, mpjpe_eachjoint_eachitem, index):
    if len(mpjpe_eachjoint_eachitem) != 0:
        for i, id in enumerate(index):
            errors_list[id].append(mpjpe_eachjoint_eachitem[i].detach().cpu())

def eval_multiscale_inference(args):
    config = ConfigParser(args.config)
    if args.config_merge != '':
        config.merge_configs(args.config_merge)
    config = config.config
    if args.output_dir != '':
        config['exper']['output_dir'] = args.output_dir
    if args.resume_checkpoint != '':
        config['exper']['resume_checkpoint'] = args.resume_checkpoint
    config['exper']['run_eval_only'] = True
    dataset_config = ConfigParser(config['data']['dataset_yaml']).config
    config['data']['dataset_info'] = dataset_config
    if 'augment' not in config['eval'].keys():
        config['eval']['augment'] = {}
    config['eval']['augment']['scale'] = 1.
    config['eval']['augment']['rot'] = 0.

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

    set_seed(config['exper']['seed'], num_gpus)
    print("Using {} GPUs".format(num_gpus))

    val_dataloader = make_hand_data_loader(config)

    mano_layer = MANO(config['data']['smplx_path'], use_pca=False, is_rhand=True)

    if config['exper']['distributed']:
        mano_layer = torch.nn.parallel.DistributedDataParallel(
            mano_layer, device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            find_unused_parameters=True,
        )

    start_eval_time = time.time()
    end = time.time()

    batch_time = AverageMeter()

    labels_list = ['n_f', 'n_r', 'h_f', 'h_r', 'f_f', 'f_r', 'fast']
    mpjpe_errors_list = [[], [], [], [], [], [], []]
    vertices_errors_list = [[], [], [], [], [], []]

    mkdir(config['exper']['output_dir'])

    file = open(os.path.join(config['exper']['output_dir'], 'error_joints.txt'), 'a')

    last_seq = 0

    pred_joints, pred_vertices = torch.load(op.join(config['exper']['output_dir'], 'pred_.pt'))

    with torch.no_grad():
        for iteration, (frames, meta_data) in enumerate(val_dataloader):

            if last_seq != str(meta_data['seq_id'][0].item()):
                last_seq = str(meta_data['seq_id'][0].item())
                print('Now for seq id: ', last_seq)

            for i in range(frames['rgb'].shape[0]):
                seq_id = meta_data['seq_id'][i].item()
                annot_id = meta_data['annot_id'][i].item()
                if seq_id in pred_joints.keys() and annot_id in pred_joints[seq_id].keys():
                    tf_c_w = meta_data['tf_w_c'][i].inverse()
                    pred_joints_tmp = (tf_c_w[:3, :3] @ pred_joints[seq_id][annot_id].transpose(0, 1) + tf_c_w[:3, 3:]).transpose(0, 1)
                    pred_vertices_tmp = (tf_c_w[:3, :3] @ pred_vertices[seq_id][annot_id].transpose(0, 1) + tf_c_w[:3, 3:]).transpose(0, 1)
                    if meta_data['seq_type'][i] == 6:
                        if meta_data['joints_2d_valid_ev'][i]:
                            pred_3d_joints_abs = pred_joints_tmp
                            pred_2d_joints = torch.mm(meta_data['K_event_r'][i],
                                                       pred_3d_joints_abs.permute(1, 0)).permute(1, 0)
                            pred_2d_joints = pred_2d_joints[:, :2] / pred_2d_joints[:, 2:]
                            gt_2d_joints = meta_data['2d_joints_event'][i]
                            pred_2d_joints_aligned = pred_2d_joints - pred_2d_joints[:1] + gt_2d_joints[:1]
                            # print('fast index', fast_index)
                            mpjpe_eachjoint_eachitem = torch.sqrt(torch.sum((pred_2d_joints_aligned - gt_2d_joints) ** 2, dim=-1))

                            update_errors_list(mpjpe_errors_list, mpjpe_eachjoint_eachitem[None, ...], meta_data['seq_type'][i:i+1])
                    else:
                        gt_3d_joints = meta_data['3d_joints'][i]
                        gt_3d_joints_sub = gt_3d_joints - gt_3d_joints[:1]
                        pred_joints_sub = pred_joints_tmp - pred_joints_tmp[:1]
                        mpjpe_eachjoint_eachitem = torch.sqrt(
                            torch.sum((pred_joints_sub - gt_3d_joints_sub) ** 2, dim=-1)) * 1000.
                        update_errors_list(mpjpe_errors_list, mpjpe_eachjoint_eachitem[None, ...],
                                           meta_data['seq_type'][i:i+1])

                        manos = meta_data['mano']
                        gt_dest_mano_output = mano_layer(
                            global_orient=manos['rot_pose'][i].reshape(-1, 3),
                            hand_pose=manos['hand_pose'][i].reshape(-1, 45),
                            betas=manos['shape'][i].reshape(-1, 10),
                            transl=manos['trans'][i].reshape(-1, 3)
                        )
                        gt_vertices_sub = gt_dest_mano_output.vertices[0] - gt_dest_mano_output.joints[0, :1, :]
                        error_vertices = torch.mean(torch.abs(pred_vertices_tmp - gt_vertices_sub))
                        update_errors_list(vertices_errors_list, error_vertices[None, ...], meta_data['seq_type'][i:i+1])
            batch_time.update(time.time() - end)
            end = time.time()
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
    print('Total training time: {} ({:.4f} s / iter)'.format(
        total_time_str, total_eval_time / (len(val_dataloader) / config['exper']['per_gpu_batch_size']))
    )


def main(args):
    # todo
    run_multiscale_inference(args)
    multiscale_fusion(args)
    eval_multiscale_inference(args)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser('Multi-Scale Evaluation')
    parser.add_argument('--config', type=str, default='./src/configs/train_evrealhands.yaml')
    parser.add_argument('--resume_checkpoint', type=str, default='')
    parser.add_argument('--config_merge', type=str, default='')
    parser.add_argument('--output_dir', type=str,
                        default='./output/test0')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    # # TODO
    # rotations = rotations[:3]
    # scale = scale[:2]
    main(args)