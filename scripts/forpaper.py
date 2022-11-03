from src.utils.dataset_utils import json_read, json_write
import os
os.chdir('/userhome/alanjjp/Project/MeshGraphormer')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import math
import imageio
from src.utils.miscellaneous import mkdir, set_seed
import pandas as pd
from pyntcloud import PyntCloud
from src.modeling._mano import MANO
from src.utils.joint_indices import indices_change


from src.configs.config_parser import ConfigParser
from src.datasets.EvRealHands import EvRealHands
from src.datasets.Interhand import Interhand

def get_config():
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--config', type=str, default='src/configs/test_for_paper.yaml')
    parser.add_argument('--config_merge', type=str, default='')
    parser.add_argument('--output_dir', type=str,
                        default='./output')
    args = parser.parse_args()
    config = ConfigParser(args.config)

    if args.config_merge != '':
        config.merge_configs(args.config_merge)
    config = config.config
    if args.output_dir != '':
        config['exper']['output_dir'] = args.output_dir
    dataset_config = ConfigParser(config['data']['dataset_yaml']).config
    config['data']['dataset_info'] = dataset_config
    config['exper']['debug'] = True
    config['exper']['run_eval_only'] = True#True
    config['eval']['fast'] = False
    return config

config = get_config()


# a = EvRealHands(config)
#
# output_dir = '/userhome/alanjjp/Project/MeshGraphormer/scripts/forpaper/evrealhands'
# mkdir(output_dir)
#
# flash = 0
# normal = 0
# highlight = 0
# fast = 0
#
# for i in range(0, len(a), 5):
#     frames, info = a[i]
#     if info[0]['seq_id'] == 0 or info[0]['seq_id'] == 1:
#         if normal > 200:
#             continue
#         normal += 1
#     if info[0]['seq_id'] == 2 or info[0]['seq_id'] == 3:
#         if highlight > 100:
#             continue
#         highlight += 1
#     if info[0]['seq_id'] == 4 or info[0]['seq_id'] == 5:
#         if flash > 100:
#             continue
#         flash += 1
#     if info[0]['seq_id'] == 6:
#         if fast > 100:
#             continue
#         fast += 1
#     rgb = frames[0]['rgb_ori'].numpy()
#     eci = frames[0]['event_ori'][0].numpy()
#     # pair = np.concatenate([rgb, eci], axis=1)
#     seq_id = str(info[0]['seq_id'].item())
#     annot_id = str(info[0]['annot_id'].item())
#     mkdir(os.path.join(output_dir, seq_id))
#     imageio.imwrite(os.path.join(output_dir, seq_id, 'image_{}.png'.format(annot_id)), (rgb * 255).astype(np.uint8))
#     imageio.imwrite(os.path.join(output_dir, seq_id, 'eci_{}.png'.format(annot_id)), (eci * 255).astype(np.uint8))
#     pass




# a = Interhand(config)
#
# davis_width = 346
# davis_height = 260
# factor = 1
# img_ori_height, img_ori_width = 512, 334
#
# # get the affine matrix from RGB frame to event frame, and this affine matrix will influence the camera intrinsics
# src_points = np.float32([[0, 0], [img_ori_width / 2 - 1, 0], [img_ori_width / 2 - 1, img_ori_height * factor]])
# dst_points = np.float32([[davis_width / 2 - img_ori_width / 2 * davis_height / factor / img_ori_height, 0],
#                          [davis_width / 2 - 1, 0],
#                          [davis_width / 2 - 1, davis_height - 1]])
# affine_matrix = cv2.getAffineTransform(src_points, dst_points)
#
# output_dir = '/userhome/alanjjp/Project/MeshGraphormer/scripts/forpaper/interhand_seq0'
# mkdir(output_dir)
#
# for i in range(0, len(a), 20):
#     frames, _ = a[i]
#     rgb = frames[0]['rgb_ori']
#     rgb_affine = cv2.warpAffine(rgb.numpy(), affine_matrix, (davis_width, davis_height))
#     eci = frames[0]['ev_frames_ori'][0]
#     rgb_affine = rgb_affine[:, 100:250, ...]
#     eci = eci[:, 100:250, ...].numpy()
#     pair = np.concatenate([rgb_affine, eci], axis=1)
#     imageio.imwrite(os.path.join(output_dir, 'eci_{}.png'.format(i)), (pair * 255).astype(np.uint8))
#     pass


dataset = EvRealHands(config)

output_dir = './scripts/forpaper/teaser'
mkdir(output_dir)

item = dataset[92]
t_l = 113 * 1e6 / 15. + dataset.data['1']['annot']['delta_time'] - 1e6 / 15.
t_r = t_l + 7 * 1e6 / 15.
events_all = dataset.data['1']['event']
# indices = np.searchsorted(
#             events_all[:, 3],
#             np.array([t_l, t_r])
#         )
#
# events = events_all[indices[0]:indices[1]]
#
# event_basic = np.concatenate([events[:, :2], events[:, 3:4]], axis=1)
# event_basic[:, 0] /= 346
# event_basic[:, 1] /= 260
# event_basic[:, 2] /= (1e6 / 15.)
# index_red = events[:, 2] == 0
# index_green = events[:, 2] == 1
# events_pc = np.concatenate([event_basic, np.ones_like(event_basic) * 255], axis=1)
# events_pc[index_red, 4:] = 0
# events_pc[index_green, 3] = 0
# events_pc[index_green, 5] = 0
# print(events_pc.shape)
# events_pc = events_pc[10::]
# data_pc = {
#     'x': events_pc[:, 0],
#     'y': events_pc[:, 1],
#     'z': events_pc[:, 2],
#     'red': events_pc[:, 3].astype(np.uint8),
#     'green': events_pc[:, 4].astype(np.uint8),
#     'blue': events_pc[:, 5].astype(np.uint8)
# }
# cloud = PyntCloud(pd.DataFrame(data=data_pc))
# cloud.to_file(os.path.join(output_dir, 'output_events.ply'))
#
mano_layer = MANO(config['data']['smplx_path'], use_pca=False, is_rhand=True)

HAND_JOINT_COLOR_Lst = [(0, 0, 0),
                        (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
                        (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
                        (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255),
                        (128, 0, 128), (128, 0, 128), (128, 0, 128), (128, 0, 128),
                        (128, 128, 0), (128, 128, 0), (128, 128, 0), (128, 128, 0)]

for step in range(len(item[1])):
    joints_2d = item[1][step]['2d_joints_rgb'][indices_change(1, 2)].numpy()
    img = (np.ones((920, 1064, 3))*255).astype(np.uint8)
    for joint_idx, kp in enumerate(joints_2d):
        cv2.circle(img, (int(kp[0]), int(kp[1])), 4, HAND_JOINT_COLOR_Lst[joint_idx], -1)
        if joint_idx % 4 == 1:
            cv2.line(img, (int(joints_2d[0][0]), int(joints_2d[0][1])), (int(kp[0]), int(kp[1])),
                     HAND_JOINT_COLOR_Lst[joint_idx], 2)  # connect to root
        elif joint_idx != 0:
            cv2.line(img, (int(joints_2d[joint_idx - 1][0]), int(joints_2d[joint_idx - 1][1])),
                     (int(kp[0]), int(kp[1])),
                     HAND_JOINT_COLOR_Lst[joint_idx], 2)  # connect to root
    imageio.imwrite(os.path.join(output_dir, 'kps_{}.png'.format(step)), img.astype(np.uint8))

# for step in range(len(item[1])):
#     manos = item[1][step]['mano']
#     mano_output = mano_layer(
#         global_orient=manos['rot_pose'].reshape(-1, 3),
#         hand_pose=manos['hand_pose'].reshape(-1, 45),
#         betas=manos['shape'].reshape(-1, 10),
#         transl=manos['trans'].reshape(-1, 3)
#     )
#     with open(os.path.join(output_dir, 'mesh_{}.obj'.format(step)), 'w') as file_object:
#         for ver in mano_output.vertices[0].detach().cpu():
#             # for ver in y['vertices_inter'][0, k, i - 3].detach().cpu():
#             print('v %f %f %f' % (ver[0], ver[1], ver[2]), file=file_object)
#         for f in mano_layer.faces:
#             print('f %d %d %d' % (f[0] + 1, f[1] + 1, f[2] + 1), file=file_object)

print('over!')
