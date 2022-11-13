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
from tqdm import tqdm

from torchvision.models import resnet

from src.configs.config_parser import ConfigParser
from src.datasets.EvRealHands import EvRealHands
from src.datasets.Interhand import Interhand

# colors_list = [(0., 0.7, 0.), (0.0, 1.0, 0.), (0.5, 0.5, 1.),
#               (0., 0., 1), (1., 0.5, 0.5), (1., 0., 0.),
#                (0.7, 0, 0.7)]
import matplotlib.patches as mpatches


colors_list = ['r', 'g', 'b', 'gold']

labels = ['EventHands', 'FastMETRO-Event', 'EvRGBHand-Large', 'EvRGBHand-Fast']
# markers = ['o', '^', '^', 's', 's', '*']
# linestyles = ['--', ':', '--', ':', '--', ':', '-']
markers = ['o', '^', ]
linestyles = ['-', '-']

font_size = 18
marker_size = 6
line_width = 2

output_dir = '/userhome/alanjjp/Project/MeshGraphormer/scripts/forpaper/performance_steps'

steps = range(1, 10)

# Large EvRGBHand
MPJPES_normal_large = [
    16.34, 18.92, 21.33, 23.71, 25.94, 28.15, 30.34, 32.38, 34.40,
]

MPJPES_stronglight_large = [
    27.71, 29.31, 30.64, 31.73, 32.97, 34.30, 35.91, 37.53, 39.56,
]

MPJPES_flash_large = [
    27.21, 30.59, 33.89, 37.33, 40.22, 43.29, 46.71, 49.81, 52.75,
]

MPVPES_normal_large = [
    6.17, 7.35, 8.42, 9.43, 10.37, 11.29, 12.18, 13.04, 13.87,
]

MPVPES_stronglight_large = [
    10.61, 11.38, 11.97, 12.39, 12.94, 13.52, 14.20, 14.87, 15.59,
]

MPVPES_flash_large = [
    10.14, 11.83, 13.42, 14.94, 16.33, 17.69, 19.18, 20.48, 21.74,
]

large_res = np.array(
    [MPJPES_normal_large, MPJPES_stronglight_large, MPJPES_flash_large, MPVPES_normal_large, MPVPES_stronglight_large, MPVPES_flash_large]
)


# Fast EvRGBHand
MPJPES_normal_fast = [
    16.49, 18.75, 20.68, 22.36, 23.84, 25.23, 26.54, 27.85, 29.05,
]

MPJPES_stronglight_fast = [
    30.45, 30.86, 31.28, 31.67, 32.13, 32.77, 33.60, 34.50, 35.41,
]

MPJPES_flash_fast = [
    27.86, 30.32, 32.91, 35.43, 37.86, 39.91, 42.01, 44.08, 46.06,
]

MPVPES_normal_fast = [
    6.22, 7.25, 8.15, 8.90, 9.57, 10.20, 10.79, 11.35, 11.88,
]

MPVPES_stronglight_fast = [
    11.71, 11.90, 12.15, 12.35, 12.62, 12.88, 13.20, 13.56, 13.91,
]

MPVPES_flash_fast = [
    10.69, 12.05, 13.43, 14.65, 15.78, 16.77, 17.71, 18.65, 19.54,
]

large_res = np.array(
    [MPJPES_normal_large, MPJPES_stronglight_large, MPJPES_flash_large, MPVPES_normal_large, MPVPES_stronglight_large, MPVPES_flash_large],
    dtype=np.float32
)

fast_res = np.array(
    [MPJPES_normal_fast, MPJPES_stronglight_fast, MPJPES_flash_fast, MPVPES_normal_fast, MPVPES_stronglight_fast, MPVPES_flash_fast],
    dtype=np.float32
)

FastMETRO_res = np.array([
    21.62, 29.06, 39.58, 8.67, 11.49, 16.51,
], dtype=np.float32)

FastMETRO_res = FastMETRO_res[:, None].repeat(9, axis=1)

EventHands_res = np.array([
    27.98, 33.47, 46.75, 11.56, 14.03, 20.01,
], dtype=np.float32)
EventHands_res = EventHands_res[:, None].repeat(9, axis=1)


y_lim_large = [35, 40, 55]

title_names = ['Normal scenes', 'Strong light scenes', 'Flash scenes']
pdf_names = ['normal', 'stronglight', 'flash']

for scene in range(3):
    lines = []

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6.5)
    ax.set_title(title_names[scene], fontsize=font_size+3, pad=30)
    ax.set_xticks(np.arange(1, 10, 1))
    ax.set_yticks(np.arange(5, y_lim_large[scene], 10))
    ax.set_xlim((0.5, 9.5))
    ax.set_ylim((5, y_lim_large[scene]))
    ax.set_xlabel('Step', fontsize=font_size)
    ax.set_ylabel('Error (mm)', fontsize=font_size)


    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(font_size-2)
    for error_type in range(2):
        index = np.arange(1, 10, 1)
        line, = ax.plot(index, EventHands_res[scene+error_type*3], color=colors_list[0], linewidth=line_width, linestyle=linestyles[error_type], marker=markers[error_type], markersize=marker_size)
        lines.append(line)
        line, = ax.plot(index, FastMETRO_res[scene+error_type*3], color=colors_list[1], linewidth=line_width, linestyle=linestyles[error_type], marker=markers[error_type], markersize=marker_size)
        lines.append(line)
        line, = ax.plot(index, large_res[scene+error_type*3], color=colors_list[2], linewidth=line_width, linestyle=linestyles[error_type], marker=markers[error_type], markersize=marker_size)
        lines.append(line)
        line, = ax.plot(index, fast_res[scene+error_type*3], color=colors_list[3], linewidth=line_width, linestyle=linestyles[error_type], marker=markers[error_type], markersize=marker_size)
        lines.append(line)

    patches = []
    for i, color in enumerate(colors_list):
        patches.append(mpatches.Patch(color=color, label=labels[i]))
    lines_new = []
    line, = ax.plot(index, fast_res[0]+100, label='MPJPE', color='gray', linewidth=line_width, linestyle=linestyles[0], marker=markers[0], markersize=marker_size)
    lines_new.append(line)
    patches.append(line)
    line, = ax.plot(index, fast_res[0]+100, label='MPVPE', color='gray', linewidth=line_width, linestyle=linestyles[1], marker=markers[1], markersize=marker_size)
    lines_new.append(line)
    patches.append(line)

    # ax.legend(patches, loc='upper right', title_fontsize=font_size - 2, prop={'size': font_size - 2})
    plt.legend(handles=patches, loc='upper left', title_fontsize=font_size - 3, prop={'size': font_size - 3})
    plt.tight_layout()
    # plt.show()
    mkdir(output_dir)
    fig.savefig(os.path.join(output_dir, '{}.pdf'.format(pdf_names[scene])))




# def get_config():
#     parser = argparse.ArgumentParser('Training')
#     parser.add_argument('--config', type=str, default='/userhome/wangbingxuan/data/hfai_output/final/p-full-iter/train.yaml')
#     #parser.add_argument('--config', type=str, default='src/configs/test_for_paper.yaml')
#     parser.add_argument('--config_merge', type=str, default='src/configs/eval_evrealhands_75.yaml')
#     # parser.add_argument('--config_merge', type=str, default='')
#     parser.add_argument('--output_dir', type=str,
#                         default='./output')
#     args = parser.parse_args()
#     config = ConfigParser(args.config)
#
#     if args.config_merge != '':
#         config.merge_configs(args.config_merge)
#     config = config.config
#     if args.output_dir != '':
#         config['exper']['output_dir'] = args.output_dir
#     dataset_config = ConfigParser(config['data']['dataset_yaml']).config
#     config['data']['dataset_info'] = dataset_config
#     config['exper']['debug'] = True
#     config['exper']['run_eval_only'] = True#True
#     config['eval']['fast'] = True
#     return config
#
# config = get_config()
#
#
# a = EvRealHands(config)
#
# output_dir = '/userhome/alanjjp/Project/MeshGraphormer/scripts/forpaper/highframerate_eci_75'
# mkdir(output_dir)
#
# flash = 0
# normal = 0
# highlight = 0
# fast = 0
# print(len(a))
# for i in range(0, len(a), 1):
#     frames, info = a[i]
#     # if info[0]['seq_id'] == 0 or info[0]['seq_id'] == 1:
#     #     if normal > 600:
#     #         continue
#     #     normal += 1
#     # if info[0]['seq_id'] == 2 or info[0]['seq_id'] == 3:
#     #     if highlight > 200:
#     #         continue
#     #     highlight += 1
#     # if info[0]['seq_id'] == 4 or info[0]['seq_id'] == 5:
#     #     if flash > 200:
#     #         continue
#     #     flash += 1
#     # if info[0]['seq_id'] == 6:
#     #     if fast > 200:
#     #         continue
#     #     fast += 1
#     for j in range(2):
#         for k in range(len(frames[j]['event_ori'])):
#             eci = frames[j]['event_ori'][k].numpy()
#             eci[..., 2] = 0
#             # pair = np.concatenate([rgb, eci], axis=1)
#             seq_id = str(info[0]['seq_id'].item())
#             annot_id = str(info[0]['annot_id'].item())
#             mkdir(os.path.join(output_dir, seq_id))
#             # imageio.imwrite(os.path.join(output_dir, seq_id, 'image_{}.png'.format(annot_id)), (rgb * 255).astype(np.uint8))
#             imageio.imwrite(os.path.join(output_dir, seq_id, 'eci_{}_step_{}_seg_{}.png'.format(annot_id, j, k)), (eci * 255).astype(np.uint8))



#     pass
#
# mano_layer = MANO(config['data']['smplx_path'], use_pca=False, is_rhand=True)
# for i in tqdm(range(len(a))):
#     item = a[i]
#     seq_id = item[1][0]['seq_id'].item()
#     annot_id = item[1][0]['annot_id'].item()
#     if (seq_id, annot_id) in [(2, 136), (0, 292), (26, 268), (53, 138)]:
#         mkdir(os.path.join(output_dir, str(seq_id)))
#         for step in range(len(item[1])):
#             manos = item[1][step]['mano']
#             mano_output = mano_layer(
#                 global_orient=manos['rot_pose'].reshape(-1, 3),
#                 hand_pose=manos['hand_pose'].reshape(-1, 45),
#                 betas=manos['shape'].reshape(-1, 10),
#                 transl=manos['trans'].reshape(-1, 3)
#             )
#             with open(os.path.join(output_dir, str(seq_id), 'mesh_{}.obj'.format(step)), 'w') as file_object:
#                 for ver in mano_output.vertices[0].detach().cpu():
#                     # for ver in y['vertices_inter'][0, k, i - 3].detach().cpu():
#                     print('v %f %f %f' % (ver[0], ver[1], ver[2]), file=file_object)
#                 for f in mano_layer.faces:
#                     print('f %d %d %d' % (f[0] + 1, f[1] + 1, f[2] + 1), file=file_object)



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


# dataset = EvRealHands(config)
#
# output_dir = './scripts/forpaper/teaser'
# mkdir(output_dir)
#
# item = dataset[92]
# t_l = 113 * 1e6 / 15. + dataset.data['1']['annot']['delta_time'] - 1e6 / 15.
# t_r = t_l + 7 * 1e6 / 15.
# events_all = dataset.data['1']['event']
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
# mano_layer = MANO(config['data']['smplx_path'], use_pca=False, is_rhand=True)
#
# HAND_JOINT_COLOR_Lst = [(0, 0, 0),
#                         (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
#                         (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
#                         (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255),
#                         (128, 0, 128), (128, 0, 128), (128, 0, 128), (128, 0, 128),
#                         (128, 128, 0), (128, 128, 0), (128, 128, 0), (128, 128, 0)]
#
# for step in range(len(item[1])):
#     joints_2d = item[1][step]['2d_joints_rgb'][indices_change(1, 2)].numpy()
#     img = (np.ones((920, 1064, 3))*255).astype(np.uint8)
#     for joint_idx, kp in enumerate(joints_2d):
#         cv2.circle(img, (int(kp[0]), int(kp[1])), 4, HAND_JOINT_COLOR_Lst[joint_idx], -1)
#         if joint_idx % 4 == 1:
#             cv2.line(img, (int(joints_2d[0][0]), int(joints_2d[0][1])), (int(kp[0]), int(kp[1])),
#                      HAND_JOINT_COLOR_Lst[joint_idx], 2)  # connect to root
#         elif joint_idx != 0:
#             cv2.line(img, (int(joints_2d[joint_idx - 1][0]), int(joints_2d[joint_idx - 1][1])),
#                      (int(kp[0]), int(kp[1])),
#                      HAND_JOINT_COLOR_Lst[joint_idx], 2)  # connect to root
#     imageio.imwrite(os.path.join(output_dir, 'kps_{}.png'.format(step)), img.astype(np.uint8))

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
