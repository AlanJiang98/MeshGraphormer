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



def plot_3D_PCK(all_joints_error_: list, colors:list, labels:list, dir:str, filename: str, name='the'):
    '''
    plot 3D PCK and return AUC list
    :param all_joints_error: list of all joints error
    :param colors: color list
    :param labels: label list
    :param dir: output direction
    :return: list of AUCs
    '''
    assert len(all_joints_error_) == len(colors)
    font_size = 18
    x_max_percent = 0.98
    x_start = 0.
    x_end = 100000.
    all_joints_error = []
    # print(len(all_joints_error_))

    for i in range(len(all_joints_error_)):
        # print(all_joints_error_[i].shape)
        all_joints_error.append(all_joints_error_[i][:, 1:].clone())
        all_joints_error[i], indices = torch.sort(all_joints_error[i].reshape(-1))
        x_end_tmp = all_joints_error[i][int(x_max_percent * len(all_joints_error[i]))]
        if x_end_tmp < x_end:
            x_end = x_end_tmp
    step = 0.1
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    x_end = 100.
    ax.set_xlim((x_start, x_end))
    ax.set_ylim((0., 1.))
    ax.grid(True)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(font_size)

    legend_labels = []
    lines = []
    AUCs = []

    for method_id in range(len(all_joints_error)):
        color = colors[method_id]
        errors = all_joints_error[method_id]
        x_axis = torch.arange(x_start, x_end, step)
        pcks = torch.searchsorted(errors, x_axis) / errors.shape[0]
        AUC = torch.sum((pcks * step) / (x_end - x_start))
        AUCs.append(AUC)
        label = labels[method_id]
        label += ' AUC:%.03f' % AUC
        line, = ax.plot(x_axis, pcks, color=color, linewidth=2)
        lines.append(line)
        legend_labels.append(label)

    legend_location = 4
    ax.legend(lines, legend_labels, loc=legend_location, title_fontsize=font_size-2,
               prop={'size': font_size-2})
    ax.set_xlabel('error (mm)', fontsize=font_size)
    ax.set_title('3D PCK on ' + name + ' scenes', fontsize=font_size+4, pad=30)
    # plt.ylabel('3D-PCK', fontsize=font_size)
    plt.tight_layout()
    # plt.show()
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, filename))
    fig.clear()
    return AUCs



event_labels_methods = ['EventHands', 'FastMETRO-Event', 'EvRGBHand-Large', 'EvRGBHand-Fast']
rgb_labels_methods = ['Mesh Graphormer', 'FastMETRO-RGB', 'EvRGBHand-Large', 'EvRGBHand-Fast']

error_dir_leaves = [
    'n_f', 'n_r',
    'h_f', 'h_r',
    'f_f', 'f_r'
]

rgb_method_dirs = [
    '/userhome/alanjjp/Project/MeshGraphormer/output/final/final/g-rgb',
    '/userhome/alanjjp/Project/MeshGraphormer/output/final/final/f-rgb',
    '/userhome/alanjjp/Project/MeshGraphormer/output/final/final/p-full-iter',
    '/userhome/alanjjp/Project/MeshGraphormer/output/final/final/p-iter-S-resnet34',
]

event_method_dirs = [
    '/userhome/alanjjp/Project/MeshGraphormer/output/final/final/evhands',
    '/userhome/alanjjp/Project/MeshGraphormer/output/final/final/f-event',
    '/userhome/alanjjp/Project/MeshGraphormer/output/final/final/p-full-iter',
    '/userhome/alanjjp/Project/MeshGraphormer/output/final/final/p-iter-S-resnet34',
]

scene_names = ['normal', 'strong light', 'flash']
file_name_list = ['normal', 'strong_light', 'flash']

colors_list = ['r', 'g', 'b', 'gold']

output_dir = '/userhome/alanjjp/Project/MeshGraphormer/scripts/forpaper/AUC'
mkdir(output_dir)

def get_method_errors(dir, error_dir_leaves, step, file_name='kps_errors.pt'):
    errors_tmp = []
    errors_return = []
    for dir_leaf in error_dir_leaves:
        error_path = os.path.join(dir, 'errors', dir_leaf+'_step'+str(step), file_name)
        errors_tmp.append(torch.load(error_path))
    for i in range(3):
        errors_return.append(torch.cat([errors_tmp[2*i], errors_tmp[2*i+1]], dim=0))
    return errors_return


rgb_errors = []
for rgb_method_dir in rgb_method_dirs:
    error_tmp = get_method_errors(rgb_method_dir, error_dir_leaves, 0)
    rgb_errors.append(error_tmp)


for i in range(3):
    plot_3D_PCK(
        [rgb_errors[0][i], rgb_errors[1][i], rgb_errors[2][i], rgb_errors[3][i]],
        colors_list,
        rgb_labels_methods,
        output_dir,
        'rgb_' + file_name_list[i]+'_auc.pdf',
        name=scene_names[i]
    )

event_errors = []
for i, event_method_dir in enumerate(event_method_dirs):
    step = 1 if (i == 2) or (i==3) else 0
    error_tmp = get_method_errors(event_method_dir, error_dir_leaves, step)
    event_errors.append(error_tmp)

for i in range(3):
    plot_3D_PCK(
        [event_errors[0][i], event_errors[1][i], event_errors[2][i], event_errors[3][i]],
        colors_list,
        event_labels_methods,
        output_dir,
        'event_' + file_name_list[i] + '_auc.pdf',
        name=scene_names[i]
    )