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


from src.configs.config_parser import ConfigParser
from src.datasets.EvRealHands import EvRealHands

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
    return config

config = get_config()

#a = Interhand(config)
dataset = EvRealHands(config)

output_dir = './scripts/forpaper/teaser'
mkdir(output_dir)

item = dataset[92]
t_l = 113 * 1e6 / 15. + dataset.data['1']['annot']['delta_time'] - 1e6 / 15.
t_r = t_l + 7 * 1e6 / 15.
events_all = dataset.data['1']['event']
indices = np.searchsorted(
            events_all[:, 3],
            np.array([t_l, t_r])
        )

events = events_all[indices[0]:indices[1]]

event_basic = np.concatenate([events[:, :2], events[:, 3:4]], axis=1)
event_basic[:, 0] /= 346
event_basic[:, 1] /= 260
event_basic[:, 2] /= (1e6 / 15.)
index_red = events[:, 2] == 0
index_green = events[:, 2] == 1
events_pc = np.concatenate([event_basic, np.ones_like(event_basic) * 255], axis=1)
events_pc[index_red, 4:] = 0
events_pc[index_green, 3] = 0
events_pc[index_green, 5] = 0
print(events_pc.shape)
events_pc = events_pc[10::]
data_pc = {
    'x': events_pc[:, 0],
    'y': events_pc[:, 1],
    'z': events_pc[:, 2],
    'red': events_pc[:, 3].astype(np.uint8),
    'green': events_pc[:, 4].astype(np.uint8),
    'blue': events_pc[:, 5].astype(np.uint8)
}
cloud = PyntCloud(pd.DataFrame(data=data_pc))
cloud.to_file(os.path.join(output_dir, 'output_events.ply'))

mano_layer = MANO(config['data']['smplx_path'], use_pca=False, is_rhand=True)

for step in range(len(item[1])):
    manos = item[1][step]['mano']
    mano_output = mano_layer(
        global_orient=manos['rot_pose'].reshape(-1, 3),
        hand_pose=manos['hand_pose'].reshape(-1, 45),
        betas=manos['shape'].reshape(-1, 10),
        transl=manos['trans'].reshape(-1, 3)
    )
    with open(os.path.join(output_dir, 'mesh_{}.obj'.format(step)), 'w') as file_object:
        for ver in mano_output.vertices[0].detach().cpu():
            # for ver in y['vertices_inter'][0, k, i - 3].detach().cpu():
            print('v %f %f %f' % (ver[0], ver[1], ver[2]), file=file_object)
        for f in mano_layer.faces:
            print('f %d %d %d' % (f[0] + 1, f[1] + 1, f[2] + 1), file=file_object)

print('over!')
