# from src.utils.dataset_utils import json_read
# annot = json_read('/userhome/alanjjp/data/EvRealHands/0/annot.json')
import os
os.chdir('/userhome/alanjjp/Project/MeshGraphormer')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

# import albumentations as A
#
# class DarkAug(object):
#     """
#     Extreme dark augmentation aiming at Aachen Day-Night
#     """
#
#     def __init__(self) -> None:
#         self.augmentor = A.Compose([
#             A.RandomBrightnessContrast(p=0.75, brightness_limit=(-0.6, 0.0), contrast_limit=(-0.5, 0.3)),
#             A.Blur(p=0.1, blur_limit=(3, 9)),
#             A.MotionBlur(p=0.2, blur_limit=(3, 25)),
#             A.RandomGamma(p=0.1, gamma_limit=(15, 65)),
#             A.HueSaturationValue(p=0.1, val_shift_limit=(-100, -40))
#         ], p=0.75)
#
#     def __call__(self, x):
#         return self.augmentor(image=x)['image']
#
#
# class MobileAug(object):
#     """
#     Random augmentations aiming at images of mobile/handhold devices.
#     """
#
#     def __init__(self):
#         self.augmentor = A.Compose([
#             # A.MotionBlur(p=1.), # GG
#             # A.ColorJitter(p=1.),
#             # A.RandomSunFlare(p=1.), # GG
#             # A.JpegCompression(p=1.0),
#             # A.ISONoise(p=1.0),
#             A.Blur(p=1.0)
#         ], p=1.0)
#
#     def __call__(self, x):
#         return self.augmentor(image=x)['image']
#
#
# def load_img(path, order='RGB'):
#     img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
#     if not isinstance(img, np.ndarray):
#         raise IOError("Fail to read %s" % path)
#     if order == 'RGB':
#         img = img[:, :, ::-1].copy()
#
#     # img = img.astype(np.float32)
#     return img
#
# img = load_img('/userhome/alanjjp/data/EvRealHands/0/images/21320028/image0009.jpg')
# import albumentations as A
#
# pass

#
# img = img
#
# plt.imshow(img/255.)
# plt.show()
#
# dark_aug = DarkAug()
# mob_aug = MobileAug()
#
# # dark_img = dark_aug(img.copy())
# mob_img = img.copy()
# for i in range(1):
#     mob_img = mob_aug(mob_img)
#
# # plt.imshow(dark_img)
# # plt.show()
#
# plt.imshow(mob_img/255.)
# plt.show()
# print('over!')
# pass

# import os
# import subprocess
# event_dir = '/userhome/alanjjp/data/EvHandFinal/data'
# target_dir = '/userhome/alanjjp/data/EvRealHands'
# ids = os.listdir(event_dir)
# for id in ids:
#     event_path = os.path.join(event_dir, id, 'event.aedat4')
#     target_path = os.path.join(target_dir, id)
#     cmd = 'cp ' + event_path + ' ' + target_path+'/'
#     subprocess.call(cmd, shell=True)



# from smplx import MANO
# from src.modeling._mano import MANO as MANO_pth
# import torch
# from src.utils.dataset_utils import json_read
# smplx_path = '/userhome/alanjjp/data/smplx_models/mano'
# layer_smplx = MANO(smplx_path, use_pca=False, is_rhand=True)
#
# annot = json_read('/userhome/alanjjp/data/EvRealHands/4/annot.json')
# camera_id = '21320028'
# K = annot['camera_info'][camera_id]['K']
# R = annot['camera_info'][camera_id]['R']
# T = annot['camera_info'][camera_id]['T']
#
# img_id = '18'
#
# mano = annot['manos'][img_id]
#
# mano_rot_pose = torch.tensor(mano['rot'], dtype=torch.float32).view(-1, 3)
# mano_hand_pose = torch.tensor(mano['hand_pose'], dtype=torch.float32).view(-1, 45)
# mano_shape = torch.tensor(mano['shape'], dtype=torch.float32).view(-1, 10)
# mano_trans = torch.tensor(mano['trans'], dtype=torch.float32).view(-1, 3)
#
# output_smplx = layer_smplx(
#     global_orient=mano_rot_pose.reshape(-1, 3),
#     hand_pose=mano_hand_pose.reshape(-1, 45),
#     betas=mano_shape.reshape(-1, 10),
#     transl=mano_trans.reshape(-1, 3)
# )
#
# print('SMPLX vertices: \n', output_smplx.vertices[0, :10])
# print('SMPLX joints: \n', output_smplx.joints[0, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]])
#
# layer_pth = MANO_pth()
# mano_pose_all = torch.cat([mano_rot_pose, mano_hand_pose], dim=1)
# vertices_pth, joints_pth = layer_pth.layer(mano_pose_all, mano_shape, mano_trans)
# joints_pth_pre = layer_pth.get_3d_joints(vertices_pth)
# print('PTH vertices: \n', vertices_pth[0, :10]/1000.)
# print('PTH joints: \n', joints_pth[0]/1000.)
# print('PTH joints regress: \n', joints_pth_pre[0]/1000.)
#
#
# print('?????')
# pass

from src.configs.config_parser import ConfigParser
from src.datasets.EvRealHands import EvRealHands

def get_config():
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--config', type=str, default='src/configs/test_evrealhands.yaml')
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
    return config

config = get_config()

a = EvRealHands(config)
item = a[160]
print('over!')