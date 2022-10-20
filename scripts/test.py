from src.utils.dataset_utils import json_read, json_write
path_ = '/userhome/alanjjp/data/EvRealHands/75/annot.json'
annot = json_read(path_)
import os
os.chdir('/userhome/alanjjp/Project/MeshGraphormer')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import math

#
# def get_3d_allviews(joint: np.ndarray, intr: np.ndarray, r: np.ndarray, t: np.ndarray):  # view为视角数，joint为二维坐标view*21*2
#     '''
#     从所有视角中直接triangulate一个3d pose
#     :param view: 视角数，一般为camera_num
#     :param joint: 二维坐标， view*N*2
#     :param intr: 内参矩阵 view*3*3
#     :param r: 旋转矩阵 view*3*3
#     :param t: 平移矩阵 view*3*1
#     :return: 三维关键点坐标
#     '''
#     joints_center = joint.mean(axis=1)
#     normalized_joints = joint - joints_center[:, None, :]
#     joints_dis = np.linalg.norm(normalized_joints, axis=-1).mean(axis=1)
#     scale = math.sqrt(2) / joints_dis
#     normalized_joints = normalized_joints * scale[:, None, None]
#
#     scaled_intr = intr.copy()
#     scaled_intr[:, :2, -1] -= joints_center
#     scaled_intr[:, :2, :] *= scale[:, None, None]
#
#     Pm = scaled_intr @ np.concatenate([r, t], axis=-1)
#     Am = np.concatenate([Pm[:, None, 2, :] * normalized_joints[:, :, 0, None] - Pm[:, None, 0, :],
#                          Pm[:, None, 2, :] * normalized_joints[:, :, 1, None] - Pm[:, None, 1, :]], axis=0).transpose(1,
#                                                                                                                       0,
#                                                                                                                       2)
#     _, D, vh = np.linalg.svd(Am, full_matrices=False)
#     # print(D)
#     x3d_h = vh[:, -1, :]
#     x3d = x3d_h[:, :3] / x3d_h[:, 3:]
#
#     return x3d
#
# camera_views = [
#     '21320027', '21320028', '21320029',
#     '21320030', '21320034', '21320035', '21320036',
# ]
# K, R, t = [], [], []
# for camera_view in camera_views:
#     K.append(np.array(annot['camera_info'][camera_view]['K'], dtype=np.float32))
#     R.append(np.array(annot['camera_info'][camera_view]['R'], dtype=np.float32))
#     t.append(np.array(annot['camera_info'][camera_view]['T'], dtype=np.float32))
#
# K = np.stack(K, axis=0)
# R = np.stack(R, axis=0)
# t = np.stack(t, axis=0)[..., None]


# for id in annot['2d_joints'][camera_views[0]].keys():
#     kps_2d = []
#     for camera_view in camera_views:
#         kps_2d.append(np.array(annot['2d_joints'][camera_view][id], dtype=np.float32))
#     kps_2d = np.stack(kps_2d, axis=0)
#     kps_3d = get_3d_allviews(kps_2d, K, R, t)
#     annot['3d_joints'][id] = kps_3d.tolist()
# json_write(path_, annot)

# for id in annot['2d_joints']['event'].keys():
#     if len(annot['2d_joints']['event'][id]) != 0 and len(annot['2d_joints']['event'][id]) != 21:
#         annot['2d_joints']['event'][id] = []
# json_write(path_, annot)
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
from src.datasets.Interhand import Interhand

def get_config():
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--config', type=str, default='src/configs/train_perceiver_super.yaml')
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
    config['exper']['run_eval_only'] = False#True
    return config

config = get_config()

#a = Interhand(config)
a = EvRealHands(config)
d = a[100]
c = a[80]
e = a[1150]
b = a[400]

# for key in a.data.keys():
#     b = a.data[key]['event']
#
# b_tmp = b[20000:30000]



# item = a[200]
# item = a[140]
# a = Interhand(config)
# item = a[1]
# item = a[100]
# a[500]
# a[1000]
print('over!')

# from src.modeling._mano import MANO
# smplx_path = '/userhome/alanjjp/data/smplx_models/mano'
# mano_layer = MANO(smplx_path, use_pca=False, is_rhand=True)
# mano_layer.to('cuda')
# manos = json_read('/userhome/alanjjp/data/Interhand/annot/InterHand2.6M_train_MANO_NeuralAnnot.json')
#
# for cap_id in manos.keys():
#     print('Capture: ', cap_id)
#     for frame_id in manos[cap_id].keys():
#         if manos[cap_id][frame_id]['right'] is not None:
#             print('frame id: {}'.format(frame_id), end='\r')
#             poses = torch.tensor(manos[cap_id][frame_id]['right']['pose'], dtype=torch.float32).view(1, -1).to('cuda')
#             shape = torch.tensor(manos[cap_id][frame_id]['right']['shape'], dtype=torch.float32).view(1, -1).to('cuda')
#             trans = torch.tensor(manos[cap_id][frame_id]['right']['trans'], dtype=torch.float32).view(1, 3).to('cuda')
#             output = mano_layer(
#                 global_orient=poses[:, :3],
#                 hand_pose=poses[:, 3:],
#                 betas=shape,
#                 transl=trans
#             )
#             manos[cap_id][frame_id]['right']['root_joint'] = output.root_joint[0].detach().cpu().tolist()
# json_write('/userhome/alanjjp/data/Interhand/annot/InterHand2.6M_train_MANO_NeuralAnnot_new.json', manos)
# pass