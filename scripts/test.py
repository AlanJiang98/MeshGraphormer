from src.utils.dataset_utils import json_read
annot = json_read('/userhome/alanjjp/data/EvRealHands/0/annot.json')

print('over!')
pass

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