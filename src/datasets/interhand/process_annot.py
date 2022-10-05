# import json
# import os
# import tqdm
# import os.path as osp
# from src.utils.dataset_utils import json_write
#
#
# def load_annotations(annot_root_dir):
#     with open(os.path.join(annot_root_dir, 'InterHand2.6M_train_MANO_NeuralAnnot.json')) as f:
#         mano_params = json.load(f)
#     with open(os.path.join(annot_root_dir, 'InterHand2.6M_train_camera.json')) as f:
#         cam_params = json.load(f)
#     with open(os.path.join(annot_root_dir, 'InterHand2.6M_train_joint_3d.json')) as f:
#         joints = json.load(f)
#     annot = {
#         'MANO': mano_params, 'camera': cam_params, 'joints': joints
#     }
#     return annot
#
# annot_root_dir = '/userhome/alanjjp/data/Interhand/annot'
# synt_dir = '/userhome/alanjjp/data/Interhand/synthetic/train'
# img_dir = '/userhome/alanjjp/data/Interhand/train'
#
# annot = load_annotations(annot_root_dir)
#
#
# cap_name_list = os.listdir(synt_dir)
# cap_name_list.sort()
# for cap_name in tqdm(cap_name_list, desc='Cap'):
#     print('Cap id is {}'.format(cap_name), end='\r')
#     ges_name_list = os.listdir(osp.join(synt_dir, cap_name))
#     ges_name_list.sort()
#     for ges_name in tqdm(ges_name_list, desc='Ges'):
#         print('Ges id is {}'.format(ges_name), end='\r')
#         cam_name_list = os.listdir(osp.join(synt_dir, cap_name, ges_name))
#         cam_name_list.sort()
#         for cam_name in tqdm(cam_name_list, desc='Cam'):
#             print('Cam id is {}'.format(cam_name), end='\r')
#             if not osp.exists(osp.join(synt_dir, cap_name, ges_name, cam_name, 'events.npz')):
#                 continue
#             image_name_list = os.listdir(osp.join(img_dir, cap_name, ges_name, cam_name))
#             image_name_list.sort()
#             frame_id_list_tmp = []
            # for image_name in image_name_list:
            #     frame_id_list_tmp.append(image_name[5:-4])
            # cap_id, ges_id, cam_id = cap_name[7:], ges_name, cam_name[3:]
            # frame_id_list = []
            # for frame_id in frame_id_list_tmp:
            #     if frame_id in annot['MANO'][cap_id].keys() and frame_id in annot['joints'][cap_id].keys():
            #         if annot['joints'][cap_id][frame_id]['hand_type'] == 'right'\
            #                 and annot['joints'][cap_id][frame_id]['hand_type_valid']:
            #             frame_id_list.append(frame_id)
            # if len(frame_id_list) == 0:
            #     continue
            #
            # output_file_name = osp.join(synt_dir, cap_name, ges_name, cam_name, 'annot.json')
            # annot_tmp = {}
            # annot_tmp['cap_id'] = cap_id
            # annot_tmp['cam_id'] = cam_id
            # annot_tmp['ges_id'] = ges_name
            # annot_tmp['hand_type'] = 'right'
            # annot_tmp['mano'] = {}
            # annot_tmp['joints'] = {}
            # for i, frame_id in enumerate(frame_id_list):
            #     annot_tmp['mano'][frame_id] = annot['MANO'][cap_id][frame_id]['right']
            #     annot_tmp['mano'][frame_id]['pca_pose'] = pose_result[i].detach().cpu().tolist()
            #     annot_tmp['joints'][frame_id] = annot['joints'][cap_id][frame_id]
            #     annot_tmp['joints'][frame_id]['pca_world_coord'] = kps[i].detach().cpu().tolist()
            # annot_tmp['camera'] = {}
            # annot_tmp['camera']['campos'] = annot['camera'][cap_id]['campos'][cam_id]
            # annot_tmp['camera']['camrot'] = annot['camera'][cap_id]['camrot'][cam_id]
            # annot_tmp['camera']['focal'] = [K_all[0, 0], K_all[1, 1]]
            # annot_tmp['camera']['princpt'] = [K_all[0, 2], K_all[1, 2]]
            #
            # json_write(output_file_name, annot_tmp)

