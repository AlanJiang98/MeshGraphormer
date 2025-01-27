import cv2
import os
import os.path as osp
import numpy as np
import copy
import torch
import json
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import multiprocessing as mp
import matplotlib.pyplot as plt
import roma
from src.utils.dataset_utils import json_read, extract_data_from_aedat4, undistortion_points, event_representations
from src.utils.joint_indices import indices_change
import albumentations as A
from src.configs.config_parser import ConfigParser
from smplx import MANO
from src.utils.augment import PhotometricAug
from src.utils.comm import is_main_process
from scipy.interpolate import interp1d


from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.utils import cameras_from_opencv_projection


def collect_data(sample):
    global samples
    for key in sample.keys():
        samples[key] = sample[key]


class Interhand(Dataset):
    def __init__(self, config):
        self.config = config
        self.data_config = None
        self.load_annotations()
        self.process_samples()
        self.mano_layer = MANO(self.config['data']['smplx_path'], use_pca=False, is_rhand=True)
        self.rgb_augment = PhotometricAug(self.config['exper']['augment']['rgb_photometry'],
                                          not self.config['exper']['run_eval_only'])
        self.event_augment = PhotometricAug(self.config['exper']['augment']['event_photometry'],
                                            not self.config['exper']['run_eval_only'])

    def load_annotations(self):
        if self.config['exper']['run_eval_only']:
            data_config = ConfigParser(self.config['data']['dataset_info']['interhand']['eval_yaml'])
        else:
            data_config = ConfigParser(self.config['data']['dataset_info']['interhand']['train_yaml'])
        self.data_config = data_config.config
        mano_params = json_read(osp.join(self.config['data']['dataset_info']['interhand']['data_dir'], self.data_config['mano_path']))
        joint_params = json_read(osp.join(self.config['data']['dataset_info']['interhand']['data_dir'], self.data_config['joint_path']))
        cam_params = json_read(osp.join(self.config['data']['dataset_info']['interhand']['data_dir'], self.data_config['cam_path']))
        self.data = {
            'mano': mano_params,
            'joint3d': joint_params,
            'cam': cam_params
        }
        self.get_synthetic_affine_matrix()
        self.ges_list = os.listdir(osp.join(self.config['data']['dataset_info']['interhand']['data_dir'], self.data_config['event_dir'], 'Capture0'))
        self.ges_list.sort()

    def get_synthetic_affine_matrix(self):
        # interhand image settings
        img_ori_height, img_ori_width = 512, 334
        davis_width, davis_height = 346, 260
        # get the affine matrix from RGB frame to event frame, and this affine matrix will influence the camera intrinsics
        src_points = np.float32([[0, 0], [img_ori_width / 2 - 1, 0], [img_ori_width / 2 - 1, img_ori_height]])
        dst_points = np.float32([[davis_width / 2 - img_ori_width / 2 * davis_height / img_ori_height, 0],
                                 [davis_width / 2 - 1, 0],
                                 [davis_width / 2 - 1, davis_height - 1]])
        affine_matrix = cv2.getAffineTransform(src_points, dst_points)
        self.K_affine = np.concatenate([affine_matrix, np.array([[0, 0, 1.0]])], axis=0)

    @staticmethod
    def get_samples_per_cap(data, config, data_config, ges_list, cap_id):
        cap_ev_dir = osp.join(config['data']['dataset_info']['interhand']['data_dir'], data_config['event_dir'], 'Capture' + cap_id)
        ges_list_ = os.listdir(cap_ev_dir)
        ges_list_.sort()
        samples = []
        for ges in ges_list_:
            if ges in ges_list:
                ges_index = ges_list.index(ges)
                for pair_id, cam_pair in enumerate(data_config['camera_pairs']):
                    if osp.exists(osp.join(cap_ev_dir, ges, 'cam' + cam_pair[0], 'events.npz')):
                        img_dir_ = osp.join(config['data']['dataset_info']['interhand']['data_dir'], data_config['img_dir'], \
                                            'Capture' + cap_id, ges, 'cam' + cam_pair[1])
                        img_name_list = os.listdir(img_dir_)
                        img_name_list.sort()
                        img_id_list = [img_name[5:-4] for img_name in img_name_list]
                        img_valid_id_list = []
                        for img_id in img_id_list:
                            if img_id in data['mano'][cap_id].keys() and img_id in data['joint3d'][cap_id].keys():
                                if data['joint3d'][cap_id][img_id]['hand_type'] == 'right' \
                                        and data['joint3d'][cap_id][img_id]['hand_type_valid']:
                                    img_valid_id_list.append(img_id)
                        if len(img_valid_id_list) <= 30:
                            continue
                        start = 24
                        for i, img_valid_id in enumerate(img_valid_id_list[start:-5]):
                            time = (int(img_valid_id) - int(img_id_list[0])) / 90. * 1000000.
                            if int(img_valid_id) - int(img_valid_id_list[start + i - 1]) == 3:
                                item = (int(cap_id), ges_index, pair_id, img_valid_id, time)
                                samples.append(item)
        return {cap_id: samples}

    def process_samples(self):
        self.samples = []
        valid_cap_ids = os.listdir(osp.join(self.config['data']['dataset_info']['interhand']['data_dir'], self.data_config['event_dir']))
        valid_cap_ids.sort()
        if self.config['exper']['debug']:
            sample_ = self.get_samples_per_cap(self.data, self.config, self.data_config, self.ges_list, valid_cap_ids[0][7:])
            self.samples += sample_[valid_cap_ids[0][7:]]
        else:
            global samples
            samples = {}
            pool = mp.Pool(mp.cpu_count())
            for cap_id in self.data_config['cap_ids']:
                if 'Capture'+str(cap_id) in valid_cap_ids:
                    cap_id = str(cap_id)
                    pool.apply_async(
                        Interhand.get_samples_per_cap,
                        args=(
                            self.data, copy.deepcopy(self.config), copy.deepcopy(self.data_config), copy.deepcopy(self.ges_list), cap_id,
                        ),
                        callback=collect_data
                    )
                    # cap_ev_dir = osp.join(self.config['data']['data_dir'], self.data_config['event_dir'], 'Capture'+cap_id)
                    # ges_list_ = os.listdir(cap_ev_dir).sort()
                    # for ges in ges_list_:
                    #     if ges in self.ges_list:
                    #         ges_index = self.ges_list.index(ges)
                    #         for pair_id, cam_pair in enumerate(self.data_config['camera_paris']):
                    #             if osp.exists(osp.join(cap_ev_dir, ges, 'cam'+cam_pair[0], 'events.npz')):
                    #                 img_dir_ = osp.join(self.config['data']['data_dir'], self.data_config['img_dir'], \
                    #                                    'Capture'+cap_id, ges, 'cam'+cam_pair[1])
                    #                 img_name_list = os.listdir(img_dir_).sort()
                    #                 img_id_list = [img_name[5:-4] for img_name in img_name_list]
                    #                 img_valid_id_list = []
                    #                 for img_id in img_id_list:
                    #                     if img_id in self.data['mano'][cap_id].keys() and img_id in self.data['joint3d'][cap_id].keys():
                    #                         if self.data['joint3d'][cap_id][img_id]['hand_type'] == 'right'\
                    #                                 and self.data['joint3d'][cap_id][img_id]['hand_type_valid']:
                    #                             img_valid_id_list.append(img_id)
                    #                 if len(img_valid_id_list) <= 30:
                    #                     continue
                    #                 start = 24
                    #                 for i, img_valid_id in enumerate(img_valid_id_list[start:-5]):
                    #                     time = (int(img_valid_id) - int(img_id_list[0])) / 90.
                    #                     if int(img_valid_id) - int(img_valid_id_list[start+i-1]) == 3:
                    #                         item = (int(cap_id), ges_index, pair_id, img_valid_id, time)
                    #                         self.samples.append(item)
            pool.close()
            pool.join()
            cap_id_list = list(samples.keys())
            cap_id_list.sort()
            for cap_id in cap_id_list:
                self.samples += samples[cap_id]

    def get_camera_params(self, cap_id, cam_id, is_event=True):
        cam_param = self.data['cam'][cap_id]
        focal = np.array(cam_param['focal'][cam_id], dtype=np.float32).reshape(2)
        princpt = np.array(cam_param['princpt'][cam_id], dtype=np.float32).reshape(2)
        K_origin = np.array([
            [focal[0], 0, princpt[0]],
            [0, focal[1], princpt[1]],
            [0, 0, 1.]
        ])
        if is_event:
            K = np.dot(self.K_affine, K_origin)
        else:
            K = K_origin
        K = torch.tensor(K, dtype=torch.float32).view(3, 3)
        R = torch.tensor(cam_param['camrot'][cam_id], dtype=torch.float32).view(3, 3)
        t = torch.tensor(cam_param['campos'][cam_id], dtype=torch.float32).view(3, 1) / 1000.
        t = (-R @ t).view(3)
        return K, R, t

    def get_annotations(self, cap_id, cam_pair, img_id):
        meta_data = {}

        meta_data['delta_time'] = 0.
        meta_data['2d_joints_event'] = torch.zeros((21, 2), dtype=torch.float32)
        meta_data['joints_2d_valid_ev'] = True
        meta_data['2d_joints_rgb'] = torch.zeros((21, 2), dtype=torch.float32)

        K_event_l, R_event_l, t_event_l = self.get_camera_params(cap_id, cam_id=cam_pair[0], is_event=True)
        meta_data.update({
            'K_event_l': K_event_l,
            'R_event_l': R_event_l,
            't_event_l': t_event_l,
            'K_event_r': K_event_l.clone(),
            'R_event_r': R_event_l.clone(),
            't_event_r': t_event_l.clone(),
        })
        K_rgb, R_rgb, t_rgb = self.get_camera_params(cap_id, cam_id=cam_pair[1], is_event=False)
        meta_data.update({
            'K_rgb': K_rgb,
            'R_rgb': R_rgb,
            't_rgb': t_rgb,
        })
        meta_data['3d_joints'] = (torch.tensor(
            self.data['joint3d'][cap_id][img_id]['world_coord'][:21],
            dtype=torch.float32
        ).view(-1, 3) / 1000.)[indices_change(0, 1)]
        mano = self.data['mano'][cap_id][img_id]['right']
        meta_data['mano'] = {}
        poses = torch.tensor(mano['pose'], dtype=torch.float32).view(-1)
        meta_data['mano']['rot_pose'] = poses[:3]
        meta_data['mano']['hand_pose'] = poses[3:]
        meta_data['mano']['shape'] = torch.tensor(mano['shape'], dtype=torch.float32).view(-1)
        meta_data['mano']['trans'] = torch.tensor(mano['trans'], dtype=torch.float32).view(3)
        meta_data['mano']['root_joint'] = torch.tensor(mano['root_joint'], dtype=torch.float32).view(3)
        # todo process root_joint
        return meta_data

    def get_l_event_indices(self, r_time, events):
        if self.config['exper']['preprocess']['event_range'] == 'time':
            l_win_range = [torch.log10(torch.tensor(x)) for x in self.config['exper']['preprocess']['left_window']]
            l_win = 10 ** (torch.rand(1)[0] * (l_win_range[1] - l_win_range[0]) + l_win_range[0])
            l_t = r_time - l_win
            indices = np.searchsorted(
                events[:, 3],
                # self.data[seq_id]['event'][:, 3],
                np.array([l_t, r_time])
            )
            index_l, index_r = indices[0], indices[1]
        else:
            timestamp = np.array(r_time, dtype=np.float32)
            index_r = np.searchsorted(
                events[:, 3],
                # self.data[seq_id]['event'][:, 3],
                timestamp
            )
            index_l = max(0, index_r - self.config['exper']['preprocess']['num_window'])
        return index_l, index_r

    def get_indices_from_timestamps(self, timestamps, event):
        timestamps = np.array(timestamps, dtype=np.float32)
        indices = np.searchsorted(
            event[:, 3],
            timestamps
        )
        index_l, index_r = self.get_l_event_indices(timestamps[1], event[indices[0]:indices[1]].copy())
        return index_l+indices[0], index_r+indices[1]

    def load_img(self, path, order='RGB'):
        img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(img, np.ndarray):
            raise IOError("Fail to read %s" % path)
        if order == 'RGB':
            img = img[:, :, ::-1].copy()

        img = img.astype(np.float32)
        return img

    def get_event_repre(self, event, indices):
        if indices[0] == 0 or indices[-1] >= event.shape[0] - 1 or indices[-1]==indices[0]:
            return torch.zeros((2, 260, 346))
        else:
            tmp_events = event[indices[0]:indices[1]].copy()
            tmp_events[:, 3] = (tmp_events[:, 3] - event[indices[0], 3]) / (event[indices[1], 3] - event[indices[0], 3])
            tmp_events = torch.tensor(tmp_events, dtype=torch.float32)
            ev_frame = event_representations(tmp_events, repre=self.config['exper']['preprocess']['ev_repre'], hw=(260, 346))
            return ev_frame

    def get_augment_param(self):
        if not self.config['exper']['run_eval_only']:
            scale_range, trans_range, rot_range = self.config['exper']['augment']['geometry']['scale'], \
                                self.config['exper']['augment']['geometry']['trans'], \
                                self.config['exper']['augment']['geometry']['rot']
        else:
            scale_range, trans_range, rot_range = 0, 0., 0.
        # augmentation for event and rgb
        augs = [[], []]
        for i in range(2):
            scale = min(2*scale_range, max(-2*scale_range, np.random.randn() * scale_range)) + 1
            trans_x = min(2*trans_range, max(-2*trans_range, np.random.randn() * trans_range))
            trans_y = min(2 * trans_range, max(-2 * trans_range, np.random.randn() * trans_range))
            rot = min(2*rot_range, max(-2*rot_range, np.random.randn() * rot_range))
            augs[i] += [trans_x, trans_y, rot, scale ]
        return augs

    def get_transform(self, R, t):
        tf = torch.eye(4)
        tf[:3, :3] = R
        tf[:3, 3] = t
        return tf

    def augment_one_view(self, augs, frame, meta_data, view='rgb'):
        if view == 'rgb' or 'event_l':
            mano_key = 'mano_rgb'
        else:
            mano_key = 'mano'
        K = meta_data['K_' + view]
        R_wm = roma.rotvec_to_rotmat(meta_data[mano_key]['rot_pose'])
        t_wm = meta_data[mano_key]['trans']

        tf_wm = self.get_transform(R_wm, t_wm + meta_data[mano_key]['root_joint'])
        tf_cw = self.get_transform(meta_data['R_' + view], meta_data['t_' + view])

        tf_cm = tf_cw @ tf_wm

        # trans augmentation
        aff_trans_2d = torch.eye(3)
        H, W, _ = frame.shape
        trans_x_2d = augs[0] * W
        trans_y_2d = augs[1] * H
        aff_trans_2d[0, 2] = trans_x_2d
        aff_trans_2d[1, 2] = trans_y_2d
        tf_trans_3d = torch.eye(4)
        trans_x_3d = trans_x_2d * tf_cm[2, 3] / meta_data['K_' + view][0, 0]
        trans_y_3d = trans_y_2d * tf_cm[2, 3] / meta_data['K_' + view][1, 1]
        tf_trans_3d[0, 3] = trans_x_3d
        tf_trans_3d[1, 3] = trans_y_3d

        # rotation
        rot_2d = roma.rotvec_to_rotmat(augs[2] * torch.tensor([0.0, 0.0, 1.0]))
        tf_rot_3d = self.get_transform(rot_2d, torch.zeros(3))
        aff_rot_2d = torch.eye(3)
        aff_rot_2d[:2, :2] = rot_2d[:2, :2]
        aff_rot_2d[0, 2] = (1 - rot_2d[0, 0]) * K[0, 2] - rot_2d[0, 1] * K[1, 2]
        aff_rot_2d[1, 2] = (1 - rot_2d[1, 1]) * K[1, 2] - rot_2d[1, 0] * K[0, 2]

        tf_3d_tmp = tf_rot_3d @ tf_trans_3d @ tf_cm
        # scale
        tf_scale_3d = torch.eye(4)
        # tf_scale_3d[2, 2] = 1.0 / augs[3]
        tf_scale_3d[2, 3] = (1.0 / augs[3] - 1.) * tf_3d_tmp[2, 3]
        aff_scale_2d = torch.eye(3)
        aff_scale_2d[0, 0], aff_scale_2d[1, 1] = augs[3], augs[3]
        aff_scale_2d[:2, 2] = (1 - augs[3]) * K[:2, 2]

        aff_2d_final = (aff_scale_2d @ aff_rot_2d @ aff_trans_2d)[:2, :]
        frame_aug = cv2.warpAffine(np.array(frame), aff_2d_final.numpy(), (W, H), flags=cv2.INTER_LINEAR)

        tf_3d_final = tf_scale_3d @ tf_rot_3d @ tf_trans_3d @ tf_cm
        tf_cw = tf_3d_final @ tf_wm.inverse()

        meta_data['R_' + view] = tf_cw[:3, :3]
        meta_data['t_' + view] = tf_cw[:3, 3]

        return frame_aug

    def get_bbox_from_joints(self, K, R, t, joints, rate=1.5):
        kps = (K @ (R @ joints.transpose(0, 1) + t.reshape(3, 1))).transpose(0, 1)
        kps = kps[:, :2] / kps[:, 2:]
        x_min = torch.min(kps[:, 0])
        x_max = torch.max(kps[:, 0])
        y_min = torch.min(kps[:, 1])
        y_max = torch.max(kps[:, 1])
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        bbox_size = torch.max(torch.tensor([x_max-x_min, y_max-y_min])) * rate
        return torch.tensor([center_x, center_y, bbox_size]).int()

    def get_bbox_list(self, meta_data, rate=1.5):
        bbox_rgb = self.get_bbox_from_joints(
            meta_data['K_rgb'],
            meta_data['R_rgb'],
            meta_data['t_rgb'],
            meta_data['3d_joints_rgb'],
            rate,
        )
        bbox_l = self.get_bbox_from_joints(
            meta_data['K_event_l'],
            meta_data['R_event_l'],
            meta_data['t_event_l'],
            meta_data['3d_joints_rgb'],
            rate,
        )
        bbox_r = self.get_bbox_from_joints(
            meta_data['K_event_r'],
            meta_data['R_event_r'],
            meta_data['t_event_r'],
            meta_data['3d_joints'],
            rate,
        )
        return bbox_l, bbox_rgb, bbox_r

    def crop(self, bbox, frame, size, hw=[260, 346]):
        lf_top = (bbox[:2] - bbox[2] / 2).int()
        rt_dom = lf_top + bbox[2].int()
        if lf_top[0] < 0 or lf_top[1] < 0 or rt_dom[0] > hw[1] or rt_dom[1] > hw[0]:
            frame = cv2.copyMakeBorder(frame, - min(0, lf_top[1].item()), max(rt_dom[1].item() - hw[0], 0),
                            -min(0, lf_top[0].item()), max(rt_dom[0].item() - hw[1], 0), cv2.BORDER_REPLICATE)
            rt_dom[1] += -min(0, lf_top[1])
            lf_top[1] += -min(0, lf_top[1])
            rt_dom[0] += -min(0, lf_top[0])
            lf_top[0] += -min(0, lf_top[0])
        frame_crop = frame[lf_top[1]: rt_dom[1], lf_top[0]:rt_dom[0]]
        scale = size / bbox[2].float()
        frame_resize = cv2.resize(np.array(frame_crop), (size, size), interpolation=cv2.INTER_AREA)
        return lf_top, scale, frame_resize

    def change_camera_view(self, meta_data):
        if self.config['exper']['preprocess']['cam_view'] == 'world':
            return self.get_transform(torch.eye(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32))
        elif self.config['exper']['preprocess']['cam_view'] == 'rgb':
            tf_rgb_w = self.get_transform(meta_data['R_rgb'], meta_data['t_rgb'])
            for mano_key in ['mano', 'mano_rgb']:
                if mano_key in meta_data.keys():
                    R_wm = roma.rotvec_to_rotmat(meta_data[mano_key]['rot_pose'])
                    t_wm = meta_data[mano_key]['trans']
                    tf_wm = self.get_transform(R_wm, t_wm + meta_data[mano_key]['root_joint'])
                    tf_rgb_m = tf_rgb_w @ tf_wm
                    meta_data[mano_key]['rot_pose'] = roma.rotmat_to_rotvec(tf_rgb_m[:3, :3])
                    meta_data[mano_key]['trans'] = tf_rgb_m[:3, 3] - meta_data[mano_key]['root_joint']
            for joints_key in ['3d_joints', '3d_joints_rgb']:
                if joints_key in meta_data.keys():
                    joints_new = (tf_rgb_w[:3, :3] @ (meta_data[joints_key].transpose(0, 1))\
                                  + tf_rgb_w[:3, 3:4]).transpose(0, 1)
                    meta_data[joints_key] = joints_new
            for event_cam in ['event_l', 'event_r']:
                tf_e_w = self.get_transform(meta_data['R_'+event_cam], meta_data['t_'+event_cam])
                tf_e_rgb = tf_e_w @ tf_rgb_w.inverse()
                meta_data['R_' + event_cam] = tf_e_rgb[:3, :3]
                meta_data['t_' + event_cam] = tf_e_rgb[:3, 3]
            meta_data['R_rgb'] = torch.eye(3, dtype=torch.float32)
            meta_data['t_rgb'] = torch.zeros(3, dtype=torch.float32)
            return tf_rgb_w.inverse()
        elif self.config['exper']['preprocess']['cam_view'] == 'event':
            tf_event_w = self.get_transform(meta_data['R_event_l'], meta_data['t_event_l'])
            for mano_key in ['mano', 'mano_rgb']:
                if mano_key in meta_data.keys():
                    R_wm = roma.rotvec_to_rotmat(meta_data[mano_key]['rot_pose'])
                    t_wm = meta_data[mano_key]['trans']
                    tf_wm = self.get_transform(R_wm, t_wm + meta_data[mano_key]['root_joint'])
                    tf_event_m = tf_event_w @ tf_wm
                    meta_data[mano_key]['rot_pose'] = roma.rotmat_to_rotvec(tf_event_m[:3, :3])
                    meta_data[mano_key]['trans'] = tf_event_m[:3, 3] - meta_data[mano_key]['root_joint']
            for joints_key in ['3d_joints', '3d_joints_rgb']:
                if joints_key in meta_data.keys():
                    joints_new = (tf_event_w[:3, :3] @ (meta_data[joints_key].transpose(0, 1))\
                                  + tf_event_w[:3, 3:4]).transpose(0, 1)
                    meta_data[joints_key] = joints_new
            for rgb_cam in ['rgb']:
                tf_rgb_w = self.get_transform(meta_data['R_'+rgb_cam], meta_data['t_'+rgb_cam])
                tf_rgb_e = tf_rgb_w @ tf_event_w.inverse()
                meta_data['R_' + rgb_cam] = tf_rgb_e[:3, :3]
                meta_data['t_' + rgb_cam] = tf_rgb_e[:3, 3]
            meta_data['R_event_l'] = torch.eye(3, dtype=torch.float32)
            meta_data['t_event_l'] = torch.zeros(3, dtype=torch.float32)
            meta_data['R_event_r'] = torch.eye(3, dtype=torch.float32)
            meta_data['t_event_r'] = torch.zeros(3, dtype=torch.float32)
            return tf_event_w.inverse()
        else:
            raise NotImplementedError('no implemention for change camera view!')

    def plotshow(self, img):
        plt.imshow(img)
        plt.show()

    def __getitem__(self, idx):
        cap_id, ges_index, pair_id, img_id, t_target = self.samples[idx]
        duration = 1
        # whether to use ere or origin res
        is_ere = torch.rand(1) < self.config['exper']['preprocess']['ere_rate']
        meta_data = self.get_annotations(str(cap_id), self.data_config['camera_pairs'][pair_id], img_id)
        events = np.load(osp.join(self.config['data']['dataset_info']['interhand']['data_dir'], self.data_config['event_dir'], \
                                  'Capture' + str(cap_id), self.ges_list[ges_index], \
                                  'cam' + self.data_config['camera_pairs'][pair_id][0], 'events.npz'
                                  ))
        events = events['events']
        events[:, :1] = 345 - events[:, :1]
        events[:, 1:2] = 259 - events[:, 1:2]
        if not is_ere:
            rgb_path = osp.join(self.config['data']['dataset_info']['interhand']['data_dir'], self.data_config['img_dir'],\
                                'Capture'+str(cap_id), self.ges_list[ges_index], \
                                'cam'+self.data_config['camera_pairs'][pair_id][1], 'image'+img_id+'.jpg')
            rgb_t = t_target

            indices_l_ev = self.get_l_event_indices(rgb_t, events)
            indices_r_ev = self.get_indices_from_timestamps([rgb_t, rgb_t], events)
            meta_data['3d_joints_rgb'] = meta_data['3d_joints'].clone()
            meta_data['mano_rgb'] = meta_data['mano'].copy()
        else:
            rgb_id = str(int(img_id) - 3*duration)
            interval = duration / 30. * 1000000
            meta_data_rgb = self.get_annotations(str(cap_id), self.data_config['camera_pairs'][pair_id], str(rgb_id))
            meta_data['3d_joints_rgb'] = meta_data_rgb['3d_joints'].clone()
            meta_data['mano_rgb'] = meta_data_rgb['mano'].copy()
            meta_data['joints_2d_valid_ev'] = meta_data_rgb['joints_2d_valid_ev']
            rgb_path = osp.join(self.config['data']['dataset_info']['interhand']['data_dir'], self.data_config['img_dir'],\
                                'Capture'+str(cap_id), self.ges_list[ges_index], \
                                'cam'+self.data_config['camera_pairs'][pair_id][1], 'image'+rgb_id+'.jpg')
            rgb_t = t_target - interval
            indices_l_ev = self.get_l_event_indices(rgb_t, events)
            indices_r_ev = self.get_indices_from_timestamps([rgb_t, rgb_t + interval], events)
        rgb = self.load_img(rgb_path)
        l_ev_frame = self.get_event_repre(events, indices_l_ev)
        l_ev_frame = torch.cat([l_ev_frame.permute(1, 2, 0), torch.zeros((260, 346, 1))], dim=2)
        r_ev_frame = self.get_event_repre(events, indices_r_ev)
        r_ev_frame = torch.cat([r_ev_frame.permute(1, 2, 0), torch.zeros((260, 346, 1))], dim=2)
        aug_params = self.get_augment_param()

        # self.render_hand(
        #     meta_data['mano_rgb'],
        #     meta_data['K_rgb'],
        #     meta_data['R_rgb'],
        #     meta_data['t_rgb'],
        #     hw=[512, 334],
        #     img_bg=rgb / 255.,
        # )
        # self.render_hand(
        #     meta_data['mano_rgb'],
        #     meta_data['K_event_l'],
        #     meta_data['R_event_l'],
        #     meta_data['t_event_l'],
        #     hw=[260, 346],
        #     img_bg=l_ev_frame,
        # )
        # self.render_hand(
        #     meta_data['mano'],
        #     meta_data['K_event_r'],
        #     meta_data['R_event_r'],
        #     meta_data['t_event_r'],
        #     hw=[260, 346],
        #     img_bg=r_ev_frame,
        # )

        rgb_aug_ori = self.augment_one_view(aug_params[1], rgb, meta_data, view='rgb')
        l_ev_frame_aug_ori = self.augment_one_view(aug_params[0], l_ev_frame, meta_data, view='event_l')
        r_ev_frame_aug_ori = self.augment_one_view(aug_params[0], r_ev_frame, meta_data, view='event_r')

        rgb_aug = self.rgb_augment(rgb_aug_ori.copy().astype(np.uint8)).astype(np.float32) / 255.
        l_ev_frame_aug = self.event_augment((l_ev_frame_aug_ori.copy() * 255).astype(np.uint8)).astype(
            np.float32) / 255.
        r_ev_frame_aug = self.event_augment((r_ev_frame_aug_ori.copy() * 255).astype(np.uint8)).astype(
            np.float32) / 255.

        bbox_l, bbox_rgb, bbox_r = self.get_bbox_list(meta_data, self.config['exper']['bbox']['rate'])

        tf_w_c = self.change_camera_view(meta_data)

        lt_rgb, sc_rgb, rgb_crop = self.crop(bbox_rgb, rgb_aug, self.config['exper']['bbox']['rgb']['size'], hw=[512, 334])
        lt_l_ev, sc_l_ev, l_ev_crop = self.crop(bbox_l, l_ev_frame_aug, self.config['exper']['bbox']['event']['size'],
                                             hw=[260, 346])
        lt_r_ev, sc_r_ev, r_ev_crop = self.crop(bbox_r, r_ev_frame_aug, self.config['exper']['bbox']['event']['size'],
                                             hw=[260, 346])

        # self.plotshow(rgb_crop)
        # self.plotshow(l_ev_crop)
        # self.plotshow(r_ev_crop)

        meta_data.update({
            'lt_rgb': lt_rgb,
            'sc_rgb': sc_rgb,
            'lt_l_ev': lt_l_ev,
            'sc_l_ev': sc_l_ev,
            'lt_r_ev': lt_r_ev,
            'sc_r_ev': sc_r_ev,
        })

        rgb = torch.tensor(rgb_crop, dtype=torch.float32)
        l_ev_frame = torch.tensor(l_ev_crop, dtype=torch.float32)
        r_ev_frame = torch.tensor(r_ev_crop, dtype=torch.float32)
        frames = {
            'rgb': rgb,
            'l_ev_frame': l_ev_frame,
            'r_ev_frame': r_ev_frame,
            # 'rgb_ori': torch.tensor(rgb_aug_ori, dtype=torch.float32),
            # 'event_l_ori': torch.tensor(l_ev_frame_aug_ori, dtype=torch.float32),
            # 'event_r_ori': torch.tensor(r_ev_frame_aug_ori, dtype=torch.float32),
        }
        if self.config['exper']['run_eval_only']:
            meta_data.update({
                'cap_id': torch.tensor(int(cap_id)),
                'ges_index': torch.tensor(ges_index),
                'annot_id': torch.tensor(int(img_id)),
                'cam_ids': torch.tensor([int(cam_id) for cam_id in self.data_config['camera_pairs'][pair_id]],
                                        dtype=torch.int32),
            })
        meta_data.update({
            'tf_w_c': tf_w_c,
        })

        return frames, meta_data

    def __len__(self):
        return len(self.samples)

    def get_render(self, hw=[920, 1064]):
        self.raster_settings = RasterizationSettings(
            image_size=(hw[0], hw[1]),
            faces_per_pixel=2,
            perspective_correct=True,
            blur_radius=0.,
        )
        self.lights = PointLights(
            location=[[0, 2, 0]],
            diffuse_color=((0.5, 0.5, 0.5),),
            specular_color=((0.5, 0.5, 0.5),)
        )
        self.render = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=self.raster_settings),
            shader=SoftPhongShader(lights=self.lights)
        )

    def render_hand(self, manos, K, R, t, hw, img_bg=None):
        self.get_render(hw)
        output = self.mano_layer(
            global_orient=manos['rot_pose'].reshape(-1, 3),
            hand_pose=manos['hand_pose'].reshape(-1, 45),
            betas=manos['shape'].reshape(-1, 10),
            transl=manos['trans'].reshape(-1, 3)
        )
        now_vertices = torch.bmm(R.reshape(-1, 3, 3), output.vertices.transpose(2, 1)).transpose(2, 1) + t.reshape(-1,
                                                                                                                   1, 3)
        faces = torch.tensor(self.mano_layer.faces.astype(np.int32)).repeat(1, 1, 1).type_as(manos['trans'])
        verts_rgb = torch.ones_like(self.mano_layer.v_template).type_as(manos['trans'])
        verts_rgb = verts_rgb.expand(1, verts_rgb.shape[0], verts_rgb.shape[1])
        textures = TexturesVertex(verts_rgb)
        mesh = Meshes(
            verts=now_vertices,
            faces=faces,
            textures=textures
        )
        cameras = cameras_from_opencv_projection(
            R=torch.eye(3).repeat(1, 1, 1).type_as(manos['shape']),
            tvec=torch.zeros(1, 3).type_as(manos['shape']),
            camera_matrix=K.reshape(-1, 3, 3).type_as(manos['shape']),
            image_size=torch.tensor([hw[0], hw[1]]).expand(1, 2).type_as(manos['shape'])
        ).to(manos['trans'].device)
        self.render.shader.to(manos['trans'].device)
        res = self.render(
            mesh,
            cameras=cameras
        )
        img = res[..., :3]
        img = img.reshape(-1, hw[0], hw[1], 3)
        # plt.imshow(img[0].detach().cpu().numpy())
        # plt.show()
        if img_bg is not None:
            mask = res[..., 3:4].reshape(-1, hw[0], hw[1], 1) != 0.
            img = torch.clip(img * mask + mask.logical_not() * img_bg[None], 0, 1)
        plt.imshow(img[0].detach().cpu().numpy())
        plt.show()
        return img