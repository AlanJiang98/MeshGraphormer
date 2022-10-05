import cv2
import os
import os.path as osp
import numpy as np
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


def collect_data(data_seq):
    global data_seqs
    if data_seq == {}:
        return
    for key in data_seq.keys():
        data_seqs[key] = data_seq[key]


class EvRealHands(Dataset):
    def __init__(self, config):
        self.config = config
        self.data = {}
        self.data_config = None
        self.load_events_annotations()
        self.process_samples()
        self.mano_layer = MANO(self.config['data']['smplx_path'], use_pca=False, is_rhand=True)
        self.rgb_augment = PhotometricAug(self.config['exper']['augment']['rgb_photometry'], not self.config['exper']['run_eval_only'])
        self.event_augment = PhotometricAug(self.config['exper']['augment']['event_photometry'], not self.config['exper']['run_eval_only'])
        if config['exper']['run_eval_only']:
            self.get_bbox_matrices_for_fast_sequences()
        pass

    @staticmethod
    def get_events_annotations_per_sequence(dir, is_train=True):
        if not osp.isdir(dir):
            raise FileNotFoundError('illegal directions for event sequence: {}'.format(dir))
        id = dir.split('/')[-1]
        annot = json_read(os.path.join(dir, 'annot.json'))
        if not is_train:
            if not annot['annoted'] and not (annot['motion_type'] == 'fast'):
                return {}
        mano_ids = [int(id) for id in annot['manos'].keys()]
        mano_ids.sort()
        mano_ids_np = np.array(mano_ids, dtype=np.int32)
        indices = np.diff(mano_ids_np) == 1
        annot['sample_ids'] = [str(id) for id in mano_ids_np[1:][indices]]
        K_old = np.array(annot['camera_info']['event']['K_old'])
        K_new = np.array(annot['camera_info']['event']['K_new'])
        dist = np.array(annot['camera_info']['event']['dist'])
        undistortion_param = [K_old, dist, K_new]
        events, _, _ = extract_data_from_aedat4(osp.join(dir, 'event.aedat4'), is_event=True)
        events = np.vstack([events['x'], events['y'], events['polarity'], events['timestamp']]).T
        if not np.all(np.diff(events[:, 3]) >= 0):
            events = events[np.argsort(events[:, 3])]
        first_event_time = events[0, 3]
        events[:, 3] = events[:, 3] - first_event_time
        events = events.astype(np.float32)
        # undistortion the events
        events[:, :2], legal_indices = undistortion_points(events[:, :2], undistortion_param[0],
                                                                undistortion_param[1],
                                                                undistortion_param[2],
                                                                set_bound=True, width=346,
                                                                height=260)
        events = events[legal_indices]
        data = {}
        data[id] = {
            'event': events,
            'annot': annot,
        }
        if is_main_process():
            print('Seq {} is over!'.format(dir))
        return data

    def load_events_annotations(self):
        if self.config['exper']['run_eval_only']:
            data_config = ConfigParser(self.config['data']['dataset_info']['evrealhands']['eval_yaml'])
        else:
            data_config = ConfigParser(self.config['data']['dataset_info']['evrealhands']['train_yaml'])
        self.data_config = data_config.config
        all_sub_2_seq_ids = json_read(
            osp.join(self.config['data']['dataset_info']['evrealhands']['data_dir'], self.data_config['seq_ids_path'])
        )
        self.seq_ids = []
        for sub_id in self.data_config['sub_ids']:
            self.seq_ids += all_sub_2_seq_ids[str(sub_id)]
        if self.config['exper']['debug']:
            if self.config['exper']['run_eval_only']:
                seq_id = '4'
            else:
                seq_id = '9'
            data = self.get_events_annotations_per_sequence(osp.join(self.config['data']['dataset_info']['evrealhands']['data_dir'], seq_id), not self.config['exper']['run_eval_only'])
            self.data = data
            self.seq_ids = [seq_id]
        else:
            global data_seqs
            data_seqs = {}
            pool = mp.Pool(mp.cpu_count())
            for seq_id in self.seq_ids:
                pool.apply_async(EvRealHands.get_events_annotations_per_sequence,
                                 args=(osp.join(self.config['data']['dataset_info']['evrealhands']['data_dir'], seq_id), not self.config['exper']['run_eval_only'], ),
                                 callback=collect_data)
            pool.close()
            pool.join()
            self.data = data_seqs

    def process_samples(self):
        '''
        get samples for the dataset
        '''
        self.sample_info = {
            'seq_ids': [],
            'cam_pair_ids': [],
            'samples_per_seq': None,
            'samples_sum': None
        }
        samples_per_seq = []
        # print(self.seq_ids)
        # print(self.data.keys())
        for seq_id in self.data.keys():
            for cam_pair_id, cam_pair in enumerate(self.data_config['camera_pairs']):
                if self.config['exper']['run_eval_only'] and self.data[seq_id]['annot']['motion_type'] == 'fast':
                    assert self.config['eval']['fast_fps'] % 15 == 0
                    num_samples = (len(self.data[seq_id]['annot']['frames'][cam_pair[1]])-1) * self.config['eval']['fast_fps'] / 15
                    samples_per_seq.append(num_samples)
                    sample_ids = np.arange(num_samples, dtype=np.int32).tolist()
                    self.data[seq_id]['annot']['sample_ids'] = [str(id) for id in sample_ids]
                else:
                    samples_per_seq.append(len(self.data[seq_id]['annot']['sample_ids']))
                self.sample_info['seq_ids'].append(seq_id)
                self.sample_info['cam_pair_ids'].append(cam_pair_id)
        self.sample_info['samples_per_seq'] = np.array(samples_per_seq, dtype=np.int32)
        self.sample_info['samples_sum'] = np.cumsum(self.sample_info['samples_per_seq'])

    def get_info_from_sample_id(self, sample_id):
        '''
        get image_id, event time stamp, annot id from sample id
        '''
        sample_info_index = np.sum(self.sample_info['samples_sum'] <= sample_id, dtype=np.int32)
        seq_id = self.sample_info['seq_ids'][sample_info_index]
        cam_pair = self.data_config['camera_pairs'][self.sample_info['cam_pair_ids'][sample_info_index]]
        if sample_info_index == 0:
            seq_loc = sample_id
        else:
            seq_loc = sample_id - self.sample_info['samples_sum'][sample_info_index-1]

        annot_id = self.data[seq_id]['annot']['sample_ids'][seq_loc]
        return seq_id, cam_pair, annot_id

    def load_img(self, path, order='RGB'):
        img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(img, np.ndarray):
            raise IOError("Fail to read %s" % path)
        if order == 'RGB':
            img = img[:, :, ::-1].copy()

        img = img.astype(np.float32)
        return img

    def get_annotations(self, seq_id, cam_pair, annot_id):
        #TODO change to which camera's coordinate?
        meta_data = {}
        meta_data['delta_time'] = self.data[seq_id]['annot']['delta_time']
        camera_pair_0 = self.data[seq_id]['annot']['camera_info'][cam_pair[0]]
        meta_data['R_event_l'] = torch.tensor(camera_pair_0['R'], dtype=torch.float32).view(3, 3)
        meta_data['K_event_l'] = torch.tensor(camera_pair_0['K'], dtype=torch.float32).view(3, 3)
        meta_data['t_event_l'] = torch.tensor(camera_pair_0['T'], dtype=torch.float32).view(3) / 1000.
        meta_data['R_event_r'], meta_data['K_event_r'], meta_data['t_event_r'] = \
            meta_data['R_event_l'].clone(), meta_data['K_event_l'].clone(), meta_data['t_event_l'].clone()
        camera_pair_1 = self.data[seq_id]['annot']['camera_info'][cam_pair[1]]
        meta_data['R_rgb'] = torch.tensor(camera_pair_1['R'], dtype=torch.float32).view(3, 3)
        meta_data['K_rgb'] = torch.tensor(camera_pair_1['K'], dtype=torch.float32).view(3, 3)
        meta_data['t_rgb'] = torch.tensor(camera_pair_1['T'], dtype=torch.float32).view(3) / 1000.

        if self.data[seq_id]['annot']['motion_type'] == 'fast' and self.data[seq_id]['annot']['2d_joints'][cam_pair[0]][annot_id] != []:
            meta_data['2d_joints_event'] = torch.tensor(self.data[seq_id]['annot']['2d_joints'][cam_pair[0]][annot_id], dtype=torch.float32).view(21, 2)[indices_change(2, 1)]
            joints_2d_valid_ev = True
        else:
            meta_data['2d_joints_event'] = torch.zeros((21, 2), dtype=torch.float32)
            joints_2d_valid_ev = False
        meta_data['joints_2d_valid_ev'] = joints_2d_valid_ev


        if annot_id not in self.data[seq_id]['annot']['2d_joints'][cam_pair[1]].keys() or \
                self.data[seq_id]['annot']['2d_joints'][cam_pair[1]][annot_id] == [] or annot_id == '-1':
            meta_data['2d_joints_rgb'] = torch.zeros((21, 2), dtype=torch.float32)
        else:
            meta_data['2d_joints_rgb'] = torch.tensor(self.data[seq_id]['annot']['2d_joints'][cam_pair[1]][annot_id], dtype=torch.float32).view(21, 2)[indices_change(2, 1)]

        if annot_id in self.data[seq_id]['annot']['3d_joints'].keys():
            meta_data['3d_joints'] = torch.tensor(self.data[seq_id]['annot']['3d_joints'][annot_id], dtype=torch.float32).view(-1, 3)[
                             indices_change(2, 1)] / 1000.
        else:
            meta_data['3d_joints'] = torch.zeros((21, 3), dtype=torch.float32)
        if annot_id in self.data[seq_id]['annot']['manos'].keys():
            mano = self.data[seq_id]['annot']['manos'][annot_id]
            meta_data['mano'] = {}
            meta_data['mano']['rot_pose'] = torch.tensor(mano['rot'], dtype=torch.float32).view(-1)
            meta_data['mano']['hand_pose'] = torch.tensor(mano['hand_pose'], dtype=torch.float32).view(-1)
            meta_data['mano']['shape'] = torch.tensor(mano['shape'], dtype=torch.float32).view(-1)
            meta_data['mano']['trans'] = torch.tensor(mano['trans'], dtype=torch.float32).view(3)
            meta_data['mano']['root_joint'] = torch.tensor(mano['root_joint'], dtype=torch.float32).view(3)
        else:
            meta_data['mano'] = {}
            meta_data['mano']['rot_pose'] = torch.zeros((3, ), dtype=torch.float32)
            meta_data['mano']['hand_pose'] = torch.zeros((45, ), dtype=torch.float32)
            meta_data['mano']['shape'] = torch.zeros((10, ), dtype=torch.float32)
            meta_data['mano']['trans'] = torch.zeros((3, ), dtype=torch.float32)
            meta_data['mano']['root_joint'] = torch.zeros((3, ), dtype=torch.float32)
        return meta_data

    def get_event_repre(self, seq_id, indices):
        event = self.data[seq_id]['event']
        if indices[0] == 0 or indices[-1] >= event.shape[0] - 1 or indices[-1]==indices[0]:
            return torch.zeros((2, 260, 346))
        else:
            tmp_events = event[indices[0]:indices[1]].copy()
            tmp_events[:, 3] = (tmp_events[:, 3] - event[indices[0], 3]) / (event[indices[1], 3] - event[indices[0], 3])
            tmp_events = torch.tensor(tmp_events, dtype=torch.float32)
            ev_frame = event_representations(tmp_events, repre=self.config['exper']['preprocess']['ev_repre'], hw=(260, 346))
            return ev_frame

    def get_augment_param(self):
        # augmentation for event and rgb
        augs = [[], []]
        if not self.config['exper']['run_eval_only']:
            scale_range, trans_range, rot_range = self.config['exper']['augment']['geometry']['scale'], \
                                                  self.config['exper']['augment']['geometry']['trans'], \
                                                  self.config['exper']['augment']['geometry']['rot']
            for i in range(2):
                scale = min(2*scale_range, max(-2*scale_range, np.random.randn() * scale_range)) + 1
                trans_x = min(2*trans_range, max(-2*trans_range, np.random.randn() * trans_range))
                trans_y = min(2 * trans_range, max(-2 * trans_range, np.random.randn() * trans_range))
                rot = min(2*rot_range, max(-2*rot_range, np.random.randn() * rot_range))
                augs[i] += [trans_x, trans_y, rot, scale ]
        else:
            for i in range(2):
                augs[i] += [0, 0, self.config['eval']['augment']['rot'], self.config['eval']['augment']['scale']]
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
        K = meta_data['K_'+view]
        R_wm = roma.rotvec_to_rotmat(meta_data[mano_key]['rot_pose'])
        t_wm = meta_data[mano_key]['trans']

        tf_wm = self.get_transform(R_wm, t_wm + meta_data[mano_key]['root_joint'])
        tf_cw = self.get_transform(meta_data['R_'+view], meta_data['t_'+view])

        tf_cm = tf_cw @ tf_wm

        # trans augmentation
        aff_trans_2d = torch.eye(3)
        H, W, _ = frame.shape
        trans_x_2d = augs[0] * W
        trans_y_2d = augs[1] * H
        aff_trans_2d[0, 2] = trans_x_2d
        aff_trans_2d[1, 2] = trans_y_2d
        tf_trans_3d = torch.eye(4)
        trans_x_3d = trans_x_2d * tf_cm[2, 3] / meta_data['K_'+view][0, 0]
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
        aff_scale_2d[:2, 2] = (1-augs[3]) * K[:2, 2]

        aff_2d_final = (aff_scale_2d @ aff_rot_2d @ aff_trans_2d)[:2, :]
        frame_aug = cv2.warpAffine(np.array(frame), aff_2d_final.numpy(), (W, H), flags=cv2.INTER_LINEAR)

        tf_3d_final = tf_scale_3d @ tf_rot_3d @ tf_trans_3d @ tf_cm
        tf_cw = tf_3d_final @ tf_wm.inverse()

        meta_data['R_'+view] = tf_cw[:3, :3]
        meta_data['t_'+view] = tf_cw[:3, 3]

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

    def plotshow(self, img):
        plt.imshow(img)
        plt.show()

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

    def get_indices_from_timestamps(self, timestamps, seq_id):
        timestamps = np.array(timestamps, dtype=np.float32)
        event = self.data[seq_id]['event']
        indices = np.searchsorted(
            event[:, 3],
            timestamps
        )
        index_l, index_r = self.get_l_event_indices(timestamps[1], event[indices[0]:indices[1]].copy())
        return index_l+indices[0], index_r+indices[1]

    def get_event_bbox_matrix_for_fast_sequences(self):
        for seq_id in self.data.keys():
            if self.data[seq_id]['annot']['motion_type'] == 'fast':
                bbox_ids = []
                for key in self.data[seq_id]['annot']['bbox']['event'].keys():
                    if self.data[seq_id]['annot']['bbox']['event'][key] != []:
                        bbox_ids.append(int(key))
                bbox_ids.sort()
                bbox_seq = np.zeros((len(bbox_ids), 4), dtype=np.float32)
                bbox_seq[:, 0] = np.array(bbox_ids, dtype=np.float32) * 1000000. / 15 + self.data[seq_id]['annot']['delta_time']
                # if the bbox is manual annotation, use it as GT bbox; else use machine annotation bbox
                K_old = np.array(self.data[seq_id]['annot']['camera_info']['event']['K_old'])
                K_new = np.array(self.data[seq_id]['annot']['camera_info']['event']['K_new'])
                dist = np.array(self.data[seq_id]['annot']['camera_info']['event']['dist'])
                undistortion_param = [K_old, dist, K_new]
                for i, id in enumerate(bbox_ids):
                    if str(id) in self.data[seq_id]['annot']['2d_joints']['event'].keys() and \
                            self.data[seq_id]['annot']['2d_joints']['event'][str(id)] != []:
                        kps2d = np.array(self.data[seq_id]['annot']['2d_joints']['event'][str(id)], dtype=np.float32)
                        top = kps2d[:, 1].min()
                        bottom = kps2d[:, 1].max()
                        left = kps2d[:, 0].min()
                        right = kps2d[:, 0].max()
                        top_left = np.array([left, top])
                        bottom_right = np.array([right, bottom])
                        center = (top_left + bottom_right) / 2.0
                        bbox_size = np.abs(
                            bottom_right - top_left).max() * self.config['exper']['bbox']['rate']
                        center, legal_indices = undistortion_points(center.reshape(-1, 2), undistortion_param[0],
                                                                    undistortion_param[1],
                                                                    undistortion_param[2],
                                                                    set_bound=True, width=260,
                                                                    height=346)
                        center = center[0]
                    else:
                        top_left = np.array(self.data[seq_id]['annot']['bbox']['event'][str(id)]['tl'], dtype=np.float32)
                        bottom_right = np.array(self.data[seq_id]['annot']['bbox']['event'][str(id)]['br'], dtype=np.float32)
                        center = (top_left + bottom_right) / 2.0
                        bbox_size = np.abs(
                            bottom_right - top_left).max()  # * self.config['preprocess']['bbox']['joints_to_bbox_rate'] / 1.5
                    bbox_seq[i, 1:3] = center
                    bbox_seq[i, 3:4] = bbox_size

                bbox_inter_f = interp1d(bbox_seq[:, 0], bbox_seq[:, 1:4], axis=0, kind='quadratic')
                self.bbox_inter[seq_id] = {'event': bbox_inter_f}

    def get_rgb_bbox_matrix_for_fast_sequences(self, rgb_camera_views):
        rgb_view = rgb_camera_views
        for seq_id in self.data.keys():
            if self.data[seq_id]['annot']['motion_type'] == 'fast':
                bbox_ids = []
                for key in self.data[seq_id]['annot']['2d_joints'][rgb_view].keys():
                    if self.data[seq_id]['annot']['2d_joints'][rgb_view][str(key)] != []:
                        bbox_ids.append(int(key))
                bbox_ids.sort()
                bbox_seq = np.zeros((len(bbox_ids), 4), dtype=np.float32)
                bbox_seq[:, 0] = np.array(bbox_ids, dtype=np.float32) * 1000000. / 15 + self.data[seq_id]['annot']['delta_time']
                for i, id in enumerate(bbox_ids):
                    if str(id) in self.data[seq_id]['annot']['2d_joints'][rgb_view].keys() and \
                            self.data[seq_id]['annot']['2d_joints'][rgb_view][str(id)] != []:
                        kps2d = np.array(self.data[seq_id]['annot']['2d_joints'][rgb_view][str(id)], dtype=np.float32)
                        top = kps2d[:, 1].min()
                        bottom = kps2d[:, 1].max()
                        left = kps2d[:, 0].min()
                        right = kps2d[:, 0].max()
                        top_left = np.array([left, top])
                        bottom_right = np.array([right, bottom])
                        center = (top_left + bottom_right) / 2.0
                        bbox_size = np.abs(
                            bottom_right - top_left).max() * self.config['exper']['bbox']['rate']
                        center = center
                        bbox_seq[i, 1:3] = center
                        bbox_seq[i, 3:4] = bbox_size
                bbox_inter_f = interp1d(bbox_seq[:, 0], bbox_seq[:, 1:4], axis=0, kind='quadratic')
                self.bbox_inter[seq_id][rgb_view] = bbox_inter_f

    def get_bbox_matrices_for_fast_sequences(self):
        self.bbox_inter = {}
        self.get_event_bbox_matrix_for_fast_sequences()
        for cam_view in self.data_config['camera_pairs']:
            self.get_rgb_bbox_matrix_for_fast_sequences(cam_view[1])

    def get_bbox_by_interpolation_for_fast_sequences(self, seq_id, camera_view, timestamp):
        # get bbox at t0 for fast sequences
        bbox = self.bbox_inter[seq_id][camera_view](timestamp)
        return torch.tensor(bbox, dtype=torch.int32)

    def get_seq_type(self, seq_id):
        if self.data[seq_id]['annot']['motion_type'] == 'fast':
            return 6
        if self.data[seq_id]['annot']['scene'] == 'normal':
            if self.data[seq_id]['annot']['gesture_type'] == 'fixed':
                return 0
            elif self.data[seq_id]['annot']['gesture_type'] == 'random':
                return 1
        elif self.data[seq_id]['annot']['scene'] == 'highlight':
            if self.data[seq_id]['annot']['gesture_type'] == 'fixed':
                return 2
            elif self.data[seq_id]['annot']['gesture_type'] == 'random':
                return 3
        elif self.data[seq_id]['annot']['scene'] == 'flash':
            if self.data[seq_id]['annot']['gesture_type'] == 'fixed':
                return 4
            elif self.data[seq_id]['annot']['gesture_type'] == 'random':
                return 5

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

    def __getitem__(self, idx):
        seq_id, cam_pair, annot_id = self.get_info_from_sample_id(idx)
        test_fast = self.config['exper']['run_eval_only'] and \
                    self.data[seq_id]['annot']['motion_type'] == 'fast'
        duration = 1

        if self.config['exper']['run_eval_only'] and test_fast:
            is_ere = True # todo change here!
        else:
            # for training
            is_ere = torch.rand(1) < self.config['exper']['preprocess']['ere_rate']

        if test_fast:
            meta_data = self.get_annotations(seq_id, cam_pair, str(int(int(annot_id) / (self.config['eval']['fast_fps'] / 15.))))
        else:
            meta_data = self.get_annotations(seq_id, cam_pair, annot_id)

        delta_time = meta_data['delta_time']

        if not is_ere:
            rgb_path = osp.join(self.config['data']['dataset_info']['evrealhands']['data_dir'], seq_id, 'images', \
                                cam_pair[1], 'image'+annot_id.rjust(4, '0')+'.jpg')
            rgb_t = delta_time + int(annot_id) * 1e6 / 15.
            indices_l_ev = self.get_l_event_indices(rgb_t, self.data[seq_id]['event'])
            indices_r_ev = self.get_indices_from_timestamps([rgb_t, rgb_t], seq_id)
            meta_data['3d_joints_rgb'] = meta_data['3d_joints'].clone()
            meta_data['mano_rgb'] = meta_data['mano'].copy()
        else:
            ## TODO add previous mesh as supervision or not???
            # TODO whether set the window distance between rgb frame and annotations
            if test_fast:
                periods = int(annot_id) / (self.config['eval']['fast_fps'] / 15.)
                rgb_id = int(periods)
                interval = (periods - rgb_id) * 1e6 / 15.
            else:
                rgb_id = int(annot_id) - duration
                interval = duration * 1e6 / 15.

            if str(rgb_id) in self.data[seq_id]['annot']['mano'].keys() or test_fast:
                meta_data_rgb = self.get_annotations(seq_id, cam_pair, str(rgb_id))
                meta_data['3d_joints_rgb'] = meta_data_rgb['3d_joints'].clone()
                meta_data['mano_rgb'] = meta_data_rgb['mano'].copy()
                meta_data['joints_2d_valid_ev'] = meta_data_rgb['joints_2d_valid_ev']

            if str(rgb_id) in self.data[seq_id]['annot']['frames'][cam_pair[1]]:
                rgb_path = osp.join(self.config['data']['dataset_info']['evrealhands']['data_dir'], seq_id, 'images', cam_pair[1],\
                                    'image' + str(rgb_id).rjust(4, '0') + '.jpg')
                rgb_t = delta_time + rgb_id * 1e6 / 15.
                ## get left time window
                indices_l_ev = self.get_l_event_indices(rgb_t, self.data[seq_id]['event'])
                indices_r_ev = self.get_indices_from_timestamps([rgb_t, rgb_t+interval], seq_id)

        rgb = self.load_img(rgb_path)
        l_ev_frame = self.get_event_repre(seq_id, indices_l_ev)
        l_ev_frame = torch.cat([l_ev_frame.permute(1, 2, 0), torch.zeros((260, 346, 1))], dim=2)
        r_ev_frame = self.get_event_repre(seq_id, indices_r_ev)
        r_ev_frame = torch.cat([r_ev_frame.permute(1,2,0), torch.zeros((260, 346, 1))], dim=2)

        aug_params = self.get_augment_param()
        # self.render_hand(
        #     meta_data['mano_rgb'],
        #     meta_data['K_rgb'],
        #     meta_data['R_rgb'],
        #     meta_data['t_rgb'],
        #     hw=[920, 1064],
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
        # print(aug_params[1])

        # geometry augmentation


        rgb_aug_ori = self.augment_one_view(aug_params[1], rgb, meta_data, view='rgb')
        l_ev_frame_aug_ori = self.augment_one_view(aug_params[0], l_ev_frame, meta_data, view='event_l')
        r_ev_frame_aug_ori = self.augment_one_view(aug_params[0], r_ev_frame, meta_data, view='event_r')
        # photometric augmentation
        rgb_aug = self.rgb_augment(rgb_aug_ori.copy().astype(np.uint8)).astype(np.float32) / 255.
        l_ev_frame_aug = self.event_augment((l_ev_frame_aug_ori.copy() * 255).astype(np.uint8)).astype(np.float32) / 255.
        r_ev_frame_aug = self.event_augment((r_ev_frame_aug_ori.copy() * 255).astype(np.uint8)).astype(np.float32) / 255.

        if test_fast:
            bbox_l = self.get_bbox_by_interpolation_for_fast_sequences(seq_id, cam_pair[0], rgb_t)
            bbox_rgb = self.get_bbox_by_interpolation_for_fast_sequences(seq_id, cam_pair[1], rgb_t)
            bbox_r = self.get_bbox_by_interpolation_for_fast_sequences(seq_id, cam_pair[0], rgb_t+interval)
        else:
            bbox_l, bbox_rgb, bbox_r = self.get_bbox_list(meta_data, self.config['exper']['bbox']['rate'])


        # self.render_hand(
        #     meta_data['mano_rgb'],
        #     meta_data['K_rgb'],
        #     meta_data['R_rgb'],
        #     meta_data['t_rgb'],
        #     hw=[920, 1064],
        #     img_bg=rgb_aug,
        # )
        # self.render_hand(
        #     meta_data['mano_rgb'],
        #     meta_data['K_event_l'],
        #     meta_data['R_event_l'],
        #     meta_data['t_event_l'],
        #     hw=[260, 346],
        #     img_bg=l_ev_frame_aug,
        # )
        # self.render_hand(
        #     meta_data['mano'],
        #     meta_data['K_event_r'],
        #     meta_data['R_event_r'],
        #     meta_data['t_event_r'],
        #     hw=[260, 346],
        #     img_bg=r_ev_frame_aug,
        # )
        lt_rgb, sc_rgb, rgb_crop = self.crop(bbox_rgb, rgb_aug, self.config['exper']['bbox']['rgb']['size'], hw=[920, 1064])
        lt_l_ev, sc_l_ev, l_ev_crop = self.crop(bbox_l, l_ev_frame_aug, self.config['exper']['bbox']['event']['size'],
                                             hw=[260, 346])
        lt_r_ev, sc_r_ev, r_ev_crop = self.crop(bbox_r, r_ev_frame_aug, self.config['exper']['bbox']['event']['size'],
                                             hw=[260, 346])

        tf_w_c = self.change_camera_view(meta_data)

        meta_data.update({
            'lt_rgb': lt_rgb,
            'sc_rgb': sc_rgb,
            'lt_l_ev': lt_l_ev,
            'sc_l_ev': sc_l_ev,
            'lt_r_ev': lt_r_ev,
            'sc_r_ev': sc_r_ev,
        })
        # self.plotshow(rgb_crop)
        # self.plotshow(l_ev_crop)
        # self.plotshow(r_ev_crop)

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
            frames.update(
                {
                    'rgb_ori': torch.tensor(rgb_aug_ori, dtype=torch.float32),
                    'event_l_ori': torch.tensor(l_ev_frame_aug_ori, dtype=torch.float32),
                    'event_r_ori': torch.tensor(r_ev_frame_aug_ori, dtype=torch.float32),
                }
            )
            meta_data.update({
                'seq_id': torch.tensor(int(seq_id)),
                'seq_type': torch.tensor(self.get_seq_type(seq_id)),
                'annot_id': torch.tensor(int(annot_id)),
            })
        meta_data.update({
            'tf_w_c': tf_w_c,
        })
        return frames, meta_data

    def __len__(self):
        return int(self.sample_info['samples_sum'][-1])

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
        now_vertices = torch.bmm(R.reshape(-1, 3, 3), output.vertices.transpose(2, 1)).transpose(2, 1) + t.reshape(-1, 1, 3)
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
