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
# import albumentations as A
from src.configs.config_parser import ConfigParser
from smplx import MANO
from src.utils.augment import PhotometricAug
from src.utils.comm import is_main_process
from scipy.interpolate import interp1d
from PIL import Image

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
import io
import tarfile
from ffrecord import PackedFolder
import pdb
import zipfile

def collect_data(data):
    global samples, bbox_inter_f,img_dict,event_dict
    for key in data[0].keys():
        samples[key] = data[0][key]
    for key in data[1].keys():
        bbox_inter_f[key] = data[1][key]
    # for key in data[2].keys():
    #     img_dict[key] = data[2][key]
    for key in data[2].keys():
        event_dict[key] = data[2][key]

# def collect_image(data_seq):
#     global img_dict
#     if data_seq == {}:
#         return
#     for key in data_seq.keys():
#         img_dict[key] = data_seq[key]

# def collect_event(data_seq):
#     global event_dict
#     if data_seq == {}:
#         return
#     for key in data_seq.keys():
#         event_dict[key] = data_seq[key]

class Interhand(Dataset):
    def __init__(self, config):
        self.config = config
        self.data_config = None
        self.tar_name = "self.config['data']['dataset_info']['interhand']['data_dir']"
        # self.tar = tarfile.open(self.config['data']['dataset_info']['interhand']['data_dir'])
        # self.img_dict = {}
        self.event_dict = {}
        self.folder = PackedFolder(self.config['data']['dataset_info']['interhand']['ffr_dir'])
        self.load_annotations()
        self.process_samples()
        self.mano_layer = MANO(self.config['data']['smplx_path'], use_pca=False, is_rhand=True)
        self.rgb_augment = PhotometricAug(self.config['exper']['augment']['rgb_photometry'],
                                          not self.config['exper']['run_eval_only'])
        self.event_augment = PhotometricAug(self.config['exper']['augment']['event_photometry'],
                                            not self.config['exper']['run_eval_only'])
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
        
        if is_main_process():
            print('Interhand length is ', self.__len__())

    def load_annotations(self):
        if self.config['exper']['run_eval_only']:
            data_config = ConfigParser(self.config['data']['dataset_info']['interhand']['eval_yaml'])
        else:
            data_config = ConfigParser(self.config['data']['dataset_info']['interhand']['train_yaml'])
        self.data_config = data_config.config
        params = self.folder.read([self.data_config['mano_path'],self.data_config['joint_path'],self.data_config['cam_path']])
        fps = [(io.BytesIO(i)).read() for i in params]
        mano_params = json.loads(fps[0].decode('utf-8'))
        joint_params = json.loads(fps[1].decode('utf-8'))
        cam_params = json.loads(fps[2].decode('utf-8'))
        # fp = io.BytesIO(folder.read_one("0/annot.json"))
        # bytestring = fp.read([self.data_config['mano_path'],self.data_config['joint_path'],self.data_config['cam_path']])
        # result = json.loads(bytestring.decode('utf-8'))
        # mano_params = json.load(self.tar.extractfile(osp.join("Interhand", self.data_config['mano_path'])))
        # joint_params = json.load(self.tar.extractfile(osp.join("Interhand", self.data_config['joint_path'])))
        # cam_params = json.load(self.tar.extractfile(osp.join("Interhand", self.data_config['cam_path'])))
        self.data = {
            'mano': mano_params,
            'joint3d': joint_params,
            'cam': cam_params
        }
        self.get_synthetic_affine_matrix()
        self.ges_list = self.folder.list((osp.join(self.data_config['event_dir'], 'Capture0')))
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
    # @staticmethod
    # def load_seq_image_to_dataset(tar_name, image_list):
    #     # print(f"loading {path}")
    #     tar = tarfile.open(tar_name,"r")
    #     data = {}
    #     # image_list = [i for i in tar.getnames() if i.endswith(".jpg") and i.startswith(path+'/')]
    #     for name in image_list:
    #         # print(name)
    #         # img = tar.extractfile(name)
    #         img = Image.open(tar.extractfile(name))
    #         img = np.array(img)
    #         data[name]=img
    #     return data
    
    @staticmethod
    def get_samples_per_cap(data, tar_name, config, data_config, ges_list, cap_id, folder=None):
        if is_main_process():
            print('Process cap {} start!'.format(cap_id))
        # tar = tarfile.open(tar_name)
        cap_ev_dir = osp.join(data_config['event_dir'], 'Capture' + cap_id)
        # ges_list_ = [i.split('/')[-1] for i in tar.getnames() if i.startswith(osp.join(cap_ev_dir,'0')) and "cam" not in i]
        ges_list_ = folder.list((osp.join(data_config['event_dir'], 'Capture0')))
        ges_list_.sort()
        # npz_list = [i for i in tar.getnames() if i.endswith('.npz')]
        samples = []
        bbox_inter_f_cap = {}
        ev_list_ = {}
        # img_list_ = {}
        for ges in ges_list_:
            # pdb.set_trace()
            if ges in ges_list:
                ges_index = ges_list.index(ges)
                # print(ges)
                for pair_id, cam_pair in enumerate(data_config['camera_pairs']):
                    # print(pair_id)
                    ev_path = osp.join(cap_ev_dir, ges, 'cam' + cam_pair[0], 'events.npz')
                    # if ev_path in npz_list:
                    if folder.exists(ev_path):
                        # ev_path = osp.join("Interhand", ev_path)
                        # events = np.load(tar.extractfile(ev_path))
                        # events = events['events']
                        fp = io.BytesIO(folder.read_one(ev_path))

                        # kwargs['allowZip64'] = True
                        np_file = zipfile.ZipFile(fp,allowZip64=True)
                        with np_file.open('events.npy',"r") as myfile:
                            events = np.load(myfile,allow_pickle=True)


                        events[:, :1] = 345 - events[:, :1]
                        events[:, 1:2] = 259 - events[:, 1:2]
                        ev_list_[ev_path] = events

                        img_dir_ = osp.join(data_config['img_dir'], \
                                            'Capture' + cap_id, ges, 'cam' + cam_pair[1])
                        # if not osp.exists(img_dir_):
                        #     continue
                        # img_name_list = [i.split('/')[-1] for i in tar.getnames() if i.startswith(img_dir_) and "jpg" in i]
                        img_name_list =  [i.split('/')[-1] for i in folder.list(img_dir_)]
                        img_name_list.sort()
                        # pdb.set_trace()
                        img_id_list = [img_name[5:-4] for img_name in img_name_list]
                        img_valid_id_list = []
                        for img_id in img_id_list:
                            # print(f"img_id: {img_id}")
                            if img_id in data['mano'][cap_id].keys() and img_id in data['joint3d'][cap_id].keys():
                                if data['joint3d'][cap_id][img_id]['hand_type'] == 'right' \
                                        and data['joint3d'][cap_id][img_id]['hand_type_valid']:
                                    img_valid_id_list.append(img_id)
                        if len(img_valid_id_list) <= 30:
                            continue
                        start = 24
                        for i, img_valid_id in enumerate(img_valid_id_list[start:-5]):
                            # print(f"img_valid_id: {img_valid_id}")
                            time = (int(img_valid_id) - int(img_id_list[0])) / 90. * 1000000.
                            if int(img_valid_id_list[start + i + 2]) - int(img_valid_id) == 6:
                                item = (int(cap_id), ges_index, pair_id, img_valid_id, time)
                                samples.append(item)
                        # for i, img_valid_id in enumerate(img_valid_id_list[start:]):
                        #     img_path = osp.join(img_dir_, 'image' + img_valid_id + '.jpg')
                        #     img = Image.open(tar.extractfile(img_path))
                        #     img = np.array(img)
                        #     img_list_[img_path]=img
                        # pdb.set_trace()
                        if ges in bbox_inter_f_cap.keys():
                            # print(list(bbox_inter_f_cap.keys()))
                            continue
                        else:
                            # print(f"ges2: {ges}")
                            valid_ids = []
                            valid_joint3d = []
                            for img_id in img_id_list:
                                # print(f"img_id2: {img_id}")
                                if img_id in data['joint3d'][cap_id].keys():
                                    # print(f"add valid id: {img_id}")
                                    valid_ids.append(int(img_id))
                                    valid_joint3d.append(copy.deepcopy(data['joint3d'][cap_id][img_id]['world_coord'][:21]))
                            ids_ = np.array(valid_ids, dtype=np.float32)
                            id_time = (ids_ - ids_[0]) / 90. * 1000000.
                            # print("?????")
                            joint3d_ = np.array(valid_joint3d, dtype=np.float32)[:, indices_change(0, 1)] / 1000.
                            # print("stop here?")
                            # print(id_time.shape, joint3d_.shape)
                            # print(id_time)
                            # print(joint3d_)
                            # np.save("id_time.npy", id_time)
                            # np.save("joint3d_.npy", joint3d_)
                            bbox_inter_f_ = interp1d(id_time, joint3d_, axis=0, kind='linear')
                            # print("?")
                            bbox_inter_f_cap[ges] = bbox_inter_f_
                            # print(f"finish ges: {ges}, imagepair: {pair_id}")
        if is_main_process():
            print('Process cap {} over!'.format(cap_id))
        return [{cap_id: samples}, {cap_id: bbox_inter_f_cap},{cap_id: ev_list_}]

    def process_samples(self):
        self.samples = []
        self.bbox_inter = {}
        valid_cap_ids = self.folder.list( self.data_config['event_dir'])
        # valid_cap_ids = [i.split('/')[-1] for i in self.tar.getnames() if i.startswith(osp.join("Interhand", self.data_config['event_dir'],'Capture')) and "00" not in i]
        # valid_cap_ids = os.listdir(osp.join(self.config['data']['dataset_info']['interhand']['data_dir'], self.data_config['event_dir']))
        valid_cap_ids.sort()
        # pdb.set_trace()
        if self.config['exper']['debug']:
            cap_id = '1'
            data_ = self.get_samples_per_cap(self.data, self.tar_name, self.config, self.data_config, self.ges_list, cap_id,self.folder)
            self.samples += data_[0][cap_id]
            self.bbox_inter[cap_id] = data_[1][cap_id]
            # self.img_dict.update(data_[2][cap_id])
            self.event_dict.update(data_[2][cap_id])
            # pdb.set_trace()
        else:
            global samples
            global bbox_inter_f
            global img_dict
            global event_dict
            samples = {}
            bbox_inter_f = {}
            img_dict = {}
            event_dict = {}
            pool = mp.Pool(9)
            for cap_id in self.data_config['cap_ids']:
                if 'Capture'+str(cap_id) in valid_cap_ids:
                    cap_id = str(cap_id)
                    pool.apply_async(
                        Interhand.get_samples_per_cap,
                        args=(
                            self.data, self.tar_name, copy.deepcopy(self.config), copy.deepcopy(self.data_config), copy.deepcopy(self.ges_list), cap_id, self.folder,
                        ),
                        callback=collect_data
                    )
            pool.close()
            pool.join()
            # for cap_id in self.data_config['cap_ids']:
            #     if 'Capture'+str(cap_id) in valid_cap_ids:
            #         cap_id = str(cap_id)
            #         # if self.config['exper']['supervision'] and int(cap_id) not in self.data_config['super_ids']:
            #         #     continue
            #         data_ = self.get_samples_per_cap(self.data, self.tar_name, self.config, self.data_config, self.ges_list, cap_id)
            #         # pdb.set_trace()
            #         self.samples += data_[0][cap_id]
            #         self.bbox_inter[cap_id] = data_[1][cap_id]
            # sample_keys = list(samples.keys())
            # if self.config['exper']['supervision']:
            #     for cap_id in sample_keys:
            #         if int(cap_id) not in self.data_config['super_ids']:
            #             samples.pop(cap_id)
            # cap_id_list = list(samples.keys())
            # cap_id_list.sort()
            # for cap_id in cap_id_list:
            #     self.samples += samples[cap_id]
            #     self.bbox_inter[cap_id] = bbox_inter_f[cap_id]
            for cap_id in self.data_config['cap_ids']:
                if 'Capture'+str(cap_id) in valid_cap_ids:
                    cap_id = str(cap_id)
                    # if self.config['exper']['supervision'] and int(cap_id) not in self.data_config['super_ids']:
                    #     continue
                    # data_ = self.get_samples_per_cap(self.data, self.tar_name, self.config, self.data_config, self.ges_list, cap_id)
                    # pdb.set_trace()
                    self.samples += samples[cap_id]
                    self.bbox_inter[cap_id] = bbox_inter_f[cap_id]
                    # self.img_dict.update(img_dict[cap_id])
                    self.event_dict.update(event_dict[cap_id])
            if is_main_process():
                print('All the sequences for Interhand items: {}'.format(len(self.samples)))
                # print(f"img_dict: {len(self.img_dict)}, event_dict: {len(self.event_dict)}")

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

        K_event, R_event, t_event = self.get_camera_params(cap_id, cam_id=cam_pair[0], is_event=True)
        meta_data.update({
            'K_event': K_event,
            'R_event': R_event,
            't_event': t_event,
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

        meta_data['joints_2d_valid_ev'] = False
        meta_data['joints_2d_valid_rgb'] = False
        meta_data['joints_3d_valid'] = True
        meta_data['mano_valid'] = True

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
            if not self.config['exper']['run_eval_only']:
                num = (np.random.rand(1) - 0.5) * 2 * self.config['exper']['preprocess']['num_var'] + \
                      self.config['exper']['preprocess']['num_window']
            else:
                num = self.config['exper']['preprocess']['num_window']
            index_l = max(0, index_r - int(num))
        return index_l, index_r

    def get_indices_from_timestamps(self, timestamps, event):
        timestamps = np.array(timestamps, dtype=np.float32)
        indices = np.searchsorted(
            event[:, 3],
            timestamps
        )
        index_l, index_r = self.get_l_event_indices(timestamps[1], event[indices[0]:indices[1]].copy())
        return index_l+indices[0], index_r+indices[0]

    def load_img(self, path, order='RGB'):
        # img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # with tarfile.open(self.tar_name) as tar:
        #     img = np.array(Image.open(tar.extractfile(path)))
        fp = io.BytesIO(self.folder.read_one(path))
        img = cv2.imdecode(np.frombuffer(fp.read(), np.uint8), cv2.IMREAD_COLOR)
        if not isinstance(img, np.ndarray):
            raise IOError("Fail to read %s" % path)
        if order == 'RGB':
            img = img[:, :, ::-1].copy()

        img = img.astype(np.float32)
        return img

    def get_event_repre(self, event, indices):
        if indices[0] == 0 or indices[-1] >= event.shape[0] - 1 or indices[-1]==indices[0]:
            return torch.zeros((260, 346, 3))
        else:
            tmp_events = event[indices[0]:indices[1]].copy()
            tmp_events[:, 3] = (tmp_events[:, 3] - event[indices[0], 3]) / (event[indices[1], 3] - event[indices[0], 3])
            tmp_events = torch.tensor(tmp_events, dtype=torch.float32)
            ev_frame = event_representations(tmp_events, repre=self.config['exper']['preprocess']['ev_repre'], hw=(260, 346))
            if ev_frame.shape[0] == 2:
                ev_frame = torch.cat([ev_frame, torch.zeros((1, 260, 346))], dim=0)
            return ev_frame.permute(1, 2, 0)

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

    # def augment_one_view(self, augs, frame, meta_data, view='rgb'):
    #     if view == 'rgb' or 'event_l':
    #         mano_key = 'mano_rgb'
    #     else:
    #         mano_key = 'mano'
    #     K = meta_data['K_' + view]
    #     R_wm = roma.rotvec_to_rotmat(meta_data[mano_key]['rot_pose'])
    #     t_wm = meta_data[mano_key]['trans']
    #
    #     tf_wm = self.get_transform(R_wm, t_wm + meta_data[mano_key]['root_joint'])
    #     tf_cw = self.get_transform(meta_data['R_' + view], meta_data['t_' + view])
    #
    #     tf_cm = tf_cw @ tf_wm
    #
    #     # trans augmentation
    #     aff_trans_2d = torch.eye(3)
    #     H, W, _ = frame.shape
    #     trans_x_2d = augs[0] * W
    #     trans_y_2d = augs[1] * H
    #     aff_trans_2d[0, 2] = trans_x_2d
    #     aff_trans_2d[1, 2] = trans_y_2d
    #     tf_trans_3d = torch.eye(4)
    #     trans_x_3d = trans_x_2d * tf_cm[2, 3] / meta_data['K_' + view][0, 0]
    #     trans_y_3d = trans_y_2d * tf_cm[2, 3] / meta_data['K_' + view][1, 1]
    #     tf_trans_3d[0, 3] = trans_x_3d
    #     tf_trans_3d[1, 3] = trans_y_3d
    #
    #     # rotation
    #     rot_2d = roma.rotvec_to_rotmat(augs[2] * torch.tensor([0.0, 0.0, 1.0]))
    #     tf_rot_3d = self.get_transform(rot_2d, torch.zeros(3))
    #     aff_rot_2d = torch.eye(3)
    #     aff_rot_2d[:2, :2] = rot_2d[:2, :2]
    #     aff_rot_2d[0, 2] = (1 - rot_2d[0, 0]) * K[0, 2] - rot_2d[0, 1] * K[1, 2]
    #     aff_rot_2d[1, 2] = (1 - rot_2d[1, 1]) * K[1, 2] - rot_2d[1, 0] * K[0, 2]
    #
    #     tf_3d_tmp = tf_rot_3d @ tf_trans_3d @ tf_cm
    #     # scale
    #     tf_scale_3d = torch.eye(4)
    #     # tf_scale_3d[2, 2] = 1.0 / augs[3]
    #     tf_scale_3d[2, 3] = (1.0 / augs[3] - 1.) * tf_3d_tmp[2, 3]
    #     aff_scale_2d = torch.eye(3)
    #     aff_scale_2d[0, 0], aff_scale_2d[1, 1] = augs[3], augs[3]
    #     aff_scale_2d[:2, 2] = (1 - augs[3]) * K[:2, 2]
    #
    #     aff_2d_final = (aff_scale_2d @ aff_rot_2d @ aff_trans_2d)[:2, :]
    #     frame_aug = cv2.warpAffine(np.array(frame), aff_2d_final.numpy(), (W, H), flags=cv2.INTER_LINEAR)
    #
    #     tf_3d_final = tf_scale_3d @ tf_rot_3d @ tf_trans_3d @ tf_cm
    #     tf_cw = tf_3d_final @ tf_wm.inverse()
    #
    #     meta_data['R_' + view] = tf_cw[:3, :3]
    #     meta_data['t_' + view] = tf_cw[:3, 3]
    #
    #     return frame_aug

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
    #
    # def get_bbox_list(self, meta_data, rate=1.5):
    #     bbox_rgb = self.get_bbox_from_joints(
    #         meta_data['K_rgb'],
    #         meta_data['R_rgb'],
    #         meta_data['t_rgb'],
    #         meta_data['3d_joints_rgb'],
    #         rate,
    #     )
    #     bbox_l = self.get_bbox_from_joints(
    #         meta_data['K_event_l'],
    #         meta_data['R_event_l'],
    #         meta_data['t_event_l'],
    #         meta_data['3d_joints_rgb'],
    #         rate,
    #     )
    #     bbox_r = self.get_bbox_from_joints(
    #         meta_data['K_event_r'],
    #         meta_data['R_event_r'],
    #         meta_data['t_event_r'],
    #         meta_data['3d_joints'],
    #         rate,
    #     )
    #     return bbox_l, bbox_rgb, bbox_r

    def get_valid_time(self, x, time):
        if x[0] >= time:
            time_ = x[0]
        elif x[-1] <= time:
            time_ = x[-1]
        else:
            time_ = time
        return time_

    def get_bbox_by_interpolation(self, seq_id, ges, timestamp, K=None, R=None, t=None):
        # get bbox at t0 for fast sequences
        timestamp = self.get_valid_time(self.bbox_inter[seq_id][ges].x, timestamp)
        joints = self.bbox_inter[seq_id][ges](timestamp)
        joints = torch.tensor(joints, dtype=torch.float32)
        bbox = self.get_bbox_from_joints(K, R, t, joints, rate=1.5)
        return bbox

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

    # def change_camera_view(self, meta_data):
    #     if self.config['exper']['preprocess']['cam_view'] == 'world':
    #         return self.get_transform(torch.eye(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32))
    #     elif self.config['exper']['preprocess']['cam_view'] == 'rgb':
    #         tf_rgb_w = self.get_transform(meta_data['R_rgb'], meta_data['t_rgb'])
    #         for mano_key in ['mano', 'mano_rgb']:
    #             if mano_key in meta_data.keys():
    #                 R_wm = roma.rotvec_to_rotmat(meta_data[mano_key]['rot_pose'])
    #                 t_wm = meta_data[mano_key]['trans']
    #                 tf_wm = self.get_transform(R_wm, t_wm + meta_data[mano_key]['root_joint'])
    #                 tf_rgb_m = tf_rgb_w @ tf_wm
    #                 meta_data[mano_key]['rot_pose'] = roma.rotmat_to_rotvec(tf_rgb_m[:3, :3])
    #                 meta_data[mano_key]['trans'] = tf_rgb_m[:3, 3] - meta_data[mano_key]['root_joint']
    #         for joints_key in ['3d_joints', '3d_joints_rgb']:
    #             if joints_key in meta_data.keys():
    #                 joints_new = (tf_rgb_w[:3, :3] @ (meta_data[joints_key].transpose(0, 1))\
    #                               + tf_rgb_w[:3, 3:4]).transpose(0, 1)
    #                 meta_data[joints_key] = joints_new
    #         for event_cam in ['event_l', 'event_r']:
    #             tf_e_w = self.get_transform(meta_data['R_'+event_cam], meta_data['t_'+event_cam])
    #             tf_e_rgb = tf_e_w @ tf_rgb_w.inverse()
    #             meta_data['R_' + event_cam] = tf_e_rgb[:3, :3]
    #             meta_data['t_' + event_cam] = tf_e_rgb[:3, 3]
    #         meta_data['R_rgb'] = torch.eye(3, dtype=torch.float32)
    #         meta_data['t_rgb'] = torch.zeros(3, dtype=torch.float32)
    #         return tf_rgb_w.inverse()
    #     elif self.config['exper']['preprocess']['cam_view'] == 'event':
    #         tf_event_w = self.get_transform(meta_data['R_event_l'], meta_data['t_event_l'])
    #         for mano_key in ['mano', 'mano_rgb']:
    #             if mano_key in meta_data.keys():
    #                 R_wm = roma.rotvec_to_rotmat(meta_data[mano_key]['rot_pose'])
    #                 t_wm = meta_data[mano_key]['trans']
    #                 tf_wm = self.get_transform(R_wm, t_wm + meta_data[mano_key]['root_joint'])
    #                 tf_event_m = tf_event_w @ tf_wm
    #                 meta_data[mano_key]['rot_pose'] = roma.rotmat_to_rotvec(tf_event_m[:3, :3])
    #                 meta_data[mano_key]['trans'] = tf_event_m[:3, 3] - meta_data[mano_key]['root_joint']
    #         for joints_key in ['3d_joints', '3d_joints_rgb']:
    #             if joints_key in meta_data.keys():
    #                 joints_new = (tf_event_w[:3, :3] @ (meta_data[joints_key].transpose(0, 1))\
    #                               + tf_event_w[:3, 3:4]).transpose(0, 1)
    #                 meta_data[joints_key] = joints_new
    #         for rgb_cam in ['rgb']:
    #             tf_rgb_w = self.get_transform(meta_data['R_'+rgb_cam], meta_data['t_'+rgb_cam])
    #             tf_rgb_e = tf_rgb_w @ tf_event_w.inverse()
    #             meta_data['R_' + rgb_cam] = tf_rgb_e[:3, :3]
    #             meta_data['t_' + rgb_cam] = tf_rgb_e[:3, 3]
    #         meta_data['R_event_l'] = torch.eye(3, dtype=torch.float32)
    #         meta_data['t_event_l'] = torch.zeros(3, dtype=torch.float32)
    #         meta_data['R_event_r'] = torch.eye(3, dtype=torch.float32)
    #         meta_data['t_event_r'] = torch.zeros(3, dtype=torch.float32)
    #         return tf_event_w.inverse()
    #     else:
    #         raise NotImplementedError('no implemention for change camera view!')

    def valid_bbox(self, bbox, hw=[260, 346]):
        if (bbox[:2] < 0).any() or bbox[0] > hw[1] or bbox[1] > hw[0] or bbox[2] < 10 or bbox[2] > 400:
            return False
        else:
            return True

    def plotshow(self, img):
        plt.imshow(img)
        plt.show()

    def get_trans_from_augment(self, augs, meta_data, H, W, view='rgb'):
        mano_key = 'mano'
        K = meta_data['K_'+view]
        R_wm = roma.rotvec_to_rotmat(meta_data[mano_key]['rot_pose'])
        t_wm = meta_data[mano_key]['trans']

        tf_wm = self.get_transform(R_wm, t_wm + meta_data[mano_key]['root_joint'])
        tf_cw = self.get_transform(meta_data['R_'+view], meta_data['t_'+view])

        tf_cm = tf_cw @ tf_wm

        # trans augmentation
        aff_trans_2d = torch.eye(3)
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
        # frame_aug = cv2.warpAffine(np.array(frame), aff_2d_final.numpy(), (W, H), flags=cv2.INTER_LINEAR)

        tf_3d_final = tf_scale_3d @ tf_rot_3d @ tf_trans_3d @ tf_cm
        tf_cw = tf_3d_final @ tf_wm.inverse()

        # meta_data['R_'+view] = tf_cw[:3, :3]
        # meta_data['t_'+view] = tf_cw[:3, 3]

        return aff_2d_final.numpy(), tf_cw

    def change_camera_view(self, meta_data):
        if self.config['exper']['preprocess']['cam_view'] == 'world':
            return self.get_transform(torch.eye(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32))
        elif self.config['exper']['preprocess']['cam_view'] == 'rgb':
            tf_rgb_w = self.get_transform(meta_data['R_rgb'], meta_data['t_rgb'])
            for mano_key in ['mano', ]:
                if mano_key in meta_data.keys():
                    R_wm = roma.rotvec_to_rotmat(meta_data[mano_key]['rot_pose'])
                    t_wm = meta_data[mano_key]['trans']
                    tf_wm = self.get_transform(R_wm, t_wm + meta_data[mano_key]['root_joint'])
                    tf_rgb_m = tf_rgb_w @ tf_wm
                    meta_data[mano_key]['rot_pose'] = roma.rotmat_to_rotvec(tf_rgb_m[:3, :3])
                    meta_data[mano_key]['trans'] = tf_rgb_m[:3, 3] - meta_data[mano_key]['root_joint']
            for joints_key in ['3d_joints', ]:
                if joints_key in meta_data.keys():
                    joints_new = (tf_rgb_w[:3, :3] @ (meta_data[joints_key].transpose(0, 1))\
                                  + tf_rgb_w[:3, 3:4]).transpose(0, 1)
                    meta_data[joints_key] = joints_new
            for event_cam in ['event', ]:
                tf_e_w = self.get_transform(meta_data['R_'+event_cam], meta_data['t_'+event_cam])
                tf_e_rgb = tf_e_w @ tf_rgb_w.inverse()
                meta_data['R_' + event_cam] = tf_e_rgb[:3, :3]
                meta_data['t_' + event_cam] = tf_e_rgb[:3, 3]
            meta_data['R_rgb'] = torch.eye(3, dtype=torch.float32)
            meta_data['t_rgb'] = torch.zeros(3, dtype=torch.float32)
            return tf_rgb_w.inverse()
        elif self.config['exper']['preprocess']['cam_view'] == 'event':
            tf_event_w = self.get_transform(meta_data['R_event'], meta_data['t_event'])
            for mano_key in ['mano', ]:
                if mano_key in meta_data.keys():
                    R_wm = roma.rotvec_to_rotmat(meta_data[mano_key]['rot_pose'])
                    t_wm = meta_data[mano_key]['trans']
                    tf_wm = self.get_transform(R_wm, t_wm + meta_data[mano_key]['root_joint'])
                    tf_event_m = tf_event_w @ tf_wm
                    meta_data[mano_key]['rot_pose'] = roma.rotmat_to_rotvec(tf_event_m[:3, :3])
                    meta_data[mano_key]['trans'] = tf_event_m[:3, 3] - meta_data[mano_key]['root_joint']
            for joints_key in ['3d_joints', ]:
                if joints_key in meta_data.keys():
                    joints_new = (tf_event_w[:3, :3] @ (meta_data[joints_key].transpose(0, 1))\
                                  + tf_event_w[:3, 3:4]).transpose(0, 1)
                    meta_data[joints_key] = joints_new
            for rgb_cam in ['rgb']:
                tf_rgb_w = self.get_transform(meta_data['R_'+rgb_cam], meta_data['t_'+rgb_cam])
                tf_rgb_e = tf_rgb_w @ tf_event_w.inverse()
                meta_data['R_' + rgb_cam] = tf_rgb_e[:3, :3]
                meta_data['t_' + rgb_cam] = tf_rgb_e[:3, 3]
            meta_data['R_event'] = torch.eye(3, dtype=torch.float32)
            meta_data['t_event'] = torch.zeros(3, dtype=torch.float32)
            return tf_event_w.inverse()
        else:
            raise NotImplementedError('no implemention for change camera view!')

    def get_default_bbox(self, hw=[260, 346], size=128):
        return torch.tensor([hw[1] / 2, hw[0] / 2, size], dtype=torch.float32)

    def __getitem__(self, idx):
        frames_output, meta_data_output = [], []
        cap_id, ges_index, pair_id, img_id, t_start = self.samples[idx]
        steps = self.config['exper']['preprocess']['steps']
        bbox_valid = True
        aug_params = self.get_augment_param()

        for step in range(steps):
            meta_data = self.get_annotations(str(cap_id), self.data_config['camera_pairs'][pair_id], img_id)
            # with tarfile.open(self.tar_name) as tar:
            #     events = np.load(tar.extractfile(osp.join("Interhand", \
            #                             self.data_config['event_dir'], \
            #                             'Capture' + str(cap_id),
            #                             self.ges_list[ges_index], \
            #                             'cam' + self.data_config['camera_pairs'][pair_id][0],
            #                             'events.npz'
            #                             )))
            #     events = events['events']
            #     events[:, :1] = 345 - events[:, :1]
            #     events[:, 1:2] = 259 - events[:, 1:2]
            events = self.event_dict[osp.join(self.data_config['event_dir'], \
                                        'Capture' + str(cap_id),
                                        self.ges_list[ges_index], \
                                        'cam' + self.data_config['camera_pairs'][pair_id][0],
                                        'events.npz'
                                        )]

            if step == 0:
                segments = 1
            else:
                segments = self.config['exper']['preprocess']['segments']

            rgb_path = osp.join( self.data_config['img_dir'],\
                                'Capture'+str(cap_id), self.ges_list[ges_index], \
                                'cam'+self.data_config['camera_pairs'][pair_id][1], 'image'+img_id+'.jpg')
            # if rgb_path == "Interhand/train/Capture1/0026_four_count/cam400053/image11075.jpg":
            #     pdb.set_trace()
            rgb = self.load_img(rgb_path)
            rgb_valid = True
            meta_data['rgb_valid'] = rgb_valid
            t_target = t_start + step * 1000000. / 30.
            T_ = 1e6 / 30.

            ev_frames = []

            for segment in range(segments):
                t_l = t_target - T_ + segment / segments * T_
                t_r = t_target - T_ + (segment + 1) / segments * T_
                indices_ev = self.get_indices_from_timestamps([t_l, t_r], events)
                # print('segment', segment, indices_ev)
                ev_frame = self.get_event_repre(events, indices_ev)
                ev_frames.append(ev_frame)

            if step == 0:
                aff_2d_rgb, tf_cw_rgb = self.get_trans_from_augment(aug_params[1], meta_data, 512, 334, view='rgb')
                aff_2d_ev, tf_cw_ev = self.get_trans_from_augment(aug_params[0], meta_data, 512, 334,
                                                                          view='event')
            meta_data['R_rgb'] = tf_cw_rgb[:3, :3]
            meta_data['t_rgb'] = tf_cw_rgb[:3, 3]
            meta_data['R_event'] = tf_cw_ev[:3, :3]
            meta_data['t_event'] = tf_cw_ev[:3, 3]

            rgb = cv2.warpAffine(np.array(rgb), aff_2d_rgb, (334, 512), flags=cv2.INTER_LINEAR)
            # self.plotshow(rgb / 255.)
            rgb = self.rgb_augment(torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.).permute(1, 2, 0)
            # self.plotshow(rgb)

            for i in range(len(ev_frames)):
                ev_frames[i] = cv2.warpAffine(np.array(ev_frames[i]), aff_2d_ev, (346, 260), flags=cv2.INTER_LINEAR)
                # self.plotshow(ev_frames[i])
                ev_frames[i] = self.event_augment(torch.tensor(ev_frames[i], dtype=torch.float32).permute(2, 0, 1)).permute(1, 2, 0)
                # self.plotshow(ev_frames[i])

            # rgb = self.render_hand(
            #     meta_data['mano'],
            #     meta_data['K_rgb'],
            #     meta_data['R_rgb'],
            #     meta_data['t_rgb'],
            #     hw=[512, 334],
            #     img_bg=rgb,
            # )[0]
            # ev_frames[-1] = self.render_hand(
            #     meta_data['mano'],
            #     meta_data['K_event'],
            #     meta_data['R_event'],
            #     meta_data['t_event'],
            #     hw=[260, 346],
            #     img_bg=ev_frames[-1],
            # )[0]

            bbox_rgb = self.get_bbox_by_interpolation(str(cap_id), self.ges_list[ges_index], t_target, meta_data['K_rgb'],
                                                      meta_data['R_rgb'], meta_data['t_rgb'])
            bbox_evs = []
            for segment in range(segments):
                t_r = t_target - T_ + (segment + 1) / segments * T_
                bbox_ev = self.get_bbox_by_interpolation(str(cap_id), self.ges_list[ges_index], t_r, meta_data['K_event'],
                                                         meta_data['R_event'], meta_data['t_event'])
                bbox_evs.append(bbox_ev)

            if not self.valid_bbox(bbox_rgb, hw=[512, 334]):
                bbox_rgb = self.get_default_bbox(hw=[512, 334], size=self.config['exper']['bbox']['rgb']['size'])
                bbox_valid = False
                # self.plotshow(rgb)
                # print('annot id', annot_id)

            for i, bbox_ev in enumerate(bbox_evs):
                if not self.valid_bbox(bbox_ev, hw=[260, 346]):
                    bbox_evs[i] = self.get_default_bbox(hw=[260, 346], size=self.config['exper']['bbox']['event']['size'])
                    bbox_valid = False
                    # self.plotshow(ev_frames[i])
                    # print('annot id', annot_id)

            lt_rgb, sc_rgb, rgb_crop = self.crop(bbox_rgb, np.array(rgb), self.config['exper']['bbox']['rgb']['size'],
                                                 hw=[512, 334])
            rgb_crop = torch.tensor(rgb_crop, dtype=torch.float32)
            # self.plotshow(rgb_crop)
            rgb_crop = self.normalize_img(rgb_crop.permute(2, 0, 1)).permute(1, 2, 0)
            # self.plotshow(rgb_crop)

            ev_frames_crop = []
            lt_evs, sc_evs = [], []
            for i, ev_frame in enumerate(ev_frames):
                lt_ev, sc_ev, ev_frame_crop = self.crop(bbox_evs[i], np.array(ev_frame),
                                                        self.config['exper']['bbox']['event']['size'],
                                                        hw=[260, 346])
                lt_evs.append(lt_ev)
                sc_evs.append(sc_ev)
                ev_frames_crop.append(torch.tensor(ev_frame_crop, dtype=torch.float32))

            tf_w_c = self.change_camera_view(meta_data)

            # self.plotshow(rgb_crop)
            # self.plotshow(ev_frames_crop[-1])

            meta_data.update({
                'lt_rgb': lt_rgb,
                'sc_rgb': sc_rgb,
                'lt_evs': lt_evs,
                'sc_evs': sc_evs,
            })

            frames = {
                'rgb': rgb_crop,
                'ev_frames': ev_frames_crop,
            }
            if self.config['exper']['run_eval_only']:
                frames.update(
                    {
                        'rgb_ori': rgb,
                        'ev_frames_ori': [torch.tensor(frame, dtype=torch.float32) for frame in ev_frames],
                    }
                )
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
            frames_output.append(frames)
            meta_data_output.append(meta_data)
            img_id = str(int(img_id) + 3)
        supervision_type = 0
        # if not self.config['exper']['run_eval_only']:
        #     if int(cap_id) not in self.data_config['super_ids']:
        #         if self.config['exper']['use_gt']:
        #             supervision_type = 1
        #         else:
        #             supervision_type = 2
        for i in range(len(meta_data_output)):
            meta_data_output[i]['bbox_valid'] = bbox_valid
            meta_data_output[i]['supervision_type'] = supervision_type
        return frames_output, meta_data_output

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