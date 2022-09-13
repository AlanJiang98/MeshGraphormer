import cv2
import os
import os.path as osp
import numpy as np
import torch
import json
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import multiprocessing as mp
from src.utils.dataset_utils import json_read, extract_data_from_aedat4, undistortion_points



class EvRealHands(Dataset):
    def __init__(self, config):
        self.config = config
        self.data = {}
        self.data_config = None
        self.load_events_annotations()
        pass

    @staticmethod
    def get_events_annotations_per_sequence(dir, data):
        if not osp.isdir(dir):
            raise FileNotFoundError('illegal directions for event sequence: {}'.format(dir))
        id = dir.split('/')[-1]
        annot = json_read(os.path.join(dir, 'annot.json'))
        mano_ids = [int(id) for id in annot['manos'].keys()]
        mano_ids.sort()
        annot['sample_ids'] = [str(id) for id in mano_ids]
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
        data[id] = {
            'event': events,
            'annot': annot,
        }

    def load_events_annotations(self):
        if self.config['exper']['run_eval_only']:
            data_config = json_read(self.config['data']['eval_yaml'])
        else:
            data_config = json_read(self.config['data']['train_yaml'])
        self.data_config = data_config
        all_sub_2_seq_ids = json_read(
            osp.join(self.config['data']['data_dir'], data_config['seq_ids_path'])
        )
        self.seq_ids = []
        for sub_id in data_config['sub_ids']:
            self.seq_ids += all_sub_2_seq_ids[sub_id]
        pool = mp.Pool(mp.cpu_count())
        for seq_id in self.seq_ids:
            pool.apply_async(EvRealHands.get_events_annotations_per_sequence,
                             args=(osp.join(self.config['data']['data_dir'], seq_id), self.data),)
        pool.close()
        pool.join()

    def process_samples(self):
        self.sample_info = {
            'seq_ids': [],
            'cam_pair_ids': [],
            'samples_per_seq': None,
            'samples_sum': None
        }
        samples_per_seq = []
        for seq_id in self.seq_ids:
            for cam_pair_id, cam_pair in enumerate(self.data_config['camera_pairs']):
                samples_per_seq.append(len(self.data[seq_id]['annot']['sample_ids']))
                self.sample_info['seq_ids'].append(seq_id)
                self.sample_info['cam_pair_ids'].append(cam_pair_id)
        self.sample_info['samples_per_seq'] = np.array(samples_per_seq, dtype=np.int32)
        self.sample_info['samples_sum'] = np.cumsum(self.sample_info['samples_per_seq'])


    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

