"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""


import os
import os.path as op
import numpy as np
import base64
import cv2
import yaml
from collections import OrderedDict
import json
from dv import AedatFile
from dv import LegacyAedatFile


def img_from_base64(imagestring):
    try:
        jpgbytestring = base64.b64decode(imagestring)
        nparr = np.frombuffer(jpgbytestring, np.uint8)
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return r
    except:
        return None


def load_labelmap(labelmap_file):
    label_dict = None
    if labelmap_file is not None and op.isfile(labelmap_file):
        label_dict = OrderedDict()
        with open(labelmap_file, 'r') as fp:
            for line in fp:
                label = line.strip().split('\t')[0]
                if label in label_dict:
                    raise ValueError("Duplicate label " + label + " in labelmap.")
                else:
                    label_dict[label] = len(label_dict)
    return label_dict


def load_shuffle_file(shuf_file):
    shuf_list = None
    if shuf_file is not None:
        with open(shuf_file, 'r') as fp:
            shuf_list = []
            for i in fp:
                shuf_list.append(int(i.strip()))
    return shuf_list


def load_box_shuffle_file(shuf_file):
    if shuf_file is not None:
        with open(shuf_file, 'r') as fp:
            img_shuf_list = []
            box_shuf_list = []
            for i in fp:
                idx = [int(_) for _ in i.strip().split('\t')]
                img_shuf_list.append(idx[0])
                box_shuf_list.append(idx[1])
        return [img_shuf_list, box_shuf_list]
    return None


def load_from_yaml_file(file_name):
    with open(file_name, 'r') as fp:
        return yaml.load(fp, Loader=yaml.CLoader)


def mkdir(path):
    if os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    else:
        raise FileNotFoundError('{} is not a legal direction.'.format(path))


def json_read(file_path):
    with open(os.path.abspath(file_path)) as f:
        data = json.load(f)
        return data
    raise ValueError("Unable to read json file: {}".format(file_path))


def json_write(file_path, data):
    directory = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    try:
        with open(os.path.abspath(file_path), 'w') as f:
            json.dump(data, f)
    except Exception:
        raise ValueError("Unable to write json file: {}".format(file_path))

def extract_data_from_aedat2(path: str):
    '''
    extract events from aedat2 data
    :param path:
    :return:
    '''
    with LegacyAedatFile(path) as f:
        events = []
        for event in f:
            events.append(np.array([[event.x, event.y, event.polarity, event.timestamp]], dtype=np.float32))
        if not events:
            return None
        else:
            events = np.vstack(events)
            return events
    return FileNotFoundError('Path {} is unavailable'.format(path))


def extract_data_from_aedat4(path: str, is_event: bool = True, is_aps: bool=False, is_trigger: bool=False):
    '''
    :param path: path to aedat4 file
    :return: events numpy array, aps numpy array
        event:
        # Access information of all events by type
        timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
        is_aps: list of frames
        is_trigger: list of triggers
        # EXTERNAL_INPUT_RISING_EDGE->2, EXTERNAL_INPUT1_RISING_EDGE->6, EXTERNAL_INPUT2_RISING_EDGE->9
        # EXTERNAL_INPUT1_PULSE->8, TIMESTAMP_RESET->1, TIMESTAMP_WRAP->0, EXTERNAL_INPUT1_FALLING_EDGE->7
    '''
    with AedatFile(path) as f:
        events, frames, triggers = None, [], [[], [], [], [], [], [], []]
        id2index = {2: 0, 6: 1, 9: 2, 8: 3, 1: 4, 0: 5, 7: 6}
        if is_event:
            events = np.hstack([packet for packet in f['events'].numpy()])
        if is_aps:
            for frame in f['frames']:
                frames.append(frame)
        if is_trigger:
            for i in f['triggers']:
                if i.type in id2index.keys():
                    triggers[id2index[i.type]].append(i)
                else:
                    print("{} at {} us is the new trigger type in this aedat4 file".format(i.type, i.timestamp))
        return events, frames, triggers
    return FileNotFoundError('Path {} is unavailable'.format(path))

def undistortion_points(xy, K_old, dist, K_new=None, set_bound=False, width=346, height=240):
    '''
    :param xy: N*2 array of event coordinates
    :param K_old: camera intrinsics
    :param dist: distortion coefficients
        such as
        mtx = np.array(
            [[252.91294004, 0, 129.63181808],
            [0, 253.08270535, 89.72598511],
            [0, 0, 1.]])
        dist = np.array(
            [-3.30783118e+01,  3.40196626e+02, -3.19491618e-04, -6.28058571e-04,
            1.67319020e+02, -3.27436981e+01,  3.29048638e+02,  2.85123812e+02,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            0.00000000e+00,  0.00000000e+00])
    :param K_new: new K for camera intrinsics
    :param set_bound: if true, set the undistorted points bounds
    :return: undistorted points
    '''
    # this function only outputs the normalized point coordinated, so we need apply a projection matrix K
    assert (xy.shape[1] == 2)
    xy = xy.astype(np.float32)
    if K_new is None:
        K_new = K_old
    und = cv2.undistortPoints(src=xy, cameraMatrix=K_old, distCoeffs=dist, P=K_new)
    und = und.reshape(-1, 2)
    und = und[:, :2]
    legal_indices = (und[:, 0] >= 0) * (und[:, 0] <= width-1) * (und[:, 1] >= 0) * (und[:, 1] <= width-1)
    if set_bound:
        und[:, 0] = np.clip(und[:, 0], 0, width-1)
        und[:, 1] = np.clip(und[:, 1], 0, height-1)
    return und, legal_indices