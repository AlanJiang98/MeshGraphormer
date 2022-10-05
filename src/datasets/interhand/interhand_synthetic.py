import numpy as np
import os
import cv2
import argparse
import subprocess
from src.utils.dataset_utils import extract_data_from_aedat2


parser = argparse.ArgumentParser('Synthetic Interhand')
parser.add_argument('--gpu', type=str,
                    default='0')
parser.add_argument('--cap_ids', default='0,1', type=str)
args = parser.parse_args()

# input data dir from interhand2.6M
data_path = '/userhome/alanjjp/data/Interhand'
# output data dir
output_dir = '/userhome/alanjjp/data/Interhand/synthetic'

# davis settings
davis_width = 346
davis_height = 260

# the factor from RGB image to davis event frames when preserving aspect ratio
factor = 1

Captures_list = [
    '0', '1', '2', '3', '5', '6', '7', '8', '9',
]

# for gestures, '00' means right single hand; '01' means left single hand
# some sequences has little data

# for cameras, '40' means rgb , 334*512; '41' means gray data 512 * 334
RGB_Cameras_list = [
    '400002', '400004', '400009', '400012', '400018', '400026',
    '400039', '400053', '400041',
]
# GRAY_Cameras_list = [
#     '410001', '410003', '410004', '410014',
#     '410019', '410027', '410053', '410214', '410231', '410237'
# ]

# interhand image settings
img_ori_height, img_ori_width = 512, 334

# get the affine matrix from RGB frame to event frame, and this affine matrix will influence the camera intrinsics
src_points = np.float32([[0, 0], [img_ori_width / 2 - 1, 0], [img_ori_width / 2 - 1, img_ori_height * factor]])
dst_points = np.float32([[davis_width / 2 - img_ori_width / 2 * davis_height / factor / img_ori_height, 0],
                         [davis_width / 2 - 1, 0],
                         [davis_width / 2 - 1, davis_height - 1]])
affine_matrix = cv2.getAffineTransform(src_points, dst_points)

tmp_Captures_list = [id for id in args.cap_ids.split(',')]

for cap_id in Captures_list:
    if cap_id not in tmp_Captures_list:
        continue
    gestures_abspath, gestures_path = [], []
    gestures_tmp = os.listdir(os.path.join(data_path, 'train', 'Capture'+cap_id))
    gestures_tmp.sort()
    # select right hand data
    for ges in gestures_tmp:
        if ges.startswith('00'):
            gestures_path.append(ges)
    for gesture in gestures_path:
        gestures_abspath.append(os.path.join(data_path, 'train', 'Capture'+cap_id, gesture))

    for ges_id, ges_abs_path in enumerate(gestures_abspath):
        # if ges_id < 8:
        #     continue
        for cam_id in RGB_Cameras_list:
            # get camera abs path
            print(40*'#')
            cam_abs_path = os.path.join(ges_abs_path, 'cam'+cam_id)
            if os.path.exists(cam_abs_path):
                frames_names = os.listdir(cam_abs_path)
                frames_names.sort()
                output_dir_tmp = os.path.join(output_dir, 'train', 'Capture'+cap_id, gestures_path[ges_id], 'cam'+cam_id)

                if not os.path.exists(output_dir_tmp):
                    os.makedirs(output_dir_tmp, exist_ok=True)
                else:
                    continue
                affine_image_dir = os.path.join(output_dir_tmp, 'images')
                if not os.path.exists(affine_image_dir):
                    os.makedirs(affine_image_dir, exist_ok=True)
                for frame_name in frames_names:
                    img_origin = cv2.imread(os.path.join(cam_abs_path, frame_name), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                    img_ori_height, img_ori_width = img_origin.shape[:2]
                    # warp Interhand images to DAVIS346 APS format
                    img_affine = cv2.warpAffine(img_origin, affine_matrix, (davis_width, davis_height))
                    cv2.imwrite(os.path.join(affine_image_dir, frame_name), img_affine)
                # call the command line to process the event data by project v2e
                # TODO you can change the paramters for your usage
                cmd = 'CUDA_VISIBLE_DEVICES=' + args.gpu + ' /userhome/alanjjp/tools/miniconda3/envs/evrgb/bin/python /userhome/alanjjp/Project/v2e/v2e.py ' \
                '--overwrite --timestamp_resolution=.0033333 --auto_timestamp_resolution=True --dvs_params=None ' \
                '--pos_thres=0.143 --neg_thres=0.225 --slomo_model=/userhome/alanjjp/Project/v2e/input/SuperSloMo39.ckpt ' \
                '--batch_size=128 --input_frame_rate=30 --dvs_exposure duration 0.008 --dvs346 --no_preview --skip_video_output'\
                ' -o ' + output_dir_tmp + ' -i ' + affine_image_dir
                #
                subprocess.call(cmd, shell=True)
                # remove the rest images
                cmd_clean = 'rm -rf ' + affine_image_dir
                subprocess.call(cmd_clean, shell=True)
                event_path = os.path.join(output_dir_tmp, 'v2e-dvs-events.aedat')
                if os.path.exists(event_path):
                    events = extract_data_from_aedat2(event_path)
                    if events is None:
                        print('No events in {} and Skip!'.format(output_dir_tmp))
                        continue
                    else:
                        np.savez(os.path.join(output_dir_tmp, 'events.npz'), events=events)
                else:
                    print('No event files in {}'.format(output_dir_tmp))
                cmd_clean = 'rm -f ' + event_path
                subprocess.call(cmd_clean, shell=True)