import os
import warnings
from argparse import ArgumentParser

import mmcv
# from xtcocotools.coco import COCO

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo
from mmpose.core import keypoints_from_heatmaps
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import json
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import copy

class SegAndKptModel(nn.Module):
    def __init__(self, load_pretrain_cpt=True, device="cpu"):
        super(SegAndKptModel,self).__init__()
        
        config_file = "/userhome/wangbingxuan/code/config.py"
        ckpt_file = "/userhome/wangbingxuan/code/res50_interhand2d_256x256_human-77b27d1a_20201029.pth" if load_pretrain_cpt else None
        pose_model = init_pose_model(config_file, ckpt_file, device='cpu')
        seg_model = models.segmentation.deeplabv3_resnet50(
                    pretrained=False,
                    num_classes=2
        )
        if load_pretrain_cpt:
            seg_state_dict = torch.load("/userhome/wangbingxuan/code/hands/checkpoint/checkpoint.ckpt")
            new_dict = {k.replace("deeplab.",""):seg_state_dict["state_dict"][k] for k in seg_state_dict["state_dict"].keys()}
            seg_model.load_state_dict(new_dict)
            pose_state_dict = torch.load("/userhome/wangbingxuan/code/res50_interhand2d_256x256_human-77b27d1a_20201029.pth")
            
        self.pose_backbone = pose_model.backbone
        self.pose_head = pose_model.keypoint_head
        self.event_pose_head = copy.deepcopy(pose_model.keypoint_head)
        self.seg_backbone = seg_model.backbone
        self.seg_head = seg_model.classifier
        self.event_seg_head = copy.deepcopy(seg_model.classifier)

    def init_dict(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            self.load_state_dict(ckpt, strict=True)
            print('load ckpt from {}'.format(ckpt_path))
        else:
            print('no ckpt is loaded')
  


    def forward(self, x):
        x = x.permute(0,3,1,2)
        input_shape = x.shape[-2:]
        pose_feat = self.pose_backbone(x)
        kpt_out = self.pose_head(pose_feat)

        seg_feature = self.seg_backbone(x)["out"]
        seg_out = self.seg_head(seg_feature)
        seg_out = F.interpolate(seg_out, size=input_shape, mode="bilinear", align_corners=False)
        return  kpt_out, seg_out