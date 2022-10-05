from copy import deepcopy
import numpy as np
import cv2
import torch
import torchvision.models as models
from src.modeling.bert import BertConfig, Graphormer
import src.modeling.data.config as cfg
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat
# todo
from src.modeling._mano import MANO
from src.modeling._mano import Mesh as MeshSampler
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


class EvRGBStereo(torch.nn.Module):
    def __init__(self, config):
        super(EvRGBStereo, self).__init__()
        self.config = config
        self.rgb_backbone, self.ev_backbone = self.create_backbone()
        self.trans_encoder = self.create_trans_encoder()
        self.mano_layer = MANO(self.config['data']['smplx_path'], use_pca=False, is_rhand=True)
        self.mesh_sampler = MeshSampler()
        self.upsampling = torch.nn.Linear(195, 778)
        self.grid_feat_dim = torch.nn.Linear(1024, 2051)

    def create_backbone(self):
        # create backbone model
        if self.config['model']['backbone']['arch'] == 'hrnet':
            hrnet_update_config(hrnet_config, self.config['model']['backbone']['hrnet_yaml'])
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=self.config['model']['backbone']['hrnet_bb'])
            # logger.info('=> loading hrnet model')
        else:
            print("=> using pre-trained model '{}'".format(self.config['model']['backbone']['arch']))
            backbone = models.__dict__[self.config['model']['backbone']['arch']](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        print('RGB Backbone total parameters: {}'.format(backbone_total_params))
        ev_backbone = deepcopy(backbone)
        ev_backbone.conv1 = torch.nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        torch.nn.init.kaiming_normal_(
            ev_backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        return backbone, ev_backbone

    def create_trans_encoder(self):
        trans_encoder = []
        input_feat_dim = self.config['model']['tfm']['input_feat_dim']
        hidden_feat_dim = self.config['model']['tfm']['hidden_feat_dim']
        output_feat_dim = input_feat_dim[1:] + [3]
        # which encoder block to have graph convs
        which_blk_graph = self.config['model']['tfm']['which_gcn']

        # init three transformer-encoder blocks in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, Graphormer
            b_config = config_class.from_pretrained(self.config['model']['bert_config'])

            b_config.output_attentions = self.config['model']['tfm']['output_attentions']
            b_config.hidden_dropout_prob = self.config['model']['tfm']['drop_out']
            b_config.img_feature_dim = input_feat_dim[i]
            b_config.output_feature_dim = output_feat_dim[i]
            b_config.output_hidden_states = self.config['model']['tfm']['output_attentions']

            if which_blk_graph[i] == 1:
                b_config.graph_conv = True
                # logger.info("Add Graph Conv")
            else:
                b_config.graph_conv = False

            b_config.mesh_type = self.config['model']['tfm']['mesh_type']
            b_config.num_hidden_layers = self.config['model']['tfm']['num_hidden_layers']
            b_config.num_attention_heads = self.config['model']['tfm']['num_attention_heads']
            b_config.hidden_size = hidden_feat_dim[i]
            b_config.intermediate_size = int(hidden_feat_dim[i] * 2)

            # init a transformer encoder and append it to a list
            assert b_config.hidden_size % b_config.num_attention_heads == 0
            model = model_class(config=b_config)
            # logger.info("Init model from scratch.")
            trans_encoder.append(model)

        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        print('trans encoders total parameters: {}'.format(total_params))
        return trans_encoder

    def forward(self, rgb, l_ev_frame, r_ev_frame, meta_data):
        batch_size = rgb.size(0)
        output = {}
        if self.config['model']['method']['framework'] == 'encoder_based':
            # Generate T-pose template mesh
            template_rot_pose = torch.zeros((1, 3)).cuda()
            template_hand_pose = torch.zeros((1, 45)).cuda()
            template_betas = torch.zeros((1, 10)).cuda()
            template_transl = torch.zeros((1, 3)).cuda()
            template_output = self.mano_layer(
                global_orient=template_rot_pose,
                hand_pose=template_hand_pose,
                betas=template_betas,
                transl=template_transl,
            )
            template_vertices = template_output.vertices
            template_3d_joints = template_output.joints

            template_vertices_sub = self.mesh_sampler.downsample(template_vertices)
            # normalize
            template_root = template_3d_joints[:, 0, :]
            template_3d_joints = template_3d_joints - template_root[:, None, :]
            template_vertices = template_vertices - template_root[:, None, :]
            template_vertices_sub = template_vertices_sub - template_root[:, None, :]
            num_joints = template_3d_joints.shape[1]
            num_vertices = template_vertices_sub.shape[1]

            # concatinate template joints and template vertices, and then duplicate to batch size
            ref_vertices = torch.cat([template_3d_joints, template_vertices_sub], dim=1)
            ref_vertices = ref_vertices.expand(batch_size, -1, -1)

            image_feat_list, grid_feat_list = [], []

            if self.config['model']['method']['ere_usage'][0]:
                image_feat_l_ev, grid_feat_l_ev = self.ev_backbone(l_ev_frame[:, :2])
                image_feat_l_ev = image_feat_l_ev.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
                grid_feat_l_ev = torch.flatten(grid_feat_l_ev, start_dim=2)
                grid_feat_l_ev = grid_feat_l_ev.transpose(1, 2)
                image_feat_list.append(image_feat_l_ev)
                grid_feat_list.append(grid_feat_l_ev)

            if self.config['model']['method']['ere_usage'][1]:
                # extract grid features and global image features using a CNN backbone
                image_feat_rgb, grid_feat_rgb = self.rgb_backbone(rgb)
                # concatinate image feat and mesh template
                image_feat_rgb = image_feat_rgb.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
                # process grid features
                grid_feat_rgb = torch.flatten(grid_feat_rgb, start_dim=2)
                grid_feat_rgb = grid_feat_rgb.transpose(1, 2)
                image_feat_list.append(image_feat_rgb)
                grid_feat_list.append(grid_feat_rgb)

            if self.config['model']['method']['ere_usage'][2]:
                image_feat_r_ev, grid_feat_r_ev = self.ev_backbone(r_ev_frame[:, :2])
                image_feat_r_ev = image_feat_r_ev.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
                grid_feat_r_ev = torch.flatten(grid_feat_r_ev, start_dim=2)
                grid_feat_r_ev = grid_feat_r_ev.transpose(1, 2)
                image_feat_list.append(image_feat_r_ev)
                grid_feat_list.append(grid_feat_r_ev)

            grid_feat = torch.cat(grid_feat_list, dim=1)
            image_feat = sum(image_feat_list) / len(image_feat_list)

            grid_feat = self.grid_feat_dim(grid_feat)
            # concatinate image feat and template mesh to form the joint/vertex queries
            features = torch.cat([ref_vertices, image_feat], dim=2)
            # prepare input tokens including joint/vertex queries and grid features
            features = torch.cat([features, grid_feat], dim=1)

            # forward pass
            att = []
            if self.config['model']['tfm']['output_attentions']:
                for sub_model in self.trans_encoder:
                    features, hidden_states, att_ = sub_model(features)
                    att.append(att_)
            else:
                features = self.trans_encoder(features)

            pred_3d_joints = features[:, :num_joints, :]
            pred_vertices_sub = features[:, num_joints:num_joints+num_vertices, :]

            temp_transpose = pred_vertices_sub.transpose(1, 2)
            pred_vertices = self.upsampling(temp_transpose)
            pred_vertices = pred_vertices.transpose(1, 2)
            output['pred_3d_joints'] = pred_3d_joints
            output['pred_vertices_sub'] = pred_vertices_sub
            output['pred_vertices'] = pred_vertices
            if self.config['model']['tfm']['output_attentions']:
                output['hidden_states'] = hidden_states
                output['att'] = att

        elif self.config['model']['method']['framework'] == 'encoder_decoder_based':
            pass
        return output

    # def forward(self, images, mesh_model, mesh_sampler, meta_masks=None, is_train=False):
    #     batch_size = images.size(0)
    #     # Generate T-pose template mesh
    #     template_pose = torch.zeros((1,48))
    #     template_pose = template_pose.cuda()
    #     template_betas = torch.zeros((1,10)).cuda()
    #     template_vertices, template_3d_joints = mesh_model.layer(template_pose, template_betas)
    #     template_vertices = template_vertices/1000.0
    #     template_3d_joints = template_3d_joints/1000.0
    #
    #     template_vertices_sub = mesh_sampler.downsample(template_vertices)
    #
    #     # normalize
    #     template_root = template_3d_joints[:,cfg.J_NAME.index('Wrist'),:]
    #     template_3d_joints = template_3d_joints - template_root[:, None, :]
    #     template_vertices = template_vertices - template_root[:, None, :]
    #     template_vertices_sub = template_vertices_sub - template_root[:, None, :]
    #     num_joints = template_3d_joints.shape[1]
    #
    #     # concatinate template joints and template vertices, and then duplicate to batch size
    #     ref_vertices = torch.cat([template_3d_joints, template_vertices_sub],dim=1)
    #     ref_vertices = ref_vertices.expand(batch_size, -1, -1)
    #
    #     # extract grid features and global image features using a CNN backbone
    #     image_feat, grid_feat = self.backbone(images)
    #     # concatinate image feat and mesh template
    #     image_feat = image_feat.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
    #     # process grid features
    #     grid_feat = torch.flatten(grid_feat, start_dim=2)
    #     grid_feat = grid_feat.transpose(1,2)
    #     grid_feat = self.grid_feat_dim(grid_feat)
    #     # concatinate image feat and template mesh to form the joint/vertex queries
    #     features = torch.cat([ref_vertices, image_feat], dim=2)
    #     # prepare input tokens including joint/vertex queries and grid features
    #     features = torch.cat([features, grid_feat],dim=1)
    #
    #     if is_train==True:
    #         # apply mask vertex/joint modeling
    #         # meta_masks is a tensor of all the masks, randomly generated in dataloader
    #         # we pre-define a [MASK] token, which is a floating-value vector with 0.01s
    #         special_token = torch.ones_like(features[:,:-49,:]).cuda()*0.01
    #         features[:,:-49,:] = features[:,:-49,:]*meta_masks + special_token*(1-meta_masks)
    #
    #     # forward pass
    #     if self.config['model']['tfm']['output_attentions']==True:
    #         features, hidden_states, att = self.trans_encoder(features)
    #     else:
    #         features = self.trans_encoder(features)
    #
    #     pred_3d_joints = features[:,:num_joints,:]
    #     pred_vertices_sub = features[:,num_joints:-49,:]
    #
    #     # learn camera parameters
    #     x = self.cam_param_fc(features[:,:-49,:])
    #     x = x.transpose(1,2)
    #     x = self.cam_param_fc2(x)
    #     x = self.cam_param_fc3(x)
    #     cam_param = x.transpose(1,2)
    #     cam_param = cam_param.squeeze()
    #
    #     temp_transpose = pred_vertices_sub.transpose(1,2)
    #     pred_vertices = self.upsampling(temp_transpose)
    #     pred_vertices = pred_vertices.transpose(1,2)
    #
    #     if self.config['model']['tfm']['output_attentions']==True:
    #         return cam_param, pred_3d_joints, pred_vertices_sub, pred_vertices, hidden_states, att
    #     else:
    #         return cam_param, pred_3d_joints, pred_vertices_sub, pred_vertices