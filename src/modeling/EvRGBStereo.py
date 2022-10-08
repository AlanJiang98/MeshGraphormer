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
from .transformer import build_transformer
from .position_encoding import build_position_encoding
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
        with torch.no_grad():
            ev_backbone.conv1.weight[:, :2] = backbone.conv1.weight[:64, :2] * 1.5
        # torch.nn.init.kaiming_normal_(
        #     ev_backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.grid_feat_dim = torch.nn.Linear(1024, 2051)
        return backbone, ev_backbone

    def create_trans_encoder(self):
        if self.config['model']['method']['framework'] == 'encoder_based':
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
        else:
            if self.config['model']['tfm']['scale'] == 'S':
                num_enc_layers = 1
                num_dec_layers = 1
            elif self.config['model']['tfm']['scale'] == 'M':
                num_enc_layers = 2
                num_dec_layers = 2
            elif self.config['model']['tfm']['scale'] == 'L':
                num_enc_layers = 3
                num_dec_layers = 3
            self.hws = []
            if self.config['model']['method']['ere_usage'][0]:
                self.hws.append([self.config['exper']['bbox']['event']['size'] / 32, self.config['exper']['bbox']['event']['size'] / 32])
            if self.config['model']['method']['ere_usage'][1]:
                self.hws.append([self.config['exper']['bbox']['rgb']['size'] / 32, self.config['exper']['bbox']['rgb']['size'] / 32])
            if self.config['model']['method']['ere_usage'][2]:
                self.hws.append([self.config['exper']['bbox']['event']['size'] / 32, self.config['exper']['bbox']['event']['size'] / 32])
            # configurations for the first transformer
            transformer_config_1 = {"model_dim": self.config['model']['tfm']['model_dim'][0], "dropout": self.config['model']['tfm']['drop_out'],
                                         "nhead": self.config['model']['tfm']['nhead'],
                                         "feedforward_dim": self.config['model']['tfm']['feedforward_dim'][0], "num_enc_layers": num_enc_layers,
                                         "num_dec_layers": num_dec_layers,
                                         "pos_type": self.config['model']['tfm']['pos_type'],
                                         "input_shapes": self.hws,
                                         "att": self.config['model']['tfm']['output_attentions'],
                                        "decoder_features": self.config['model']['tfm']['decoder_features']
                                        }
            # configurations for the second transformer
            transformer_config_2 = {"model_dim": self.config['model']['tfm']['model_dim'][1], "dropout": self.config['model']['tfm']['drop_out'],
                                         "nhead": self.config['model']['tfm']['nhead'],
                                         "feedforward_dim": self.config['model']['tfm']['feedforward_dim'][1], "num_enc_layers": num_enc_layers,
                                         "num_dec_layers": num_dec_layers,
                                         "pos_type": self.config['model']['tfm']['pos_type'],
                                    "input_shapes": self.hws,
                                    "att": self.config['model']['tfm']['output_attentions'],
                                    "decoder_features": self.config['model']['tfm']['decoder_features']
                                    }
            # build transformers
            self.transformer_1 = build_transformer(transformer_config_1)
            self.transformer_2 = build_transformer(transformer_config_2)
            # dimensionality reduction
            self.dim_reduce_enc = torch.nn.Linear(transformer_config_1["model_dim"],
                                            transformer_config_2["model_dim"])
            self.dim_reduce_dec_l = torch.nn.Linear(transformer_config_1["model_dim"],
                                            transformer_config_2["model_dim"])
            self.dim_reduce_dec_r = torch.nn.Linear(transformer_config_1["model_dim"],
                                            transformer_config_2["model_dim"])
            # token embeddings
            self.joint_token_embed_l = torch.nn.Embedding(21, transformer_config_1["model_dim"])
            self.vertex_token_embed_l = torch.nn.Embedding(195, transformer_config_1["model_dim"])
            self.joint_token_embed_r = torch.nn.Embedding(21, transformer_config_1["model_dim"])
            self.vertex_token_embed_r = torch.nn.Embedding(195, transformer_config_1["model_dim"])
            # positional encodings
            self.position_encoding_1 = build_position_encoding(pos_type=transformer_config_1['pos_type'],
                                                               hidden_dim=transformer_config_1['model_dim'])
            self.position_encoding_2 = build_position_encoding(pos_type=transformer_config_2['pos_type'],
                                                               hidden_dim=transformer_config_2['model_dim'])
            # estimators
            self.xyz_regressor_l = torch.nn.Linear(transformer_config_2["model_dim"], 3)
            self.xyz_regressor_r = torch.nn.Linear(transformer_config_2["model_dim"], 3)

            # 1x1 Convolution
            self.conv_1x1_ev = torch.nn.Conv2d(1024, transformer_config_1["model_dim"], kernel_size=1)
            self.conv_1x1_rgb = torch.nn.Conv2d(1024, transformer_config_1["model_dim"], kernel_size=1)

            # attention mask
            zeros_1 = torch.tensor(np.zeros((195, 21)).astype(bool))
            zeros_2 = torch.tensor(np.zeros((21, (21 + 195))).astype(bool))
            adjacency_indices = torch.load('./src/modeling/data/mano_195_adjmat_indices.pt')
            adjacency_matrix_value = torch.load('./src/modeling/data/mano_195_adjmat_values.pt')
            adjacency_matrix_size = torch.load('./src/modeling/data/mano_195_adjmat_size.pt')
            adjacency_matrix = torch.sparse_coo_tensor(adjacency_indices, adjacency_matrix_value,
                                                       size=adjacency_matrix_size).to_dense()
            temp_mask_1 = (adjacency_matrix == 0)
            temp_mask_2 = torch.cat([zeros_1, temp_mask_1], dim=1)
            self.attention_mask = torch.cat([zeros_2, temp_mask_2], dim=0)

    def forward(self, rgb, l_ev_frame, r_ev_frame, meta_data):
        batch_size = rgb.size(0)
        device = rgb.device
        output = {}
        if self.config['model']['method']['framework'] == 'encoder_based':
            # Generate T-pose template mesh
            template_rot_pose = torch.zeros((1, 3), device=device)
            template_hand_pose = torch.zeros((1, 45), device=device)
            template_betas = torch.zeros((1, 10), device=device)
            template_transl = torch.zeros((1, 3), device=device)
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
            output['pred_3d_joints_l'] = pred_3d_joints
            output['pred_vertices_sub_l'] = pred_vertices_sub
            output['pred_vertices_l'] = pred_vertices
            if self.config['model']['tfm']['output_attentions']:
                output['att'] = att

        elif self.config['model']['method']['framework'] == 'encoder_decoder_based':
            # preparation
            jv_tokens_l = torch.cat([self.joint_token_embed_l.weight, self.vertex_token_embed_l.weight], dim=0).unsqueeze(
                1).repeat(1, batch_size, 1)
            jv_tokens_r = torch.cat([self.joint_token_embed_r.weight, self.vertex_token_embed_r.weight], dim=0).unsqueeze(
                1).repeat(1, batch_size, 1)
            attention_mask = self.attention_mask.to(device)

            grid_feat_list = []
            hws = []

            if self.config['model']['method']['ere_usage'][0]:
                image_feat_l_ev, grid_feat_l_ev = self.ev_backbone(l_ev_frame[:, :2])
                _, _, h, w= grid_feat_l_ev.shape
                hws.append([h, w])
                grid_feat_l_ev = self.conv_1x1_ev(grid_feat_l_ev).flatten(2).permute(2, 0, 1)
                grid_feat_list.append(grid_feat_l_ev)

            if self.config['model']['method']['ere_usage'][1]:
                # extract grid features and global image features using a CNN backbone
                image_feat_rgb, grid_feat_rgb = self.rgb_backbone(rgb)
                _, _, h, w= grid_feat_rgb.shape
                hws.append([h, w])
                grid_feat_rgb = self.conv_1x1_rgb(grid_feat_rgb).flatten(2).permute(2, 0, 1)
                grid_feat_list.append(grid_feat_rgb)

            if self.config['model']['method']['ere_usage'][2]:
                image_feat_r_ev, grid_feat_r_ev = self.ev_backbone(r_ev_frame[:, :2])
                _, _, h, w= grid_feat_r_ev.shape
                hws.append([h, w])
                grid_feat_r_ev = self.conv_1x1_ev(grid_feat_r_ev).flatten(2).permute(2, 0, 1)
                grid_feat_list.append(grid_feat_r_ev)
            img_features = torch.cat(grid_feat_list, dim=0)

            # positional encodings
            pos_enc_1 = self.position_encoding_1(batch_size, hws, device).flatten(2).permute(2, 0, 1)
            pos_enc_2 = self.position_encoding_2(batch_size, hws, device).flatten(2).permute(2, 0, 1)

            # first transformer encoder-decoder
            output_1 = self.transformer_1(img_features, [jv_tokens_l, jv_tokens_r], pos_enc_1, attention_mask=attention_mask)

            # progressive dimensionality reduction
            reduced_enc_img_features_1 = self.dim_reduce_enc(output_1['e_outputs'])
            reduced_jv_features_1_l = self.dim_reduce_dec_l(output_1['jv_features_l'])
            reduced_jv_features = [reduced_jv_features_1_l]
            if len(self.hws) == 3:
                reduced_jv_features_1_r = self.dim_reduce_dec_r(output_1['jv_features_r'])
                reduced_jv_features += [reduced_jv_features_1_r]
            # second transformer encoder-decoder
            output_2 = self.transformer_2(reduced_enc_img_features_1,
                                                                  reduced_jv_features, pos_enc_2,
                                                                  attention_mask=attention_mask)

            # estimators
            pred_3d_coordinates_l = self.xyz_regressor_l(output_2['jv_features_l'].transpose(0, 1))
            pred_3d_joints_l = pred_3d_coordinates_l[:, :21, :]
            pred_vertices_sub_l = pred_3d_coordinates_l[:, 21:, :]
            pred_vertices_l = self.upsampling(pred_vertices_sub_l.transpose(1,2))
            pred_vertices_l = pred_vertices_l.transpose(1, 2)
            output['pred_3d_joints_l'] = pred_3d_joints_l
            output['pred_vertices_sub_l'] = pred_vertices_sub_l
            output['pred_vertices_l'] = pred_vertices_l
            if len(self.hws) == 3:
                pred_3d_coordinates_r = self.xyz_regressor_r(output_2['jv_features_r'].transpose(0, 1))
                pred_3d_joints_r = pred_3d_coordinates_r[:, :21, :]
                pred_vertices_sub_r = pred_3d_coordinates_r[:, 21:, :]
                pred_vertices_r = self.upsampling(pred_vertices_sub_r.transpose(1, 2))
                pred_vertices_r = pred_vertices_r.transpose(1, 2)
                output['pred_3d_joints_r'] = pred_3d_joints_r
                output['pred_vertices_sub_r'] = pred_vertices_sub_r
                output['pred_vertices_r'] = pred_vertices_r
            if self.config['model']['tfm']['output_attentions']:
                output['att'] = [output_1['d_atts_l'], output_2['d_atts_l']]
                if len(self.hws) == 3:
                    output['att'] += [output_1['d_atts_r'], output_2['d_atts_r']]
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