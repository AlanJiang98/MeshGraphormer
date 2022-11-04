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
from .perceiver import Encoder, Decoder, build_transformer_block, PerceiverBlock
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
from src.modeling.resnet.resnet_gridfeat import resnet50gridfeat, resnet34gridfeat


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
        if self.config['model']['method']['framework'] == 'eventhands':
            res50_pretrained = models.resnet50(pretrained=False)
            res50_state_dict = torch.load(self.config['model']['backbone']['hrnet_bb'])
            res50_pretrained.load_state_dict(res50_state_dict, strict=False)
            self.eventhands_encoder_main = torch.nn.Sequential(*list(res50_pretrained.children())[:-1])
            self.eventhands_encoder_fc = torch.nn.Linear(2048, 58)
            return None, None
        if self.config['model']['backbone']['arch'] == 'hrnet':
            hrnet_update_config(hrnet_config, self.config['model']['backbone']['hrnet_yaml'])
            backbone = get_cls_net_gridfeat(hrnet_config, pretrained=self.config['model']['backbone']['hrnet_bb'])
            # logger.info('=> loading hrnet model')
        elif self.config['model']['backbone']['arch'] == 'resnet50':
            backbone = resnet50gridfeat()
            backbone.load_state_dict(torch.load(self.config['model']['backbone']['hrnet_bb']),strict=False)
            # logger.info('=> loading resnet50 model')
        elif self.config['model']['backbone']['arch'] == 'resnet34':
            backbone = resnet34gridfeat()
            backbone.load_state_dict(torch.load(self.config['model']['backbone']['hrnet_bb']),strict=False)
        else:
            print("=> using pre-trained model '{}'".format(self.config['model']['backbone']['hrnet_bb']))
            backbone = models.__dict__[self.config['model']['backbone']['arch']](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        print('RGB Backbone total parameters: {}'.format(backbone_total_params))
        ev_backbone = deepcopy(backbone)
        # todo change backbone
        # ev_backbone.conv1 = torch.nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1,
        #                        bias=False)
        # with torch.no_grad():
        #     ev_backbone.conv1.weight[:, :2] = backbone.conv1.weight[:64, :2] * 1.5
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
        elif self.config['model']['method']['framework'] == 'encoder_decoder_based':
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
        elif self.config['model']['method']['framework'] == 'perceiver':
            if self.config['model']['tfm']['scale'] == 'S':
                num_layers = 1
            elif self.config['model']['tfm']['scale'] == 'M':
                num_layers = 2
            elif self.config['model']['tfm']['scale'] == 'L':
                num_layers = 3

            hw_ev = [self.config['exper']['bbox']['event']['size'] / 32, self.config['exper']['bbox']['event']['size'] / 32]
            hw_rgb = [self.config['exper']['bbox']['rgb']['size'] / 32, self.config['exper']['bbox']['rgb']['size'] / 32]

            # configurations for the first transformer
            transformer_config_1 = {"model_dim": self.config['model']['tfm']['model_dim'][0],
                                    "dropout": self.config['model']['tfm']['drop_out'],
                                    "nhead": self.config['model']['tfm']['nhead'],
                                    "feedforward_dim": self.config['model']['tfm']['feedforward_dim'][0],
                                    "num_layers": num_layers,
                                    'n_self_att_layers': self.config['model']['tfm']['n_self_att_layers'],
                                    'pos_type': self.config['model']['tfm']['pos_type'],
                                    'perceiver_layers': self.config['model']['tfm']['perceiver_layers']
                                        }
            transformer_config_2 = {"model_dim": self.config['model']['tfm']['model_dim'][1],
                                    "dropout": self.config['model']['tfm']['drop_out'],
                                    "nhead": self.config['model']['tfm']['nhead'],
                                    "feedforward_dim": self.config['model']['tfm']['feedforward_dim'][1],
                                    "num_layers": num_layers,
                                    'n_self_att_layers': self.config['model']['tfm']['n_self_att_layers'],
                                    'pos_type': self.config['model']['tfm']['pos_type'],
                                    'perceiver_layers': self.config['model']['tfm']['perceiver_layers']
                                        }

            if 'scene_weight' in self.config['model']['tfm'].keys() and self.config['model']['tfm']['scene_weight']:
                self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.fc_scene_weight = torch.nn.Linear(1024, 2)
                self.softmax = torch.nn.Softmax(dim=-1)

            self.stereo_encoders = torch.nn.ModuleList()
            self.event_encoders = torch.nn.ModuleList()
            self.perveivers = torch.nn.ModuleList()
            self.decoders = torch.nn.ModuleList()

            for config_ in [transformer_config_1, transformer_config_2]:
                self.stereo_encoders.append(build_transformer_block('encoder', config_))
                self.event_encoders.append(build_transformer_block('encoder', config_))
                self.perveivers.append(build_transformer_block('perceiver', config_))
                self.decoders.append(build_transformer_block('decoder', config_))

            # dimensionality reduction
            self.dim_reduce_stereo = torch.nn.Linear(transformer_config_1["model_dim"],
                                            transformer_config_2["model_dim"])
            self.dim_reduce_event = torch.nn.Linear(transformer_config_1["model_dim"],
                                                  transformer_config_2["model_dim"])
            self.dim_reduce_dec = torch.nn.Linear(transformer_config_1["model_dim"],
                                            transformer_config_2["model_dim"])

            # token embeddings
            self.joint_token_embed = torch.nn.Embedding(21, transformer_config_1["model_dim"])
            self.vertex_token_embed = torch.nn.Embedding(195, transformer_config_1["model_dim"])

            # positional encodings
            self.position_encoding_1 = build_position_encoding(pos_type=transformer_config_1['pos_type'],
                                                               hidden_dim=transformer_config_1['model_dim'])
            self.position_encoding_2 = build_position_encoding(pos_type=transformer_config_2['pos_type'],
                                                               hidden_dim=transformer_config_2['model_dim'])

            # estimators
            self.xyz_regressor = torch.nn.Linear(transformer_config_2["model_dim"], 3)

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
            pass

    def forward(self, frames, return_att=True, decode_all=False):
        batch_size = frames[0]['rgb'].size(0)
        device = frames[0]['rgb'].device
        output = []
        if self.config['model']['method']['framework'] == 'eventhands':
            x = self.eventhands_encoder_main(frames[0]['ev_frames'][0].permute(0, 3, 1, 2))
            x = self.eventhands_encoder_fc(x.flatten(1))
            atts = None
            mano_output = self.mano_layer(
                global_orient=x[:, :3],
                hand_pose=x[:, 3:48],
                betas=x[:, 48:58],
                transl=torch.zeros((batch_size, 3), device=device),
            )
            pred_3d_joints = mano_output.joints - mano_output.joints[:, :1]
            pred_vertices = mano_output.vertices - mano_output.joints[:, :1]

            output.append([{
                'pred_3d_joints': pred_3d_joints,
                'pred_vertices': pred_vertices,
                'pred_rot_pose': x[:, :3],
                'pred_hand_pose': x[:, 3:48],
                'pred_shape': x[:, 48:58],
            }])
            if not self.config['exper']['run_eval_only']:
                pred_vertices_sub = self.mesh_sampler.downsample(mano_output.vertices) - mano_output.joints[:, :1]
                output[0][0].update({
                    'pred_vertices_sub': pred_vertices_sub,
                })

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
                image_feat_l_ev, grid_feat_l_ev = self.ev_backbone(frames[0]['ev_frames'][0].permute(0, 3, 1, 2))#[:, :2])
                image_feat_l_ev = image_feat_l_ev.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
                grid_feat_l_ev = torch.flatten(grid_feat_l_ev, start_dim=2)
                grid_feat_l_ev = grid_feat_l_ev.transpose(1, 2)
                image_feat_list.append(image_feat_l_ev)
                grid_feat_list.append(grid_feat_l_ev)

            if self.config['model']['method']['ere_usage'][1]:
                # extract grid features and global image features using a CNN backbone
                image_feat_rgb, grid_feat_rgb = self.rgb_backbone(frames[0]['rgb'].permute(0, 3, 1, 2))
                # concatinate image feat and mesh template
                image_feat_rgb = image_feat_rgb.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
                # process grid features
                grid_feat_rgb = torch.flatten(grid_feat_rgb, start_dim=2)
                grid_feat_rgb = grid_feat_rgb.transpose(1, 2)
                image_feat_list.append(image_feat_rgb)
                grid_feat_list.append(grid_feat_rgb)

            if self.config['model']['method']['ere_usage'][2]:
                image_feat_r_ev, grid_feat_r_ev = self.ev_backbone(frames[1]['ev_frames'][-1].permute(0, 3, 1, 2))#[:, :2])
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
            atts = ()
            if self.config['model']['tfm']['output_attentions']:
                for sub_model in self.trans_encoder:
                    features, hidden_states, att_ = sub_model(features)
                    atts += (att_,)
            else:
                features = self.trans_encoder(features)

            pred_3d_joints = features[:, :num_joints, :]
            pred_vertices_sub = features[:, num_joints:num_joints+num_vertices, :]

            temp_transpose = pred_vertices_sub.transpose(1, 2)
            pred_vertices = self.upsampling(temp_transpose)
            pred_vertices = pred_vertices.transpose(1, 2)
            # output['pred_3d_joints_l'] = pred_3d_joints
            # output['pred_vertices_sub_l'] = pred_vertices_sub
            # output['pred_vertices_l'] = pred_vertices
            output.append([{
                'pred_3d_joints': pred_3d_joints,
                'pred_vertices': pred_vertices,
                'pred_vertices_sub': pred_vertices_sub,
            }])
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
                image_feat_l_ev, grid_feat_l_ev = self.ev_backbone(frames[0]['ev_frames'][0].permute(0, 3, 1, 2))#[:, :2])
                _, _, h, w= grid_feat_l_ev.shape
                hws.append([h, w])
                grid_feat_l_ev = self.conv_1x1_ev(grid_feat_l_ev).flatten(2).permute(2, 0, 1)
                grid_feat_list.append(grid_feat_l_ev)

            if self.config['model']['method']['ere_usage'][1]:
                # extract grid features and global image features using a CNN backbone
                image_feat_rgb, grid_feat_rgb = self.rgb_backbone(frames[0]['rgb'].permute(0, 3, 1, 2))
                _, _, h, w= grid_feat_rgb.shape
                hws.append([h, w])
                grid_feat_rgb = self.conv_1x1_rgb(grid_feat_rgb).flatten(2).permute(2, 0, 1)
                grid_feat_list.append(grid_feat_rgb)

            if self.config['model']['method']['ere_usage'][2]:
                image_feat_r_ev, grid_feat_r_ev = self.ev_backbone(frames[1]['ev_frames'][-1].permute(0, 3, 1, 2))#[:, :2])
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
            output = []
            atts = ()
            output.append([{
                'pred_3d_joints': pred_3d_joints_l,
                'pred_vertices': pred_vertices_l,
                'pred_vertices_sub': pred_vertices_sub_l,
            }])
            atts += (output_1['d_atts_l'], output_2['d_atts_l'])
            # output['pred_3d_joints_l'] = pred_3d_joints_l
            # output['pred_vertices_sub_l'] = pred_vertices_sub_l
            # output['pred_vertices_l'] = pred_vertices_l
            if len(self.hws) == 3:
                pred_3d_coordinates_r = self.xyz_regressor_r(output_2['jv_features_r'].transpose(0, 1))
                pred_3d_joints_r = pred_3d_coordinates_r[:, :21, :]
                pred_vertices_sub_r = pred_3d_coordinates_r[:, 21:, :]
                pred_vertices_r = self.upsampling(pred_vertices_sub_r.transpose(1, 2))
                pred_vertices_r = pred_vertices_r.transpose(1, 2)

                output.append([{
                    'pred_3d_joints': pred_3d_joints_r,
                    'pred_vertices': pred_vertices_r,
                    'pred_vertices_sub': pred_vertices_sub_r,
                }])
                atts += (output_1['d_atts_r'], output_2['d_atts_r'])
                # output['pred_3d_joints_r'] = pred_3d_joints_r
                # output['pred_vertices_sub_r'] = pred_vertices_sub_r
                # output['pred_vertices_r'] = pred_vertices_r
            # if self.config['model']['tfm']['output_attentions']:
            #     output['att'] = [output_1['d_atts_l'], output_2['d_atts_l']]
            #     if len(self.hws) == 3:
            #         output['att'] += [output_1['d_atts_r'], output_2['d_atts_r']]
        elif self.config['model']['method']['framework'] == 'perceiver':
            jv_tokens = torch.cat([self.joint_token_embed.weight, self.vertex_token_embed.weight], dim=0).unsqueeze(1).repeat(1, batch_size, 1)
            attention_mask = self.attention_mask.to(device)
            grid_feat_list = []
            scene_weight_list = []
            hws = []

            atts = ()
            # stereo feature
            _, grid_feat_ev = self.ev_backbone(frames[0]['ev_frames'][0].permute(0, 3, 1, 2))#[:, :2])
            _, _, h, w = grid_feat_ev.shape
            if 'scene_weight' in self.config['model']['tfm'].keys() and self.config['model']['tfm']['scene_weight']:
                scene_weight_list.append(self.avgpool(grid_feat_ev).flatten(2))
            hws.append([h, w])
            grid_feat_ev = self.conv_1x1_ev(grid_feat_ev).flatten(2).permute(2, 0, 1)
            grid_feat_list.append(grid_feat_ev)

            # extract grid features and global image features using a CNN backbone
            _, grid_feat_rgb = self.rgb_backbone(frames[0]['rgb'].permute(0, 3, 1, 2))
            _, _, h, w = grid_feat_rgb.shape
            if 'scene_weight' in self.config['model']['tfm'].keys() and self.config['model']['tfm']['scene_weight']:
                scene_weight_list.append(self.avgpool(grid_feat_rgb).flatten(2))
            hws.append([h, w])
            grid_feat_rgb = self.conv_1x1_rgb(grid_feat_rgb).flatten(2).permute(2, 0, 1)
            grid_feat_list.append(grid_feat_rgb)

            stereo_features = torch.cat(grid_feat_list, dim=0)

            scene_weight = torch.ones((*stereo_features.shape[:2], 1), device=stereo_features.device)

            if 'scene_weight' in self.config['model']['tfm'].keys() and self.config['model']['tfm']['scene_weight']:
                scene_weight_feat = torch.cat(scene_weight_list, dim=2).permute(0, 2, 1)
                scene_weight_pre = self.fc_scene_weight(scene_weight_feat)
                scene_weight_ = self.softmax(scene_weight_pre)
                scene_weight[:hws[0][0] * hws[0][1]] *= scene_weight_[:, :1, 1][None, ...]#.repeat(stereo_features.shape[0], 1, 1)
                scene_weight[hws[0][0] * hws[0][1]:] *= scene_weight_[:, 1:, 1][None, ...]#.repeat(stereo_features.shape[0], 1, 1)

            stereo_features *= scene_weight

            if 'mask' in self.config['model']['tfm'].keys() and not self.config['exper']['run_eval_only']:
                grid_mask = torch.ones((*stereo_features.shape[:2], 1), device=stereo_features.device)
                len_rand = (torch.rand(1) * self.config['model']['tfm']['mask'] * stereo_features.shape[0]).int()
                index = torch.randperm(stereo_features.shape[0])[:len_rand]
                grid_mask[index] = 0
                stereo_features *= grid_mask

            pos_enc_1 = self.position_encoding_1(batch_size, hws, device).flatten(2).permute(2, 0, 1)
            pos_enc_2 = self.position_encoding_2(batch_size, hws, device).flatten(2).permute(2, 0, 1)
            pos_enc_ev_2 = self.position_encoding_2(batch_size, hws[:1], device).flatten(2).permute(2, 0, 1)
            pos_enc_ev_1 = self.position_encoding_1(batch_size, hws[:1], device).flatten(2).permute(2, 0, 1)

            def infer_encode(modules, features, pos, return_att, dim_reduce):
                latent_1, att_1 = modules[0](features, pos[0], return_att=return_att)
                reduced_latent_1 = dim_reduce(latent_1)
                latent_2, att_2 = modules[1](reduced_latent_1, pos[1], return_att=return_att)
                att_ = (att_1, att_2)
                return latent_1, latent_2, att_

            def infer_decode(modules, jv_tokens, pos, latent, return_att, dim_reduce, attention_mask, regress, upsampling):
                hidden_1, att_1 = modules[0](latent[0], jv_tokens, pos[0], attention_mask=attention_mask,
                                                      return_att=return_att)
                reduced_hidden_1 = dim_reduce(hidden_1)
                hidden_2, att_2 = modules[1](latent[1], reduced_hidden_1, pos[1], attention_mask=attention_mask,
                                                      return_att=return_att)
                pred_3d_coordinates = regress(hidden_2.transpose(0, 1))
                pred_3d_joints = pred_3d_coordinates[:, :21, :]
                pred_vertices_sub = pred_3d_coordinates[:, 21:, :]
                pred_vertices = upsampling(pred_vertices_sub.transpose(1, 2))
                pred_vertices = pred_vertices.transpose(1, 2)
                return pred_3d_joints, pred_vertices, pred_vertices_sub, (att_1, att_2)

            # stereo encoder
            latent_1, latent_2, att_ = infer_encode(self.stereo_encoders, stereo_features, [pos_enc_1, pos_enc_2], return_att, self.dim_reduce_stereo)
            if 'latent' in self.config['model']['tfm'].keys() and self.config['model']['tfm']['latent'] == 'rgb':
                start_index = hws[0][0] * hws[0][1]
                latent_1 = latent_1[start_index:]
                latent_2 = latent_2[start_index:]
                pos_enc_1 = pos_enc_1[start_index:]
                pos_enc_2 = pos_enc_2[start_index:]

            # stereo decoder
            pred_3d_joints_0, pred_vertices_0, pred_vertices_sub_0, att_0 = infer_decode(
                self.decoders, jv_tokens, [pos_enc_1, pos_enc_2], [latent_1, latent_2], return_att, self.dim_reduce_dec, attention_mask, self.xyz_regressor, self.upsampling
            )
            atts += (att_, att_0)

            output.append([{
                'pred_3d_joints': pred_3d_joints_0,
                'pred_vertices': pred_vertices_0,
                'pred_vertices_sub': pred_vertices_sub_0
            }])
            if 'scene_weight' in self.config['model']['tfm'].keys() and self.config['model']['tfm']['scene_weight']:
                output[0][0].update({
                    'scene_weight': scene_weight_pre,
                })

            for i in range(1, len(frames)):
                for j in range(len(frames[i]['ev_frames'])):
                    _, grid_feat_ev = self.ev_backbone(frames[i]['ev_frames'][j].permute(0, 3, 1, 2))#[:, :2])
                    ev_update_features = self.conv_1x1_ev(grid_feat_ev).flatten(2).permute(2, 0, 1)
                    ev_latent_1, ev_latent_2, att_ = infer_encode(self.event_encoders, ev_update_features,
                                                            [pos_enc_ev_1, pos_enc_ev_2], return_att, self.dim_reduce_event)

                    latent_1, att_1 = self.perveivers[0](latent_1, ev_latent_1, self.config['model']['tfm']['iterations'],
                                                         pos_enc_ev_1, return_att)
                    latent_2, att_2 = self.perveivers[1](latent_2, ev_latent_2, self.config['model']['tfm']['iterations'],
                                                         pos_enc_ev_2, return_att)
                    atts += ((att_,att_1, att_2), )
                    if decode_all:
                        pred_3d_joints_, pred_vertices_, pred_vertices_sub_, att_ = infer_decode(
                            self.decoders, jv_tokens, [pos_enc_1, pos_enc_2], [latent_1, latent_2], return_att,
                            self.dim_reduce_dec, attention_mask, self.xyz_regressor, self.upsampling
                        )
                        res = {
                            'pred_3d_joints': pred_3d_joints_,
                            'pred_vertices': pred_vertices_,
                            'pred_vertices_sub': pred_vertices_sub_
                        }
                        if j == 0:
                            output.append([res])
                        else:
                            output[i].append(res)
                        atts += (att_)
                if not decode_all:
                    pred_3d_joints_, pred_vertices_, pred_vertices_sub_, att_d = infer_decode(
                        self.decoders, jv_tokens, [pos_enc_1, pos_enc_2], [latent_1, latent_2], return_att,
                        self.dim_reduce_dec, attention_mask, self.xyz_regressor, self.upsampling
                    )
                    atts += (att_d, )
                    output.append([{
                        'pred_3d_joints': pred_3d_joints_,
                        'pred_vertices': pred_vertices_,
                        'pred_vertices_sub': pred_vertices_sub_
                    }])
        return output, atts
