import numpy as np
import torch
from src.modeling._mano import MANO
from src.modeling._mano import Mesh as MeshSampler
from pytorch3d.renderer import (BlendParams, HardFlatShader, MeshRasterizer,
                                MeshRenderer, PointLights,
                                RasterizationSettings, PerspectiveCameras,
                                TexturesVertex,
                                SoftSilhouetteShader)
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.structures import Meshes


class Loss(torch.nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.mano_layer = MANO(self.config['data']['smplx_path'], use_pca=False, is_rhand=True)
        self.mesh_sampler = MeshSampler()
        self.init_criterion()
        self.init_silouette_render()

    def init_silouette_render(self):
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        event_raster_settings = RasterizationSettings(
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
            faces_per_pixel=2,
            perspective_correct=False
        )
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=event_raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend_params),
        )

    def init_criterion(self):
        # define loss function (criterion) and optimizer
        self.criterion_2d_joints = torch.nn.MSELoss(reduction='none').cuda(self.config['exper']['device'])
        self.criterion_3d_joints = torch.nn.MSELoss(reduction='none').cuda(self.config['exper']['device'])
        self.criterion_vertices = torch.nn.L1Loss(reduction='none').cuda(self.config['exper']['device'])

    def get_3d_joints_loss(self, gt_3d_joints, pred_3d_joints, mask):
        gt_root = gt_3d_joints[:, 0, :]
        gt_3d_aligned = gt_3d_joints - gt_root[:, None, :]
        pred_root = pred_3d_joints[:, 0, :]
        pred_3d_aligned = pred_3d_joints - pred_root[:, None, :]
        if mask.any():
            return (self.criterion_3d_joints(gt_3d_aligned, pred_3d_aligned)[mask]).mean()
        else:
            return torch.tensor(0, device=gt_3d_joints.device)

    # todo check the confidence of each joints
    def get_2d_joints_loss(self, gt_2d_joints, pred_2d_joints, mask, prob):
        if mask.any():
            loss = ((self.criterion_2d_joints(gt_2d_joints, pred_2d_joints) * prob[..., None])[mask]).mean()
        else:
            loss = torch.tensor(0, device=gt_2d_joints.device)
        return loss

    def get_vertices_loss(self, gt_vertices, pred_vertices, mask):
        loss = self.criterion_vertices(gt_vertices, pred_vertices)
        if mask.any():
            return (loss[mask]).mean()
        else:
            return torch.tensor(0, device=gt_vertices.device)

    # todo  DICE loss for segmentation here
    def dice_loss(self):
        pass

    def forward(self, meta_data, preds):
        # if self.config['model']['method']['framework'] != 'perceiver':
        #     manos = meta_data['mano_rgb']
        #     gt_rgb_mano_output = self.mano_layer(
        #         global_orient=manos['rot_pose'].reshape(-1, 3),
        #         hand_pose=manos['hand_pose'].reshape(-1, 45),
        #         betas=manos['shape'].reshape(-1, 10),
        #         transl=manos['trans'].reshape(-1, 3)
        #     )
        #     manos = meta_data['mano']
        #     gt_dest_mano_output = self.mano_layer(
        #         global_orient=manos['rot_pose'].reshape(-1, 3),
        #         hand_pose=manos['hand_pose'].reshape(-1, 45),
        #         betas=manos['shape'].reshape(-1, 10),
        #         transl=manos['trans'].reshape(-1, 3)
        #     )
        #     gt_dest_vertices_sub = self.mesh_sampler.downsample(gt_dest_mano_output.vertices)
        #     gt_dest_root = meta_data['3d_joints'][:, 0, :]
        #     gt_dest_3d_joints = meta_data['3d_joints'] - gt_dest_root[:, None, :]
        #     gt_dest_vertices = gt_dest_mano_output.vertices - gt_dest_mano_output.joints[:, :1, :]
        #     gt_dest_vertices_sub = gt_dest_vertices_sub - gt_dest_mano_output.joints[:, :1, :]
        #
        #     gt_rgb_vertices_sub = self.mesh_sampler.downsample(gt_rgb_mano_output.vertices)
        #     gt_rgb_root = gt_rgb_mano_output.joints[:, 0, :]
        #     gt_rgb_3d_joints = gt_rgb_mano_output.joints - gt_rgb_root[:, None, :]
        #     gt_rgb_vertices = gt_rgb_mano_output.vertices - gt_rgb_root[:, None, :]
        #     gt_rgb_vertices_sub = gt_rgb_vertices_sub - gt_rgb_root[:, None, :]
        #
        # if self.config['model']['method']['framework'] == 'encoder_based':
        #     pred_3d_joints_from_mesh = self.mano_layer.get_3d_joints(preds['pred_vertices'])
        #
        #     # todo directly predicted 3d joints loss
        #     loss_3d_joints = self.get_3d_joints_loss(gt_dest_3d_joints, preds['pred_3d_joints'])
        #     # todo predicted (from mesh) 3d joints loss
        #     loss_3d_joints_reg = self.get_3d_joints_loss(gt_dest_3d_joints, pred_3d_joints_from_mesh)
        #     # todo vertices loss here
        #     loss_vertices = self.get_vertices_loss(gt_dest_vertices, preds['pred_vertices'])
        #     loss_vertices_sub = self.get_vertices_loss(gt_dest_vertices_sub, preds['pred_vertices_sub'])
        #
        #     # todo sum up the losses
        #     loss_sum = self.config['exper']['loss']['vertices'] * loss_vertices +\
        #         self.config['exper']['loss']['vertices_sub'] * loss_vertices_sub +\
        #         self.config['exper']['loss']['3d_joints'] * loss_3d_joints +\
        #         self.config['exper']['loss']['3d_joints_from_mesh'] * loss_3d_joints_reg
        #     loss_items = {
        #         'loss_vertices': loss_vertices,
        #         'loss_vertices_sub': loss_vertices_sub,
        #         'loss_3d_joints': loss_3d_joints,
        #         'loss_3d_joints_reg': loss_3d_joints_reg,
        #     }
        # elif self.config['model']['method']['framework'] == 'encoder_decoder_based':
        #     pred_3d_joints_from_mesh_l = self.mano_layer.get_3d_joints(preds['pred_vertices_l'])
        #     loss_3d_joints_l = self.get_3d_joints_loss(gt_rgb_3d_joints, preds['pred_3d_joints_l'])
        #     loss_3d_joints_reg_l = self.get_3d_joints_loss(gt_rgb_3d_joints, pred_3d_joints_from_mesh_l)
        #     loss_vertices_l = self.get_vertices_loss(gt_rgb_vertices, preds['pred_vertices_l'])
        #     loss_vertices_sub_l = self.get_vertices_loss(gt_rgb_vertices_sub, preds['pred_vertices_sub_l'])
        #     loss_sum = self.config['exper']['loss']['vertices'] * loss_vertices_l + \
        #                self.config['exper']['loss']['vertices_sub'] * loss_vertices_sub_l + \
        #                self.config['exper']['loss']['3d_joints'] * loss_3d_joints_l + \
        #                self.config['exper']['loss']['3d_joints_from_mesh'] * loss_3d_joints_reg_l
        #     loss_items = {
        #         'loss_vertices_l': loss_vertices_l,
        #         'loss_vertices_sub_l': loss_vertices_sub_l,
        #         'loss_3d_joints_l': loss_3d_joints_l,
        #         'loss_3d_joints_reg_l': loss_3d_joints_reg_l,
        #     }
        #     if self.config['model']['method']['ere_usage'][2]:
        #         pred_3d_joints_from_mesh_r = self.mano_layer.get_3d_joints(preds['pred_vertices_r'])
        #         loss_3d_joints_r = self.get_3d_joints_loss(gt_dest_3d_joints, preds['pred_3d_joints_r'])
        #         loss_3d_joints_reg_r = self.get_3d_joints_loss(gt_dest_3d_joints, pred_3d_joints_from_mesh_r)
        #         loss_vertices_r = self.get_vertices_loss(gt_dest_vertices, preds['pred_vertices_r'])
        #         loss_vertices_sub_r = self.get_vertices_loss(gt_dest_vertices_sub, preds['pred_vertices_sub_r'])
        #         loss_sum += self.config['exper']['loss']['vertices'] * loss_vertices_r + \
        #                    self.config['exper']['loss']['vertices_sub'] * loss_vertices_sub_r + \
        #                    self.config['exper']['loss']['3d_joints'] * loss_3d_joints_r + \
        #                    self.config['exper']['loss']['3d_joints_from_mesh'] * loss_3d_joints_reg_r
        #         loss_items.update({
        #             'loss_vertices_r': loss_vertices_r,
        #             'loss_vertices_sub_r': loss_vertices_sub_r,
        #             'loss_3d_joints_r': loss_3d_joints_r,
        #             'loss_3d_joints_reg_r': loss_3d_joints_reg_r,
        #         })
        # if self.config['model']['method']['framework'] == 'perceiver':
        steps = len(meta_data)
        loss_sum = 0
        loss_items = {}
        device = preds[0][-1]['pred_vertices'].device
        for step in range(steps):
            bbox_valid = meta_data[step]['bbox_valid']
            mano_valid = meta_data[step]['mano_valid'] * bbox_valid
            joints_3d_valid = meta_data[step]['joints_3d_valid'] * bbox_valid
            super_3d_valid = meta_data[step]['supervision_type'] == 0
            super_2d_valid = torch.logical_not(super_3d_valid)
            manos = meta_data[step]['mano']
            gt_dest_mano_output = self.mano_layer(
                global_orient=manos['rot_pose'].reshape(-1, 3),
                hand_pose=manos['hand_pose'].reshape(-1, 45),
                betas=manos['shape'].reshape(-1, 10),
                transl=manos['trans'].reshape(-1, 3)
            )
            gt_dest_vertices_sub = self.mesh_sampler.downsample(gt_dest_mano_output.vertices)
            gt_dest_root = meta_data[step]['3d_joints'][:, 0, :]
            gt_dest_3d_joints_aligned = meta_data[step]['3d_joints'] - gt_dest_root[:, None, :]
            gt_dest_vertices_aligned = gt_dest_mano_output.vertices - gt_dest_mano_output.joints[:, :1, :]
            gt_dest_vertices_sub_aligned = gt_dest_vertices_sub - gt_dest_mano_output.joints[:, :1, :]

            pred_3d_joints_from_mesh = self.mano_layer.get_3d_joints(preds[step][-1]['pred_vertices'])

            if super_3d_valid.any():
                loss_3d_joints = self.get_3d_joints_loss(gt_dest_3d_joints_aligned[super_3d_valid],
                                                         preds[step][-1]['pred_3d_joints'][super_3d_valid],
                                                         joints_3d_valid[super_3d_valid])
                loss_3d_joints_reg = self.get_3d_joints_loss(gt_dest_3d_joints_aligned[super_3d_valid],
                                                             pred_3d_joints_from_mesh[super_3d_valid],
                                                             joints_3d_valid[super_3d_valid])
                loss_vertices = self.get_vertices_loss(gt_dest_vertices_aligned[super_3d_valid],
                                                       preds[step][-1]['pred_vertices'][super_3d_valid],
                                                       mano_valid[super_3d_valid])
                loss_vertices_sub = self.get_vertices_loss(gt_dest_vertices_sub_aligned[super_3d_valid],
                                                           preds[step][-1]['pred_vertices_sub'][super_3d_valid],
                                                           mano_valid[super_3d_valid])
                if step == 0:
                    ratio = 2
                else:
                    ratio = 1
                loss_sum += self.config['exper']['loss']['vertices'] * ratio * loss_vertices + \
                           self.config['exper']['loss']['vertices_sub'] * ratio * loss_vertices_sub + \
                           self.config['exper']['loss']['3d_joints'] * ratio * loss_3d_joints + \
                           self.config['exper']['loss']['3d_joints_from_mesh'] * ratio * loss_3d_joints_reg
                loss_items.update({
                    'loss_vertices_'+str(step): loss_vertices,
                    'loss_vertices_sub_'+str(step): loss_vertices_sub,
                    'loss_3d_joints_'+str(step): loss_3d_joints,
                    'loss_3d_joints_reg_'+str(step): loss_3d_joints_reg,
                })
            if super_2d_valid.any():
                batch_size = super_2d_valid.sum()
                faces = torch.tensor(self.mano_layer.faces.astype(np.int32)).repeat(batch_size, 1, 1).to(device)
                verts_rgb = torch.ones_like(self.mano_layer.v_template).to(device)
                verts_rgb = verts_rgb.expand(batch_size, verts_rgb.shape[0], verts_rgb.shape[1])
                textures = TexturesVertex(verts_rgb)

                # for event
                K_event = meta_data[step]['K_event'][super_2d_valid]
                R_event = meta_data[step]['R_event'][super_2d_valid]
                t_event = meta_data[step]['t_event'][super_2d_valid]
                K_event[:, :2, 2] -= meta_data[step]['lt_evs'][-1][super_2d_valid]
                K_event[:, :2] *= meta_data[step]['sc_evs'][-1][super_2d_valid]

                event_cameras = cameras_from_opencv_projection(
                    R=torch.eye(3).repeat(batch_size, 1, 1).to(device),
                    tvec=torch.zeros(batch_size, 3).to(device),
                    camera_matrix=K_event,
                    image_size=torch.tensor([self.config['exper']['bbox']['event']['size'],
                                             self.config['exper']['bbox']['event']['size']]).expand(batch_size, 2).to(device)
                ).to(device)
                # todo
                mesh = Meshes(
                    verts=None, #now_vertices,
                    faces=faces,
                    textures=textures
                )

        return loss_sum, loss_items
