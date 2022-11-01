import numpy as np
import torch
import torch.nn.functional as F
from src.modeling._mano import MANO
from src.modeling._mano import Mesh as MeshSampler
from pytorch3d.renderer import (BlendParams, HardFlatShader, MeshRasterizer,
                                MeshRenderer, PointLights,
                                RasterizationSettings, PerspectiveCameras,
                                TexturesVertex,
                                SoftSilhouetteShader)
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt


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
            image_size=(self.config['exper']['bbox']['event']['size'], self.config['exper']['bbox']['event']['size']),
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
            faces_per_pixel=20,
            perspective_correct=False
        )
        rgb_raster_settings = RasterizationSettings(
            image_size=(self.config['exper']['bbox']['rgb']['size'], self.config['exper']['bbox']['rgb']['size']),
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
            faces_per_pixel=20,
            perspective_correct=False
        )
        self.event_silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=event_raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend_params),
        )
        self.rgb_silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=rgb_raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend_params),
        )

    def init_criterion(self):
        # define loss function (criterion) and optimizer
        self.criterion_2d_joints = torch.nn.L1Loss(reduction='none').cuda(self.config['exper']['device'])
        self.criterion_3d_joints = torch.nn.L2Loss(reduction='none').cuda(self.config['exper']['device'])
        self.criterion_vertices = torch.nn.L1Loss(reduction='none').cuda(self.config['exper']['device'])
        if 'scene_weight' in self.config['model']['tfm'].keys() and self.config['model']['tfm']['scene_weight']:
            self.criterion_scene = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=self.config['exper']['loss']['label_smoothing']).cuda(self.config['exper']['device'])


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
    def get_2d_joints_loss(self, gt_2d_joints, pred_2d_joints, mask, prob=None):
        if mask.any():
            loss = (self.criterion_2d_joints(gt_2d_joints, pred_2d_joints)[mask]).mean()
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
    def get_dice_loss(self, seg1, seg2, mask):
        seg1_ = seg1.flatten(start_dim=1)
        seg2_ = seg2.flatten(start_dim=1)
        intersection = (seg1_ * seg2_).sum(-1).sum()
        score = (2. * intersection + 1e-6) / (seg1_.sum() + seg2_.sum() + 1e-6)
        return 1 - score

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
        device = preds[0][-1]['pred_vertices'].device
        loss_sum = torch.tensor([0.], device=device, dtype=torch.float32)
        loss_items = {}
        for step in range(steps):
            bbox_valid = meta_data[step]['bbox_valid']
            mano_valid = meta_data[step]['mano_valid'] * bbox_valid
            joints_3d_valid = meta_data[step]['joints_3d_valid'] * bbox_valid
            super_3d_valid = meta_data[step]['supervision_type'] == 0
            super_2d_valid = torch.logical_not(super_3d_valid)
            manos = meta_data[step]['mano']

            if self.config['model']['method']['framework'] == 'eventhands':
                loss_hand_pose = F.mse_loss(preds[step][-1]['pred_hand_pose'], manos['hand_pose'].squeeze(dim=1))
                loss_shape = F.mse_loss(preds[step][-1]['pred_shape'], manos['shape'].squeeze(dim=1))
                loss_rot_pose = F.mse_loss(preds[step][-1]['pred_rot_pose'], manos['rot_pose'].squeeze(dim=1))

                loss_items.update({
                    'loss_hand_pose_' + str(step): loss_hand_pose,
                    'loss_shape_' + str(step): loss_shape,
                    'loss_rot_pose_' + str(step): loss_rot_pose,
                })

                loss_sum += loss_hand_pose * 5. + loss_shape * 5. + loss_rot_pose * 20
                continue

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

                if step == 0:
                    if 'scene_weight' in self.config['model']['tfm'].keys() and self.config['model']['tfm']['scene_weight']:
                        loss_scene_ev = (self.criterion_scene(preds[step][-1]['scene_weight'][:, 0], meta_data[step]['scene_weight'][:, 0].long())[super_3d_valid]).mean()
                        loss_scene_rgb = (self.criterion_scene(preds[step][-1]['scene_weight'][:, 1],
                                                               meta_data[step]['scene_weight'][:, 1].long())[
                            super_3d_valid]).mean()
                        loss_sum += self.config['exper']['loss']['scene'] * ratio * (loss_scene_rgb + loss_scene_ev)
                        loss_items.update(
                            {
                                'scene_weight_rgb_'+str(step): loss_scene_rgb,
                                'scene_weight_ev_' + str(step): loss_scene_ev,
                            }
                        )
            if super_2d_valid.any():
                batch_size = super_2d_valid.sum()
                faces = torch.tensor(self.mano_layer.faces.astype(np.int32)).repeat(batch_size, 1, 1).to(device)
                verts_rgb = torch.ones_like(self.mano_layer.v_template).to(device)
                verts_rgb = verts_rgb.expand(batch_size, verts_rgb.shape[0], verts_rgb.shape[1])
                textures = TexturesVertex(verts_rgb)

                mesh = Meshes(
                    verts=preds[step][-1]['pred_vertices'][super_2d_valid] + gt_dest_mano_output.joints[super_2d_valid][:, :1, :],
                    faces=faces,
                    textures=textures
                )
                mesh_gt = Meshes(
                    verts=gt_dest_mano_output.vertices[super_2d_valid],
                    faces=faces,
                    textures=textures
                )

                # for event
                K_event = meta_data[step]['K_event'][super_2d_valid]
                R_event = meta_data[step]['R_event'][super_2d_valid]
                t_event = meta_data[step]['t_event'][super_2d_valid]
                K_event[:, :2, 2] -= meta_data[step]['lt_evs'][-1][super_2d_valid]
                K_event[:, :2] *= meta_data[step]['sc_evs'][-1][super_2d_valid][..., None, None]

                event_cameras = cameras_from_opencv_projection(
                    R=R_event,
                    tvec=t_event,
                    camera_matrix=K_event,
                    image_size=torch.tensor([self.config['exper']['bbox']['event']['size'],
                                             self.config['exper']['bbox']['event']['size']]).expand(batch_size, 2).to(device)
                ).to(device)

                event_silouette = self.event_silhouette_renderer(
                    mesh, device=device, cameras=event_cameras
                )
                # if step == 0:
                #     plt.imshow(event_silouette[0, ..., 3].detach().cpu().numpy())
                #     plt.show()
                if meta_data[step]['supervision_type'][super_2d_valid][0] == 1:
                    gt_event_silouette = self.event_silhouette_renderer(
                        mesh_gt, device=device, cameras=event_cameras
                    )
                    loss_seg_event = self.get_dice_loss(event_silouette[..., 3], gt_event_silouette[..., 3], super_2d_valid[super_2d_valid])
                    # print(loss_seg_event)
                else:
                    pass

                # for rgb
                K_rgb = meta_data[step]['K_rgb'][super_2d_valid]
                R_rgb = meta_data[step]['R_rgb'][super_2d_valid]
                t_rgb = meta_data[step]['t_rgb'][super_2d_valid]
                K_rgb[:, :2, 2] -= meta_data[step]['lt_rgb'][super_2d_valid]
                K_rgb[:, :2] *= meta_data[step]['sc_rgb'][super_2d_valid][..., None, None]

                rgb_cameras = cameras_from_opencv_projection(
                    R=R_rgb,
                    tvec=t_rgb,
                    camera_matrix=K_rgb,
                    image_size=torch.tensor([self.config['exper']['bbox']['rgb']['size'],
                                             self.config['exper']['bbox']['rgb']['size']]).expand(batch_size, 2).to(
                        device)
                ).to(device)

                rgb_silouette = self.rgb_silhouette_renderer(
                    mesh, device=device, cameras=rgb_cameras
                )

                if meta_data[step]['supervision_type'][super_2d_valid][0] == 1:
                    gt_rgb_silouette = self.rgb_silhouette_renderer(
                        mesh_gt, device=device, cameras=rgb_cameras
                    )
                    loss_seg_rgb = self.get_dice_loss(rgb_silouette[..., 3], gt_rgb_silouette[..., 3], super_2d_valid[super_2d_valid])
                else:
                    pass

                ## for 2d joints

                pred_joints = (preds[step][-1]['pred_3d_joints'] + meta_data[step]['3d_joints'][:, :1, :])[super_2d_valid]

                pred_joints_event_ = torch.bmm(R_event, pred_joints.transpose(2, 1)).transpose(2, 1) + t_event.reshape(-1, 1, 3)
                pred_2d_joints_event = torch.bmm(K_event, pred_joints_event_.permute(0, 2, 1)).permute(0, 2, 1)
                pred_2d_joints_event = pred_2d_joints_event[:, :, :2] / pred_2d_joints_event[:, :, 2:]
                ev_mask = torch.max(torch.abs(pred_2d_joints_event), dim=2).values < self.config['exper']['bbox']['event']['size']
                if meta_data[step]['supervision_type'][super_2d_valid][0] == 1:
                    gt_joints_event_ = torch.bmm(R_event, meta_data[step]['3d_joints'][super_2d_valid].transpose(2, 1)).transpose(2,
                                                                                                   1) + t_event.reshape(
                        -1, 1, 3)
                    gt_2d_joints_event = torch.bmm(K_event, gt_joints_event_.permute(0, 2, 1)).permute(0, 2, 1)
                    gt_2d_joints_event = gt_2d_joints_event[:, :, :2] / gt_2d_joints_event[:, :, 2:]
                    loss_2d_event = self.get_2d_joints_loss(gt_2d_joints_event, pred_2d_joints_event, ev_mask) / (self.config['exper']['bbox']['event']['size'] ** 2)
                else:
                    pass

                pred_joints_rgb_ = torch.bmm(R_rgb, pred_joints.transpose(2, 1)).transpose(2, 1) + t_rgb.reshape(-1, 1, 3)
                pred_2d_joints_rgb = torch.bmm(K_rgb, pred_joints_rgb_.permute(0, 2, 1)).permute(0, 2, 1)
                pred_2d_joints_rgb = pred_2d_joints_rgb[:, :, :2] / pred_2d_joints_rgb[:, :, 2:]
                rgb_mask = torch.max(torch.abs(pred_2d_joints_rgb), dim=2).values < \
                          self.config['exper']['bbox']['rgb']['size']
                if meta_data[step]['supervision_type'][super_2d_valid][0] == 1:
                    gt_joints_rgb_ = torch.bmm(R_rgb, meta_data[step]['3d_joints'][super_2d_valid].transpose(2, 1)).transpose(2, 1) + t_rgb.reshape(-1, 1, 3)
                    gt_2d_joints_rgb = torch.bmm(K_rgb, gt_joints_rgb_.permute(0, 2, 1)).permute(0, 2, 1)
                    gt_2d_joints_rgb = gt_2d_joints_rgb[:, :, :2] / gt_2d_joints_rgb[:, :, 2:]
                    loss_2d_rgb = self.get_2d_joints_loss(gt_2d_joints_rgb, pred_2d_joints_rgb, rgb_mask) / (self.config['exper']['bbox']['rgb']['size'] ** 2)
                else:
                    pass

                loss_sum += self.config['exper']['loss']['seg_rgb'] * loss_seg_rgb + \
                           self.config['exper']['loss']['seg_event'] * loss_seg_event + \
                           self.config['exper']['loss']['2d_joints_rgb'] * loss_2d_rgb + \
                           self.config['exper']['loss']['2d_joints_event'] * loss_2d_event
                loss_items.update({
                    'loss_seg_rgb_'+str(step): loss_seg_rgb,
                    'loss_seg_event_'+str(step): loss_seg_event,
                    'loss_2d_joints_rgb_'+str(step): loss_2d_rgb,
                    'loss_2d_joints_event_'+str(step): loss_2d_event,
                })
        return loss_sum, loss_items
