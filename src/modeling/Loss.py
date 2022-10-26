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
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F
from mmpose.core import keypoints_from_heatmaps


def meanIoU(logits, labels):
    """
    Computes the mean intersection over union (mIoU).
    
    Args:
        logits: tensor of shape [bs, c, h, w].
        labels: tensor of shape [bs, h, w].
    
    Output:
        miou: scalar.
    """
    num_classes = logits.shape[1]
    preds = F.softmax(logits, 1)
    preds_oh = F.one_hot(preds.argmax(1), num_classes).permute(0, 3, 1, 2).to(torch.float32) # [bs, c, h, w] 
    labels_oh = F.one_hot(labels, num_classes).permute(0, 3, 1, 2).to(torch.float32) # [bs, c, h, w]
    tps = (preds_oh * labels_oh).sum(-1).sum(-1) # true positives [bs, c]
    fps = (preds_oh * (1 - labels_oh)).sum(-1).sum(-1) # false positives [bs, c]
    fns = ((1 - preds_oh) * labels_oh).sum(-1).sum(-1) # false negatives [bs, c]
    iou = tps / (tps + fps + fns + 1e-8) # [bs, c]
    return iou.mean(-1).mean(0)

def get_max_preds_torch(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = torch.argmax(heatmaps_reshaped, 2)
    maxvals = torch.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = torch.tile(idx, (1, 1, 2)).float()

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)

    pred_mask = torch.tile(torch.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.float()

    preds *= pred_mask
    return preds, maxvals

def get_final_preds_torch(batch_heatmaps, scale):
    coords, maxvals = get_max_preds_torch(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    coords_diff = coords.clone()
    coords_diff = torch.floor(coords_diff+0.5).long()
    mask1 = coords_diff > 1
    mask2 = coords_diff < heatmap_width-1
    mask = torch.logical_and(mask1,mask2)
    mask = torch.logical_and(mask[...,0],mask[...,1])

    coords_diff = coords_diff[mask]

    sign_diff_x = torch.sign(batch_heatmaps[mask][torch.arange(0,coords_diff.shape[0]).long(),(coords_diff)[...,1],(coords_diff+1)[...,0]] - \
                    batch_heatmaps[mask][torch.arange(0,coords_diff.shape[0]).long(),(coords_diff)[...,1],(coords_diff-1)[...,0]])
    sign_diff_y = torch.sign(batch_heatmaps[mask][torch.arange(0,coords_diff.shape[0]).long(),(coords_diff+1)[...,1],(coords_diff)[...,0]] - \
                    batch_heatmaps[mask][torch.arange(0,coords_diff.shape[0]).long(),(coords_diff-1)[...,1],(coords_diff)[...,0]])
    sign_diff = torch.stack([sign_diff_x,sign_diff_y],dim=1)
    coords[mask] = coords[mask] + sign_diff*0.25
    return coords*scale


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
    def get_2d_joints_loss(self, gt_2d_joints, pred_2d_joints, mask, prob=None):
        if mask.any():
            loss = (self.criterion_2d_joints(gt_2d_joints, pred_2d_joints)[mask]).mean()
        else:
            loss = torch.tensor(0, device=gt_2d_joints.device)
        return loss

    def generate_heatmap_gt(self, joints_2d, heatmap_size, sigma=2.0):
        """Generate the target heatmap via "MSRA" approach.
        Args:
            cfg (dict): data config
            joints_2d: np.ndarray ([num_joints, 2)
            heatmap size: int size of heatmap)
            sigma: Sigma of heatmap gaussian
        Returns:
            tuple: A tuple containing targets.
            - target: Target heatmaps.
            - target_weight: (1: visible, 0: invisible)
        """
        num_joints = joints_2d.shape[1]
        batch_size = joints_2d.shape[0]
        assert num_joints == 21, 'Only support 21 joints'
        feat_stride = 4.0
        W, H = heatmap_size, heatmap_size
        # joint_weights = cfg['joint_weights']
        use_different_joint_weights = False
        target_weight = torch.where(joints_2d[:, :, -1] > 0, torch.ones_like(joints_2d[:, :, -1]), torch.zeros_like(joints_2d[:, :, -1])).to(joints_2d.device)
        # target_weight = torch.zeros((num_joints, 1), dtype=torch.float32,device=joints_3d.device)
        # target = torch.zeros((num_joints, H, W), dtype=torch.float32, device=joints_3d.device)

        # 3-sigma rule
        tmp_size = sigma * 3
        joints_heatmap = (joints_2d / feat_stride).view(batch_size, num_joints, 1,1,2)
        x,y = torch.meshgrid(torch.arange(0, W), torch.arange(0, H),indexing="xy")#.to("cpu").unsqueeze(0).unsqueeze(0).repeat(10,21,1,1)
        target = torch.stack((x, y), 2).view((1, 1, H, W, 2)).to(joints_2d.device).repeat(batch_size,num_joints,1,1,1)
        # target = torch.meshgrid(torch.arange(0, W), torch.arange(0, H),indexing="xy").to(joints_2d.device).unsqueeze(0).unsqueeze(0).repeat(batch_size,num_joints,1,1)
        # target.expand(2, -1, -1)
        # assert joints_heatmap.shape == target.shape
        target = torch.exp(-(((target - joints_heatmap) ** 2).sum(dim=-1) / (2 * sigma ** 2)))
        return target, target_weight


    def get_heatmap_loss(self, gt_2d_joints, pred_heatmap, mask=None, prob=None):
        # if not mask.any():
        #     return torch.tensor(0, device=gt_2d_joints.device)
        # get gt_heatmap from gt_2d_joints
        heatmap_size = pred_heatmap.shape[-1]
        # assert heatmap_size == self.config['exper']['bbox']['rgb']['size']/4 or heatmap_size == self.config['exper']['bbox']['event']['size']/4
        
        gt_heatmap = torch.zeros((gt_2d_joints.shape[0], 21, heatmap_size,
                                      heatmap_size),
                                     device=gt_2d_joints.device)
            
        gt_heatmap, gt_heatmap_weight = self.generate_heatmap_gt(gt_2d_joints, heatmap_size)
        # pdb.set_trace()
        pred_kpt = get_final_preds_torch(pred_heatmap,4)
        # gt_kpt = get_final_preds_torch(gt_heatmap,4)
        mpjperr = (pred_kpt - gt_2d_joints).norm(dim=-1).mean()
        # test_err = (gt_2d_joints - gt_kpt).norm(dim=-1).mean()
        
        # pdb.set_trace()
        # calculate mpjpe:
        
        return self.criterion_2d_joints(gt_heatmap, pred_heatmap).mean(), mpjperr



    def get_vertices_loss(self, gt_vertices, pred_vertices, mask):
        loss = self.criterion_vertices(gt_vertices, pred_vertices)
        if mask.any():
            return (loss[mask]).mean()
        else:
            return torch.tensor(0, device=gt_vertices.device)

    # todo  DICE loss for segmentation here
    def get_dice_loss(self, seg1, seg2, mask):
        # pdb.set_trace()
        seg1_ = seg1.flatten(start_dim=1)
        seg2_ = seg2.flatten(start_dim=1)
        intersection = (seg1_ * seg2_).sum(-1).sum()
        score = (2. * intersection + 1e-6) / (seg1_.sum() + seg2_.sum() + 1e-6)
        return 1 - score

    def get_bce_loss(self, seg, seg_gt, mask):
        # assert seg.shape[2:] == seg_gt.shape[1:], "seg shape {} and seg_gt shape {} are not avalible".format(seg.shape, seg_gt.shape)
        # pdb.set_trace()
        loss = F.binary_cross_entropy_with_logits(seg, seg_gt)
        y_hat = F.softmax(seg, 1).detach()
        miou = meanIoU(y_hat, seg_gt.argmax(1))
        return loss, miou

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
        if self.config['model']['method']['framework'] == 'pretrain':
            # loss for pretrain model
            steps = len(meta_data)
            assert steps == 1
            device = preds[0][-1]['pred_heatmap'].device
            loss_sum = torch.tensor([0.], device=device, dtype=torch.float32)
            loss_items = {}
            for step in range(steps):
                bbox_valid = meta_data[step]['bbox_valid']
                mano_valid = meta_data[step]['mano_valid'] * bbox_valid
                joints_3d_valid = meta_data[step]['joints_3d_valid'] * bbox_valid
                # only use 2d ground truth
                super_3d_valid = meta_data[step]['supervision_type'] < 0
                super_2d_valid = torch.logical_not(super_3d_valid)
                # MANO is not needed for pretrian
                # manos = meta_data[step]['mano']
                # gt_dest_mano_output = self.mano_layer(
                #     global_orient=manos['rot_pose'].reshape(-1, 3),
                #     hand_pose=manos['hand_pose'].reshape(-1, 45),
                #     betas=manos['shape'].reshape(-1, 10),
                #     transl=manos['trans'].reshape(-1, 3)
                # )
                # gt_dest_vertices_sub = self.mesh_sampler.downsample(gt_dest_mano_output.vertices)
                # gt_dest_root = meta_data[step]['3d_joints'][:, 0, :]
                # gt_dest_3d_joints_aligned = meta_data[step]['3d_joints'] - gt_dest_root[:, None, :]
                # gt_dest_vertices_aligned = gt_dest_mano_output.vertices - gt_dest_mano_output.joints[:, :1, :]
                # gt_dest_vertices_sub_aligned = gt_dest_vertices_sub - gt_dest_mano_output.joints[:, :1, :]

                # pred_3d_joints_from_mesh = self.mano_layer.get_3d_joints(preds[step][-1]['pred_vertices'])

                if super_3d_valid.any():
                    assert False, '3d supervision is not supported for pretrain'
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
                    # faces = torch.tensor(self.mano_layer.faces.astype(np.int32)).repeat(batch_size, 1, 1).to(device)
                    # verts_rgb = torch.ones_like(self.mano_layer.v_template).to(device)
                    # verts_rgb = verts_rgb.expand(batch_size, verts_rgb.shape[0], verts_rgb.shape[1])
                    # textures = TexturesVertex(verts_rgb)

                    # mesh = Meshes(
                    #     verts=preds[step][-1]['pred_vertices'][super_2d_valid] + gt_dest_mano_output.joints[super_2d_valid][:, :1, :],
                    #     faces=faces,
                    #     textures=textures
                    # )
                    # mesh_gt = Meshes(
                    #     verts=gt_dest_mano_output.vertices[super_2d_valid],
                    #     faces=faces,
                    #     textures=textures
                    # )

                    # for event
                    K_event = meta_data[step]['K_event'][super_2d_valid]
                    R_event = meta_data[step]['R_event'][super_2d_valid]
                    t_event = meta_data[step]['t_event'][super_2d_valid]
                    K_event[:, :2, 2] -= meta_data[step]['lt_evs'][-1][super_2d_valid]
                    K_event[:, :2] *= meta_data[step]['sc_evs'][-1][super_2d_valid][..., None, None]

                    # event_cameras = cameras_from_opencv_projection(
                    #     R=R_event,
                    #     tvec=t_event,
                    #     camera_matrix=K_event,
                    #     image_size=torch.tensor([self.config['exper']['bbox']['event']['size'],
                    #                             self.config['exper']['bbox']['event']['size']]).expand(batch_size, 2).to(device)
                    # ).to(device)

                    # event_silouette = self.event_silhouette_renderer(
                    #     mesh, device=device, cameras=event_cameras
                    # )
                    # if step == 0:
                    #     plt.imshow(event_silouette[0, ..., 3].detach().cpu().numpy())
                    #     plt.show()
                    # if meta_data[step]['supervision_type'][super_2d_valid][0] == 1:
                    #     gt_event_silouette = self.event_silhouette_renderer(
                    #         mesh_gt, device=device, cameras=event_cameras
                    #     )
                    #     loss_seg_event = self.get_dice_loss(event_silouette[..., 3], gt_event_silouette[..., 3], super_2d_valid[super_2d_valid])
                    #     # print(loss_seg_event)
                    # else:
                    #     pass
                    
                    gt_event_silouette = meta_data[step]["ev_mask"][super_2d_valid]
                    # assert gt_event_silouette.shape == (batch_size, 3, self.config['exper']['bbox']['event']['size'], self.config['exper']['bbox']['event']['size']), \
                    #     "gt_event_silouette.shape: {}, batch_size: {}".format(gt_event_silouette.shape, batch_size)
                    # pdb.set_trace()
                    loss_seg_event = self.get_dice_loss(F.softmax(preds[step][-1]['pred_seg_ev'],1)[:,1,:,:], gt_event_silouette[..., -1], super_2d_valid[super_2d_valid])
                    loss_bce_event, miou_event = self.get_bce_loss(preds[step][-1]['pred_seg_ev'], torch.stack([1-gt_event_silouette[..., -1],gt_event_silouette[..., -1]],dim=1), super_2d_valid[super_2d_valid])
                    # for rgb
                    K_rgb = meta_data[step]['K_rgb'][super_2d_valid]
                    R_rgb = meta_data[step]['R_rgb'][super_2d_valid]
                    t_rgb = meta_data[step]['t_rgb'][super_2d_valid]
                    K_rgb[:, :2, 2] -= meta_data[step]['lt_rgb'][super_2d_valid]
                    K_rgb[:, :2] *= meta_data[step]['sc_rgb'][super_2d_valid][..., None, None]

                    # rgb_cameras = cameras_from_opencv_projection(
                    #     R=R_rgb,
                    #     tvec=t_rgb,
                    #     camera_matrix=K_rgb,
                    #     image_size=torch.tensor([self.config['exper']['bbox']['rgb']['size'],
                    #                             self.config['exper']['bbox']['rgb']['size']]).expand(batch_size, 2).to(
                    #         device)
                    # ).to(device)

                    # rgb_silouette = self.rgb_silhouette_renderer(
                    #     mesh, device=device, cameras=rgb_cameras
                    # )

                    # if meta_data[step]['supervision_type'][super_2d_valid][0] == 1:
                    #     gt_rgb_silouette = self.rgb_silhouette_renderer(
                    #         mesh_gt, device=device, cameras=rgb_cameras
                    #     )
                    #     loss_seg_rgb = self.get_dice_loss(rgb_silouette[..., 3], gt_rgb_silouette[..., 3], super_2d_valid[super_2d_valid])
                    # else:
                    #     pass
                    gt_rgb_silouette = meta_data[step]["mask"][super_2d_valid]
                    loss_seg_rgb = self.get_dice_loss(F.softmax(preds[step][-1]['pred_seg'],1)[:,1,:,:], gt_rgb_silouette[..., -1], super_2d_valid[super_2d_valid])
                    loss_bce_rgb, miou_rgb = self.get_bce_loss(preds[step][-1]['pred_seg'], torch.stack([1-gt_rgb_silouette[..., -1],gt_rgb_silouette[..., -1]],dim=1), super_2d_valid[super_2d_valid])
                    
                    ## for 2d joints

                    # pred_joints = (preds[step][-1]['pred_3d_joints'] + meta_data[step]['3d_joints'][:, :1, :])[super_2d_valid]

                    # pred_joints_event_ = torch.bmm(R_event, pred_joints.transpose(2, 1)).transpose(2, 1) + t_event.reshape(-1, 1, 3)
                    # pred_2d_joints_event = torch.bmm(K_event, pred_joints_event_.permute(0, 2, 1)).permute(0, 2, 1)
                    # pred_2d_joints_event = pred_2d_joints_event[:, :, :2] / pred_2d_joints_event[:, :, 2:]
                    # ev_mask = torch.max(torch.abs(pred_2d_joints_event), dim=2).values < self.config['exper']['bbox']['event']['size']
                    # if meta_data[step]['supervision_type'][super_2d_valid][0] == 1:
                    gt_joints_event_ = torch.bmm(R_event, meta_data[step]['3d_joints'][super_2d_valid].transpose(2, 1)).transpose(2,
                                                                                                1) + t_event.reshape(
                        -1, 1, 3)
                    gt_2d_joints_event = torch.bmm(K_event, gt_joints_event_.permute(0, 2, 1)).permute(0, 2, 1)
                    gt_2d_joints_event = gt_2d_joints_event[:, :, :2] / gt_2d_joints_event[:, :, 2:]
                    # loss_2d_event = self.get_2d_joints_loss(gt_2d_joints_event, pred_2d_joints_event, ev_mask) / (self.config['exper']['bbox']['event']['size'] ** 2)
                    loss_2d_event, mpjpe_ev = self.get_heatmap_loss(gt_2d_joints_event, preds[step][-1]['pred_heatmap_ev'])# / (self.config['exper']['bbox']['event']['size'] ** 2)
                    # else:
                    #     pass

                    # pred_joints_rgb_ = torch.bmm(R_rgb, pred_joints.transpose(2, 1)).transpose(2, 1) + t_rgb.reshape(-1, 1, 3)
                    # pred_2d_joints_rgb = torch.bmm(K_rgb, pred_joints_rgb_.permute(0, 2, 1)).permute(0, 2, 1)
                    # pred_2d_joints_rgb = pred_2d_joints_rgb[:, :, :2] / pred_2d_joints_rgb[:, :, 2:]
                    # rgb_mask = torch.max(torch.abs(pred_2d_joints_rgb), dim=2).values < \
                    #         self.config['exper']['bbox']['rgb']['size']
                    # if meta_data[step]['supervision_type'][super_2d_valid][0] == 1:
                    gt_joints_rgb_ = torch.bmm(R_rgb, meta_data[step]['3d_joints'][super_2d_valid].transpose(2, 1)).transpose(2, 1) + t_rgb.reshape(-1, 1, 3)
                    gt_2d_joints_rgb = torch.bmm(K_rgb, gt_joints_rgb_.permute(0, 2, 1)).permute(0, 2, 1)
                    gt_2d_joints_rgb = gt_2d_joints_rgb[:, :, :2] / gt_2d_joints_rgb[:, :, 2:]
                    # loss_2d_rgb = self.get_2d_joints_loss(gt_2d_joints_rgb, pred_2d_joints_rgb, rgb_mask) / (self.config['exper']['bbox']['rgb']['size'] ** 2)
                    loss_2d_rgb, mpjpe_rgb = self.get_heatmap_loss(gt_2d_joints_rgb, preds[step][-1]['pred_heatmap'])# / (self.config['exper']['bbox']['rgb']['size'] ** 2)
                    # else:
                    #     pass
                    # pdb.set_trace()
                    loss_sum += self.config['exper']['loss']['seg_rgb'] * loss_seg_rgb + \
                            self.config['exper']['loss']['seg_event'] * loss_seg_event + \
                            self.config['exper']['loss']['2d_joints_rgb'] * loss_2d_rgb + \
                            self.config['exper']['loss']['2d_joints_event'] * loss_2d_event + \
                            self.config['exper']['loss']['bce_rgb'] * loss_bce_rgb + \
                            self.config['exper']['loss']['bce_event'] * loss_bce_event
                    loss_items.update({
                        'loss_seg_rgb_'+str(step): loss_seg_rgb,
                        'loss_seg_event_'+str(step): loss_seg_event,
                        'loss_2d_joints_rgb_'+str(step): loss_2d_rgb,
                        'loss_2d_joints_event_'+str(step): loss_2d_event,
                        'loss_bce_rgb_'+str(step): loss_bce_rgb,
                        'loss_bce_event_'+str(step): loss_bce_event,
                        'miou_rgb_'+str(step): miou_rgb,
                        'miou_event_'+str(step): miou_event,
                        'mpjpe_rgb_'+str(step): mpjpe_rgb,
                        'mpjpe_ev_'+str(step): mpjpe_ev,
                    })
            return loss_sum, loss_items

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
                    # loss_test_debug = self.get_heatmap_loss(gt_2d_joints_event, torch.ones((gt_2d_joints_event.shape[0],32,32)), ev_mask) / (self.config['exper']['bbox']['event']['size'] ** 2)
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
                    # loss_test_debug_rgb = self.get_heatmap_loss(gt_2d_joints_rgb, torch.ones((gt_2d_joints_rgb.shape[0],gt_2d_joints_rgb.shape[1],48,48)), rgb_mask) / (self.config['exper']['bbox']['rgb']['size'] ** 2)
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
