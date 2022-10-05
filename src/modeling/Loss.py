import torch
from src.modeling._mano import MANO
from src.modeling._mano import Mesh as MeshSampler


class Loss(torch.nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.mano_layer = MANO(self.config['data']['smplx_path'], use_pca=False, is_rhand=True)
        self.mesh_sampler = MeshSampler()
        self.init_criterion()

    def init_criterion(self):
        # define loss function (criterion) and optimizer
        self.criterion_2d_joints = torch.nn.MSELoss(reduction='none').cuda(self.config['exper']['device'])
        self.criterion_3d_joints = torch.nn.MSELoss(reduction='none').cuda(self.config['exper']['device'])
        self.criterion_vertices = torch.nn.L1Loss().cuda(self.config['exper']['device'])

    def get_3d_joints_loss(self, gt_3d_joints, pred_3d_joints):
        gt_root = gt_3d_joints[:, 0, :]
        gt_3d_aligned = gt_3d_joints - gt_root[:, None, :]
        pred_root = pred_3d_joints[:, 0, :]
        pred_3d_aligned = pred_3d_joints - pred_root[:, None, :]
        return (self.criterion_3d_joints(gt_3d_aligned, pred_3d_aligned)).mean()

    def get_2d_joints_loss(self, gt_2d_joints, pred_2d_joints):
        loss = self.criterion_2d_joints(gt_2d_joints, pred_2d_joints).mean()
        return loss

    def get_vertices_loss(self, gt_vertices, pred_vertices):
        loss = self.criterion_vertices(gt_vertices, pred_vertices)
        return loss

    def forward(self, meta_data, preds):
        manos = meta_data['mano_rgb']
        gt_rgb_mano_output = self.mano_layer(
            global_orient=manos['rot_pose'].reshape(-1, 3),
            hand_pose=manos['hand_pose'].reshape(-1, 45),
            betas=manos['shape'].reshape(-1, 10),
            transl=manos['trans'].reshape(-1, 3)
        )
        manos = meta_data['mano']
        gt_dest_mano_output = self.mano_layer(
            global_orient=manos['rot_pose'].reshape(-1, 3),
            hand_pose=manos['hand_pose'].reshape(-1, 45),
            betas=manos['shape'].reshape(-1, 10),
            transl=manos['trans'].reshape(-1, 3)
        )
        gt_dest_vertices_sub = self.mesh_sampler.downsample(gt_dest_mano_output.vertices)
        gt_dest_root = meta_data['3d_joints'][:, 0, :]
        gt_dest_3d_joints = meta_data['3d_joints'] - gt_dest_root[:, None, :]
        gt_dest_vertices = gt_dest_mano_output.vertices - gt_dest_mano_output.joints[:, :1, :]
        gt_dest_vertices_sub = gt_dest_vertices_sub - gt_dest_mano_output.joints[:, :1, :]

        gt_rgb_vertices_sub = self.mesh_sampler.downsample(gt_rgb_mano_output.vertices)
        gt_rgb_root = gt_rgb_mano_output.joints[:, 0, :]
        gt_rgb_3d_joints = gt_rgb_mano_output.joints - gt_rgb_root[:, None, :]
        gt_rgb_vertices = gt_rgb_mano_output.vertices - gt_rgb_root[:, None, :]
        gt_rgb_vertices_sub = gt_rgb_vertices_sub - gt_rgb_root[:, None, :]

        pred_3d_joints_from_mesh = self.mano_layer.get_3d_joints(preds['pred_vertices'])

        # todo directly predicted 3d joints loss
        loss_3d_joints = self.get_3d_joints_loss(gt_dest_3d_joints, preds['pred_3d_joints'])
        # todo predicted (from mesh) 3d joints loss
        loss_3d_joints_reg = self.get_3d_joints_loss(gt_dest_3d_joints, pred_3d_joints_from_mesh)
        # todo vertices loss here
        loss_vertices = self.get_vertices_loss(gt_dest_vertices, preds['pred_vertices'])
        loss_vertices_sub = self.get_vertices_loss(gt_dest_vertices_sub, preds['pred_vertices_sub'])

        # todo sum up the losses
        loss_sum = self.config['exper']['loss']['vertices'] * loss_vertices +\
            self.config['exper']['loss']['vertices_sub'] * loss_vertices_sub +\
            self.config['exper']['loss']['3d_joints'] * loss_3d_joints +\
            self.config['exper']['loss']['3d_joints_from_mesh'] * loss_3d_joints_reg
        loss_items = {
            'loss_vertices': loss_vertices,
            'loss_vertices_sub': loss_vertices_sub,
            'loss_3d_joints': loss_3d_joints,
            'loss_3d_joints_reg': loss_3d_joints_reg,
        }
        return loss_sum, loss_items
