import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from options.config_utils import load_config
# import pointnet2_ops.pointnet2_modules as pointnet2
from timm.models.layers import DropPath, trunc_normal_
from .build import MODELS
from models.losses import ContactGraspNetLoss
                        
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'Pointnet_Pointnet2_pytorch'))
from Pointnet_Pointnet2_pytorch.models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

@MODELS.register_module()
class ContactGraspNet(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.cfg = load_config('base_model_param_cgnet.yaml')
        self.model_cfg = self.cfg['MODEL']
        
        self.sa_module_cfg = self.model_cfg['pointnet_sa_modules']
        self.fp_module_cfg = self.model_cfg['pointnet_fp_modules']
        self.offset_bins = self.model_cfg['offset_bins']
        
        
        self.sa1_module = PointNetSetAbstractionMsg(
            npoint=self.sa_module_cfg[0]['npoint'],
            radius_list=self.sa_module_cfg[0]['radius_list'],
            nsample_list=self.sa_module_cfg[0]['nsample_list'],
            mlp_list=self.sa_module_cfg[0]['mlp_list'],
            in_channel=0
        )
        sa1_out_channel = sum([self.sa_module_cfg[0]['mlp_list'][i][-1] for i in range(len(self.sa_module_cfg[0]['mlp_list']))])
        
        self.sa2_module = PointNetSetAbstractionMsg(
            npoint=self.sa_module_cfg[1]['npoint'],
            radius_list=self.sa_module_cfg[1]['radius_list'],
            nsample_list=self.sa_module_cfg[1]['nsample_list'],
            mlp_list=self.sa_module_cfg[1]['mlp_list'],
            in_channel=sa1_out_channel
        )
        sa2_out_channel = sum([self.sa_module_cfg[1]['mlp_list'][i][-1] for i in range(len(self.sa_module_cfg[1]['mlp_list']))])
        
        self.sa3_module = PointNetSetAbstractionMsg(
            npoint=self.sa_module_cfg[2]['npoint'],
            radius_list=self.sa_module_cfg[2]['radius_list'],
            nsample_list=self.sa_module_cfg[2]['nsample_list'],
            mlp_list=self.sa_module_cfg[2]['mlp_list'],
            in_channel=sa2_out_channel
        )
        sa3_out_channel = sum([self.sa_module_cfg[2]['mlp_list'][i][-1] for i in range(len(self.sa_module_cfg[2]['mlp_list']))])

        self.sa4_module = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            mlp=self.model_cfg['pointnet_sa_module']['mlp'],
            in_channel=3+sa3_out_channel,
            group_all=self.model_cfg['pointnet_sa_module']['group_all']
        )
        
        self.fp1_module = PointNetFeaturePropagation(mlp=self.fp_module_cfg[0]['mlp'], 
                                                     in_channel=self.fp_module_cfg[1]['mlp'][-1]+sa1_out_channel)
        self.fp2_module = PointNetFeaturePropagation(mlp=self.fp_module_cfg[1]['mlp'],
                                                     in_channel=self.fp_module_cfg[2]['mlp'][-1]+sa2_out_channel)
        self.fp3_module = PointNetFeaturePropagation(mlp=self.fp_module_cfg[2]['mlp'],
                                                     in_channel=self.model_cfg['pointnet_sa_module']['mlp'][-1]+sa3_out_channel)
        
        #* define grasp prediction heads
        # head for grasp direction
        self.grasp_dir_head = nn.Sequential(
            nn.Conv1d(self.fp_module_cfg[0]['mlp'][-1], 128, 1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 3, 1, padding=0)
        )
        # head for grasp approach
        self.grasp_app_head = nn.Sequential(
            nn.Conv1d(self.fp_module_cfg[0]['mlp'][-1], 128, 1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 3, 1, padding=0)
        )
        # head for garsp width
        self.grasp_width_head = nn.Sequential(
            nn.Conv1d(self.fp_module_cfg[0]['mlp'][-1], 128, 1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, len(self.offset_bins)-1, 1, padding=0)
        )
        # head for contact points
        self.binary_seg_head = nn.Sequential(
            nn.Conv1d(self.fp_module_cfg[0]['mlp'][-1], 128, 1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, 1, 1, padding=0)
        )
        self.build_loss_func()

    
    def build_loss_func(self):
        # self.loss_func = L1_Distance()
        # self.loss_func = Bi_Grasp_Chamfer_Distance_w_Quality_v2()
        self.loss_func = ContactGraspNetLoss(self.cfg)

    def get_loss(self, pred, data, epoch, is_train=True):
        
        loss = self.loss_func(pred, data, epoch, is_train=is_train)
        
        return loss
    
    def forward(self, pc):
        
        if self.model_cfg['raw_num_points'] != self.model_cfg['ndataset_points']:
            inds = np.random.choice(pc.shape[1], self.model_cfg['ndataset_points'], replace=False)
            pc = pc[:, inds, :]
        # print('pc shape: ', pc.shape)
        
        
        pc = pc.permute(0, 2, 1).contiguous() # (B, N, 3) -> (B, 3, N)
        l0_xyz = pc[:, :3, :]
        l0_points = None
        
        #* SA Layers
        l1_xyz, l1_points = self.sa1_module(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2_module(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3_module(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4_module(l3_xyz, l3_points)
        
        #* FP Layers
        l3_points = self.fp3_module(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp2_module(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp1_module(l1_xyz, l2_xyz, l1_points, l2_points)
        
        l0_points = l1_points
        pred_points = l1_xyz

        #* grasp heads
        # grasp dir head
        grasp_dir_head = self.grasp_dir_head(l0_points)
        grasp_dir_head_normed = F.normalize(grasp_dir_head, p=2, dim=1)
        # grasp app head
        grasp_app_head = self.grasp_app_head(l0_points) # (1, 3, 2048)=(B, 3, N)
        grasp_app_head = grasp_app_head - torch.sum(grasp_dir_head_normed * grasp_app_head, dim=1, keepdim=True) * grasp_dir_head_normed
        grasp_app_head_orthog = F.normalize(grasp_app_head, p=2, dim=1)
        # grasp width head
        grasp_width_head = self.grasp_width_head(l0_points)
        # binary seg head
        binary_seg_head = self.binary_seg_head(l0_points)
        
        grasp_bin_val = self.get_bin_vals() # (num_bins)
        argmax_indices = torch.argmax(grasp_width_head, dim=1, keepdim=True) # (B, 1, N)
        width_bin_pred_vals = grasp_bin_val[argmax_indices]
        
        # get pred grasp transformation
        pred_grasps = self.build_6d_grasp(grasp_app_head_orthog.permute(0, 2, 1),
                                          grasp_dir_head_normed.permute(0, 2, 1),
                                          pred_points.permute(0, 2, 1),
                                          width_bin_pred_vals.permute(0, 2, 1),
                                          use_torch=True)
        # get pred score
        pred_scores = torch.sigmoid(binary_seg_head).permute(0,2,1)
        # pred_scores = binary_seg_head.permute(0, 2, 1)
        # get pred points
        pred_points = pred_points.permute(0, 2, 1)
        # get pred width
        pred_width = width_bin_pred_vals.permute(0, 2, 1)
        
        pred = dict(
            pred_grasps=pred_grasps,
            pred_scores=pred_scores,
            pred_points=pred_points,
            pred_width=pred_width,
            grasp_width_head=grasp_width_head # for loss compute
        )

        return pred
    
        
    def get_bin_vals(self):
        """
        
        Create bin values for grasp width prediction defined in config file
        
        """
        
        bins_bounds = np.array(self.model_cfg['offset_bins'])
        bin_vals = (bins_bounds[1:] + bins_bounds[:-1]) / 2
        bin_vals[-1] = bins_bounds[-1]
        bin_vals = np.minimum(bin_vals, self.model_cfg['gripper_width']-0.005)
        bin_vals = torch.tensor(bin_vals, dtype=torch.float32).cuda()
        
        return bin_vals
    
    def build_6d_grasp(self, grasp_app, grasp_dir, contact_pts, thickness, use_torch=False, base_depth=0.02):
        """
        Build 6-Dof grasp + width from point wise network prediciton
        """
        if use_torch:
            grasp_R = torch.stack([grasp_dir, torch.cross(grasp_app, grasp_dir), grasp_app], dim=3) # (B, N, 3, 3)
            grasp_t = contact_pts.unsqueeze(3)
            ones = torch.ones((contact_pts.shape[0], contact_pts.shape[1], 1, 1), dtype=torch.float32).to(grasp_t.device)
            zeros = torch.zeros((contact_pts.shape[0], contact_pts.shape[1], 1, 3), dtype=torch.float32).to(grasp_t.device)
            homog_vec = torch.cat([zeros, ones], dim=3) # (B, N, 1, 4)
            grasps = torch.cat([torch.cat([grasp_R, grasp_t], dim=3), homog_vec], dim=2) # (B, N, 4, 4)
        else:
            grasps = []
            for i in range(len(contact_pts)):
                grasp = np.eye(4)
                grasp[:3, 0] = grasp_dir[i] / np.linalg.norm(grasp_dir[i])
                grasp[:3, 2] = grasp_app[i] / np.linalg.norm(grasp_app[i])
                grasp_y = np.cross(grasp[:3, 2], grasp[:3, 0])
                grasp[:3, 1] = grasp_y / np.linalg.norm(grasp_y)
                grasp[:3, 3] = contact_pts[i]
                grasps.append(grasp)
            grasps = np.array(grasps)
        
        return grasps