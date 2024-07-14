import torch
import numpy as np
# import open3d as o3d
import math
import torch.nn as nn
import torch.nn.functional as F
import time
from scipy.optimize import linear_sum_assignment
# import trimesh
import time
from tqdm import tqdm
import utils.utils as utils

class ContactGraspNetLoss(nn.Module):
    def __init__(self, config):
        super(ContactGraspNetLoss, self).__init__()
        self.config = config['MODEL']
        bin_weights = self.config['bin_weights']
        self.bin_weights = torch.tensor(bin_weights)
        self.bin_vals = self._get_bin_vals()
        
        gripper_control_points = utils.get_dc6d_control_point_tensor(1, use_torch=True, symmetric=False)
        gripper_control_points_sym = utils.get_dc6d_control_point_tensor(1, use_torch=True, symmetric=True)
        self.gripper_control_points = gripper_control_points
        self.gripper_control_points_sym = gripper_control_points_sym
        
        
    def forward(self, pred, data, epoch, is_train=True):
        """_summary_
        """
        # prediction values
        pred_grasps = pred['pred_grasps'] # (B, N, 4, 4)
        pred_scores = pred['pred_scores'] # (B, N, 1)
        pred_points = pred['pred_points'] # (B, N, 3)
        grasp_width_head = pred['grasp_width_head'].permute(0,2,1) # (B, N, len(bin_vals))
        
        # gt values
        pos_contact_points = data['pos_contact_points'] # (B, M, 3)
        pos_contact_width = data['pos_contact_width'] # (B, M)
        pos_contact_rot = data['pos_contact_rot'] # (B, M, 3, 3)
        pos_contact_trans = data['pos_contact_trans'] # (B, M, 3)
        
        # #* check is the gt values are not wrong
        # for i in range(pred_grasps.shape[0]):
        #     for j in range(128):
        #         gripper_control_points = utils.get_dc6d_control_point_tensor(1, use_torch=False, symmetric=False).squeeze()
        #         gt_rot = pos_contact_rot[i, j].detach().cpu().numpy() # (3, 3)
        #         gt_trans = pos_contact_trans[i, j].detach().cpu().numpy() # (3)
        #         gt_cp = np.matmul(gt_rot, gripper_control_points.T).T + gt_trans
        #         tmp_gt_cp = []
        #         mid_gt_grasp_point = (gt_cp[1] + gt_cp[2]) / 2
        #         tmp_gt_cp.append(gt_cp[0])
        #         tmp_gt_cp.append(mid_gt_grasp_point)
        #         tmp_gt_cp.append(gt_cp[1])
        #         tmp_gt_cp.append(gt_cp[2])
        #         tmp_gt_cp.append(gt_cp[3])
        #         tmp_gt_cp.append(gt_cp[4])
        #         gt_cp = np.asarray(tmp_gt_cp)
        #         gt_line_set = o3d.geometry.LineSet()
        #         gt_line_set.points = o3d.utility.Vector3dVector(gt_cp)
        #         gt_line_set.lines = o3d.utility.Vector2iVector([[3,5], [2,4], [2,3], [0,1]])
        #         gt_line_set.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0] for i in range(4)]))
                
        #         pc = data['pc'][i].detach().cpu().numpy()
        #         pc_o3d = o3d.geometry.PointCloud()
        #         pc_o3d.points = o3d.utility.Vector3dVector(pc)
        #         o3d.visualization.draw_geometries([gt_line_set, pc_o3d])

        #     exit()
        
        
        width_label, grasp_success_label, contact_rot_label, contact_trans_label = self._compute_labels(pred_points=pred_points,
                                                                                                        pos_contact_points=pos_contact_points,
                                                                                                        pos_contact_width=pos_contact_width,
                                                                                                        pos_contact_rot=pos_contact_rot,
                                                                                                        pos_contact_trans=pos_contact_trans)
                
        #* check how many success label is in train data after compute label
        # print(grasp_success_label.shape)
        # tmp_grasp_success_label = grasp_success_label.squeeze(2)
        # pos_idx = torch.nonzero(tmp_grasp_success_label, as_tuple=False)
        # print(pos_idx.shape)
        
        min_geom_loss_divisor = float(1)
        pos_grasps_in_view = torch.clamp(grasp_success_label.sum(dim=1), min=min_geom_loss_divisor) # (B, )
        #* Grasp Confidence Loss
        
        bin_ce_loss = F.binary_cross_entropy(pred_scores, grasp_success_label, reduction='none') # (B, N, 1)
        
        # a number of positive grasp success label is too small, we do not compute the grasp confidence loss for all points
        # sample the negative grasp success label to balance the number of positive and negative grasp success label
        # then select bin_ce_loss using sampled neg grasp idx
        bin_ce_loss_tmp = 0
        negative_grasp_success_label = 1 - grasp_success_label
        for i in range(grasp_success_label.shape[0]):
            neg_grasp_idx = torch.nonzero(negative_grasp_success_label[i].squeeze(1), as_tuple=False)
            # sampled neg grasp idx same as pos_grrasp_in_view
            sampled_neg_grasp_idx = torch.randperm(neg_grasp_idx.shape[0])[:int(pos_grasps_in_view[i])]
            sampled_neg_grasp_idx = neg_grasp_idx[sampled_neg_grasp_idx]
            pos_grasp_idx = torch.nonzero(grasp_success_label[i].squeeze(1), as_tuple=False)
            if pos_grasp_idx.shape[0] == 0:
                pos_grasp_idx = torch.randperm(neg_grasp_idx.shape[0])[:int(pos_grasps_in_view[i])]
                pos_grasp_idx = neg_grasp_idx[pos_grasp_idx]

            sampled_grasp_idx = torch.cat([pos_grasp_idx, sampled_neg_grasp_idx], dim=0).squeeze(1)

            sampled_bin_ce_loss = bin_ce_loss[i, sampled_grasp_idx]
            sampled_bin_ce_loss = torch.mean(sampled_bin_ce_loss)
            bin_ce_loss_tmp += sampled_bin_ce_loss
            
        bin_ce_loss = bin_ce_loss_tmp / grasp_success_label.shape[0]
        
        # print('pred_score', pred_scores[0])
        # print(torch.nonzero(pred_scores[0] >= 0.5, as_tuple=False).shape)
        # if pred_scores.device == torch.device('cuda:0'):
        #     print('pred grasp True: ', torch.nonzero(pred_scores[0].squeeze(1) >= 0.5, as_tuple=False).shape[0], '  pred grasp False: ', torch.nonzero(pred_scores[0].squeeze(1) < 0.5, as_tuple=False).shape[0])
        #     print('gt grasp True: ', torch.nonzero(grasp_success_label[0].squeeze(1) == 1, as_tuple=False).shape[0], '   gt grasp False: ', torch.nonzero(grasp_success_label[0].squeeze(1) == 0, as_tuple=False).shape[0])
        #     print('-------------------------------------------')
                
                
        # print('pred grasp False', torch.nonzero(pred_scores[0].squeeze(1) < 0.5, as_tuple=False).shape[0])
        
        # print('pred grasp True: ', torch.nonzero(pred_scores[0] >= 0.5, as_tuple=False).shape)
        # top k confidence
        # if is_train:
        #     bin_ce_loss, ce_topk_idx = torch.topk(bin_ce_loss.squeeze(), k=256)
        #     bin_ce_loss = torch.mean(bin_ce_loss)
        #     if pred_scores.device == torch.device('cuda:0'):
        #         print('topk pred grasp True: ', torch.nonzero(pred_scores[0].squeeze(1)[ce_topk_idx[0]] >= 0.5, as_tuple=False).shape[0], '  topk pred grasp False: ', torch.nonzero(pred_scores[0].squeeze(1)[ce_topk_idx[0]] < 0.5, as_tuple=False).shape[0])
        #         print('topk gt grasp True: ', torch.nonzero(grasp_success_label[0].squeeze(1)[ce_topk_idx[0]] == 1, as_tuple=False).shape[0], '   topk gt grasp False: ', torch.nonzero(grasp_success_label[0].squeeze(1)[ce_topk_idx[0]] == 0, as_tuple=False).shape[0])
        #         print('-------------------------------------------')
        # else:
        #     bin_ce_loss = torch.mean(bin_ce_loss)
        
        # no top_k confidence
        # bin_ce_loss = torch.mean(torch.mean(bin_ce_loss.squeeze(2), dim=-1), dim=-1)
        
        #* Grasp Width Loss
        # convert to multihot
        bin_vals = self.config['offset_bins']
        grasp_width_labels_multihot = self._bin_labels_to_multihot(width_label, bin_vals) # (B, N, 10)
        width_loss = F.binary_cross_entropy_with_logits(grasp_width_head,
                                                        grasp_width_labels_multihot,
                                                        reduction='none') # (B, N, 1)
        bin_weights = self.bin_weights[None, None, :].cuda()
        width_loss = (bin_weights * width_loss).mean(axis=2)
        masked_width_loss = width_loss * grasp_success_label.squeeze()
        width_loss = torch.mean(torch.sum(masked_width_loss, axis=1, keepdim=True) / pos_grasps_in_view)
        
        
        #* Grasp 6D Pose Loss
        # select positive grasp
        success_mask_rot = grasp_success_label.bool()[:, :, :, None] # (B, N, 1, 1)
        success_mask_rot = torch.broadcast_to(success_mask_rot, contact_rot_label.shape) # (B, N, 3, 3)
        pos_contact_rot_label = torch.where(success_mask_rot, contact_rot_label, torch.ones_like(contact_rot_label)*100000) # (B, N, 3, 3)
        success_mask_trans = grasp_success_label.bool() # (B, N, 1)
        success_mask_trans = torch.broadcast_to(success_mask_trans, contact_trans_label.shape) # (B, N, 3)
        pos_contact_trans_label = torch.where(success_mask_trans, contact_trans_label, torch.ones_like(contact_trans_label)*100000) # (B, N, 3)
        
        # expand gripper control points to match the number of the points
        self.gripper_control_points = self.gripper_control_points.cuda()
        self.gripper_control_points_sym = self.gripper_control_points_sym.cuda()
        
        control_points = self.gripper_control_points.unsqueeze(1) # (1, 1, 5, 3)
        control_points = control_points.repeat(pred_points.shape[0], pred_points.shape[1], 1, 1) # (B, N, 5, 3)
        sym_control_points = self.gripper_control_points_sym.unsqueeze(1) # (1, 1, 5, 3)
        sym_control_points = sym_control_points.repeat(pred_points.shape[0], pred_points.shape[1], 1, 1) # (B, N, 5, 3)
        
        # transform pred and gt control points
        pred_contact_rot = pred_grasps[:, :, :3, :3] # (B, N, 3, 3)
        pred_contact_trans = pred_grasps[:, :, :3, 3] # (B, N, 3)
        pred_control_points = torch.matmul(pred_contact_rot, control_points.permute(0,1,3,2)).permute(0,1,3,2) + pred_contact_trans.unsqueeze(2)
        gt_control_points = torch.matmul(pos_contact_rot_label, control_points.permute(0,1,3,2)).permute(0,1,3,2) + pos_contact_trans_label.unsqueeze(2)
        sym_gt_control_points = torch.matmul(pos_contact_rot_label, sym_control_points.permute(0,1,3,2)).permute(0,1,3,2) + pos_contact_trans_label.unsqueeze(2)
        
        # #* visualize pred control points and gt control points with pred points
        # for i in range(pred_control_points.shape[0]):
        #     pred_line_sets = []
        #     gt_line_sets = []
        #     for j in range(128):
        #         if grasp_success_label[i, j] == 0:
        #             continue
        #         else:
        #             pred_cp = pred_control_points[i, j].detach().cpu().numpy()
        #             gt_cp = gt_control_points[i, j].detach().cpu().numpy() # (5, 3)
        #             tmp_pred_cp = []
        #             mid_pred_grasp_point = (pred_cp[1] + pred_cp[2]) / 2
        #             tmp_pred_cp.append(pred_cp[0])
        #             tmp_pred_cp.append(mid_pred_grasp_point)
        #             tmp_pred_cp.append(pred_cp[1])
        #             tmp_pred_cp.append(pred_cp[2])
        #             tmp_pred_cp.append(pred_cp[3])
        #             tmp_pred_cp.append(pred_cp[4])
        #             pred_cp = np.asarray(tmp_pred_cp)
        #             tmp_gt_cp = []
        #             mid_gt_grasp_point = (gt_cp[1] + gt_cp[2]) / 2
        #             tmp_gt_cp.append(gt_cp[0])
        #             tmp_gt_cp.append(mid_gt_grasp_point)
        #             tmp_gt_cp.append(gt_cp[1])
        #             tmp_gt_cp.append(gt_cp[2])
        #             tmp_gt_cp.append(gt_cp[3])
        #             tmp_gt_cp.append(gt_cp[4])
        #             gt_cp = np.asarray(tmp_gt_cp)
                    
        #             pred_line_set = o3d.geometry.LineSet()
        #             pred_line_set.points = o3d.utility.Vector3dVector(pred_cp)
        #             pred_line_set.lines = o3d.utility.Vector2iVector([[3,5], [2,4], [2,3], [0,1]])
        #             pred_line_set.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0] for i in range(4)]))
        #             pred_line_sets.append(pred_line_set)
        #             gt_line_set = o3d.geometry.LineSet()
        #             gt_line_set.points = o3d.utility.Vector3dVector(gt_cp)
        #             gt_line_set.lines = o3d.utility.Vector2iVector([[3,5], [2,4], [2,3], [0,1]])
        #             gt_line_set.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0] for i in range(4)]))
        #             gt_line_sets.append(gt_line_set)
                    
        #             pc = data['pc'][i].detach().cpu().numpy()
        #             pc_o3d = o3d.geometry.PointCloud()
        #             pc_o3d.points = o3d.utility.Vector3dVector(pc)
                    
        #             # print pred width and gt width
        #             # print('pred width: ', grasp_width_head[i, j])
        #             print('pred width idx: ', torch.argmax(grasp_width_head[i, j]))
                    
        #             print('gt width idx: ', torch.argmax(grasp_width_labels_multihot[i, j]))
                    
                    
        #             o3d.visualization.draw_geometries([gt_line_set, pc_o3d, pred_line_set])
        #     exit()
        
        
        
        # compute dist btw pred and gt control points
        expanded_pred_control_points = pred_control_points.unsqueeze(2) # (B, N, 1, 5, 3)
        expanded_gt_control_points = gt_control_points.unsqueeze(1) # (B, 1, M, 5, 3)
        expanded_sym_gt_control_points = sym_gt_control_points.unsqueeze(1) # (B, 1, M, 5, 3)
        
        # sum of squared dist btw all points
        squared_add = torch.sum((expanded_pred_control_points - expanded_gt_control_points)**2, dim=(3,4)) # (B, N, M)
        sym_squared_add = torch.sum((expanded_pred_control_points - expanded_sym_gt_control_points)**2, dim=(3,4)) # (B, N, M

        # combine dist btw gt and sym gt grasp and take min dist to gt grasp for each pred grasp
        squared_adds = torch.concat([squared_add, sym_squared_add], dim=-1) # (B, N, 2*M)
        squared_adds_k = torch.topk(squared_adds, k=1, dim=2, largest=False, sorted=False)[0] # (B, N, 1)

        # mask negative grasp
        sum_grasp_success_labels = torch.sum(grasp_success_label, dim=2, keepdim=True) # (B, N, 1)
        binary_grasp_success_labels = torch.clamp(sum_grasp_success_labels, 0, 1)
        min_adds = binary_grasp_success_labels * torch.sqrt(squared_adds_k) # (B, N, 1)
        adds_loss = torch.sum(pred_scores * min_adds, dim=(1), keepdim=True) # (B, 1, 1)
        adds_loss = adds_loss.squeeze() / pos_grasps_in_view.squeeze() # B
        adds_loss = torch.mean(adds_loss)
        total_loss = 1*bin_ce_loss + 1*width_loss + 3*adds_loss
        # total_loss = bin_ce_loss
        # loss_info = {
        #     'bin_ce_loss': bin_ce_loss,
        #     'width_loss': width_loss,
        #     'adds_loss': adds_loss
        # }
        return total_loss, bin_ce_loss, width_loss, adds_loss
        
    
    def _get_bin_vals(self):
        """
        Creates bin values for grasping widths according to bounds defined in config
        """
        bins_bounds = np.array(self.config['offset_bins'])
        bin_vals = (bins_bounds[1:] + bins_bounds[:-1]) / 2
        bin_vals[-1] = bins_bounds[-1]
        bin_vals = np.minimum(bin_vals, self.config['gripper_width']-0.005)
        bin_vals = torch.tensor(bin_vals, dtype=torch.float32)
        
        return bin_vals

    def _compute_labels(self, pred_points, pos_contact_points, pos_contact_width, pos_contact_rot, pos_contact_trans):
        """_summary_
        Project gt grasp labels on the predicted points
        All points w/o nearby successful grasp contact are considered negative contact points
        
        Args:
            pred_points (_type_): _description_
            pos_contact_points (_type_): _description_
            pos_contact_width (_type_): _description_
            pos_contact_rot (_type_): _description_
            pos_contact_trans (_type_): _description_

        Returns:
            _type_: _description_
        """
        nsample = 1
        radius = 0.005
        filter_z = True
        z_val = -0.1
        
        _, N, _ = pred_points.shape
        B, M, _ = pos_contact_points.shape
        
        # make grasp width B, M, 1
        pos_contact_width = pos_contact_width[:, :, None]
        
        # if filter_z:
        #     # filter out direction that are too far
        
        # compute distance
        pred_points = pred_points.unsqueeze(2) # (B, N, 1, 3)
        pos_contact_points = pos_contact_points.unsqueeze(1) # (B, 1, M, 3)
        squared_dist = torch.sum((pred_points - pos_contact_points)**2, dim=-1) # (B, N, M)
        squared_dist_k, close_contact_pt_idcs = torch.topk(squared_dist, k=nsample, dim=2, largest=False, sorted=False) # (B, N, nsample)
        
        # group labels
        grouped_contact_width = utils.index_points(pos_contact_width, close_contact_pt_idcs) # (B, N, nsample, 1)
        grouped_contact_rot = utils.index_points(pos_contact_rot, close_contact_pt_idcs) # (B, N, nsample, 3, 3)
        grouped_contact_trans = utils.index_points(pos_contact_trans, close_contact_pt_idcs) # (B, N, nsample, 3)
        # compute label
        width_label = grouped_contact_width.mean(dim=2) # (B, N, 3)
        grasp_success_label = torch.mean(squared_dist_k, dim=2, keepdim=True) < radius**2 # (B, N, 1)
        grasp_success_label = grasp_success_label.type(torch.float32)
        grasp_contact_rot_label = grouped_contact_rot.mean(dim=2) # (B, N, 3, 3)
        grasp_contact_trans_label = grouped_contact_trans.mean(dim=2) # (B, N, 3)
        
        return width_label, grasp_success_label, grasp_contact_rot_label, grasp_contact_trans_label
        
    def _bin_labels_to_multihot(self, cont_labels, bin_boundaries):
        """_summary_
        Computes binned grasp width labels from continous labels and bin boundaries
        
        Args:
            garsp_width_labels (_type_): _description_
            bin_vals (_type_): _description_

        Returns:
            _type_: _description_
        """
        bins = []
        for b in range(len(bin_boundaries)-1):
            bins.append(torch.logical_and(torch.greater_equal(cont_labels, bin_boundaries[b]), torch.less(cont_labels, bin_boundaries[b+1])))
        multi_hot_labels = torch.cat(bins, dim=2)
        multi_hot_labels = multi_hot_labels.to(torch.float32).cuda()
        
        return multi_hot_labels
