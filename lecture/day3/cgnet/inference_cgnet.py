import torch
import cgnet.utils.utils as utils
import numpy as np
from cgnet.utils.collision_detector import ModelFreeCollisionDetector

def inference_cgnet(pc, model, device):
    pc_torch = torch.from_numpy(pc).float().to(device)
    pred = model(pc_torch.unsqueeze(0))
    pred_grasps = pred['pred_grasps'].detach().cpu() #(B, N, 4, 4)
    pred_scores = pred['pred_scores'] # (B, N)
    pred_points = pred['pred_points'].detach().cpu() # (B, N, 3)
    pred_graps_width_bin = pred['pred_width'] # (B, N, 1)

    pred_rot = pred_grasps[:, :, :3, :3] # (B, N, 3, 3)
    pred_trans = pred_grasps[:, :, :3, 3] # (B, N, 3)
    
    sorted_pred_score, sorted_idx = torch.topk(pred_scores.squeeze(), k=2048, largest=True)
    sorted_idx = sorted_idx.detach().cpu()
    sorted_pred_rot = pred_rot[:, sorted_idx, :, :]
    sorted_pred_trans = pred_trans[:, sorted_idx, :]
    sorted_pred_width_bin = pred_graps_width_bin[:, sorted_idx, :]
    
    sorted_pred_rot = sorted_pred_rot.detach().cpu().numpy()[0] #(2048, 3, 3)
    sorted_pred_trans = sorted_pred_trans.detach().cpu().numpy()[0] #(2048, 3)
    sorted_pred_score = sorted_pred_score.detach().cpu().numpy()
    sorted_pred_width_bin = sorted_pred_width_bin.detach().cpu().numpy()[0, :, 0]
    
    
    #* visualize non-filtered grasps, this check model inference is correctely working
    import open3d as o3d
    from graspnetAPI.graspnetAPI import GraspNet, GraspNetEval, GraspGroup, Grasp
    # from graspnetAPI import GraspNet, GraspNetEval, GraspGroup, Grasp
    
    g_array = []
    for i in range(len(sorted_idx)):
        score = sorted_pred_score[i]
        width = sorted_pred_width_bin[i]
        rot = sorted_pred_rot[i] # (3, 3)
        trans = sorted_pred_trans[i].reshape(-1) # (3,)
        
        trans = trans.reshape(1, 3) - (rot @ np.array([[0.21,0,0]]).reshape(3, 1)).reshape(1,3)
        trans = trans.reshape(-1)
        
        rot = rot.reshape(-1)
        g_array.append([score, width, 0.02, 0.02, *rot, *trans, -1])
    
    g_array = np.array(g_array)
    gg = GraspGroup(g_array)
    # check collisoin
    mfcdetector = ModelFreeCollisionDetector(pc, voxel_size=0.005)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
    gg = gg[~collision_mask]
    gg = gg.nms()
    gg = gg.sort_by_score()
    if gg.__len__() > 10:
        gg = gg[:10]
    gg_vis = gg[:1]
    gg_vis_trans = gg[:1]
    gg_vis_trans.translations += (gg_vis_trans.rotation_matrices[0] @ np.array([[0.21, 0, 0]]).reshape(3, 1)).reshape(1,3)
    grippers = gg_vis.to_open3d_geometry_list()
    grippers_trans = gg_vis_trans.to_open3d_geometry_list()
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pc_o3d, *grippers, *grippers_trans])    
    
    tcp_trans = np.array([[[0.21, 0, 0]]]).reshape(1,3,1)
    pre_tcp_trans = np.array([[0.31, 0, 0]]).reshape(1,3,1)
    tcp_trans = np.repeat(tcp_trans, 2048, axis=0)
    pre_tcp_trans = np.repeat(pre_tcp_trans, 2048, axis=0)
    sorted_pred_pre_trans = sorted_pred_trans.copy()
    sorted_pred_trans -= (sorted_pred_rot @ tcp_trans).reshape(-1,3)
    sorted_pred_pre_trans -= (sorted_pred_rot @ pre_tcp_trans).reshape(-1,3)
    
    return sorted_pred_rot, sorted_pred_trans, sorted_pred_pre_trans, sorted_pred_width_bin
    