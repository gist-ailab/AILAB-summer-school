import torch
import numpy as np

def index_points(points, idx):
    """
    From pointnet2 repo

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def get_dc6d_control_point_tensor(batch_size, use_torch=True, device="cpu", symmetric=False):
    if not symmetric:
        control_points = np.array([[-0.044, 0, 0],[-0.02, -0.08, 0], [-0.02, 0.08, 0],
                                   [0.01999, -0.08, 0], [0.01999, 0.08, 0]])
    else:
        control_points = np.array([[-0.044, 0, 0], [-0.02, 0.08, 0], [-0.02, -0.08, 0],
                                   [0.01999, 0.08, 0], [0.01999, -0.08, 0]])
    control_points = np.asarray(control_points, dtype=np.float32)
    control_points = np.tile(np.expand_dims(control_points, 0), [batch_size, 1, 1])
    if use_torch:
        return torch.tensor(control_points)
    return control_points