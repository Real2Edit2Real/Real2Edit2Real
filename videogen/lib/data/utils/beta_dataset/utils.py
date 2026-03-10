import torch
import numpy as np
import random
# from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix

import kornia
def depth_to_canny(depth, low_thresh=0.0, high_thresh=0.5):
    """
    Input: depth (B, 3, H, W), value range [-1, 1]
    Output: edges (B, 3, H, W), binary image with value range [-1, 1]
    """
    if depth.shape[1] != 3:
        raise ValueError("Input depth map must be 3-channel (B, 3, H, W)")

    # Take single channel
    depth_single = depth[:, 0:1, :, :]   # (B, 1, H, W)

    # Normalize to [0, 1]
    depth_norm = (depth_single + 1) / 2.0

    # Canny detection
    canny = kornia.filters.Canny(low_thresh, high_thresh)
    _, edges = canny(depth_norm)  # edges in [0, 1]

    # Convert to [-1, 1]
    edges = edges * 2 - 1  # (B, 1, H, W)

    # Repeat to 3 channels
    edges = edges.repeat(1, 3, 1, 1)  # (B, 3, H, W)

    return edges


def add_depth_noise(depth, mode="gaussian", **kwargs):
    """
    Add noise to batched depth images (supports torch.Tensor)

    Args:
        depth (torch.Tensor): Input depth map, shape (B, 3, H, W), float
        mode (str): "gaussian" or "missing"
        kwargs: Noise parameters
            - gaussian: sigma (float, default=0.01)
            - missing: drop_prob (float, default=0.05), missing_val (default=-1)
    Returns:
        noisy_depth (torch.Tensor): Depth map after adding noise, shape (B, 3, H, W)
    """
    if not torch.is_tensor(depth):
        raise TypeError("depth must be torch.Tensor")
    if depth.shape[1] != 3:
        raise ValueError("Input depth map must be 3-channel (B, 3, H, W)")

    # Take single channel
    depth_single = depth[:, 0, :, :]  # (B, H, W)

    if mode == "gaussian":
        sigma = kwargs.get("sigma", 0.01)
        noise = torch.randn_like(depth_single) * sigma
        noisy = depth_single + noise

    elif mode == "missing":
        drop_prob = kwargs.get("drop_prob", 0.05)
        missing_val = kwargs.get("missing_val", -1)
        mask = torch.rand_like(depth_single) < drop_prob
        noisy = depth_single.masked_fill(mask, missing_val)

    else:
        raise ValueError(f"Unknown noise mode: {mode}")
    
    noisy = torch.clamp(noisy, min=-1, max=1)  # (B, H, W)
    # Copy back to 3 channels
    noisy = noisy.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, 3, H, W)

    return noisy


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Copied from https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L43C1-L72C54

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def gen_batch_ray_parellel(intrinsic,c2w,W,H):
    batch_size = intrinsic.shape[0]
    
    fx, fy, cx, cy = intrinsic[:,0,0].unsqueeze(1).unsqueeze(2), intrinsic[:,1,1].unsqueeze(1).unsqueeze(2), intrinsic[:,0,2].unsqueeze(1).unsqueeze(2), intrinsic[:,1,2].unsqueeze(1).unsqueeze(2)
    i, j = torch.meshgrid(torch.linspace(0.5, W-0.5, W, device=c2w.device), torch.linspace(0.5, H-0.5, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    i = i.unsqueeze(0).repeat(batch_size,1,1)
    j = j.unsqueeze(0).repeat(batch_size,1,1)
    dirs = torch.stack([(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:,np.newaxis,np.newaxis, :3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_o = c2w[:, :3, -1].unsqueeze(1).unsqueeze(2).repeat(1,H,W,1)
    viewdir = rays_d/torch.norm(rays_d,dim=-1,keepdim=True)
    return rays_d, rays_o, viewdir


def intrinsic_transform(intrinsic, original_res, size, transform_mode):
    fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
    original_height = original_res[0]
    original_width = original_res[1]
    if transform_mode == 'resize':
        resize_height = size[0]
        resize_width = size[1]

        scale_height = resize_height / original_height
        scale_width = resize_width / original_width

        fx_new = fx * scale_width
        fy_new = fy * scale_height
        cx_new = cx * scale_width
        cy_new = cy * scale_height

    elif transform_mode == 'center_crop_resize':
        if original_height <= original_width:
            scale_ratio = min(size) / original_height
        else:
            scale_ratio = min(size) / original_width
        resize_height = scale_ratio * original_height
        resize_width = scale_ratio * original_width

        fx_new = fx * scale_ratio
        fy_new = fy * scale_ratio
        cx_new = cx * scale_ratio
        cy_new = cy * scale_ratio

        crop_height = size[0]
        crop_width = size[1]
        cx_new = cx_new * (crop_width / resize_width)
        cy_new = cy_new * (crop_height / resize_height)
    
    else:
        raise NotImplementedError('No such transformation mode for image!')
    
    return torch.tensor([[fx_new, 0, cx_new],
                         [0, fy_new, cy_new],
                         [0, 0, 1]])




def intrinsic_transform_batch(intrinsic, original_res, size, transform_mode):
    b = intrinsic.shape[0]
    fx, fy, cx, cy = intrinsic[:,0,0], intrinsic[:,1,1], intrinsic[:,0,2], intrinsic[:,1,2]
    original_height = original_res[0]
    original_width = original_res[1]
    if transform_mode == 'resize':
        resize_height = size[0]
        resize_width = size[1]

        scale_height = resize_height / original_height
        scale_width = resize_width / original_width

        fx_new = fx * scale_width
        fy_new = fy * scale_height
        cx_new = cx * scale_width
        cy_new = cy * scale_height

    elif transform_mode == 'center_crop_resize':
        if original_height <= original_width:
            scale_ratio = min(size) / original_height
        else:
            scale_ratio = min(size) / original_width
        resize_height = scale_ratio * original_height
        resize_width = scale_ratio * original_width

        fx_new = fx * scale_ratio
        fy_new = fy * scale_ratio
        cx_new = cx * scale_ratio
        cy_new = cy * scale_ratio

        crop_height = size[0]
        crop_width = size[1]
        cx_new = cx_new * (crop_width / resize_width)
        cy_new = cy_new * (crop_height / resize_height)
    
    else:
        raise NotImplementedError('No such transformation mode for image!')
    

    fx_expanded = fx_new
    fy_expanded = fy_new
    cx_expanded = cx_new
    cy_expanded = cy_new

    intrinsic_matrices = torch.zeros((b, 3, 3), dtype=fx.dtype, device=fx.device)
    intrinsic_matrices[:, 0, 0] = fx_expanded
    intrinsic_matrices[:, 1, 1] = fy_expanded
    intrinsic_matrices[:, 0, 2] = cx_expanded
    intrinsic_matrices[:, 1, 2] = cy_expanded
    intrinsic_matrices[:, 2, 2] = 1
    
    return intrinsic_matrices


def gen_crop_config(tensor):
    _, _, h, w = tensor.shape
    h_start = random.randint(0,h//8)
    w_start = random.randint(0,w//8)
    h_crop = random.randint(7*h//8,h-h_start)
    w_crop = random.randint(7*w//8,w-w_start)
    return h_start, w_start, h_crop, w_crop


def crop_tensor(tensor, h_start, w_start, h_crop, w_crop):
    cropped_tensor = tensor[:,:,h_start:h_start+h_crop,w_start:w_start+w_crop]
    return cropped_tensor


def intrin_crop_transform(intrinsic, h_start, w_start):
    fx, fy, cx, cy = intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2]
    cx_new = cx - w_start
    cy_new = cy - h_start
    return torch.tensor([[fx,0,cx_new],[0,fy,cy_new],[0,0,1]])

def get_transformation_matrix_from_quat(xyz_quat):
    ### xyz_quat: tensor, (b, 7)
    rot_quat = xyz_quat[:, 3:]
    ### in pytorch3d, quaternion_to_matrix takes wxyz-quat as input
    rot_quat = rot_quat[:, [3,0,1,2]]
    rot = quaternion_to_matrix(rot_quat)
    trans = xyz_quat[:, :3]
    output = torch.eye(4).unsqueeze(0).repeat(xyz_quat.shape[0], 1, 1)
    output[:,:3,:3] = rot
    output[:,:3, 3] = trans
    return output