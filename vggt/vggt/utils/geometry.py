# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
import cv2


from vggt.utils.distortion import apply_distortion, iterative_undistortion, single_undistortion

def inpaint_depth_map(depth_map, invalid_mask=None, method='nearest'):
    """
    Fill invalid depth regions using interpolation.

    Args:
        depth_map (np.ndarray): (H, W), input depth map with holes (0 or NaN).
        invalid_mask (np.ndarray or None): Mask indicating which regions are invalid (True=hole).
        method (str): 'nearest', 'linear', 'median', 'bilateral', etc.

    Returns:
        np.ndarray: Filled depth map.
    """
    depth = depth_map.copy()
    if invalid_mask is None:
        invalid_mask = (depth == 0) | ~np.isfinite(depth)

    # OpenCV's inpaint requires an 8-bit single-channel mask
    mask = invalid_mask.astype(np.uint8)

    # Normalize depth to [0, 1] for inpaint
    max_val = np.nanmax(depth)
    depth_norm = depth / max_val
    depth_norm[invalid_mask] = 0

    if method == 'nearest':
        # Use nearest neighbor fill (fast)
        from scipy.ndimage import distance_transform_edt
        valid = ~invalid_mask
        nearest_idx = np.array(np.nonzero(valid)).T
        invalid_idx = np.array(np.nonzero(invalid_mask)).T
        from scipy.spatial import cKDTree
        tree = cKDTree(nearest_idx)
        dists, indices = tree.query(invalid_idx)
        filled = depth.copy()
        for i, (y, x) in enumerate(invalid_idx):
            ny, nx = nearest_idx[indices[i]]
            filled[y, x] = depth[ny, nx]
        return filled

    elif method == 'cv2_inpaint':
        depth_8u = (depth_norm * 255).astype(np.uint8)
        inpainted = cv2.inpaint(depth_8u, mask, 3, cv2.INPAINT_NS)
        return inpainted.astype(np.float32) / 255. * max_val

    elif method == 'median':
        # Median filter can alleviate small holes
        depth_filled = depth.copy()
        depth_filled[invalid_mask] = cv2.medianBlur(depth, 5)[invalid_mask]
        return depth_filled

    else:
        raise ValueError(f"Unknown inpaint method: {method}")

def project_world_points_to_depth_map(
    world_points: np.ndarray,
    extrinsics_cam: np.ndarray,
    intrinsics_cam: np.ndarray,
    image_size=None,
    extrinsics_is_c2w: bool = False,
    invalid_value: float = 0.0,
    keep_z_positive_only: bool = True,
) -> np.ndarray:
    """
    Project batch of world-space 3D points into per-camera depth maps (z in camera coords).

    Args:
        world_points (np.ndarray or torch.Tensor):
            - Shape (S, H, W, 3)  -> grid-aligned input; image_size inferred = (H, W)
            - Shape (S, N, 3)     -> unstructured point cloud; must pass `image_size`
        extrinsics_cam (np.ndarray or torch.Tensor):
            Batch of camera extrinsics of shape (S, 3, 4).
            Interpretation controlled by `extrinsics_is_c2w`:
                True  -> [R|t] maps camera coords -> world coords (c2w).
                False -> [R|t] maps world coords  -> camera coords (w2c).
        intrinsics_cam (np.ndarray or torch.Tensor):
            Batch of intrinsics of shape (S, 3, 3). Assumed pinhole; K = [[fx, s, cx],[0, fy, cy],[0,0,1]].
            Skew `s` supported but usually 0.
        image_size (tuple or None):
            (H, W). Required if `world_points` is (S, N, 3). Ignored if grid input.
        extrinsics_is_c2w (bool):
            See above. Use True to match your `depth_to_world_coords_points`, which
            typically consumes c2w when going depth->world.
        invalid_value (float):
            Fill value for pixels receiving no valid projected points.
        keep_z_positive_only (bool):
            If True, discard points with z_cam <= 0.

    Returns:
        np.ndarray: Depth maps of shape (S, H, W) (float32).
    """
    # ---- to numpy ----------------------------------------------------------
    if isinstance(world_points, torch.Tensor):
        world_points = world_points.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    world_points = np.asarray(world_points)
    extrinsics_cam = np.asarray(extrinsics_cam)
    intrinsics_cam = np.asarray(intrinsics_cam)

    S = world_points.shape[0]

    # Infer / validate image size
    if world_points.ndim == 4:            # (S, H, W, 3)
        H, W = world_points.shape[1:3]
    else:                                 # (S, N, 3)
        if image_size is None:
            raise ValueError("image_size=(H,W) must be provided when world_points is unstructured (S,N,3).")
        H, W = image_size

    depth_maps = np.full((S, H, W), invalid_value, dtype=np.float32)

    # ---- per-frame ---------------------------------------------------------
    for i in range(S):
        pts_w = world_points[i]
        if pts_w.ndim == 3:   # grid
            pts_w = pts_w.reshape(-1, 3)  # (H*W,3)
        # Extrinsics
        Rt = extrinsics_cam[i]  # (3,4)
        R = Rt[:, :3]
        t = Rt[:, 3]

        if extrinsics_is_c2w:
            # Given X_c -> X_w = R_c2w * X_c + t_c2w
            # We need w2c: X_c = R_w2c * X_w + t_w2c
            R_c2w = R
            t_c2w = t
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
        else:
            # Already world->cam
            R_w2c = R
            t_w2c = t

        # Transform world -> cam
        pts_cam = (R_w2c @ pts_w.T) + t_w2c[:, None]  # (3,N)
        x_c, y_c, z_c = pts_cam

        if keep_z_positive_only:
            valid = z_c > 0
        else:
            valid = np.ones_like(z_c, dtype=bool)

        # Project using intrinsics
        K = intrinsics_cam[i]
        fx, s, cx = K[0, 0], K[0, 1], K[0, 2]
        fy, cy = K[1, 1], K[1, 2]

        # normalized
        inv_z = np.empty_like(z_c)
        inv_z[valid] = 1.0 / z_c[valid]
        xn = x_c * inv_z
        yn = y_c * inv_z

        # pixel coords
        u = fx * xn + s * yn + cx
        v = fy * yn + cy

        # Round/truncate to integer pixel index
        u_i = np.round(u).astype(int)
        v_i = np.round(v).astype(int)

        # In-bounds mask
        in_bounds = (
            valid &
            (u_i >= 0) & (u_i < W) &
            (v_i >= 0) & (v_i < H)
        )

        if not np.any(in_bounds):
            continue

        u_i = u_i[in_bounds]
        v_i = v_i[in_bounds]
        z_vals = z_c[in_bounds]

        # Z-buffer: keep nearest depth per pixel
        # We'll flatten depth map to 1D for scatter-min
        flat_idx = v_i * W + u_i
        flat_depth = depth_maps[i].reshape(-1)

        # Initialize those pixels to +inf so min works correctly
        # (Only for those we're about to update; avoids touching full array each time.)
        # Actually, we can compare against invalid_value; but invalid_value may be 0 so we need sentinel.
        # Strategy: gather current, compare, write min.
        cur_vals = flat_depth[flat_idx]
        needs_init = cur_vals == invalid_value
        # For uninitialized, just assign; for existing, take min
        flat_depth[flat_idx[needs_init]] = z_vals[needs_init]
        # For pixels that already had something, pick min
        better = (~needs_init) & (z_vals < cur_vals)
        flat_depth[flat_idx[better]] = z_vals[better]

    return depth_maps


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsics_cam: np.ndarray, intrinsics_cam: np.ndarray
) -> np.ndarray:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (np.ndarray): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (np.ndarray): Batch of camera extrinsic matrices of shape (S, 3, 4)
        intrinsics_cam (np.ndarray): Batch of camera intrinsic matrices of shape (S, 3, 3)

    Returns:
        np.ndarray: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1), extrinsics_cam[frame_idx], intrinsics_cam[frame_idx]
        )
        world_points_list.append(cur_world_points)
    world_points_array = np.stack(world_points_list, axis=0)

    return world_points_array


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


# TODO: this code can be further cleaned up


def project_world_points_to_camera_points_batch(world_points, cam_extrinsics):
    """
    Transforms 3D points to 2D using extrinsic and intrinsic parameters.
    Args:
        world_points (torch.Tensor): 3D points of shape BxSxHxWx3.
        cam_extrinsics (torch.Tensor): Extrinsic parameters of shape BxSx3x4.
    Returns:
    """
    # TODO: merge this into project_world_points_to_cam
    
    # device = world_points.device
    # with torch.autocast(device_type=device.type, enabled=False):
    ones = torch.ones_like(world_points[..., :1])  # shape: (B, S, H, W, 1)
    world_points_h = torch.cat([world_points, ones], dim=-1)  # shape: (B, S, H, W, 4)

    # extrinsics: (B, S, 3, 4) -> (B, S, 1, 1, 3, 4)
    extrinsics_exp = cam_extrinsics.unsqueeze(2).unsqueeze(3)

    # world_points_h: (B, S, H, W, 4) -> (B, S, H, W, 4, 1)
    world_points_h_exp = world_points_h.unsqueeze(-1)

    # Now perform the matrix multiplication
    # (B, S, 1, 1, 3, 4) @ (B, S, H, W, 4, 1) broadcasts to (B, S, H, W, 3, 1)
    camera_points = torch.matmul(extrinsics_exp, world_points_h_exp).squeeze(-1)

    return camera_points



def project_world_points_to_cam(
    world_points,
    cam_extrinsics,
    cam_intrinsics=None,
    distortion_params=None,
    default=0,
    only_points_cam=False,
):
    """
    Transforms 3D points to 2D using extrinsic and intrinsic parameters.
    Args:
        world_points (torch.Tensor): 3D points of shape Px3.
        cam_extrinsics (torch.Tensor): Extrinsic parameters of shape Bx3x4.
        cam_intrinsics (torch.Tensor): Intrinsic parameters of shape Bx3x3.
        distortion_params (torch.Tensor): Extra parameters of shape BxN, which is used for radial distortion.
    Returns:
        torch.Tensor: Transformed 2D points of shape BxNx2.
    """
    device = world_points.device
    # with torch.autocast(device_type=device.type, dtype=torch.double):
    with torch.autocast(device_type=device.type, enabled=False):
        N = world_points.shape[0]  # Number of points
        B = cam_extrinsics.shape[0]  # Batch size, i.e., number of cameras
        world_points_homogeneous = torch.cat(
            [world_points, torch.ones_like(world_points[..., 0:1])], dim=1
        )  # Nx4
        # Reshape for batch processing
        world_points_homogeneous = world_points_homogeneous.unsqueeze(0).expand(
            B, -1, -1
        )  # BxNx4

        # Step 1: Apply extrinsic parameters
        # Transform 3D points to camera coordinate system for all cameras
        cam_points = torch.bmm(
            cam_extrinsics, world_points_homogeneous.transpose(-1, -2)
        )

        if only_points_cam:
            return None, cam_points

        # Step 2: Apply intrinsic parameters and (optional) distortion
        image_points = img_from_cam(cam_intrinsics, cam_points, distortion_params, default=default)

        return image_points, cam_points



def img_from_cam(cam_intrinsics, cam_points, distortion_params=None, default=0.0):
    """
    Applies intrinsic parameters and optional distortion to the given 3D points.

    Args:
        cam_intrinsics (torch.Tensor): Intrinsic camera parameters of shape Bx3x3.
        cam_points (torch.Tensor): 3D points in camera coordinates of shape Bx3xN.
        distortion_params (torch.Tensor, optional): Distortion parameters of shape BxN, where N can be 1, 2, or 4.
        default (float, optional): Default value to replace NaNs in the output.

    Returns:
        pixel_coords (torch.Tensor): 2D points in pixel coordinates of shape BxNx2.
    """

    # Normalized device coordinates (NDC)
    cam_points = cam_points / cam_points[:, 2:3, :]
    ndc_xy = cam_points[:, :2, :]

    # Apply distortion if distortion_params are provided
    if distortion_params is not None:
        x_distorted, y_distorted = apply_distortion(distortion_params, ndc_xy[:, 0], ndc_xy[:, 1])
        distorted_xy = torch.stack([x_distorted, y_distorted], dim=1)
    else:
        distorted_xy = ndc_xy

    # Prepare cam_points for batch matrix multiplication
    cam_coords_homo = torch.cat(
        (distorted_xy, torch.ones_like(distorted_xy[:, :1, :])), dim=1
    )  # Bx3xN
    # Apply intrinsic parameters using batch matrix multiplication
    pixel_coords = torch.bmm(cam_intrinsics, cam_coords_homo)  # Bx3xN

    # Extract x and y coordinates
    pixel_coords = pixel_coords[:, :2, :]  # Bx2xN

    # Replace NaNs with default value
    pixel_coords = torch.nan_to_num(pixel_coords, nan=default)

    return pixel_coords.transpose(1, 2)  # BxNx2




def cam_from_img(pred_tracks, intrinsics, extra_params=None):
    """
    Normalize predicted tracks based on camera intrinsics.
    Args:
    intrinsics (torch.Tensor): The camera intrinsics tensor of shape [batch_size, 3, 3].
    pred_tracks (torch.Tensor): The predicted tracks tensor of shape [batch_size, num_tracks, 2].
    extra_params (torch.Tensor, optional): Distortion parameters of shape BxN, where N can be 1, 2, or 4.
    Returns:
    torch.Tensor: Normalized tracks tensor.
    """

    # We don't want to do intrinsics_inv = torch.inverse(intrinsics) here
    # otherwise we can use something like
    #     tracks_normalized_homo = torch.bmm(pred_tracks_homo, intrinsics_inv.transpose(1, 2))

    principal_point = intrinsics[:, [0, 1], [2, 2]].unsqueeze(-2)
    focal_length = intrinsics[:, [0, 1], [0, 1]].unsqueeze(-2)
    tracks_normalized = (pred_tracks - principal_point) / focal_length

    if extra_params is not None:
        # Apply iterative undistortion
        try:
            tracks_normalized = iterative_undistortion(
                extra_params, tracks_normalized
            )
        except:
            tracks_normalized = single_undistortion(
                extra_params, tracks_normalized
            )

    return tracks_normalized


