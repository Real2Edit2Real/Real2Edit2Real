# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
from termcolor import cprint

import torch_scatter
import torch.nn.functional as F

try:
    from cuml import DBSCAN
    cprint("Using GPU DBSCAN from cuML", "green")
except:
    from sklearn.cluster import DBSCAN
    cprint("Fallback to CPU DBSCAN. Run `conda install rapids=25.06 -c rapidsai` to install the GPU version.", "red")
    
    
def unproject_depth_map_to_point_map(
    depth_map: torch.Tensor, extrinsics_cam: torch.Tensor, intrinsics_cam: torch.Tensor
) -> torch.Tensor:
    """
    Unproject a batch of depth maps to 3D world coordinates using PyTorch.

    Args:
        depth_map (torch.Tensor): Batch of depth maps with shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (torch.Tensor): Batch of camera extrinsic matrices with shape (S, 3, 4)
        intrinsics_cam (torch.Tensor): Batch of camera intrinsic matrices with shape (S, 3, 3)

    Returns:
        torch.Tensor: Batch of 3D world coordinates with shape (S, H, W, 3)
    """
    # Ensure inputs are PyTorch tensors
    if isinstance(depth_map, np.ndarray):
        depth_map = torch.from_numpy(depth_map)
    if isinstance(extrinsics_cam, np.ndarray):
        extrinsics_cam = torch.from_numpy(extrinsics_cam)
    if isinstance(intrinsics_cam, np.ndarray):
        intrinsics_cam = torch.from_numpy(intrinsics_cam)

    # Process depth map dimensions
    if depth_map.dim() == 4 and depth_map.shape[-1] == 1:
        depth_map = depth_map.squeeze(-1)  # Convert to (S, H, W)
    
    S, H, W = depth_map.shape
    
    # Prepare grid coordinates (S, H, W)
    u = torch.arange(W, device=depth_map.device).repeat(S, H, 1)  # (S, H, W)
    v = torch.arange(H, device=depth_map.device).unsqueeze(1).repeat(S, 1, W)  # (S, H, W)
    
    # Extract intrinsic parameters
    fu = intrinsics_cam[:, 0, 0].unsqueeze(1).unsqueeze(1)  # (S, 1, 1)
    fv = intrinsics_cam[:, 1, 1].unsqueeze(1).unsqueeze(1)  # (S, 1, 1)
    cu = intrinsics_cam[:, 0, 2].unsqueeze(1).unsqueeze(1)  # (S, 1, 1)
    cv = intrinsics_cam[:, 1, 2].unsqueeze(1).unsqueeze(1)  # (S, 1, 1)
    
    # Calculate camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map
    
    # Stack to form camera coordinates (S, H, W, 3)
    cam_coords = torch.stack((x_cam, y_cam, z_cam), dim=-1)
    
    # Calculate inverse of extrinsics
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsics_cam)  # (S, 4, 4)
    
    # Extract rotation and translation components
    R_cam_to_world = cam_to_world_extrinsic[:, :3, :3]  # (S, 3, 3)
    t_cam_to_world = cam_to_world_extrinsic[:, :3, 3]    # (S, 3)
    
    # Apply rotation and translation (S, H, W, 3)
    world_coords = torch.einsum('sij,shwj->shwi', R_cam_to_world, cam_coords) + t_cam_to_world.unsqueeze(1).unsqueeze(1)
    
    return world_coords

def closed_form_inverse_se3(se3: torch.Tensor, R=None, T=None) -> torch.Tensor:
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch using PyTorch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 tensor of SE3 matrices.
        R (optional): Nx3x3 tensor of rotation matrices.
        T (optional): Nx3x1 tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.
    """
    # Validate shapes
    if se3.shape[-2:] not in [(4, 4), (3, 4)]:
        raise ValueError(f"se3 must be of shape (N,4,4) or (N,3,4), got {se3.shape}.")

    # Extract R and T from se3 if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        # Handle both 3x4 and 4x4 cases
        if se3.shape[-2:] == (3, 4):
            T = se3[:, :3, 3:4]  # (N,3,1)
        else:  # 4x4
            T = se3[:, :3, 3:4]  # (N,3,1)

    # Transpose R
    R_transposed = R.transpose(1, 2)  # (N,3,3)
    
    # Calculate -R^T t
    top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
    
    # Create 4x4 identity matrix and populate
    inverted_matrix = torch.eye(4, device=se3.device, dtype=se3.dtype).unsqueeze(0).repeat(se3.size(0), 1, 1)
    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:4] = top_right

    return inverted_matrix

# --------------------------------------------------------------------------
# Helper function: PyTorch tensorized nearest neighbor filling (using convolution for diffusion)
# --------------------------------------------------------------------------
def _nearest_neighbor_fill_tensorized_batch(img: torch.Tensor, max_iterations: int = 10, max_val: float = 10., kernel_size: int = 3, padding: int = 1):
    """
    Iteratively fill regions marked as inf in N images using tensorized operations (pooling) with nearest neighbor interpolation.

    Args:
        img: torch.Tensor, shape (N, H, W), a batch of images containing inf values (e.g., depth maps).
        max_iterations: int, maximum number of iterations.
    
    Returns:
        filled_img: Filled image, shape (N, H, W).
    """
    if img.dim() != 3:
        raise ValueError("Input tensor must have shape (N, H, W).")
        
    if not img.is_floating_point():
        img = img.to(torch.float32)

    # Convert (N, H, W) to (N, 1, H, W) to fit the (B, C, H, W) format for conv2d/pool2d
    # B=N, C=1
    filled_img = img.clone().unsqueeze(1)
    
    for _ in range(max_iterations):
        # 1. Find pixels to be filled
        # Mask shape: (N, 1, H, W)
        invalid_mask = torch.isinf(filled_img)

        # Optimization: If there are no inf values in any batch, exit the loop
        if not torch.any(invalid_mask):
            break

        # 2. Create a "source" image for current valid values
        source = filled_img.clone()
        # Replace inf with Max_Val to ensure inf doesn't become the minimum in the neighborhood
        source[invalid_mask] = max_val

        # 3. Use -MaxPool2d(-X) to implement MinPool2d(X)
        negative_source = -source

        # MaxPool2d is performed on (N, 1, H, W), calculated independently for each batch
        min_depths_propagated = -F.max_pool2d(
            negative_source, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding
        )
        
        # 4. Update condition masks
        # a) Pixel must currently be inf (invalid_mask)
        # b) Propagated value must be a valid depth (i.e., a valid value exists in the neighborhood, propagated value < max_val)
        valid_propagated_mask = min_depths_propagated < max_val
        update_mask = invalid_mask & valid_propagated_mask

        # 5. Update: Use the propagated minimum depth value to update filled_img
        filled_img[invalid_mask] = min_depths_propagated[invalid_mask]
        
    # Convert back to (N, H, W)
    return filled_img.squeeze(1)
# --------------------------------------------------------------------------

def project_pcd2image(
    pcd: torch.Tensor, 
    w2c: torch.Tensor, 
    intrinsic: torch.Tensor,
    image_bg: torch.Tensor = None,  # Receive background image (B, H, W) or None
    shape: tuple = (294, 518),
    invalid_default: float = 3.0,
    max_iterations: int = 10,
) -> torch.Tensor:
    """
    Project point cloud to multiple image planes and fuse with background image, without explicit Python loops.

    Args:
        pcd: torch.Tensor, shape (N, 3), point cloud in world coordinate system [x, y, z].
        w2c: torch.Tensor, shape (B, 4, 4), B world-to-camera transformation matrices.
        intrinsic: torch.Tensor, shape (B, 3, 3), B camera intrinsic matrices.
        image_bg: torch.Tensor or None, initial depth map. Expected shape is (B, H, W).
        shape: tuple, output image shape (H, W).
        invalid_default: float, default fill value for invalid positions in the depth map.

    Returns:
        output_depths: Projected depth map, shape (B, H, W).
    """
    device = pcd.device
    dtype = pcd.dtype
    B = w2c.shape[0]
    N = pcd.shape[0]

    # --- 1. Initialize output depth map (including background) ---
    if image_bg is not None:
        # Use background map to determine H, W and as the starting point for output
        if image_bg.dim() == 2: # Assume (H, W), broadcast to all batches
            H, W = image_bg.shape
            output_depths = image_bg.unsqueeze(0).expand(B, H, W).clone()
        elif image_bg.dim() == 3: # Assume (B, H, W)
            H, W = image_bg.shape[1:]
            # Ensure background map is floating point for comparison with depth values
            output_depths = image_bg.to(dtype=dtype).clone() 
        else:
            raise ValueError("image_bg must have shape (H, W) or (B, H, W).")
    else:
        # If no background map, initialize with shape and inf
        H, W = shape
        output_depths = torch.full((B, H, W), float('inf'), dtype=dtype, device=device)
    
    
    # ------------------ (Geometric transformation and projection part - same as before) ------------------
    
    # 1. Expand point cloud (N, 3) -> (B, N, 4)
    xyz_world = pcd[:, :3]
    ones = torch.ones((N, 1), device=device, dtype=dtype)
    xyz_homo = torch.cat([xyz_world, ones], dim=1)  # (N, 4)
    xyz_homo_b = xyz_homo.unsqueeze(0).expand(B, -1, -1) 
    
    # 2. World coordinates -> Camera coordinates (B, N, 3)
    xyz_cam_homo_b = torch.bmm(w2c, xyz_homo_b.transpose(1, 2)).transpose(1, 2)
    xyz_cam_b = xyz_cam_homo_b[..., :3]
    depth_z_b = xyz_cam_b[..., 2]
    valid_mask_bn = depth_z_b > 1e-6
    
    # 3. Camera coordinates -> Pixel coordinates (B, N, 3)
    proj_b = torch.bmm(intrinsic, xyz_cam_b.transpose(1, 2)).transpose(1, 2)
    
    # 4. Normalization and depth extraction
    u_b = proj_b[..., 0] / proj_b[..., 2] # (B, N)
    v_b = proj_b[..., 1] / proj_b[..., 2] # (B, N)
    depth_b = proj_b[..., 2]              # (B, N)
    
    # 5. Pixel filtering (within image boundaries)
    u_b_int = torch.round(u_b).to(torch.int64)
    v_b_int = torch.round(v_b).to(torch.int64)
    
    valid_mask_pixel = (u_b_int >= 0) & (u_b_int < W) & \
                       (v_b_int >= 0) & (v_b_int < H)
                       
    final_valid_mask = valid_mask_bn & valid_mask_pixel # (B, N)

    # 6. Extract all valid u, v, depth, and b indices (flattened)
    u_flat = u_b_int[final_valid_mask]
    v_flat = v_b_int[final_valid_mask]
    depth_flat = depth_b[final_valid_mask]
    
    b_indices = torch.arange(B, device=device).unsqueeze(-1).expand_as(u_b_int)
    b_flat = b_indices[final_valid_mask]

    # ------------------ (Conflict resolution and fusion part) ------------------
    
    # 7. Create flattened global pixel indices (B * H * W)
    # Target index = Batch index * (H * W) + V coordinate * W + U coordinate
    global_pixel_indices = b_flat * (H * W) + v_flat * W + u_flat
    
    # 8. Initialize a flattened depth map with all inf for scatter_min 
    # **Note:** min_depths_flat here must be initialized to INF because we only care about the minimum depth of the point cloud projection.
    min_depths_flat_pcd = torch.full((B * H * W,), float('inf'), device=device, dtype=dtype)
    
    # 9. Use scatter_min to resolve depth conflicts within the point cloud
    min_depths_resolved_pcd, _ = torch_scatter.scatter_min(
        src=depth_flat, 
        index=global_pixel_indices,
        out=min_depths_flat_pcd,
        dim=0, 
    )
    
    # 10. Reshape back to (B, H, W)
    min_depths_2d_pcd = min_depths_resolved_pcd.reshape(B, H, W)
    
    # 11. Fusion: Compare the minimum depth of the point cloud with the background depth and take the minimum of both
    # output_depths is already (B, H, W) and contains image_bg
    output_depths = torch.min(output_depths, min_depths_2d_pcd)
    output_depths = _nearest_neighbor_fill_tensorized_batch(output_depths, max_iterations=max_iterations, max_val=invalid_default)
    invalid_mask = torch.isinf(output_depths)
    
    # 12. Fill invalid/infinite depth values (if neither provides a valid depth)
    if torch.any(invalid_mask):
        output_depths[invalid_mask] = invalid_default
    return output_depths


def check_bbox_intersection(bboxes_a: np.ndarray, bboxes_b: np.ndarray, expand_dim: bool=True) -> np.ndarray:
    """
    Parallelly compute the intersection between K sets of BBoxes and N sets of BBoxes.

    Args:
        bboxes_a: NumPy array with shape (K, 2, 3).
                  (K, 0, 0) is the x_min of the Kth bbox
                  (K, 1, 0) is the x_max of the Kth bbox
        bboxes_b: NumPy array with shape (N, 2, 3).

    Returns:
        Boolean array with shape (K, N), True if A[i] intersects B[j].
    """
    
    # ----------------------------------------------
    # 1. Prepare data and perform broadcast dimension reshaping (K, 1, 2, 3) and (1, N, 2, 3)
    # ----------------------------------------------
    if expand_dim:
        # Reshape A to (K, 1, 2, 3) for broadcasting with N
        A = bboxes_a[:, np.newaxis, :, :]
        
        # Reshape B to (1, N, 2, 3) for broadcasting with K
        B = bboxes_b[np.newaxis, :, :, :]
    else:
        A = bboxes_a
        B = bboxes_b
    # ----------------------------------------------
    # 2. Extract Min/Max coordinates, both with shape (K, N, 3)
    # ----------------------------------------------
    
    # Extract min coordinates for all bbox_a (x_min, y_min, z_min)
    # A_min shape is (K, N, 3). N axis is from broadcasting.
    A_min = A[..., 0, :]  # A[:, :, 0, :]
    
    # Extract max coordinates for all bbox_a (x_max, y_max, z_max)
    A_max = A[..., 1, :]  # A[:, :, 1, :]
    
    # Extract min coordinates for all bbox_b
    B_min = B[..., 0, :]  # B[:, :, 0, :]
    
    # Extract max coordinates for all bbox_b
    B_max = B[..., 1, :]  # B[:, :, 1, :]

    # ----------------------------------------------
    # 3. Compute axial intersection condition (K, N, 3)
    # ----------------------------------------------
    
    # Condition 1: A_min <= B_max 
    # i.e., A's starting point <= B's ending point
    # Shape (K, N, 3)
    cond1 = A_min <= B_max
    
    # Condition 2: A_max >= B_min
    # i.e., A's ending point >= B's starting point
    # Shape (K, N, 3)
    cond2 = A_max >= B_min

    # Both conditions must be met to intersect
    # intersection_axes shape (K, N, 3)
    intersection_axes = cond1 & cond2

    # ----------------------------------------------
    # 4. Final decision: intersect on all axes (K, N)
    # ----------------------------------------------
    
    # Use np.all() along the last axis (axis=2, i.e., x, y, z axes) for logical AND operation
    # Only when (x AND y AND z) are all True, the final result is True
    # intersection_result shape (K, N)
    intersection_result = np.all(intersection_axes, axis=-1)
    
    return intersection_result

def check_bbox_y_disjoint_relative(
    bboxes_object,
    bboxes_target,
    target_to_object, 
    expand_dim=True
):
    """
    Parallelly compute the [overall] relative relationship along the Y axis between K sets of Object BBoxes and N sets of Target BBoxes.
    
    [Key Conventions]: 
    1. Y+ direction is defined as "left".
    2. True (disjoint) only if one BBox is entirely on one side of another.
    
    Relationship definitions:
    - "left": Target is entirely in the Y+ direction (left side) of Object.
              Target y_min > Object y_max
    - "right": Target is entirely in the Y- direction (right side) of Object.
               Target y_max < Object y_min

    Args:
        bboxes_object: NumPy array with shape (K, 2, 3) (main object BBoxes).
        bboxes_target: NumPy array with shape (N, 2, 3) (target BBoxes for comparison).
        target_to_object: string, defines the direction of Target relative to Object.
        expand_dim: Whether to perform broadcast dimension reshaping. Default is True.

    Returns:
        Boolean array with shape (K, N).
    """
    
    # ----------------------------------------------
    # 1. Prepare data and perform broadcast dimension reshaping
    # ----------------------------------------------
    if expand_dim:
        # Reshape Object to (K, 1, 2, 3)
        Object = bboxes_object[:, np.newaxis, :, :]
        # Reshape Target to (1, N, 2, 3)
        Target = bboxes_target[np.newaxis, :, :, :]
    else:
        Object = bboxes_object
        Target = bboxes_target

    # ----------------------------------------------
    # 2. Extract Y-axis Min/Max coordinates, both with shape (K, N)
    # ----------------------------------------------
    
    # Object's y_min and y_max (shape: K, N)
    Object_y_min = Object[..., 0, 1] 
    Object_y_max = Object[..., 1, 1] 
    
    # Target's y_min and y_max (shape: K, N)
    Target_y_min = Target[..., 0, 1] 
    Target_y_max = Target[..., 1, 1] 

    # ----------------------------------------------
    # 3. Compute relationship (K, N) - overall comparison logic
    # ----------------------------------------------
    
    if target_to_object == "left":
        # Target is on the 'left' side (Y+ direction) of Object
        # Target y_min must be greater than Object y_max
        comparison_result = Target_y_min > Object_y_max
        
    elif target_to_object == "right":
        # Target is on the 'right' side (Y- direction) of Object
        # Target y_max must be less than Object y_min
        comparison_result = Target_y_max < Object_y_min
        
    else:
        raise ValueError("target_to_object must be 'left' or 'right'")

    return comparison_result

    return comparison_result

def cluster_filter_pointcloud(pointcloud, eps=0.03, min_samples=35, return_mask=True):
    """
    Filter out noise points in the point cloud using clustering algorithm
    
    Args:
        pointcloud: Point cloud data (N, 6) [x, y, z, r, g, b]
        eps: DBSCAN neighborhood radius (default 3cm, stricter for edge filtering)
        min_samples: DBSCAN minimum samples (default 35, avoid sparse edge clusters)
        
    Returns:
        filtered_pointcloud: Filtered point cloud (noise points removed, all valid clusters kept)
    """
    if pointcloud is None or len(pointcloud) == 0:
        return pointcloud
    
    # Extract point coordinates
    points = pointcloud[:, :3]  # (N, 3)
    
    # Use DBSCAN for clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(points)
    
    # Cluster statistics
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)  # Exclude noise
    n_noise = np.sum(cluster_labels == -1)
        
    # Keep all non-noise points (remove -1 label)
    keep_mask = cluster_labels != -1
    filtered_points = points[keep_mask]
    
    original_count = len(pointcloud)
    filtered_count = len(filtered_points)
    removed_count = original_count - filtered_count
    # cprint(f"original {original_count}, filtered {filtered_count}, removed {removed_count}", "green")
    # Merge back to point cloud format
    filtered_pointcloud = filtered_points
    if return_mask:
        return filtered_pointcloud, keep_mask
    else:
        return filtered_pointcloud