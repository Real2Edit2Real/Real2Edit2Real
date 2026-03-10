import torch
import numpy as np
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, quaternion_multiply, Transform3d
from scipy.spatial.transform import Rotation as R

def transform_points(c2w: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Apply camera-to-world transformation to 3D points.
    
    Args:
        c2w: Camera-to-world transformation matrix of shape (4, 4)
        points: 3D points in camera coordinates, shape (..., 3)
        
    Returns:
        Transformed points in world coordinates, shape (..., 3)
    """
    # Convert points to homogeneous coordinates (N, 4) by adding ones
    transformed_points = points.clone()
    
    homogeneous_points = torch.cat([points[..., :3], torch.ones_like(points[..., :1])], dim=-1)
    
    # Apply transformation: (N, 4) * (4, 4) -> (N, 4)
    transformed_homogeneous = homogeneous_points @ c2w.T  # Using transpose for proper matrix multiplication
    
    # Convert back to 3D coordinates by dropping the last dimension
    transformed_points[..., :3] = transformed_homogeneous[..., :3]
    
    return transformed_points

def transform_trans_quat(c2w, trans_c, quat_c, quat_type="xyzw"):
    '''
    c2w: torch.tensor, 4x4
    trans_c: torch.tensor, [N, 3]
    quat_c: torch.tensor, [N, 4], wxyz or xyzw
    trans_w: torch.tensor, [N, 3]
    quat_w: torch.tensor, [N, 4], wxyz or xyzw
    quat_type: xyzw or wxyz
    '''
    if quat_type == "wxyz":
        quat_c_in = quat_c
    elif quat_type == "xyzw":
        # xyzw --> wxyz
        quat_c_in = quat_c[:, [3, 0, 1, 2]]
    
    trans_w = transform_points(c2w, trans_c)
    c2w_quat = matrix_to_quaternion(c2w[:3, :3].unsqueeze(0))
    quat_w = quaternion_multiply(c2w_quat, quat_c_in)
    if quat_type == "xyzw":
        # wxyz --> xyzw
        quat_w = quat_w[:, [1, 2, 3, 0]]
    return trans_w, quat_w

def expand_extrinsic(extrinsic):
    """
    Convert extrinsic from 3x4 to 4x4
    """
    if extrinsic.shape == (3, 4):
        extrinsic_4x4 = np.eye(4)
        extrinsic_4x4[:3, :4] = extrinsic
        return extrinsic_4x4
    
    return extrinsic

def action2mat(actions, quat_type="xyzw"):
    '''
    actions: torch.tensor, [N, 7], xyz, xyzw
    '''
    trans_r = actions[:, :3]
    quat_r = actions[:, 3:7]
    if quat_type == "xyzw":
        quat_in = quat_r[:, [3, 0, 1, 2]]
    else:
        quat_in = quat_r
    rot_mat_r = quaternion_to_matrix(quat_in) # N x 3 x 3
    mat_3x4_r = torch.cat([rot_mat_r, trans_r.unsqueeze(2)], dim=2)
    zero_one = torch.tensor([[0, 0, 0, 1]]).float().to(mat_3x4_r.device).unsqueeze(0).repeat(len(mat_3x4_r), 1, 1)
    mat_4x4_r = torch.cat([mat_3x4_r, zero_one], dim=1)
    return mat_4x4_r


def rotate_pose_around_point(pose_xyz, pose_xyzw, rotation_center_xyz, rotation_xyzw):
    """
    Rotate a given (xyz, xyzw) pose around the Z-axis corresponding to a given 3D point by a specified angle.

    Args:
        pose_xyz (np.ndarray): Original position of the pose [x, y, z].
        pose_xyzw (np.ndarray): Original quaternion of the pose [x, y, z, w] (Scalar-Last).
        rotation_center_xyz (np.ndarray): 3D coordinates of the rotation center [x_r, y_r, z_r].
        rotation_xyzw (np.ndarray): Quaternion [x, y, z, w].

    Returns:
        tuple: (new_xyz, new_xyzw), the new position and quaternion of the pose.
    """
    
    # ----------------- 1. Define rotation center P_rot and rotation quaternion q_rot -----------------
    
    # Rotation center P_rot (given)
    center = rotation_center_xyz.copy()
    
    # Create rotation object R_rot
    R_rot = R.from_quat(rotation_xyzw)
    
    # ----------------- 2. Rotation and translation part (position) -----------------
    
    # (a) Translate to origin: t_centered = t_orig - P_rot
    t_centered = pose_xyz - center
    
    # (b) Apply rotation: t_rotated_centered = R_rot * t_centered
    # Apply rotation to the position vector
    t_rotated_centered = R_rot.apply(t_centered)
    
    # (c) Translate back to original position: t_new = t_rotated_centered + P_rot
    new_xyz = t_rotated_centered + center
    
    # ----------------- 3. Rotation of pose part (quaternion) -----------------
    
    # New pose q_new = q_rot * q_orig (rotation around the world frame is left multiplication)
    
    # Original pose q_orig
    q_orig = R.from_quat(pose_xyzw)
    
    # Rotation pose
    q_rot = R_rot # Use the instance of the R_rot object created above
    
    # Quaternion multiplication (SciPy uses * operator for composition)
    q_new = q_rot * q_orig
    
    # Extract new quaternion [x, y, z, w]
    new_xyzw = q_new.as_quat()
    
    return new_xyz, new_xyzw

def rotate_n_aabbs_with_n_angles_around_center_z(n_aabbs, rots):
    """
    Perform rotation around the Z-axis of each AABB's center for N AABBs, with each AABB corresponding to an angle.
    Use np.einsum for vectorized matrix multiplication, resolving dimension matching issues with Rotation.apply.

    Input format: 
        n_aabbs (N, 2, 3) 
        rots (N, 4)

    Args:
        n_aabbs (np.ndarray): AABB array with shape (N, 2, 3).
        rots (np.ndarray): Quaternion array with shape (N, 4).

    Returns:
        tuple: (new_aabbs, n_rotated_corners)
               new_aabbs: New axis-aligned bounding box array with shape (N, 2, 3).
               n_rotated_corners: Rotated corner point array with shape (N, 8, 3).
    """
    N = n_aabbs.shape[0]

    # 1. Vectorized calculation of N AABB center points (N, 3)
    centers = (n_aabbs[:, 0, :] + n_aabbs[:, 1, :]) / 2.0  

    # 2. Vectorized generation of 8 corners for N AABBs (N, 8, 3)
    min_p = n_aabbs[:, 0, :]
    max_p = n_aabbs[:, 1, :]
    corners = np.stack([
        min_p,
        np.array([max_p[:, 0], min_p[:, 1], min_p[:, 2]]).T,
        np.array([min_p[:, 0], max_p[:, 1], min_p[:, 2]]).T,
        np.array([min_p[:, 0], min_p[:, 1], max_p[:, 2]]).T,
        np.array([max_p[:, 0], max_p[:, 1], min_p[:, 2]]).T,
        np.array([max_p[:, 0], min_p[:, 1], max_p[:, 2]]).T,
        np.array([min_p[:, 0], max_p[:, 1], max_p[:, 2]]).T,
        max_p
    ], axis=1)

    # 3. Translate to respective centers (p - c)
    # centers (N, 3) broadcasted to corners (N, 8, 3)
    points_centered = corners - centers[:, np.newaxis, :]  # Shape: (N, 8, 3)

    # 4. Construct N Z-axis rotation matrix stacks (N, 3, 3)
    R_Z_stack = R.from_quat(rots).as_matrix() # Shape: (N, 3, 3)

    # 5. Vectorized rotation (R @ (p - c))
    # Using np.einsum: R_Z_stack[n, i, j] * points_centered[n, k, j] -> n_rotated_corners_centered[n, k, i]
    # 'nij, nkj -> nki' represents matrix multiplication for N sets of (3x3) and (8x3)
    n_rotated_corners_centered = np.einsum('nij, nkj->nki', R_Z_stack, points_centered) 
    # Shape: (N, 8, 3)

    # 6. Translate back to respective centers (R * (p - c) + c)
    n_rotated_corners = n_rotated_corners_centered + centers[:, np.newaxis, :] # Shape: (N, 8, 3)

    # 7. Vectorized calculation of N new AABBs 
    new_min_point = np.min(n_rotated_corners, axis=1) # Shape: (N, 3)
    new_max_point = np.max(n_rotated_corners, axis=1) # Shape: (N, 3)
    
    # Stack into (N, 2, 3) output format
    new_aabbs = np.stack([new_min_point, new_max_point], axis=1)

    return new_aabbs