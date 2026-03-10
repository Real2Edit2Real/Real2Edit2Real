import numpy as np
import os
import h5py
from scipy.spatial.transform import Rotation
from lib.data.utils.beta_dataset.traj_vis_statistics import EEF2CamLeft, EEF2CamRight

def normalize_angles(radius):
    radius_normed = np.mod(radius, 2 * np.pi) - 2 * np.pi * (np.mod(radius, 2 * np.pi) > np.pi)
    return radius_normed


def get_actions(gripper, all_ends_p=None, all_ends_o=None, slices=None, delta_act_sidx=None):

    if delta_act_sidx is None:
        delta_act_sidx = 1

    if slices is None:
        ### the first frame is repeated to fill memory
        n = all_ends_p.shape[0]-1+delta_act_sidx
        slices = [0,]*(delta_act_sidx-1) + list(range(all_ends_p.shape[0]))
    else:
        n = len(slices)

    all_left_rpy = []
    all_right_rpy = []
    all_left_quat = []
    all_right_quat = []

    ### cam eef 30...CAM_ANGLE...
    cvt_vis_l = Rotation.from_euler("xyz", np.array(EEF2CamLeft))
    cvt_vis_r = Rotation.from_euler("xyz", np.array(EEF2CamRight))
    for i in slices:
        
        rot_l = Rotation.from_quat(all_ends_o[i, 0])
        rot_vis_l = rot_l*cvt_vis_l
        left_vis_quat = np.concatenate((all_ends_p[i,0], rot_vis_l.as_quat()), axis=0)
        left_rpy = np.concatenate((all_ends_p[i,0], rot_l.as_euler("xyz", degrees=False)), axis=0)

        rot_r = Rotation.from_quat(all_ends_o[i, 1])
        rot_vis_r = rot_r*cvt_vis_r
        right_vis_quat = np.concatenate((all_ends_p[i,1], rot_vis_r.as_quat()), axis=0)
        right_rpy = np.concatenate((all_ends_p[i,1], rot_r.as_euler("xyz", degrees=False)), axis=0)

        all_left_rpy.append(left_rpy)
        all_right_rpy.append(right_rpy)
        all_left_quat.append(left_vis_quat)
        all_right_quat.append(right_vis_quat)

    ### xyz, rpy
    all_left_rpy = np.stack(all_left_rpy)
    all_right_rpy = np.stack(all_right_rpy)
    ### xyz, xyzw
    all_left_quat = np.stack(all_left_quat)
    all_right_quat = np.stack(all_right_quat)

    ### xyz, xyzw, gripper
    all_abs_actions = np.zeros([n, 16])
    ### xyz, rpy, gripper
    all_delta_actions = np.zeros([n-delta_act_sidx, 14])
    for i in range(0, n):
        all_abs_actions[i, 0:7] = all_left_quat[i, :7]
        all_abs_actions[i, 7] = gripper[slices[i], 0]
        all_abs_actions[i, 8:15] = all_right_quat[i, :7]
        all_abs_actions[i, 15] = gripper[slices[i], 1]
        if i >= delta_act_sidx:
            all_delta_actions[i-delta_act_sidx, 0:6] = all_left_rpy[i, :6] - all_left_rpy[i-1, :6]
            all_delta_actions[i-delta_act_sidx, 3:6] = normalize_angles(all_delta_actions[i-delta_act_sidx, 3:6])
            all_delta_actions[i-delta_act_sidx, 6] = gripper[slices[i], 0] / 120.0
            all_delta_actions[i-delta_act_sidx, 7:13] = all_right_rpy[i, :6] - all_right_rpy[i-1, :6]
            all_delta_actions[i-delta_act_sidx, 10:13] = normalize_angles(all_delta_actions[i-delta_act_sidx, 10:13])
            all_delta_actions[i-delta_act_sidx, 13] = gripper[slices[i], 1] / 120.0

    return all_abs_actions, all_delta_actions


def parse_h5(h5_file, slices=None, delta_act_sidx=1):
    """
    read and parse .h5 file, and obtain the absolute actions and the action differences
    """
    with h5py.File(h5_file, "r") as fid:
        all_abs_gripper = np.array(fid[f"state/effector/position"], dtype=np.float32)
        # print(all_abs_gripper.shape)
        all_ends_p = np.array(fid["state/end/position"], dtype=np.float32)
        all_ends_o = np.array(fid["state/end/orientation"], dtype=np.float32)

    all_abs_actions, all_delta_actions = get_actions(
        gripper=all_abs_gripper,
        slices=slices,
        delta_act_sidx=delta_act_sidx,
        all_ends_p=all_ends_p,
        all_ends_o=all_ends_o,
    )
    return all_abs_actions, all_delta_actions