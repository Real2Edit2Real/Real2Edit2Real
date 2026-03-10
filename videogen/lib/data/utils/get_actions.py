import numpy as np
import os
import h5py
from scipy.spatial.transform import Rotation
import torch

### convert from Link7_l to Link_hand_l
EEF2CamLeft = [0,0,-0.5236]
### convert from Link7_r to Link_hand_r
EEF2CamRight = [0,0,0.5236]


def normalize_angles(radius):
    radius_normed = np.mod(radius, 2 * np.pi) - 2 * np.pi * (np.mod(radius, 2 * np.pi) > np.pi)
    return radius_normed


def get_actions(gripper, all_ends_p=None, all_ends_o=None, slices=None, delta_act_sidx=None, get_delta_act=False):

    if slices is None:
        n = all_ends_p.shape[0]
        slices = list(range(n))
    else:
        n = len(slices)
    
    if get_delta_act:
        if delta_act_sidx is None:
            delta_act_sidx = 1
        else:
            assert(delta_act_sidx > 1)

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
        left_vis_rpy = np.concatenate((all_ends_p[i,0], rot_l.as_euler("xyz", degrees=False)), axis=0)

        rot_r = Rotation.from_quat(all_ends_o[i, 1])
        rot_vis_r = rot_r*cvt_vis_r
        right_vis_quat = np.concatenate((all_ends_p[i,1], rot_vis_r.as_quat()), axis=0)
        right_vis_rpy = np.concatenate((all_ends_p[i,1], rot_r.as_euler("xyz", degrees=False)), axis=0)

        all_left_rpy.append(left_vis_rpy)
        all_right_rpy.append(right_vis_rpy)
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
    if get_delta_act:
        ### xyz, rpy, gripper
        all_delta_actions = np.zeros([n-delta_act_sidx, 14])
    else:
        all_delta_actions = None
    for i in range(0, n):
        all_abs_actions[i, 0:7] = all_left_quat[i, :7]
        all_abs_actions[i, 7] = gripper[slices[i], 0]
        all_abs_actions[i, 8:15] = all_right_quat[i, :7]
        all_abs_actions[i, 15] = gripper[slices[i], 1]
        if get_delta_act:
            if i >= delta_act_sidx:
                all_delta_actions[i-delta_act_sidx, 0:6] = all_left_rpy[i, :6] - all_left_rpy[i-1, :6]
                all_delta_actions[i-delta_act_sidx, 3:6] = normalize_angles(all_delta_actions[i-delta_act_sidx, 3:6])
                all_delta_actions[i-delta_act_sidx, 6] = gripper[slices[i], 0] / 120.0
                all_delta_actions[i-delta_act_sidx, 7:13] = all_right_rpy[i, :6] - all_right_rpy[i-1, :6]
                all_delta_actions[i-delta_act_sidx, 10:13] = normalize_angles(all_delta_actions[i-delta_act_sidx, 10:13])
                all_delta_actions[i-delta_act_sidx, 13] = gripper[slices[i], 1] / 120.0

    return all_abs_actions, all_delta_actions


def parse_h5(h5_file, slices=None, delta_act_sidx=1, get_delta_act=False):
    """
    read and parse .h5 file, and obtain the absolute actions and the action differences
    """
    with h5py.File(h5_file, "r") as fid:
        if "effector" in fid["state"]:
            all_abs_gripper = np.array(fid[f"state/effector/position"], dtype=np.float32)
        else:
            all_abs_gripper_l = np.array(fid[f"state/left_effector/position"], dtype=np.float32)
            all_abs_gripper_r = np.array(fid[f"state/right_effector/position"], dtype=np.float32)
            all_abs_gripper = np.concatenate((all_abs_gripper_l, all_abs_gripper_r), axis=1)
        all_ends_p = np.array(fid["state/end/position"], dtype=np.float32)
        all_ends_o = np.array(fid["state/end/orientation"], dtype=np.float32)

    all_abs_actions, all_delta_actions = get_actions(
        gripper=all_abs_gripper,
        slices=slices,
        delta_act_sidx=delta_act_sidx,
        all_ends_p=all_ends_p,
        all_ends_o=all_ends_o,
        get_delta_act=get_delta_act
    )
    return all_abs_actions, all_delta_actions



def parse_npy(npy_file, slices=None, delta_act_sidx=1, get_delta_act=False):
    
    n = len(slices)
    abs_act = torch.FloatTensor(torch.load(npy_file, weights_only=False))[slices, :]
    abs_act = torch.cat((
        abs_act[:, 28:35],
        abs_act[:, 13:14],
        abs_act[:, 35:42],
        abs_act[:, 27:28],
    ),dim=1)

    if get_delta_act:
        ### convert link_hand back to link_7
        cvtback_vis_l = Rotation.from_euler("xyz", -1*np.array(EEF2CamLeft))
        cvtbcak_vis_r = Rotation.from_euler("xyz", -1*np.array(EEF2CamRight))
        rpy_act_l = []
        rpy_act_r = []
        for i in range(n):
            if i>=delta_act_sidx-1:
                i_rpy_l = (Rotation.from_quat(abs_act[i, 3:7])*cvtback_vis_l).as_euler("xyz", degrees=False)
                i_rpy_r = (Rotation.from_quat(abs_act[i, 11:15])*cvtbcak_vis_r).as_euler("xyz", degrees=False)
                rpy_act_l.append(torch.FloatTensor(i_rpy_l))
                rpy_act_r.append(torch.FloatTensor(i_rpy_r))

        delta_act = torch.zeros(n-delta_act_sidx, 14).float()
        for i in range(n-delta_act_sidx):
            
            delta_act[i, 0:3] = abs_act[i+delta_act_sidx, 0:3]-abs_act[i+delta_act_sidx-1, 0:3]
            delta_act[i, 3:6] = rpy_act_l[i+1]-rpy_act_l[i]
            delta_act[i, 6] = abs_act[i+delta_act_sidx, 7]/120.0

            delta_act[i, 7:10] = abs_act[i+delta_act_sidx, 8:11]-abs_act[i+delta_act_sidx-1, 8:11]
            delta_act[i, 10:13] = rpy_act_r[i+1]-rpy_act_r[i]
            delta_act[i, 13] = abs_act[i+delta_act_sidx, 14]/120.0

    else:
        delta_act = None
            
    return abs_act, delta_act

