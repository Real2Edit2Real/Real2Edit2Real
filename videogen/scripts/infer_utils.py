StatisticInfo = {
    "agibotworld": {
        "mean": [
            2.949369387467578e-05,
            -2.4703978620566167e-09,
            -3.214520512208581e-05,
            -0.00010488863132125087,
            6.125598648383931e-06,
            9.914418478635093e-05,
            2.5206249689794423e-05,
            2.386436953835608e-07,
            -3.6538445098187414e-05,
            -0.0001170051017729845,
            9.653312579896854e-06,
            -0.0001260203458056615
        ],
        "std": [
            0.0008946039574389581,
            0.0011014855217635904,
            0.0010237522979384508,
            0.010402864950142636,
            0.0049353766216338945,
            0.010759416673279353,
            0.000943621432585622,
            0.0013782102901814636,
            0.0011818999106504536,
            0.007539298011753274,
            0.00432025757794631,
            0.00814967736114264
        ],
        "max": [
            0.01856310599640343,
            0.03177942708584999,
            0.018952805852612054,
            3.0865110456701466,
            0.1787040522510046,
            3.105806772923132,
            0.016432826227830488,
            0.026423381566343274,
            0.02303190146027201,
            2.568170220208282,
            0.10684547518938814,
            2.559759198673244
        ],
        "min": [
            -0.015601498225847443,
            -0.02534869126080025,
            -0.02138581039999865,
            -3.0028421423096114,
            -0.18314801566285777,
            -2.9975689101543246,
            -0.018059042080331267,
            -0.03005789448249585,
            -0.031778412202921325,
            -3.0621243038649673,
            -0.09253045909575608,
            -3.079023907601185
        ],
        "q01": [
            -0.0027658579522003824,
            -0.0031980807400494816,
            -0.003292214464989222,
            -0.018250476162738016,
            -0.01660465544913178,
            -0.021129296758958454,
            -0.00297611722802993,
            -0.004738506975966939,
            -0.003575277531292638,
            -0.017990562138815742,
            -0.013047587006098461,
            -0.021908274158528515
        ],
        "q99": [
            0.003141418866128098,
            0.0039143937835317754,
            0.0033975673798853686,
            0.018703783106675375,
            0.01459661027523962,
            0.021610170142870165,
            0.002918105104810307,
            0.0040321486424608165,
            0.003686890595378792,
            0.017915328572414856,
            0.014709717457957535,
            0.01960137494174451
        ],
    }
}

import torch
import numpy as np
import torchvision.transforms as transforms
import cv2
from einops import rearrange
from lib.data.utils.beta_dataset.utils import intrinsic_transform, get_transformation_matrix_from_quat
from lib.data.utils.beta_dataset.traj_vis_statistics import ColorMapLeft, ColorMapRight, ColorListLeft, ColorListRight, EndEffectorPts, Gripper2EEFCvt

def transform_video(video, specific_transforms_resize, intrinsic, sample_size, depth=None, canny=None):
        """
        crop (optional) and resize the videos, and modify the intrinsic accordingly
        """
        c, t, h, w = video.shape
        if depth is not None and not depth.shape == video.shape:
            resize = transforms.Resize(video.shape[-2:])
            depth = resize(depth)
        if canny is not None and not canny.shape == video.shape:
            resize = transforms.Resize(video.shape[-2:])
            canny = resize(canny)

        intrinsic = intrinsic_transform(intrinsic, (h, w), sample_size, 'resize')
        video = specific_transforms_resize(video)
        if depth is not None:
            depth = specific_transforms_resize(depth)
        if canny is not None:
            canny = specific_transforms_resize(canny)
        return video, intrinsic, depth, canny

def normalize_video(video, specific_transforms_norm):
    """
    input video should have shape (c,t,h,w)
    """
    video = specific_transforms_norm(video.permute(1,0,2,3)).permute(1,0,2,3)
    return video

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


def transform_extrinsic_sequence(extrinsic, new_extrinsic):
    """
    Pre-multiply each matrix in the extrinsic sequence with a new extrinsic matrix.
    from cam_to_cam2 and cam2_to_cam3 compute cam_to_cam3.

    Args:
        extrinsic: np.ndarray, shape (S, 4, 4) (cam_to_cam2)
        new_extrinsic: np.ndarray, shape (S, 4, 4) (cam2_to_cam3)

    Returns:
        new_extrinsic_seq: np.ndarray, shape (S, 4, 4)
    """
    S = extrinsic.shape[0]
    if len(new_extrinsic.shape) == 2:
        new_extrinsic = new_extrinsic.unsqueeze(0).repeat(S, 1, 1)
    return torch.bmm(new_extrinsic, extrinsic)

def get_traj(sample_size, pose, w2c, c2w, intrinsic, radius=50):
        """
        this function takes camera info. and eef. poses as inputs, and outputs the trajectory maps.
        output traj map shape: (c, t, h, w)
        """        
        h, w = sample_size

        if isinstance(pose, np.ndarray):
            pose = torch.tensor(pose, dtype=torch.float32)
        
        ee_key_pts = torch.tensor(EndEffectorPts, dtype=torch.float32, device=pose.device).view(1,4,4).permute(0,2,1)

        ### t, 4, 4
        pose_l_mat = get_transformation_matrix_from_quat(pose[:, 0:7])
        pose_r_mat = get_transformation_matrix_from_quat(pose[:, 8:15])

        ### t, 4, 4
        ee2cam_l = torch.matmul(w2c, pose_l_mat)
        ee2cam_r = torch.matmul(w2c, pose_r_mat)

        cvt_matrix = torch.tensor(Gripper2EEFCvt, dtype=torch.float32, device=pose.device).view(1,4,4)
        ee2cam_l = torch.matmul(ee2cam_l, cvt_matrix)
        ee2cam_r = torch.matmul(ee2cam_r, cvt_matrix)
        
        ### t, 4, 4
        pts_l = torch.matmul(ee2cam_l, ee_key_pts)
        pts_r = torch.matmul(ee2cam_r, ee_key_pts)
        
        ### 1, 3, 3
        intrinsic = intrinsic.unsqueeze(0)

        ### t, 3, 4
        uvs_l = torch.matmul(intrinsic, pts_l[:,:3,:])
        uvs_l = (uvs_l / pts_l[:,2:3,:])[:,:2,:].permute(0,2,1).to(dtype=torch.int64)

        ### t, 3, 4
        uvs_r = torch.matmul(intrinsic, pts_r[:,:3,:])
        uvs_r = (uvs_r / pts_r[:,2:3,:])[:,:2,:].permute(0,2,1).to(dtype=torch.int64)

        img_list = []

        for i in range(pose.shape[0]):
            
            img = np.zeros((h, w, 3), dtype=np.uint8) + 50

            ###
            ### Gripper Range in AgiBotWorld < 120
            normalized_value_l = pose[i, 7].item() / 120
            normalized_value_r = pose[i, 15].item() / 120
            color_l = ColorMapLeft(normalized_value_l)[:3]  # Get RGB values
            color_r = ColorMapRight(normalized_value_r)[:3]  # Get RGB values
            color_l = tuple(int(c * 255) for c in color_l)
            color_r = tuple(int(c * 255) for c in color_r)

            i_coord_list = []
            for points, color, colors, lr_tag in zip([uvs_l[i], uvs_r[i]], [color_l, color_r], [ColorListLeft, ColorListRight], ["left", "right"]):
                base = np.array(points[0])
                if base[0]<0 or base[0]>=w or base[1]<0 or base[1]>=h:
                    continue
                point = np.array(points[0][:2])
                cv2.circle(img, tuple(point), radius, color, -1)
                

            for points, color, colors, lr_tag in zip([uvs_l[i], uvs_r[i]], [color_l, color_r], [ColorListLeft, ColorListRight], ["left", "right"]):
                base = np.array(points[0]) # points:[4,3]
                if base[0]<0 or base[0]>=w or base[1]<0 or base[1]>=h:
                    continue
                for i, point in enumerate(points):
                    point = np.array(point[:2])
                    if i == 0:
                        continue
                    else:
                        cv2.line(img, tuple(base), tuple(point), colors[i-1], 8)

            img_list.append(img/255.)

        img_list = np.stack(img_list, axis=0) ### t,h,w,c
        img_list = rearrange(torch.tensor(img_list), "t h w c -> c t h w").float()

        return img_list