
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import random
import math
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from einops import rearrange
import glob
from moviepy.editor import VideoFileClip
import torchvision.transforms as transforms
import jsonlines
from tqdm import tqdm
import torch.nn.functional as F
import cv2
import imageio

from lib.data.utils.beta_dataset.domain_table import DomainTable
from lib.data.utils.beta_dataset.statistics import StatisticInfo
from lib.data.utils.beta_dataset.traj_vis_statistics import ColorMapLeft, ColorMapRight, ColorListLeft, ColorListRight, EndEffectorPts, Gripper2EEFCvt
from lib.data.utils.beta_dataset.utils import intrinsic_transform, gen_crop_config, intrin_crop_transform, get_transformation_matrix_from_quat, add_depth_noise, depth_to_canny
from lib.data.utils.beta_dataset.get_actions import parse_h5

from lib.data.utils.beta_dataset.utils_geometry import closed_form_inverse_se3, transform_extrinsic_sequence


class AgiBotWorld(Dataset):
    def __init__(self,
        data_roots,
        preprocess_root,
        domains,
        specific_tasks=None,
        sample_size=(320,512), 
        sample_n_frames=64,
        preprocess = 'resize',
        valid_cam = 'head',
        chunk=1,
        n_previous=-1,
        previous_pick_mode='uniform',
        action_dim=7,
        random_crop=True,
        min_sep=1,
        max_sep=3,
        fps=2,
        split='train',
        use_unified_prompt=True, 
        use_ori_extrinsic=True,
        depth_aug_prob=0.0,
        **kwargs
    ):
        self.random_crop = random_crop
        self.action_dim = action_dim
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.valid_cam = valid_cam
        self.preprocess_root = preprocess_root
        self.use_unified_prompt = use_unified_prompt
        self.decode_type = 'cpu'
        self.ignore_seek = False
        self.use_ori_extrinsic = use_ori_extrinsic
        self.depth_aug_prob = depth_aug_prob

        self.data_roots = data_roots
        self.dataset = []
        for _data_root, _domain_name in zip(self.data_roots, domains):
            valid_tasks = os.listdir(preprocess_root)
            # valid_tasks.sort()
            # split dataset
            split_ratio = 0.9
            split_index = int(len(valid_tasks) * split_ratio)
            if split == 'train':
                valid_tasks = valid_tasks[:split_index]
            elif split == 'validation':
                valid_tasks = valid_tasks[split_index:]
                print(valid_tasks)
            print(f"{split} task num: {len(valid_tasks)}")
            for task in tqdm(valid_tasks):
                if (specific_tasks is None) or (task in specific_tasks):
                    episode_list = glob.glob(os.path.join(_data_root, "observations", task, "*"))
                    episode_list.sort()
                    for episode in episode_list:
                        if not self.check_episode_valid(_data_root, episode):
                            continue
                        episode_id = os.path.basename(episode)
                        info = [
                            episode,
                            os.path.join(_data_root, "parameters", task, episode_id),
                            os.path.join(_data_root, "proprio_stats", task, episode_id),
                            _domain_name, DomainTable[_domain_name],
                            os.path.join(preprocess_root, task, episode_id)
                        ]
                        self.dataset.append(info)
        self.length = len(self.dataset)
        print(f"{split} episode num: {len(self.dataset)}")

        self.chunk = chunk
        self.sample_n_frames = sample_n_frames
        
        self.sample_size = sample_size

        if preprocess == 'center_crop_resize':
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(min(sample_size)),  # the size of shape (1,) means the smaller edge will be resized to it and the img will keep the h-w ratio.
                transforms.CenterCrop(sample_size),
            ])
        if preprocess == 'resize':
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(sample_size),
            ])
        self.pixel_transforms_norm = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.onechannel_transforms_norm = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5], inplace=True),
        ])
        self.preprocess = preprocess
        self.random_erasing = None #transforms.RandomErasing(p=0.8, value=(-1.,0.38,-0.5), inplace=False)

        if n_previous > 1:
            self.n_previous = n_previous
            self.previous_pick_mode = previous_pick_mode
        else:
            self.n_previous = self.sample_n_frames - self.chunk
            self.previous_pick_mode = 'uniform'

        self.fps = fps
    def check_episode_valid(self, data_root, episode):
        """
        check if the episode is valid
        """
        if isinstance(self.valid_cam, str):
            valid_cam = [self.valid_cam]
        else:
            valid_cam = self.valid_cam
        
        for cam_name in valid_cam:
            # check if the episode has valid preprocessed results
            preprocess_path = episode.replace(os.path.join(data_root, "observations"), self.preprocess_root)
            parameter_path = episode.replace(os.path.join(data_root, "observations"), os.path.join(data_root, "parameters"))
            if not (os.path.exists(os.path.join(preprocess_path, cam_name+"_depth_ori")) and os.path.exists(os.path.join(preprocess_path, cam_name+"_depth_canny"))):
                return False
            if not os.path.exists(os.path.join(parameter_path, "parameters", "camera", cam_name+"_intrinsic_params.json")):
                return False
            if not os.path.exists(os.path.join(parameter_path, "parameters", "camera", cam_name+"_extrinsic_params_aligned.json")):
                return False
            if not os.path.exists(os.path.join(preprocess_path, cam_name+"_extrinsic.npy")):
                return False
        return True

    def get_total_timesteps(self, data_root, cam_name):
        with open(os.path.join(data_root, "parameters", "camera", cam_name+"_extrinsic_params_aligned.json"), "r") as f:
            info = json.load(f)
        total_frames = len(info)
        return total_frames


    def get_frame_indexes(self, total_frames, sep=1, ):
        """
        select self.n_previous memory frames and self.chunk prediction frmaes
        1. randomly select the end frame
        2. take frames from {end-chunk*sep} to {end} as the prediction frames
        3. uniformly/randomly select memory frames from {end-self.sample_n_frames*sep} to {end-chunk*sep}
        """
        chunk_end = random.randint(self.chunk*sep, total_frames)
        indexes = np.array(list(range(chunk_end-self.sample_n_frames*sep, chunk_end, sep)))
        indexes = np.clip(indexes, a_min=1, a_max=total_frames-1).tolist()
        video_end = indexes[-self.chunk:]
        mem_candidates = [indexes[int(i)] for i in np.linspace(0, self.sample_n_frames-self.chunk-1, self.sample_n_frames-self.chunk).tolist()]
        if self.previous_pick_mode == 'uniform':
            mem_indexes = [mem_candidates[int(i)] for i in np.linspace(0, len(mem_candidates)-1, self.n_previous).tolist()]

        elif self.previous_pick_mode == 'random':
            mem_indexes = [indexes[i] for i in sorted(np.random.choice(list(range(0,len(mem_candidates)-1)), size=self.n_previous-1, replace=False).tolist())] + [mem_candidates[-1]]
            """
            if random.random() < 0.5:
                mem_indexes = [indexes[i] for i in sorted(np.random.choice(list(range(0,len(mem_candidates)-1)), size=self.n_previous-1, replace=False).tolist())] + [mem_candidates[-1]]
            else:
                mem_indexes = [indexes[i] for i in sorted(np.random.choice(list(range(0,len(mem_candidates))), size=self.n_previous, replace=False).tolist())]
            """
        else:
            raise NotImplementedError(f"unsupported previous_pick_mode: {self.previous_pick_mode}")
        frame_indexes = mem_indexes + video_end
        return frame_indexes


    def get_action_bias_std(self, domain_name):
        return torch.tensor(StatisticInfo[domain_name]['mean']).unsqueeze(0), torch.tensor(StatisticInfo[domain_name]['std']).unsqueeze(0)


    def get_action(self, h5_file, slices, domain_name):
        """
        1. extract actions from .h5 files
        2. obatin End Effector actions and delta_action:
           action (t, 16)                      : {xyz, quat(xyzw), gripper} * 2
           delta_action (t-self.n_previous, 14): {xyz, quat(rpy),  gripper} * 2
        """
        action, delta_action = parse_h5(h5_file, slices=slices, delta_act_sidx=self.n_previous)
        action = torch.FloatTensor(action)
        delta_action = torch.FloatTensor(delta_action)
        delta_act_meanv, delta_act_stdv = self.get_action_bias_std(domain_name)
        delta_action[:, :6] = (delta_action[:, :6] - self.max_sep*delta_act_meanv[:, :6]) / (self.max_sep*delta_act_stdv[:, :6])
        delta_action[:, 7:13] = (delta_action[:, 7:13] - self.max_sep*delta_act_meanv[:, 6:]) / (self.max_sep*delta_act_stdv[:, 6:])

        return action, delta_action

    def get_depth(self, preprocess_episode, cam_name, slices, norm_depth=True):
        """
        get depth video frames according to the input slices;
        output video shape: (c,t,h,w)
        """
        depth_root = os.path.join(preprocess_episode, cam_name+"_depth_ori")
        depth = []
        for idx in slices:
            depth_path = os.path.join(depth_root, f"{idx}.png")
            depth_mm = imageio.imread(depth_path).astype(np.float32)
            depth_m = depth_mm / 1000.0
            if norm_depth:
                depth_min = depth_m.min()
                depth_max = depth_m.max()
                depth_m = (depth_m - depth_min) / (depth_max - depth_min)
            depth.append(depth_m)
        depth =  torch.from_numpy(np.stack(depth ,axis=0)).unsqueeze(0).contiguous()
        return depth.repeat(3, 1, 1, 1)
    
    def normalize_depth_list_global(self, depth_list, specific_transforms_norm):
        """
        Globally normalize the entire depth list to [0, 1], and then further normalize to [-1, 1].
        
        Args:
            depth_list (list[torch.Tensor]): 
                Length V, where each element has shape (C, T, H, W), C=3.
        Returns:
            list[torch.Tensor]: Normalized depth list.
        """
        # First find the global minimum and maximum values (look at the first channel only)
        min_val = float('inf')
        max_val = float('-inf')
        for depth in depth_list:
            d = depth[0]  # Only take the first channel (T, H, W)
            min_val = min(min_val, d.min().item())
            max_val = max(max_val, d.max().item())

        # Avoid division by zero
        if max_val == min_val:
            raise ValueError("All depth values are the same, cannot normalize.")

        # Perform normalization
        norm_list = []
        for depth in depth_list:
            norm_depth = (depth - min_val) / (max_val - min_val)
            norm_depth = specific_transforms_norm(norm_depth.permute(1,0,2,3)).permute(1,0,2,3)
            norm_list.append(norm_depth)
        
        return norm_list
    
    def normalize_depth_list_per_timestep(self, depth_list, specific_transforms_norm):
        """
        Normalize depth values to [0, 1] for each time step across all views.
        C=3 (RGB-like) but all channels are identical copies of depth.
        Args:
            depth_list (list[torch.Tensor]):
                Length V, each element is shape (C, T, H, W).
        Returns:
            list[torch.Tensor]: Normalized depth list.
        """
        V = len(depth_list)
        C, T, H, W = depth_list[0].shape

        # Find per-time-step min and max across all views (only first channel)
        min_vals = torch.full((T,), float('inf'), dtype=torch.float32, device=depth_list[0].device)
        max_vals = torch.full((T,), float('-inf'), dtype=torch.float32, device=depth_list[0].device)

        for depth in depth_list:
            d = depth[0]  # shape: (T, H, W)
            min_vals = torch.minimum(min_vals, d.view(T, -1).min(dim=1).values)
            max_vals = torch.maximum(max_vals, d.view(T, -1).max(dim=1).values)
        
        # Avoid division by zero
        same_mask = (max_vals == min_vals)
        if same_mask.any():
            raise ValueError(f"Some time steps have constant depth values, cannot normalize: {same_mask.nonzero().squeeze()}")

        # Normalize each time step independently across all views
        norm_list = []
        for depth in depth_list:
            d = depth.clone()
            for t in range(T):
                d[:, t] = (d[:, t] - min_vals[t]) / (max_vals[t] - min_vals[t])
            d = specific_transforms_norm(d.permute(1,0,2,3)).permute(1,0,2,3)
            norm_list.append(d)

        return norm_list

    def get_canny(self, preprocess_episode, cam_name, slices):
        """
        get canny video frames according to the input slices;
        output video shape: (c,t,h,w)
        """
        canny_root = os.path.join(preprocess_episode, cam_name+"_depth_canny")
        canny = []
        for idx in slices:
            canny_path = os.path.join(canny_root, f"{idx}.png")
            canny_edge = imageio.imread(canny_path).astype(np.uint8)
            canny.append(canny_edge)
        canny =  torch.from_numpy(np.stack(canny ,axis=0)).unsqueeze(0).contiguous()
        canny = canny.float() / 255.0
        return canny.repeat(3, 1, 1, 1)


    def seek_mp4(self, video_root, cam_name, slices):
        """
        seek video frames according to the input slices;
        output video shape: (c,t,h,w)
        """
        video_reader = VideoFileClip(os.path.join(video_root, "videos", cam_name+'_color.mp4'))
        fps = video_reader.fps
        video = []
        for idx in slices:
            video.append(video_reader.get_frame(float(idx)/fps))
        video = torch.from_numpy(np.stack(video)).permute(3, 0, 1, 2).contiguous()
        video = video.float()/255.
        video_reader.close()
        return video

    def get_intrin_and_extrin(self, cam_name, data_root, slices,):
        """
        get the intrinsic (3x3), c2ws (Tx4x4) and w2cs (Tx4x4) tensors
        """
        with open(os.path.join(data_root, "parameters", "camera", cam_name+"_intrinsic_params.json"), "r") as f:
            info = json.load(f)["intrinsic"]
        intrinsic = torch.eye(3, dtype=torch.float)
        intrinsic[0,0] = info["fx"]
        intrinsic[1,1] = info["fy"]
        intrinsic[0,2] = info["ppx"]
        intrinsic[1,2] = info["ppy"]

        with open(os.path.join(data_root, "parameters", "camera", cam_name+"_extrinsic_params_aligned.json"), "r") as f:
            info = json.load(f)
        c2ws = []
        w2cs = []
        for _i in slices:
            _i_info = info[_i]
            c2w = torch.eye(4, dtype=torch.float)
            c2w[:3, :3] = torch.FloatTensor(_i_info["extrinsic"]["rotation_matrix"])
            c2w[:3, -1] = torch.FloatTensor(_i_info["extrinsic"]["translation_vector"])
            w2c = torch.linalg.inv(c2w)
            c2ws.append(c2w)
            w2cs.append(w2c)
        c2ws = torch.stack(c2ws, dim=0)
        w2cs = torch.stack(w2cs, dim=0)
        return intrinsic, c2ws, w2cs
    
    def load_pred_extrinsics_from_npy(self, cam_name, preprocess_episode, slices):
        """
        load the preprocessed extrinsics from .npy files (w2c)
        select extrinsics with slices and transform to c2w format
        output shape: (T, 4, 4)
        """
        extrinsic_path = os.path.join(preprocess_episode, cam_name+"_extrinsic.npy")
        extrinsics = np.load(extrinsic_path)
        extrinsics = torch.from_numpy(extrinsics[slices]).float()
        extrinsics = closed_form_inverse_se3(extrinsics)  # convert w2c to c2w

        return extrinsics


    def transform_video(self, video, specific_transforms_resize, intrinsic, sample_size, depth=None, canny=None):
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

        if self.random_crop:
            h_start, w_start, h_crop, w_crop = gen_crop_config(video)
            video = video[:,:,h_start:h_start+h_crop,w_start:w_start+w_crop]
            if depth is not None:
                depth = depth[:,:,h_start:h_start+h_crop,w_start:w_start+w_crop]
            if canny is not None:
                canny = canny[:,:,h_start:h_start+h_crop,w_start:w_start+w_crop]
            intrinsic = intrin_crop_transform(intrinsic, h_start, w_start)
            h, w = h_crop, w_crop
        intrinsic = intrinsic_transform(intrinsic, (h, w), sample_size, self.preprocess)
        video = specific_transforms_resize(video)
        if depth is not None:
            depth = specific_transforms_resize(depth)
        if canny is not None:
            canny = specific_transforms_resize(canny)
        return video, intrinsic, depth, canny


    def normalize_video(self, video, specific_transforms_norm):
        """
        input video should have shape (c,t,h,w)
        """
        video = specific_transforms_norm(video.permute(1,0,2,3)).permute(1,0,2,3)
        return video


    def get_transform(self, ):
        sample_size = self.sample_size
        specific_transforms_resize = self.pixel_transforms_resize
        specific_transforms_norm = self.pixel_transforms_norm
        return sample_size, specific_transforms_resize, specific_transforms_norm


    def get_traj(self, pose, w2c, c2w, intrinsic, radius=50):
        """
        this function takes camera info. and eef. poses as inputs, and outputs the trajectory maps.
        output traj map shape: (c, t, h, w)
        """        
        h, w = self.sample_size

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


    def get_batch_new(self, idx, debug=False):
        video_root = self.dataset[idx][0]
        caminfo_root = self.dataset[idx][1]
        h5_file = os.path.join(self.dataset[idx][2], "proprio_stats.h5")
        domain_name = self.dataset[idx][3]
        domain_id = self.dataset[idx][4]
        preprocess_episode = self.dataset[idx][5]

        if self.use_unified_prompt:
            caption = "best quality, consistent and smooth motion, realistic, clear and distinct,"
        else:
            caption = ''

        if isinstance(self.valid_cam, str):
            total_frames = self.get_total_timesteps(caminfo_root, self.valid_cam)
        else:
            total_frames = self.get_total_timesteps(caminfo_root, self.valid_cam[0])
        ### 
        ### random action-speed
        sep = random.randint(self.min_sep, self.max_sep)

        sample_size, specific_transforms_resize, specific_transforms_norm = self.get_transform()
        video_indexes = self.get_frame_indexes(total_frames, sep=sep, )
        action, delta_action = self.get_action(h5_file, video_indexes, domain_name)
        # get video, traj, depth, canny from multiview
        depth = []
        canny = []
        video = []
        traj_maps = []
        intrinsics = []
        c2ws = []
        w2cs = []
        if isinstance(self.valid_cam, str):
            valid_cam = [self.valid_cam]
        else:
            valid_cam = self.valid_cam
        for cam_name in valid_cam:
            ## get depth and canny
            depth_v = self.get_depth(preprocess_episode, cam_name, video_indexes, norm_depth=False) # (c, t, h, w)
            canny_v = self.get_canny(preprocess_episode, cam_name, video_indexes)
            
            intrinsics_v, c2ws_v, w2cs_v = self.get_intrin_and_extrin(cam_name, caminfo_root, video_indexes)
            if not self.use_ori_extrinsic:
                pred_extrinsics_v = self.load_pred_extrinsics_from_npy(cam_name, preprocess_episode, video_indexes) # (T, 4, 4) c2w
                # fix head_extrinsic to identity matrix
                if cam_name == 'head':
                    head_c2ws = c2ws_v
                else:
                    c2ws_v = transform_extrinsic_sequence(pred_extrinsics_v, head_c2ws)
                    w2cs_v = torch.linalg.inv(c2ws_v)
                
            ### c, total_frames, h, w
            video_v = self.seek_mp4(video_root, cam_name, video_indexes)
            video_v, intrinsics_v, depth_v, canny_v = self.transform_video(
                video_v, specific_transforms_resize, intrinsics_v, sample_size, depth_v, canny_v
            )

            if isinstance(video_v, (list, tuple)):
                video_v = torch.stack(video_v, dim=1)

            traj_maps_v = self.get_traj(action, w2cs_v, c2ws_v, intrinsics_v)

            video_v = self.normalize_video(video_v, specific_transforms_norm)
            traj_maps_v = self.normalize_video(traj_maps_v, specific_transforms_norm)
            # depth_v = self.normalize_video(depth_v, specific_transforms_norm)
            canny_v = self.normalize_video(canny_v, specific_transforms_norm)
            
            video.append(video_v)
            traj_maps.append(traj_maps_v)
            depth.append(depth_v)
            canny.append(canny_v)
            intrinsics.append(intrinsics_v)
            c2ws.append(c2ws_v)
            w2cs.append(w2cs_v)
        
        # depth = self.normalize_depth_list_per_timestep(depth, specific_transforms_norm)
        depth = self.normalize_depth_list_global(depth, specific_transforms_norm)

        video = torch.stack(video, dim=0).permute(1, 0, 2, 3, 4)  # (c, v, t, h, w)
        traj_maps = torch.stack(traj_maps, dim=0).permute(1, 0, 2, 3, 4)  # (c, v, t, h, w)
        depth = torch.stack(depth, dim=0).permute(1, 0, 2, 3, 4)  # (c, v, t, h, w)
        canny = torch.stack(canny, dim=0).permute(1, 0, 2, 3, 4)  # (c, v, t, h, w)
        intrinsics = torch.stack(intrinsics, dim=0) # (v, 3, 3)
        c2ws = torch.stack(c2ws, dim=0) # (v, t, 4, 4)
        w2cs = torch.stack(w2cs, dim=0) # (v, t, 4, 4)

        if self.depth_aug_prob > 0:
            c, v, t, h, w = depth.shape
            depth = depth.permute(1, 2, 0, 3, 4).reshape(v*t, c, h, w)
            if random.random() < self.depth_aug_prob:
                depth = add_depth_noise(depth, mode="gaussian")
                
            if random.random() < self.depth_aug_prob:
                depth = add_depth_noise(depth, mode="missing")
            
            canny = depth_to_canny(depth)
            depth = depth.reshape(v, t, c, h, w).permute(2, 0, 1, 3, 4)
            canny = canny.reshape(v, t, c, h, w).permute(2, 0, 1, 3, 4)
            

        cond_id = -(self.n_previous+self.chunk)

        fps = self.fps

        return video, video_root, cond_id, intrinsics, c2ws, domain_id, action, traj_maps, delta_action, fps, depth, canny, caption

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        while True:
            try:
                video, video_root, cond_id, intrinsics, extrinsics, domain_id, action, traj_maps, delta_action, fps, depth, canny, caption = self.get_batch_new(idx)
                break
            except Exception as e:
                ### 
                idx = random.randint(0, self.length-1)
        
        # video, video_root, cond_id, intrinsics, extrinsics, domain_id, action, traj_maps, delta_action, fps, depth, canny = self.get_batch_new(idx)
        sample = dict(
            video=video, path=video_root,
            cond_id=cond_id, intrinsic=intrinsics, extrinsic=extrinsics, domain_id=domain_id,
            action=action, trajs=traj_maps, delta_action=delta_action, fps=fps, depths=depth, cannys=canny, caption=caption
        )
        
        return sample


