import numpy as np
import copy
import os
from termcolor import cprint
import imageio
from tqdm import tqdm
import h5py
import open3d as o3d
import json
import cv2
from PIL import Image
import pickle
import shutil
import sys
from scipy.spatial.transform import Rotation as R
import time

import torch

from pytorch3d.transforms import (
    quaternion_to_matrix, 
    matrix_to_quaternion, 
    quaternion_multiply, 
    Transform3d
)

from demo_generation.a2d_solver import (
    A2D_URDF_Processor,
    DualArmA2DSolver,
    LEFT_ARM_LINK_NAMES,
    RIGHT_ARM_LINK_NAMES
)
from demo_generation.geometry_utils import (
    unproject_depth_map_to_point_map, 
    project_pcd2image, 
    check_bbox_intersection, 
    check_bbox_y_disjoint_relative,
    cluster_filter_pointcloud,
)
from demo_generation.format_utils import (
    extrinsic_to_json,
    h5_to_dict,
    dict_to_h5,
    save_dict_to_h5,
)
from demo_generation.pose_utils import (
    transform_points, 
    transform_trans_quat,
    expand_extrinsic,
    action2mat,
    rotate_pose_around_point,
    rotate_n_aabbs_with_n_angles_around_center_z,
)
from demo_generation.vis_plotly import (
    visualize_pcd_and_bbox,
)
from demo_generation.vis_utils import (
    visualize_depth, 
    depth_canny,
    visualize_pre_computed_canny_video, 
    visualize_depth_video,
)

CAMERAS = ["head", "hand_left", "hand_right"]
arm_index = {
    "left": 0,
    "right": 1,
}

class AgibotReplayBuffer:
    def __init__(self, episode_path, preprocessed_episode_path, fixed_arms=["left"], 
                 target_cameras=["head"], mask_names=dict(), load_cache=False, 
                 relax_thresholds=None, disturbance=False, device=torch.device("cuda:0"), 
                 demogen_rank=0, cfg=None):
        self.buffer = dict()
        self.target_cameras = target_cameras
        self.mask_names = mask_names
        self.episode_path = episode_path
        self.preprocessed_episode_path = preprocessed_episode_path
        self.frame_ids = sorted([int(x) for x in os.listdir(f'{episode_path}/camera') if os.path.isdir(os.path.join(f'{episode_path}/camera', x))])
        self.device = device
        self.load_urdf(disturbance=disturbance)
        self.load_camera_params(episode_path)
        self.load_h5(episode_path, fixed_arms)
        self.load_desktop_plane(episode_path)
        self.fixed_arms = fixed_arms
        self.relax_thresholds = relax_thresholds
        self.cfg = cfg
        if load_cache:
            cache_path = os.path.join(self.preprocessed_episode_path, "0", "cache.pkl")
            if not os.path.exists(cache_path):
                self.load_all_ply(episode_path, target_cameras)
                if demogen_rank == 0:
                    cprint(f"only save cache for rank {demogen_rank}", "yellow")
                    self.save_cache(cache_path)
            else:
                self.load_cache(cache_path)
        else:
            self.load_all_ply(episode_path, target_cameras)

    def load_cache(self, cache_path):
        with open(cache_path, 'rb') as f:
            pickle_buffer = pickle.load(f)
            for key in pickle_buffer:
                self.buffer[key] = pickle_buffer[key]
            
    def save_cache(self, cache_path):
        with open(cache_path, 'wb') as f:
            pickle.dump(self.buffer, f)
    
    def load_urdf(self, urdf_path='data/A2D_120s/A2D_fixed_hw.urdf', urdf_path_mask='data/A2D_120s/A2D_fixed_hw_fixed.urdf', disturbance=False):
        #------------------------------------------
        # Init URDF processor, IK Solver and robot states
        #------------------------------------------
        self.urdf_processor = A2D_URDF_Processor(urdf_path=urdf_path, device=self.device)
        self.urdf_processor_mask = A2D_URDF_Processor(urdf_path=urdf_path_mask, device=self.device)
        self.ik_solver = DualArmA2DSolver(ik_type="motion_gen", disturbance=disturbance,)

    def get_base2w(self):
        lbase2w = self.urdf_processor.robot._scene.graph.get(frame_to='base_link_l', frame_from=None)[0]
        rbase2w = self.urdf_processor.robot._scene.graph.get(frame_to='base_link_r', frame_from=None)[0]
        self.lbase2w = torch.tensor(lbase2w).float().to(self.device)
        self.rbase2w = torch.tensor(rbase2w).float().to(self.device)
    
    def depthmap_to_world(self, depth_maps: np.ndarray, extrs: np.ndarray, intrs: np.ndarray, flatten: bool=True):
        depth_maps = torch.from_numpy(depth_maps).float().to(self.device)
        extrs = torch.from_numpy(extrs).float().to(self.device)
        intrs = torch.from_numpy(intrs).float().to(self.device)
        world_points = unproject_depth_map_to_point_map(
            depth_maps, 
            extrs,
            intrs,
        )
        if flatten:
            world_points = world_points.reshape(-1, 3)
        return world_points
    
    def load_world_point_frameid(self, frame_id, flatten: bool=True, to_numpy:bool=True, with_color=False):
        saved_dir = os.path.join(self.preprocessed_episode_path, frame_id)
        
        depth_maps = []
        extrs = []
        intrs = []
        if with_color:
            images = []
        for camera_name in self.target_cameras:
            image_path = os.path.join(saved_dir, f"{camera_name}_preprocessed_image.png")
            if frame_id == "background":
                suffix = "_depth_scaled"
                cam_frame_id = 0
            else:
                suffix = "_depth"
                cam_frame_id = int(frame_id)
            depth_map = np.load(os.path.join(saved_dir, f"{camera_name}{suffix}.npz"))["data"]
            extr = self.buffer["extrinsics_cam"][cam_frame_id, CAMERAS.index(camera_name)]
            intr = self.buffer["intrinsics_cam"][cam_frame_id, CAMERAS.index(camera_name)]
            if with_color:
                image = imageio.imread(image_path) / 255
            depth_maps.append(depth_map)
            extrs.append(extr)
            intrs.append(intr)
            if with_color:
                images.append(image)
        depth_maps = np.array(depth_maps)
        extrs = np.array(extrs)
        intrs = np.array(intrs)
        if with_color:
            images = np.array(images)
            if flatten:
                images = images.reshape(-1, 3)
        world_points = self.depthmap_to_world(depth_maps, extrs, intrs, flatten=flatten)
        head_c2w = torch.tensor(self.buffer["head_extrinsics_c2w"]).float().to(self.device)
        world_points = transform_points(head_c2w, world_points)
        if to_numpy:
            world_points = world_points.cpu().numpy()
            if with_color:
                world_points = np.concatenate([world_points, images], axis=-1)
        return world_points
    
    def load_all_ply(self, episode_path, target_cameras):
        #------------------------------------------
        # load depth and pcd from background
        #------------------------------------------
        self.buffer["bg_point_cloud"] = self.load_world_point_frameid("background", flatten=True, with_color=self.cfg.get("with_color", False))

        #------------------------------------------
        # load depth and pcd from every time
        #------------------------------------------
        self.buffer["world_point_cloud"] = []
        self.buffer["filter_world_point_cloud"] = []
        self.buffer["left_arm_ee_bbox"] = []
        self.buffer["right_arm_ee_bbox"] = []
        self.buffer["left_arm_bbox"] = []
        self.buffer["right_arm_bbox"] = []
        for frame_id in tqdm(self.frame_ids, desc="Loading PLY..."):
            saved_dir = os.path.join(self.preprocessed_episode_path, str(frame_id))
            world_points_list = []
            filter_world_points_list = []
            
            world_points = self.load_world_point_frameid(str(frame_id), flatten=False, to_numpy=False)
            world_points_head = world_points[CAMERAS.index("head")]
            if self.cfg.get("sam_for_all_frames", False):
                assert self.cfg.task_n_object == 2 and len(self.cfg.fixed_arms) == 0
                if frame_id >= self.cfg.parsing_frames.get("right-motion-1") or frame_id == 0:
                    print(f"saving sam for all frames {frame_id}")
                    self.buffer["world_point_cloud"].append(world_points.cpu().numpy())
                else:
                    self.buffer["world_point_cloud"].append(None)
            else:
                if frame_id == 0:
                    self.buffer["world_point_cloud"].append(world_points.cpu().numpy())

            head_c2w = torch.tensor(self.buffer["head_extrinsics_c2w"]).float().to(self.device)
            for camera_name in target_cameras:
                filter_point_clouds = o3d.io.read_point_cloud(os.path.join(saved_dir, f"{camera_name}_vggt_ori.ply"))
                filter_points = torch.tensor(np.asarray(filter_point_clouds.points)).float().to(self.device)
                filter_points = transform_points(head_c2w, filter_points).cpu().numpy()
                if self.cfg.get("with_color", False):
                    filter_colors = np.asarray(filter_point_clouds.colors)
                    filter_points = np.concatenate([filter_points, filter_colors], axis=-1)
                filter_world_points_list.append(filter_points)
            self.buffer["filter_world_point_cloud"].append(filter_world_points_list)
            
            # Compute arm meshes mask with URDF 
            state_angles = [self.state_dict["waist"]["position"][frame_id][1], self.state_dict["waist"]["position"][frame_id][0]] + self.state_dict["joint"]["position"][frame_id]
            state_angles_dict = self.urdf_processor.get_joint_angles_dict(joint_angles=state_angles, joint_names="whole")
            self.urdf_processor.update_robot(state_angles_dict)
            self.urdf_processor_mask.update_robot(state_angles_dict)
            c2w_head = self.buffer["head_extrinsics_c2w"] @ np.linalg.inv(expand_extrinsic(self.buffer["extrinsics_cam"][frame_id, CAMERAS.index("head")]))
            c2w_head = torch.tensor(c2w_head).float().to(self.device)
            intr_head = torch.tensor(self.buffer["intrinsics_cam"][frame_id, CAMERAS.index("head")]).float().to(self.device)
            _, mask_arm_links, _ = self.urdf_processor_mask.render_link_depth_and_mask(camera_intrinsics=intr_head, 
                                                                                        camera_extrinsics=c2w_head,
                                                                                        image_size=(518, 294),
                                                                                        link_names="left_right_arm")
            left_arm_link_bboxes = []
            right_arm_link_bboxes = []
            for link_idx, link_mask in enumerate(mask_arm_links):
                link_points = world_points_head[link_mask]
                if len(link_points) > 0:
                    link_points_min = torch.min(link_points, dim=0).values.cpu().numpy()
                    link_points_max = torch.max(link_points, dim=0).values.cpu().numpy()
                    link_bbox = np.array([link_points_min, link_points_max])
                    if link_idx < 7:
                        left_arm_link_bboxes.append(link_bbox)
                    else:
                        right_arm_link_bboxes.append(link_bbox)
            if len(left_arm_link_bboxes) == 0:
                left_arm_link_bboxes = np.empty((0, 2, 3))
            else:
                left_arm_link_bboxes = np.array(left_arm_link_bboxes)
            if len(right_arm_link_bboxes) == 0:
                right_arm_link_bboxes = np.empty((0, 2, 3))
            else:
                right_arm_link_bboxes = np.array(right_arm_link_bboxes)
            left_ee_bbox = self.urdf_processor_mask.get_link_bbox(link_names='left_ee')
            left_arm_ee_bbox = np.concatenate([left_arm_link_bboxes, np.expand_dims(left_ee_bbox, axis=0)], axis=0)
            right_ee_bbox = self.urdf_processor_mask.get_link_bbox(link_names='right_ee')
            right_arm_ee_bbox = np.concatenate([right_arm_link_bboxes, np.expand_dims(right_ee_bbox, axis=0)], axis=0)
            self.buffer["left_arm_ee_bbox"].append(left_arm_ee_bbox)
            self.buffer["left_arm_bbox"].append(left_arm_link_bboxes)
            self.buffer["right_arm_ee_bbox"].append(right_arm_ee_bbox)
            self.buffer["right_arm_bbox"].append(right_arm_link_bboxes)
            
    def load_object_or_target_ply(self, object_or_target, frame_id=None):        
        episode_path = self.episode_path
        object_or_target_clustereds = []
        camera_name = 'head'
        used_frame_id = self.frame_ids[0] if frame_id is None else frame_id
        saved_dir = os.path.join(self.preprocessed_episode_path, str(used_frame_id))
        care_obj_name = self.mask_names[object_or_target]
        care_obj_mask = cv2.imread(os.path.join(saved_dir, f"{camera_name}_{care_obj_name}.jpg"), cv2.IMREAD_GRAYSCALE)
        care_obj_mask = (care_obj_mask > 127).astype(bool)
        care_obj_mask = care_obj_mask.reshape(-1)
        world_points_colors = self.buffer["world_point_cloud"][used_frame_id][0].reshape(-1, 3)
        object_or_target_points = world_points_colors[care_obj_mask]
        object_or_target_clustered = cluster_filter_pointcloud(object_or_target_points, eps=0.03, min_samples=40, return_mask=False)
        object_or_target_clustereds.append(object_or_target_clustered)
        return object_or_target_clustered
            
    def load_h5(self, episode_path, fixed_arms=["left"]):
        '''
        state is absolute ee pose
        action is delta ee pose
        '''
        h5_file = os.path.join(episode_path, "aligned_joints.h5")
        with h5py.File(h5_file, 'r') as f:
            action_dict = h5_to_dict(f)
            self.state_dict = action_dict["state"]
            self.all_dict = action_dict
            joint_state_all = np.asarray(action_dict["state"]["joint"]["position"])
            state_all = np.concatenate([np.asarray(action_dict["state"]["end"]["position"]), 
                                        np.asarray(action_dict["state"]["end"]["orientation"])], axis=2)
            action_all = state_all.copy()
            state_angles = [self.state_dict["waist"]["position"][0][1], self.state_dict["waist"]["position"][0][0]] + self.state_dict["joint"]["position"][0]
            state_angles_dict = self.urdf_processor.get_joint_angles_dict(joint_angles=state_angles, joint_names="whole")
            self.urdf_processor.update_robot(state_angles_dict)
            self.urdf_processor_mask.update_robot(state_angles_dict)
        self.buffer["state"] = state_all
        self.buffer["action"] = action_all
        action_mat_left = action2mat(torch.tensor(action_all[:, 0, :]).float().to(self.device), quat_type='xyzw').unsqueeze(1)
        action_mat_right = action2mat(torch.tensor(action_all[:, 1, :]).float().to(self.device), quat_type='xyzw').unsqueeze(1)
        action_mat = torch.cat([action_mat_left, action_mat_right], dim=1)
        action_mat = action_mat.cpu().numpy()
        self.buffer["action_mat"] = action_mat
        self.buffer["joint_state"] = joint_state_all

    def load_desktop_plane(self, episode_path):
        saved_dir = os.path.join(self.preprocessed_episode_path, str(self.frame_ids[0]))
        self.desktop_plane = json.load(open(os.path.join(saved_dir, "desktop_plane.json")))
        plane_c = [self.desktop_plane["a"], \
                   self.desktop_plane["b"], \
                   self.desktop_plane["c"], \
                   self.desktop_plane["d"]]
        plane_c = np.array(plane_c)
        head_c2w = self.buffer["head_extrinsics_c2w"]
        head_w2c = np.linalg.inv(head_c2w)
        head_w2cT = head_w2c.T
        plane_w = head_w2cT @ plane_c
        self.desktop_plane_w = plane_w
        
    def load_camera_params(self, episode_path):
        head_extrinsics_path = os.path.join(episode_path, "parameters", "camera", "head_extrinsic_params_aligned.json")
        head_extrinsics_dict = json.load(open(head_extrinsics_path, 'r'))
        self.buffer["head_extrinsics_c2w"] = np.eye(4)
        self.buffer["head_extrinsics_c2w"][:3, :3] = head_extrinsics_dict[0]["extrinsic"]["rotation_matrix"]
        self.buffer["head_extrinsics_c2w"][:3,  3] = head_extrinsics_dict[0]["extrinsic"]["translation_vector"]
        
        self.buffer["extrinsics_cam"] = []
        self.buffer["intrinsics_cam"] = []
        
        for frame_id in self.frame_ids:
            saved_dir = os.path.join(self.preprocessed_episode_path, str(frame_id))
            cam_params_filename = f"camera_params.npz"
            cam_params_path = os.path.join(saved_dir, cam_params_filename)
            cam_params = np.load(cam_params_path)
            self.buffer["extrinsics_cam"].append(cam_params["extrinsics"])
            self.buffer["intrinsics_cam"].append(cam_params["intrinsics"])
        
        self.buffer["extrinsics_cam"] = np.array(self.buffer["extrinsics_cam"])
        self.buffer["intrinsics_cam"] = np.array(self.buffer["intrinsics_cam"])
    
    def get_episode(self, episode_idx):
        return self.buffer
    
class DemoGen:
    def __init__(self, cfg, device=None, demogen_rank=None):
        self.cfg = cfg
        self.data_root = cfg.data_root
        self.source_name = cfg.source_name
        self.output_root = cfg.output_root
        self.preprocess_root = cfg.preprocess_root
        self.task_n_object = cfg.task_n_object
        self.parsing_frames = cfg.parsing_frames
        self.mask_names = cfg.mask_names
        self.gen_name = cfg.generation.range_name
        self.object_trans_range = cfg.trans_range[self.gen_name]["object"]
        if self.task_n_object >= 2:
            self.target_trans_range = cfg.trans_range[self.gen_name]["target"]
        self.n_gen_per_source = cfg.generation.n_gen_per_source
        self.render_video = cfg.generation.get("render_video", False)
        self.render_depth = cfg.generation.get("render_depth", False)
        self.render_canny = cfg.generation.get("render_canny", False)
        self.render_file = cfg.generation.get("render_file", False)
        self.save_pcd = cfg.generation.get("save_pcd", False)
        self.disturbance = cfg.generation.get("disturbance", False)
        self.adjust_speed = cfg.generation.get("adjust_speed", False)
        self.prepare_only = cfg.generation.get("prepare_only", False)
        self.check_multiply = cfg.generation.get("check_multiply", 4)
        self.object_rot_range = cfg.rot_range[self.gen_name].get("object", [0.0, 0.0])
        if self.task_n_object >= 2:
            self.target_to_object = cfg.get("target_to_object", "left")
            self.target_rot_range = cfg.rot_range[self.gen_name].get("target", [0.0, 0.0])
        self.canny_thres2 = cfg.generation.get("canny_thres2", 10)
        if self.render_video:
            cprint("[NOTE] Rendering video is enabled. It takes ~10s to render a single generated trajectory.", "yellow")
        self.gen_mode = cfg.generation.mode
        if device is None:
            self.device = torch.device("cuda:0")
        else:
            self.device = device
        if demogen_rank is None:
            self.demogen_rank = 0
        else:
            self.demogen_rank = demogen_rank
        self._load_from_agibot(self.data_root, 
                                self.preprocess_root,
                                cfg.get("fixed_arms", []), 
                                cfg.get("target_cameras", ["head"]), 
                                cfg.get("mask_names", dict()),
                                cfg.get("load_cache", True),
                                cfg.get("relax_thresholds", None),
                                disturbance=self.disturbance)
    
    def _load_from_agibot(self, episode_path, preprocessed_episode_path, fixed_arms=["left"], target_cameras=["head"], mask_names=dict(), load_cache=True, relax_thresholds=None, disturbance=False):
        cprint(f"Loading data from {episode_path}", "blue")
        self.replay_buffer = AgibotReplayBuffer(episode_path, preprocessed_episode_path, fixed_arms, target_cameras, mask_names, load_cache=load_cache, relax_thresholds=relax_thresholds, disturbance=disturbance, device=self.device, demogen_rank=self.demogen_rank, cfg=self.cfg)
        self.n_source_episodes = 1
        self.demo_name = self.source_name
        
    def generate_trans_vectors(self, trans_range, n_demos, mode="random"):
        """
        Argument: trans_range: (2, 3)
            [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        Return: A list of translation vectors. (n_demos, 3)
        """
        x_min, x_max, y_min, y_max = trans_range[0][0], trans_range[1][0], trans_range[0][1], trans_range[1][1]
        if mode == "grid":
            n_side = int(np.sqrt(n_demos))
            # print(f"n_side: {n_side}, n_demos: {n_demos}")
            if n_side ** 2 != n_demos or n_demos == 1:
                raise ValueError("In grid mode, n_demos must be a squared number larger than 1")
            x_values = [x_min + i / (n_side - 1) * (x_max - x_min) for i in range(n_side)]
            y_values = [y_min + i / (n_side - 1) * (y_max - y_min) for i in range(n_side)]
            xyz = list(set([(x, y, 0) for x in x_values for y in y_values]))
            # assert len(xyz) == n_demos
            return np.array(xyz)
        elif mode == "random" or mode == "full_random":
            xyz = []
            for _ in range(n_demos):
                x = np.random.random() * (x_max - x_min) + x_min
                y = np.random.random() * (y_max - y_min) + y_min
                xyz.append([x, y, 0])
            return np.array(xyz)
        else:
            raise NotImplementedError
        
    def generate_offset_trans_vectors(self, offsets, trans_range, n_demos, mode="grid"):
        """
        For each point (translation vector) generate n_demos in trans_range.
        points_pos: (2, n_points)
            [[x1, x2, ..., xn], [y1, y2, ..., yn]]
        # NOTE: Small-range offsets are used in the experiments in our paper. However, we later found it is 
            in many times unnecessary, if we add random jitter augmentations to the point clouds when training the policy.
        """
        trans_vectors = []

        for x_offset, y_offset in zip(offsets[0], offsets[1]):
            trans_range_offset = copy.deepcopy(trans_range)
            trans_range_offset[0][0] += x_offset
            trans_range_offset[1][0] += x_offset
            trans_range_offset[0][1] += y_offset
            trans_range_offset[1][1] += y_offset
            trans_vector = self.generate_trans_vectors(trans_range_offset, n_demos, mode)
            trans_vectors.append(trans_vector)

        return np.concatenate(trans_vectors, axis=0)
    
    def get_objects_pcd_from_sam_mask(self, pcd, demo_idx, object_or_target="object", frame_id=None):
        return self.replay_buffer.load_object_or_target_ply(object_or_target, frame_id=frame_id)
    
    def clean_buffer(self):
        self.traj_states = []
        self.traj_actions = []
        self.traj_current_joint = []
        self.traj_current_lgripper = []
        self.traj_current_rgripper = []
        self.traj_pcds = []
        self.traj_depth_map = []
        self.traj_c2w_list = []
    
    def log_buffer(self, step_action, step_state, step_joint, step_lgripper, step_rgripper, cur_pcd, cur_depth_list, cur_c2w_list):
        self.traj_actions.append(step_action)
        self.traj_states.append(step_action)
        self.traj_current_joint.append(step_joint)
        self.traj_current_lgripper.append(step_lgripper)
        self.traj_current_rgripper.append(step_rgripper)
        self.traj_pcds.append(cur_pcd)
        self.traj_depth_map.append(cur_depth_list)
        self.traj_c2w_list.append(cur_c2w_list)
        
    def generate_demo(self):
        if self.task_n_object == 1:
            if len(self.cfg.get("fixed_arms", [])) == 0:
                self.edit_dual_arm_single_object(self.n_gen_per_source, self.render_video, self.gen_mode)
            else:
                self.edit_single_arm_single_object(self.n_gen_per_source, self.render_video, self.gen_mode)
        elif self.task_n_object == 2 or self.task_n_object == 3:
            if len(self.cfg.get("fixed_arms", [])) == 0:
                self.edit_dual_arm_multi_objects(self.n_gen_per_source, self.render_video, self.gen_mode)
            else:
                self.edit_single_arm_multi_objects(self.n_gen_per_source, self.render_video, self.gen_mode)
        else:
            raise NotImplementedError
    
    def depth_edit_kernel(self, source_demo, current_frame, obj_bbox, tar_bbox, obj_trans_vec_step, tar_trans_vec_step, 
                     left_arm_transform, left_camera_transform, left_arm_joints,
                     right_arm_transform, right_camera_transform, right_arm_joints, skill_type='motion-1',
                     support_obj_bbox=None, support_obj_trans_vec_step=None, fixed_arms=[]):
        task_n_object = self.task_n_object
        assert task_n_object in [3, 2, 1], f"Unsupported task_n_object {task_n_object}"
        source_pcd = np.concatenate(source_demo["filter_world_point_cloud"][current_frame], axis=0)
        obj_center = np.mean(obj_bbox, axis=0)
        obj_center_gpu = torch.tensor(obj_center).float().to(self.device)
        obj_trans_vec_step_gpu = torch.tensor(obj_trans_vec_step).float().to(self.device)
        if task_n_object in [2, 3]:
            tar_center = np.mean(tar_bbox, axis=0)
            tar_center_gpu = torch.tensor(tar_center).float().to(self.device)
            tar_trans_vec_step_gpu = torch.tensor(tar_trans_vec_step).float().to(self.device)
            if task_n_object == 3:
                support_obj_trans_vec_step_gpu = torch.tensor(support_obj_trans_vec_step).float().to(self.device)
        # 1. remove the arms' point clouds
        left_arm_sub_bboxes = source_demo["left_arm_bbox"][current_frame]
        right_arm_sub_bboxes = source_demo["right_arm_bbox"][current_frame]
        if "left" not in fixed_arms and self.cfg.get("no_arm_clip", False):
            print("no clip left")
        else:
            source_pcd = self.pcd_divide(source_pcd, left_arm_sub_bboxes)[-1]
        if "right" not in fixed_arms and self.cfg.get("no_arm_clip", False):
            print("no clip right")
        else:
            source_pcd = self.pcd_divide(source_pcd, right_arm_sub_bboxes)[-1]
        source_pcd_gpu = torch.tensor(source_pcd).float().to(self.device)
        
        # 2. find the points higher than desktop.
        direction_points = source_pcd_gpu[..., :3] @ torch.tensor(self.replay_buffer.desktop_plane_w[:3]).float().to(self.device) + self.replay_buffer.desktop_plane_w[3]
        if self.replay_buffer.desktop_plane_w[2] > 0:
            # if normal vector points to z+, use former face
            pcd_others = source_pcd_gpu[direction_points > self.cfg.relax_thresholds.get("desktop_z_thres", 0.02)]
        else:
            # if normal vector points to z-, use back face
            pcd_others = source_pcd_gpu[direction_points < -self.cfg.relax_thresholds.get("desktop_z_thres", 0.02)]
        pcd_others = pcd_others.cpu().numpy()
        
        # 3. divide pcd
        fixed_arm_ee_bboxes = np.empty((0, 2, 3))
        for fixed_arm in fixed_arms:
            fixed_arm_ee_bbox = source_demo[f"{fixed_arm}_arm_ee_bbox"][current_frame].copy()
            fixed_arm_ee_bbox[-1] = self.relax_bbox(fixed_arm_ee_bbox[-1], relax_val=np.array(self.cfg.relax_thresholds.get("ee", [0.05, 0.05, 0.05])))
            fixed_arm_ee_bboxes = np.concatenate([fixed_arm_ee_bboxes, fixed_arm_ee_bbox], axis=0)
        fixed_arm_ee_bboxes_tolist = []
        for bbox in fixed_arm_ee_bboxes:
            fixed_arm_ee_bboxes_tolist.append(bbox)
        fixed_arm_ee_bboxes = fixed_arm_ee_bboxes_tolist
        
        if "left" not in fixed_arms:
            arm_transform = left_arm_transform
        else:
            arm_transform = right_arm_transform
            
        if skill_type == "motion-1":
            if task_n_object == 3:
                cared_bboxes = [obj_bbox, tar_bbox, support_obj_bbox]
            elif task_n_object == 2:
                cared_bboxes = [obj_bbox, tar_bbox]
            elif task_n_object == 1:
                cared_bboxes = [obj_bbox]
            divided_results = self.pcd_divide(pcd_others, cared_bboxes + fixed_arm_ee_bboxes)
            # obj, tar, support obj, fixed arms, remained arm
            pcd_fixed_arm = divided_results[task_n_object:-1]
            pcd_remained_arm = divided_results[-1]
            pcd_obj = divided_results[0]
            pcd_obj_gpu = torch.tensor(pcd_obj).float().to(self.device)
            pcd_obj = self.pcd_transform(pcd_obj_gpu, obj_center_gpu, obj_trans_vec_step_gpu).cpu().numpy()
            if task_n_object >= 2:
                pcd_tar = divided_results[1]
                pcd_tar_gpu = torch.tensor(pcd_tar).float().to(self.device)
                pcd_tar = self.pcd_transform(pcd_tar_gpu, tar_center_gpu, tar_trans_vec_step_gpu).cpu().numpy()
                cared_pcd = [pcd_obj, pcd_tar]
                if task_n_object == 3:
                    pcd_support_obj = divided_results[2]
                    pcd_support_obj_gpu = torch.tensor(pcd_support_obj).float().to(self.device)
                    pcd_support_obj = self.pcd_transform(pcd_support_obj_gpu, obj_center_gpu, support_obj_trans_vec_step_gpu).cpu().numpy()
                    cared_pcd.append(pcd_support_obj)
            elif task_n_object == 1:
                cared_pcd = [pcd_obj]
            pcd_remained_arm_gpu = torch.tensor(pcd_remained_arm).float().to(self.device)
            pcd_remained_arm = transform_points(arm_transform, pcd_remained_arm_gpu).cpu().numpy() # self.pcd_transform(pcd_right_ee, pcd_transform_mat[j-motion_1_frame])
            cur_pcd = np.concatenate(cared_pcd + pcd_fixed_arm + [pcd_remained_arm], axis=0)
        elif skill_type == "skill-1" or skill_type == "motion-2":
            if task_n_object == 3:
                cared_objs = [tar_bbox, support_obj_bbox]
            elif task_n_object == 2:
                cared_objs = [tar_bbox]
            elif task_n_object == 1:
                cared_objs = []
            divided_results = self.pcd_divide(pcd_others, cared_objs + fixed_arm_ee_bboxes)
            # tar, support obj, fixed arms, remained arm
            if task_n_object >= 2:
                pcd_tar = divided_results[0]
                pcd_tar_gpu = torch.tensor(pcd_tar).float().to(self.device)
                pcd_tar = self.pcd_transform(pcd_tar_gpu, tar_center_gpu, tar_trans_vec_step_gpu).cpu().numpy()
                cared_pcd = [pcd_tar]
                if task_n_object == 3:
                    pcd_support_obj = divided_results[1]
                    pcd_support_obj_gpu = torch.tensor(pcd_support_obj).float().to(self.device)
                    pcd_support_obj = self.pcd_transform(pcd_support_obj_gpu, obj_center_gpu, support_obj_trans_vec_step_gpu).cpu().numpy()
                    cared_pcd.append(pcd_support_obj)
            else:
                cared_pcd = []
            pcd_fixed_arm = divided_results[(task_n_object-1):-1]
            pcd_remained_arm = divided_results[-1]
            pcd_remained_arm_gpu = torch.tensor(pcd_remained_arm).float().to(self.device)
            pcd_remained_arm = transform_points(arm_transform, pcd_remained_arm_gpu).cpu().numpy()
            cur_pcd = np.concatenate(cared_pcd + pcd_fixed_arm + [pcd_remained_arm], axis=0)
        elif skill_type == "skill-2":
            assert task_n_object in [2,3], f"Unsupported task_n_object {task_n_object} for skill-2"
            if task_n_object == 3:
                cared_objs = [support_obj_bbox]
            elif task_n_object == 2:
                cared_objs = []
            cared_pcd = []
            divided_results = self.pcd_divide(pcd_others, cared_objs + fixed_arm_ee_bboxes)
            if task_n_object == 3:
                pcd_support_obj = divided_results[0]
                pcd_support_obj_gpu = torch.tensor(pcd_support_obj).float().to(self.device)
                pcd_support_obj = self.pcd_transform(pcd_support_obj_gpu, obj_center_gpu, support_obj_trans_vec_step_gpu).cpu().numpy()
                cared_pcd.append(pcd_support_obj)
            pcd_fixed_arm = divided_results[(task_n_object-2):-1]
            pcd_remained_arm = divided_results[-1]
            pcd_remained_arm_gpu = torch.tensor(pcd_remained_arm).float().to(self.device)
            pcd_remained_arm = transform_points(arm_transform, pcd_remained_arm_gpu).cpu().numpy()
            cur_pcd = np.concatenate(cared_pcd + pcd_fixed_arm + [pcd_remained_arm], axis=0)
        elif skill_type == "copy":
            cur_pcd = pcd_others
        elif skill_type == "motion-1-dual":
            if task_n_object == 2:
                cared_bboxes = [tar_bbox, obj_bbox]
            elif task_n_object == 1:
                cared_bboxes = [obj_bbox]
            divide_results = self.pcd_divide(pcd_others, cared_bboxes + fixed_arm_ee_bboxes)
            pcd_fixed_arm = np.concatenate(divide_results[1:-1], axis=0)
            pcd_remained_arm = divide_results[-1]
            
            if task_n_object == 2:
                # for scan code, fix arm shouldn't be transformed
                pass
            elif task_n_object == 1:
                pcd_obj = divide_results[0]
                pcd_obj_gpu = torch.tensor(pcd_obj).float().to(self.device)
                pcd_obj = self.pcd_transform(pcd_obj_gpu, obj_center_gpu, obj_trans_vec_step_gpu).cpu().numpy()
                pcd_fixed_arm_gpu = torch.tensor(pcd_fixed_arm).float().to(self.device)
                pcd_fixed_arm = self.pcd_transform(pcd_fixed_arm_gpu, obj_center_gpu, obj_trans_vec_step_gpu).cpu().numpy()
            if task_n_object == 2:
                pcd_tar = divide_results[0]
                pcd_tar_gpu = torch.tensor(pcd_tar).float().to(self.device)
                pcd_tar = self.pcd_transform(pcd_tar_gpu, tar_center_gpu, tar_trans_vec_step_gpu).cpu().numpy()
                cared_pcd = [pcd_tar]
            elif task_n_object == 1:
                cared_pcd = [pcd_obj]
            pcd_remained_arm_gpu = torch.tensor(pcd_remained_arm).float().to(self.device)
            pcd_remained_arm = transform_points(arm_transform, pcd_remained_arm_gpu).cpu().numpy() # self.pcd_transform(pcd_right_ee, pcd_transform_mat[j-motion_1_frame])
            cur_pcd = np.concatenate(cared_pcd + [pcd_fixed_arm, pcd_remained_arm], axis=0)
        elif skill_type == "skill-1-dual" or skill_type == "motion-2-dual":
            if task_n_object == 2:
                cared_objs = [obj_bbox]
            elif task_n_object == 1:
                cared_objs = []
            divided_results = self.pcd_divide(pcd_others, cared_objs + fixed_arm_ee_bboxes)
            if task_n_object == 2:
                cared_pcd = [obj_bbox]
            elif task_n_object == 1:
                cared_pcd = []
            pcd_fixed_arm = np.concatenate(divided_results[0:-1], axis=0)
            if task_n_object == 2:
                # for scan code, fix arm shouldn't be transformed
                pass
            elif task_n_object == 1:
                pcd_fixed_arm_gpu = torch.tensor(pcd_fixed_arm).float().to(self.device)
                pcd_fixed_arm = self.pcd_transform(pcd_fixed_arm_gpu, obj_center_gpu, obj_trans_vec_step_gpu).cpu().numpy()
            pcd_remained_arm = divided_results[-1]
            pcd_remained_arm_gpu = torch.tensor(pcd_remained_arm).float().to(self.device)
            pcd_remained_arm = transform_points(arm_transform, pcd_remained_arm_gpu).cpu().numpy()
            cur_pcd = np.concatenate(cared_pcd + [pcd_fixed_arm, pcd_remained_arm], axis=0)

        ## 4. compute c2w
        c2w_head = self.replay_buffer.buffer["head_extrinsics_c2w"] @ np.linalg.inv(expand_extrinsic(self.replay_buffer.buffer["extrinsics_cam"][current_frame, 0]))
        c2w_left_arm = left_camera_transform @ self.replay_buffer.buffer["head_extrinsics_c2w"] @ np.linalg.inv(expand_extrinsic(self.replay_buffer.buffer["extrinsics_cam"][current_frame, 1]))
        c2w_right_arm = right_camera_transform @ self.replay_buffer.buffer["head_extrinsics_c2w"] @ np.linalg.inv(expand_extrinsic(self.replay_buffer.buffer["extrinsics_cam"][current_frame, 2]))
        c2w_list = [c2w_head, c2w_left_arm, c2w_right_arm]
        cur_c2w_list = np.array(c2w_list)
        
        urdf_processor = self.replay_buffer.urdf_processor
        urdf_processor_mask = self.replay_buffer.urdf_processor_mask
        joint_angles = self.replay_buffer.state_dict["joint"]["position"][0].copy()
        joint_angles[0:7] = left_arm_joints
        joint_angles[7:14] = right_arm_joints
        state_angles = [self.replay_buffer.state_dict["waist"]["position"][0][1], self.replay_buffer.state_dict["waist"]["position"][0][0]] + joint_angles
        state_angles_dict = urdf_processor_mask.get_joint_angles_dict(joint_angles=state_angles, joint_names="whole")
        urdf_processor.update_robot(state_angles_dict)
        urdf_processor_mask.update_robot(state_angles_dict)
        
        ## 5. compute depth
        cur_depth_list = []
        bg_pcd = torch.tensor(self.replay_buffer.buffer['bg_point_cloud'][..., :3]).float().to(self.device)
        fg_pcd = torch.tensor(cur_pcd[..., :3]).float().to(self.device)
        c2ws = []
        intrs = []
        for camera_i in range(len(self.replay_buffer.target_cameras)):
            c2ws.append(c2w_list[camera_i])
            intrs.append(self.replay_buffer.buffer["intrinsics_cam"][current_frame, camera_i])
        c2ws = np.array(c2ws)
        intrs = np.array(intrs)
        c2w = torch.tensor(c2ws).float().to(self.device)
        w2c = torch.linalg.inv(c2w)
        intr = torch.tensor(intrs).float().to(self.device)
        depth_map = project_pcd2image(
            bg_pcd,
            w2c=w2c,
            intrinsic=intr,
        )
        depth_map = project_pcd2image(
            fg_pcd,
            w2c=w2c,
            intrinsic=intr,
            image_bg=depth_map,
        )
        # NOTE: Optional, render gripper base for moving arm
        # if "left" in fixed_arms:
        #     depth_arm_head_left, mask_arm_head_left, _ = urdf_processor_mask.render_link_depth_and_mask(camera_intrinsics=intr[0:2], 
        #                                                                             camera_extrinsics=c2w[0:2],
        #                                                                             image_size=(518, 294),
        #                                                                             concat_meshes=True,
        #                                                                             link_names="left_right_arm")
        #     depth_arm_right, mask_arm_right, _ = urdf_processor_mask.render_link_depth_and_mask(camera_intrinsics=intr[2:3], 
        #                                             camera_extrinsics=c2w[2:3],
        #                                             image_size=(518, 294),
        #                                             concat_meshes=True,
        #                                             link_names='left_right_arm')   
        #     depth_arm = torch.cat([depth_arm_head_left, depth_arm_right], dim=0)                                               
        #     mask_arm = torch.cat([mask_arm_head_left, mask_arm_right], dim=0)
        # else:
        #     assert "right" in fixed_arms
        #     depth_arm_head_right, mask_arm_head_right, _ = urdf_processor_mask.render_link_depth_and_mask(camera_intrinsics=intr[[0,2]], 
        #                                                                             camera_extrinsics=c2w[[0,2]],
        #                                                                             image_size=(518, 294),
        #                                                                             concat_meshes=True,
        #                                                                             link_names="left_right_arm")
        #     depth_arm_left, mask_arm_left, _ = urdf_processor_mask.render_link_depth_and_mask(camera_intrinsics=intr[1:2], 
        #                                             camera_extrinsics=c2w[1:2],
        #                                             image_size=(518, 294),
        #                                             concat_meshes=True,
        #                                             link_names='left_right_arm')  
        #     depth_arm = torch.cat([depth_arm_head[0:1], depth_arm_left, depth_arm_head_right[1:2]], dim=0)                                               
        #     mask_arm = torch.cat([mask_arm_head[0:1], mask_arm_left, mask_arm_head_right[1:2]], dim=0)
        if not self.cfg.get("no_arm_rendering", False):
            depth_arm, mask_arm, _ = urdf_processor_mask.render_link_depth_and_mask(camera_intrinsics=intr, 
                                                                                    camera_extrinsics=c2w,
                                                                                    image_size=(518, 294),
                                                                                    concat_meshes=True,
                                                                                    link_names="left_right_arm")
            depth_map[mask_arm] = depth_arm[mask_arm]
        cur_depth_list = depth_map.cpu().numpy()
        cur_pcd = np.concatenate([self.replay_buffer.buffer['bg_point_cloud'], cur_pcd], axis=0)
        return cur_pcd, cur_c2w_list, cur_depth_list
    
    def prepare_pcd_and_bboxes(self):
        source_demo = self.replay_buffer.get_episode(0)
        pcds = source_demo["world_point_cloud"]
        pcd_obj = self.get_objects_pcd_from_sam_mask(pcds[0][0], 0, "object")
        pcd_tar = None
        pcd_support_obj = None
        obj_bbox = self.pcd_bbox(pcd_obj)
        obj_center = None
        tar_bbox = None
        tar_center = None
        support_obj_bbox = None
        
        if self.task_n_object >= 2:
            pcd_tar = self.get_objects_pcd_from_sam_mask(pcds[0][0], 0, "target")
            tar_bbox = self.pcd_bbox(pcd_tar)
            pcd_obj, pcd_tar, _ = self.pcd_divide(pcds[0].reshape(-1, 3), [obj_bbox, tar_bbox])
            if self.task_n_object == 3:
                pcd_support_obj = self.get_objects_pcd_from_sam_mask(pcds[0][0], 0, "support_object")
                pcd_support_obj = cluster_filter_pointcloud(pcd_support_obj, eps=0.03, min_samples=40, return_mask=False)
                support_obj_bbox = self.pcd_bbox(pcd_support_obj)
            pcd_tar = cluster_filter_pointcloud(pcd_tar, eps=0.03, min_samples=40, return_mask=False)
            target_bbox_thres = self.cfg.relax_thresholds.target
            tar_bbox = self.pcd_bbox(pcd_tar, x_lower=target_bbox_thres[0], x_upper=target_bbox_thres[1], y_lower=target_bbox_thres[2], y_upper=target_bbox_thres[3], z_upper=target_bbox_thres[5])
            tar_center = np.mean(tar_bbox, axis=0)
        else:
            pcd_obj, _ = self.pcd_divide(pcds[0].reshape(-1, 3), [obj_bbox])
            
        pcd_obj = cluster_filter_pointcloud(pcd_obj, eps=0.03, min_samples=40, return_mask=False)
        obj_bbox_thres = self.cfg.relax_thresholds.object
        obj_bbox = self.pcd_bbox(pcd_obj, x_lower=obj_bbox_thres[0], x_upper=obj_bbox_thres[1], y_lower=obj_bbox_thres[2], y_upper=obj_bbox_thres[3], z_lower=obj_bbox_thres[4], z_upper=obj_bbox_thres[5])
        
        urdf_processor = self.replay_buffer.urdf_processor
        urdf_processor_mask = self.replay_buffer.urdf_processor_mask
        state_angles = [self.replay_buffer.state_dict["waist"]["position"][0][1], self.replay_buffer.state_dict["waist"]["position"][0][0]] + self.replay_buffer.state_dict["joint"]["position"][0]
        state_angles_dict = urdf_processor.get_joint_angles_dict(joint_angles=state_angles, joint_names="whole")
        urdf_processor.update_robot(state_angles_dict)
        urdf_processor_mask.update_robot(state_angles_dict)
        self.replay_buffer.get_base2w()
        obj_center = np.mean(obj_bbox, axis=0)
        
        return pcds, pcd_obj, pcd_tar, pcd_support_obj, \
               obj_bbox, obj_center, tar_bbox, tar_center, support_obj_bbox
               
    def stage_prepare_initial_layout(self, 
                                    obj_bbox, tar_bbox, support_obj_bbox,
                                    prepare_frame, 
                                    obj_rot_angle, obj_trans_vec, tar_rot_angle, tar_trans_vec, 
                                    output_path, fixed_arms=[]):
        ############# stage {prepare} starts #############
        source_demo = self.replay_buffer.get_episode(0)
        fixed_arms = list(set(self.cfg.get("fixed_arms", []) + fixed_arms))
        for j in tqdm(range(prepare_frame), desc='prepare'):
            obj_rot_step = R.from_euler(seq='z', angles=obj_rot_angle / (prepare_frame) * j, degrees=True).as_quat()
            # xyz + xyzw
            obj_trans_vec_step = np.concatenate([obj_trans_vec / (prepare_frame) * j, obj_rot_step])
            tar_rot_step = None
            tar_trans_vec_step = None
            support_obj_trans_vec_step = None
            if self.task_n_object >= 2:
                tar_rot_step = R.from_euler(seq='z', angles=tar_rot_angle / (prepare_frame) * j, degrees=True).as_quat()
                tar_trans_vec_step = np.concatenate([tar_trans_vec / (prepare_frame) * j, tar_rot_step])

            if self.task_n_object == 3:
                # xyz + xyzw
                support_obj_trans_vec_step = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) # obj_trans_vec_step
                if self.cfg.get("change_z", False):
                    ori_obj_trans_vec = obj_trans_vec - self.x_extra_motion - self.z_extra_motion
                    if j < (prepare_frame // 3):
                        # xyz + xyzw
                        obj_rot_step = R.from_euler(seq='z', angles=obj_rot_angle / (prepare_frame // 3) * j, degrees=True).as_quat()
                        obj_trans_vec_step = np.concatenate([ori_obj_trans_vec / (prepare_frame // 3) * j, obj_rot_step])
                    else:
                        obj_rot_step = R.from_euler(seq='z', angles=obj_rot_angle, degrees=True).as_quat()
                        if j >= (prepare_frame // 3) and j < (2 * prepare_frame // 3):
                            motion = self.z_extra_motion
                            last_motion = np.array([0, 0, 0])
                            motion_ratio = (j - prepare_frame // 3) / (prepare_frame // 3)
                        elif j >= (2 * prepare_frame // 3):
                            motion = self.x_extra_motion
                            last_motion = self.z_extra_motion
                            motion_ratio = (j - 2 * prepare_frame // 3) / (prepare_frame // 3)
                        obj_trans_vec_step = np.concatenate([ori_obj_trans_vec + motion * motion_ratio + last_motion, obj_rot_step])
                else:
                    obj_rot_step = R.from_euler(seq='z', angles=obj_rot_angle / (prepare_frame) * j, degrees=True).as_quat()
                    # xyz + xyzw
                    obj_trans_vec_step = np.concatenate([obj_trans_vec / (prepare_frame) * j, obj_rot_step])
            
            
            left_right_buffer_list = [
                torch.eye(4).float().to(self.device),
                np.eye(4),
                source_demo['joint_state'][j][0:7],
                torch.eye(4).float().to(self.device),
                np.eye(4),
                source_demo['joint_state'][j][7:14],
            ]
            cur_pcd, cur_c2w_list, cur_depth_list = self.depth_edit_kernel(source_demo, 0, obj_bbox, tar_bbox, obj_trans_vec_step, tar_trans_vec_step, 
                *left_right_buffer_list,
                support_obj_bbox=support_obj_bbox if self.task_n_object == 3 else None,
                support_obj_trans_vec_step=support_obj_trans_vec_step,
                fixed_arms=fixed_arms,
            )
            step_action = source_demo["state"][0]
            step_joint = self.replay_buffer.all_dict['state']['joint']['position'][0].copy()
            step_lgripper = self.replay_buffer.all_dict['action']['left_effector']['position'][0].copy()
            step_rgripper = self.replay_buffer.all_dict['action']['right_effector']['position'][0].copy()
            self.log_buffer(
                step_action, step_action, step_joint, step_lgripper, step_rgripper, cur_pcd, cur_depth_list, cur_c2w_list
            )
            self.save_output_file(
                j, output_path, cur_depth_list, cur_pcd
            )
        ############# stage {prepare} ends #############
        
    def stage_motion(self,
                     start_frame,
                     end_frame,
                     obj_trans_vec,
                     tar_trans_vec,
                     obj_center,
                     obj_rot,
                     tar_center,
                     tar_rot,
                     obj_bbox,
                     tar_bbox,
                     support_obj_bbox,
                     cur_joint_states,
                     output_frameid_offset,
                     output_path,
                     skill_type="motion-1",
                     fixed_arms=[],
                     dual_arm_follow=False,
                     dual_arm_last_transform_mat_motion=None,
                     dual_joint=None,
                     is_barcode_motion2=False,
                     ):
        ############# stage {motion-1} starts #############
        source_demo = self.replay_buffer.get_episode(0)
        fixed_arms = list(set(self.cfg.get("fixed_arms", []) + fixed_arms))
        if "left" not in fixed_arms:
            used_single_arm = "left"
            unused_single_arm = "right"
        if "right" not in fixed_arms:
            used_single_arm = "right"
            unused_single_arm = "left"
        used_arm_index = arm_index[used_single_arm]
        unused_arm_index = arm_index[unused_single_arm]
        end_pos = source_demo["state"][end_frame-1, used_arm_index, :3]
        end_quat = source_demo["state"][end_frame-1, used_arm_index, 3:7]
        if skill_type == "motion-1":
            center = obj_center
            rotation = obj_rot
            start_trans_vec = np.zeros_like(obj_trans_vec)
            end_trans_vec = obj_trans_vec.copy()
        elif skill_type == "motion-2":
            if is_barcode_motion2:
                center = np.array([0, 0, 0])
                rotation = np.array([0, 0, 0, 1])
                start_trans_vec = obj_trans_vec.copy()
                end_trans_vec = np.zeros_like(obj_trans_vec)
            else:
                center = tar_center
                rotation = tar_rot
                start_trans_vec = obj_trans_vec.copy()
                end_trans_vec = tar_trans_vec.copy()
        elif skill_type == "motion-1-dual":
            assert dual_arm_follow, "dual_arm_follow must == True, if dual in skill_type"
            if self.task_n_object == 2:
                center = tar_center
                rotation = tar_rot
                start_trans_vec = np.zeros_like(tar_trans_vec)
                end_trans_vec = tar_trans_vec.copy()
            elif self.task_n_object == 1:
                center = obj_center
                rotation = obj_rot
                start_trans_vec = np.zeros_like(obj_trans_vec)
                end_trans_vec = obj_trans_vec.copy()
        elif skill_type == "motion-2-dual":
            assert dual_arm_follow, "dual_arm_follow must == True, if dual in skill_type"
            if self.task_n_object == 2:
                center = np.array([0, 0, 0])
                rotation = np.array([0, 0, 0, 1])
                start_trans_vec = tar_trans_vec.copy()
                end_trans_vec = np.zeros_like(tar_trans_vec)
                
        end_pos, end_quat = rotate_pose_around_point(end_pos, end_quat, center, rotation)
        end_pos += end_trans_vec.copy()
        solved_joint, solved_end_trans, solved_end_quat, solved_action_mat, pcd_transform_mat, adjusted_idxs = self.generate_continuous_path(
            end_pos,
            end_quat,
            cur_joint_states, # 7 dim
            arm_name=used_single_arm,
            ori_start_frame=start_frame,
            ori_end_frame=end_frame,
            ori_start_offset=start_trans_vec,
        )
        if solved_joint is None:
            if dual_arm_follow:
                return None, None, None, None
            return None, None, None
        
        if self.adjust_speed:
            motion_idxs = adjusted_idxs
        else:
            motion_idxs = np.arange(start_frame, end_frame)
            
        if dual_arm_follow:
            solved_dual_joint = dual_joint
            unused_skill_mat_ori = action2mat(torch.tensor(source_demo["state"][start_frame:end_frame, unused_arm_index]).float().to(self.device))
            unused_skill_mat_new = dual_arm_last_transform_mat_motion @ unused_skill_mat_ori
            unused_skill_trans_new = unused_skill_mat_new[:, :3, 3]
            # wxyz --> xyzw
            unused_skill_quat_new = matrix_to_quaternion(unused_skill_mat_new[:, :3, :3])[:, [1, 2, 3, 0]]
            unused_skill_action_new = torch.cat([unused_skill_trans_new, unused_skill_quat_new], dim=1).cpu().numpy()
            
        for motion_adjusted_idx, j in enumerate(tqdm(motion_idxs, desc=skill_type)):
            # xyz + xyzw
            step_action = source_demo["state"][j].copy()
            step_joint = self.replay_buffer.all_dict['state']['joint']['position'][j].copy()
            step_lgripper = self.replay_buffer.all_dict['action']['left_effector']['position'][j].copy()
            step_rgripper = self.replay_buffer.all_dict['action']['right_effector']['position'][j].copy()
            step_action[used_arm_index] = torch.cat([solved_end_trans[motion_adjusted_idx], solved_end_quat[motion_adjusted_idx]], dim=0).cpu().numpy()
            step_joint[used_arm_index*7:(used_arm_index+1)*7] = solved_joint[motion_adjusted_idx]
            if dual_arm_follow:
                step_action[unused_arm_index] = unused_skill_action_new[j-start_frame].copy()
            obj_trans_vec_step = np.concatenate([obj_trans_vec, obj_rot])
            if tar_trans_vec is None or tar_rot is None:
                tar_trans_vec_step = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            else:
                tar_trans_vec_step = np.concatenate([tar_trans_vec, tar_rot])
            support_obj_trans_vec_step = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            
            left_right_buffer_list = [
                torch.eye(4).float().to(self.device),
                np.eye(4),
                source_demo['joint_state'][j][0:7],
                torch.eye(4).float().to(self.device),
                np.eye(4),
                source_demo['joint_state'][j][7:14],
            ]
            
            left_right_buffer_list[used_arm_index*3:(used_arm_index+1)*3] = [
                pcd_transform_mat[motion_adjusted_idx],
                pcd_transform_mat[motion_adjusted_idx].cpu().numpy(),
                solved_joint[motion_adjusted_idx]
            ]
            
            if dual_arm_follow:
                solved_dual_joint = self.generate_stepbystep_path(
                    step_action[unused_arm_index, :3],
                    step_action[unused_arm_index, 3:7],
                    solved_dual_joint[-1], # 7 dim
                    arm_name=unused_single_arm,
                )
                left_right_buffer_list[unused_arm_index*3:(unused_arm_index+1)*3] = [
                    dual_arm_last_transform_mat_motion[0],
                    dual_arm_last_transform_mat_motion[0].cpu().numpy(),
                    solved_dual_joint[-1]
                ]
                step_joint[unused_arm_index*7:(unused_arm_index+1)*7] = solved_dual_joint[-1]
                if self.task_n_object == 2:
                    pcds = source_demo["world_point_cloud"]
                    pcd_obj = self.get_objects_pcd_from_sam_mask(pcds[j][0], 0, "object", frame_id=int(j))
                    obj_bbox = self.pcd_bbox(pcd_obj)
                    pcd_obj, _ = self.pcd_divide(pcds[j].reshape(-1, 3), [obj_bbox])
                    pcd_obj = cluster_filter_pointcloud(pcd_obj, eps=0.03, min_samples=40, return_mask=False)
                    obj_bbox_thres = self.cfg.relax_thresholds.object_rightmoving
                    obj_bbox = self.pcd_bbox(pcd_obj, x_lower=obj_bbox_thres[0], x_upper=obj_bbox_thres[1], y_lower=obj_bbox_thres[2], y_upper=obj_bbox_thres[3], z_lower=obj_bbox_thres[4], z_upper=obj_bbox_thres[5])

            cur_pcd, cur_c2w_list, cur_depth_list = self.depth_edit_kernel(source_demo, j, obj_bbox, tar_bbox, obj_trans_vec_step, tar_trans_vec_step, 
                                                                            *left_right_buffer_list,
                                                                            skill_type=skill_type,
                                                                            support_obj_bbox=support_obj_bbox if self.task_n_object == 3 else None,
                                                                            support_obj_trans_vec_step=support_obj_trans_vec_step if self.task_n_object == 3 else None,
                                                                            fixed_arms=fixed_arms,
                                                                            )
            self.log_buffer(
                step_action, step_action, step_joint, step_lgripper, step_rgripper, cur_pcd, cur_depth_list, cur_c2w_list
            )
            self.save_output_file(motion_adjusted_idx+output_frameid_offset, output_path, cur_depth_list, cur_pcd)
        ############## stage {motion-1} ends #############
        
        return_list = [pcd_transform_mat[-1:], solved_joint[-1:], len(motion_idxs)]
        if dual_arm_follow:
            return_list.append(solved_dual_joint[-1:])
        return return_list
    
    def stage_skill(self,
                    start_frame,
                    end_frame,
                    obj_trans_vec,
                    tar_trans_vec,
                    obj_center,
                    obj_rot,
                    tar_center,
                    tar_rot,
                    obj_bbox,
                    tar_bbox,
                    support_obj_bbox,
                    last_transform_mat_motion,
                    last_joint,
                    output_frameid_offset,
                    output_path,
                    skill_type="skill-1",
                    fixed_arms=[],
                    dual_arm_follow=False,
                    dual_arm_last_transform_mat_motion=None,
                    dual_joint=None,
                    ):
        source_demo = self.replay_buffer.get_episode(0)
        fixed_arms = list(set(self.cfg.get("fixed_arms", []) + fixed_arms))
        if "left" not in fixed_arms:
            used_single_arm = "left"
            unused_single_arm = "right"
        if "right" not in fixed_arms:
            used_single_arm = "right"
            unused_single_arm = "left"
        used_arm_index = arm_index[used_single_arm]
        unused_arm_index = arm_index[unused_single_arm]
        ############# stage {skill-1} starts #############
        skill_mat_ori = action2mat(torch.tensor(source_demo["state"][start_frame:end_frame, used_arm_index]).float().to(self.device))
        skill_mat_new = last_transform_mat_motion @ skill_mat_ori
        skill_trans_new = skill_mat_new[:, :3, 3]
        # wxyz --> xyzw
        skill_quat_new = matrix_to_quaternion(skill_mat_new[:, :3, :3])[:, [1, 2, 3, 0]]
        skill_action_new = torch.cat([skill_trans_new, skill_quat_new], dim=1).cpu().numpy()

        to_continue = False
        skill_idxs = np.arange(start_frame, end_frame)
        solved_joint = last_joint
            
        if dual_arm_follow:
            solved_dual_joint = dual_joint
            unused_skill_mat_ori = action2mat(torch.tensor(source_demo["state"][start_frame:end_frame, unused_arm_index]).float().to(self.device))
            unused_skill_mat_new = dual_arm_last_transform_mat_motion @ unused_skill_mat_ori
            unused_skill_trans_new = unused_skill_mat_new[:, :3, 3]
            # wxyz --> xyzw
            unused_skill_quat_new = matrix_to_quaternion(unused_skill_mat_new[:, :3, :3])[:, [1, 2, 3, 0]]
            unused_skill_action_new = torch.cat([unused_skill_trans_new, unused_skill_quat_new], dim=1).cpu().numpy()
            
        for skill_adjusted_idx, j in enumerate(tqdm(skill_idxs, desc=skill_type)):
            # xyz + xyzw
            step_action = source_demo["state"][j].copy()
            step_action[used_arm_index] = skill_action_new[j-start_frame].copy()
            if dual_arm_follow:
                step_action[unused_arm_index] = unused_skill_action_new[j-start_frame].copy()
            obj_trans_vec_step = np.concatenate([obj_trans_vec, obj_rot])
            if tar_trans_vec is None or tar_rot is None:
                tar_trans_vec_step = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            else:
                tar_trans_vec_step = np.concatenate([tar_trans_vec, tar_rot])
            support_obj_trans_vec_step = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

            solved_joint = self.generate_stepbystep_path(
                step_action[used_arm_index, :3],
                step_action[used_arm_index, 3:7],
                solved_joint[-1], # 7 dim
                arm_name=used_single_arm,
            )
            if solved_joint is None:
                if dual_arm_follow:
                    return None, None, None
                return None, None
            
            left_right_buffer_list = [
                torch.eye(4).float().to(self.device),
                np.eye(4),
                source_demo['joint_state'][j][0:7],
                torch.eye(4).float().to(self.device),
                np.eye(4),
                source_demo['joint_state'][j][7:14],
            ]
            
            left_right_buffer_list[used_arm_index*3:(used_arm_index+1)*3] = [
                last_transform_mat_motion[0],
                last_transform_mat_motion[0].cpu().numpy(),
                solved_joint[-1]
            ]

            if dual_arm_follow:
                solved_dual_joint = self.generate_stepbystep_path(
                    step_action[unused_arm_index, :3],
                    step_action[unused_arm_index, 3:7],
                    solved_dual_joint[-1], # 7 dim
                    arm_name=unused_single_arm,
                )
                left_right_buffer_list[unused_arm_index*3:(unused_arm_index+1)*3] = [
                    dual_arm_last_transform_mat_motion[0],
                    dual_arm_last_transform_mat_motion[0].cpu().numpy(),
                    solved_dual_joint[-1]
                ]
                if self.task_n_object == 2:
                    pcds = source_demo["world_point_cloud"]
                    pcd_obj = self.get_objects_pcd_from_sam_mask(pcds[j][0], 0, "object", frame_id=int(j))
                    obj_bbox = self.pcd_bbox(pcd_obj)
                    pcd_obj, _ = self.pcd_divide(pcds[j].reshape(-1, 3), [obj_bbox])
                    pcd_obj = cluster_filter_pointcloud(pcd_obj, eps=0.03, min_samples=40, return_mask=False)
                    obj_bbox_thres = self.cfg.relax_thresholds.object_rightmoving
                    obj_bbox = self.pcd_bbox(pcd_obj, x_lower=obj_bbox_thres[0], x_upper=obj_bbox_thres[1], y_lower=obj_bbox_thres[2], y_upper=obj_bbox_thres[3], z_lower=obj_bbox_thres[4], z_upper=obj_bbox_thres[5])

            step_lgripper = self.replay_buffer.all_dict['action']['left_effector']['position'][j].copy()
            step_rgripper = self.replay_buffer.all_dict['action']['right_effector']['position'][j].copy()
            step_joint = self.replay_buffer.all_dict['state']['joint']['position'][j].copy()
            step_joint[used_arm_index*7:(used_arm_index+1)*7] = solved_joint[-1]
            if dual_arm_follow:
                step_joint[unused_arm_index*7:(unused_arm_index+1)*7] = solved_dual_joint[-1]
            cur_pcd, cur_c2w_list, cur_depth_list = self.depth_edit_kernel(source_demo, j, obj_bbox, tar_bbox, obj_trans_vec_step, tar_trans_vec_step, 
                                                                            *left_right_buffer_list,
                                                                            skill_type=skill_type,
                                                                            support_obj_bbox=support_obj_bbox if self.task_n_object == 3 else None,
                                                                            support_obj_trans_vec_step=support_obj_trans_vec_step if self.task_n_object == 3 else None,
                                                                            fixed_arms=fixed_arms,
                                                                            )
            
            self.log_buffer(
                step_action, step_action, step_joint, step_lgripper, step_rgripper, cur_pcd, cur_depth_list, cur_c2w_list
            )
            self.save_output_file(skill_adjusted_idx+output_frameid_offset, output_path, cur_depth_list, cur_pcd)
        ############## stage {skill-1} ends #############
        return_list = [solved_joint[-1:], len(skill_idxs)]
        if dual_arm_follow:
            return_list.append(solved_dual_joint[-1:])
        return return_list
    
    def generate_continuous_path(self,
                                 end_trans,
                                 end_quat,
                                 cur_joint_states, # 7 dim
                                 arm_name,
                                 ori_start_frame=None,
                                 ori_end_frame=None,
                                 ori_start_offset=None,
                                 check_mode=False,
                                 ):
        source_demo = self.replay_buffer.get_episode(0)
        ik_solver = self.replay_buffer.ik_solver
        if ori_start_frame is not None:
            ori_start_trans = source_demo["state"][ori_start_frame, arm_index[arm_name], :3]
        if ori_start_offset is not None:
            start_trans = ori_start_trans + ori_start_offset
        if ori_end_frame is not None:
            ori_end_trans = source_demo["state"][ori_end_frame-1, arm_index[arm_name], :3]
        if ori_start_frame is not None and ori_end_frame is not None:
            ori_steps = (ori_end_frame - ori_start_frame)
        else:
            ori_steps = None
        if self.adjust_speed and ori_start_frame is not None and ori_end_frame is not None:
            # ori_distance / ori_steps == new_distance / target_steps
            if ori_steps is None:
                raise RuntimeError(f"Bad ori_steps {ori_steps} when planning motion.")
            ori_distance = np.sqrt(np.sum((ori_end_trans - ori_start_trans) ** 2)) # original distance between the two waypoints.
            target_distance = np.sqrt(np.sum(( end_trans - start_trans) ** 2)) # the distance of (target + target offset) <--> (object + object offset)
            target_steps = int(target_distance * ori_steps / ori_distance)
            cprint("Adjusting speed: ori_distance {:.2f}, target_distance {:.2f}, ori_steps {:d}, target_steps {:d}".format(ori_distance, target_distance, (ori_end_frame - ori_start_frame), target_steps))
        else:
            target_steps = ori_steps
        base2w = self.replay_buffer.lbase2w if arm_name == "left" else self.replay_buffer.rbase2w
        end_trans_arm, end_quat_arm = transform_trans_quat(torch.linalg.inv(base2w), 
                                            torch.tensor(end_trans).unsqueeze(0).float().to(self.device), 
                                            torch.tensor(end_quat).unsqueeze(0).float().to(self.device), 
                                            quat_type="xyzw")
        ik_result = ik_solver.solve_ik_by_motion_gen_single_arm(curr_joint_state=cur_joint_states,
                                                                target_trans=end_trans_arm[0].cpu().numpy(),
                                                                target_quat=end_quat_arm[0].cpu().numpy(),
                                                                arm=arm_name,
                                                                quat_type="xyzw",
                                                                steps=target_steps)
        if ik_result is None:
            return None, None, None, None, None, None
        if check_mode:
            return ik_result, None, None, None, None, None
        
        solved_joint = ik_result
        
        solved_end_trans_arm, solved_end_quat_arm = ik_solver.compute_fk_single_arm(
            ik_result,
            arm=arm_name,
            quat_type="xyzw",
        )
        solved_end_trans, solved_end_quat = transform_trans_quat(
            base2w, 
            torch.from_numpy(solved_end_trans_arm).float().to(self.device),
            torch.from_numpy(solved_end_quat_arm).float().to(self.device),
            quat_type="xyzw"
        )
        
        solved_action_mat = action2mat(torch.cat([solved_end_trans, solved_end_quat], dim=1))
        if ori_start_frame is None or ori_end_frame is None:
            pcd_transform_mat = None
            adjusted_idxs = None
        else:
            if self.adjust_speed:
                original_frameids = np.arange(ori_start_frame, ori_end_frame)
                original_indices = np.arange(len(original_frameids))
                target_indices = np.linspace(0, len(original_frameids)-1, target_steps)
                adjusted_idxs = np.interp(target_indices, original_indices, original_frameids).astype(int)
            else:
                adjusted_idxs = np.arange(ori_start_frame, ori_end_frame)
            pcd_transform_mat = solved_action_mat @ torch.linalg.inv(torch.from_numpy(source_demo["action_mat"][adjusted_idxs, arm_index[arm_name]]).float().to(self.device))
        return solved_joint, solved_end_trans, solved_end_quat, solved_action_mat, pcd_transform_mat, adjusted_idxs
         
    def generate_stepbystep_path(self,
                                 end_trans,
                                 end_quat,
                                 cur_joint_states, # 7 dim
                                 arm_name):
        base2w = self.replay_buffer.lbase2w if arm_name == "left" else self.replay_buffer.rbase2w
        ik_solver = self.replay_buffer.ik_solver
        end_trans_arm, end_quat_arm = transform_trans_quat(torch.linalg.inv(base2w), 
                                            torch.tensor(end_trans).unsqueeze(0).float().to(self.device), 
                                            torch.tensor(end_quat).unsqueeze(0).float().to(self.device), 
                                            quat_type="xyzw")
        ik_result = ik_solver.solve_ik_by_motion_gen_single_arm(curr_joint_state=cur_joint_states,
                                                                target_trans=end_trans_arm[0].cpu().numpy(),
                                                                target_quat=end_quat_arm[0].cpu().numpy(),
                                                                arm=arm_name,
                                                                quat_type="xyzw",
                                                                steps=None)
        return ik_result
    
    def generate_random_layouts(self, gen_mode, n_demos,
                                object_arm_keyframe_pairs, 
                                obj_bbox,
                                obj_center,
                                target_arm_keyframe_pairs=[], 
                                tar_bbox=None,
                                tar_center=None,):
        source_demo = self.replay_buffer.get_episode(0)
        fixed_arms = self.cfg.get("fixed_arms", [])
        ik_solver = self.replay_buffer.ik_solver
        trans_vectors, success_obj_rots, success_obj_rot_angles, success_tar_rots, success_tar_rot_angles = None, None, None, None, None
        if gen_mode == "full_random":
            obj_xyz = self.generate_trans_vectors(self.object_trans_range, self.cfg.generation.n_gen_per_source * self.check_multiply, mode=gen_mode)
        else:
            obj_xyz = self.generate_trans_vectors(self.object_trans_range, self.cfg.generation.n_grid_obj ** 2, mode=gen_mode)
        if self.task_n_object == 3:
            if self.cfg.get("change_z", False):
                self.x_extra_motion = np.array([0.2, 0, 0])
                self.z_extra_motion = np.array([0, 0, 0.08])
                obj_xyz = obj_xyz + self.x_extra_motion + self.z_extra_motion
        obj_rot_mu = (self.object_rot_range[0] + self.object_rot_range[1]) / 2.0
        obj_rot_sigma = (self.object_rot_range[1] - self.object_rot_range[0]) / 6.0
        obj_rot_angles = np.random.normal(
            loc=obj_rot_mu, 
            scale=obj_rot_sigma, 
            size=len(obj_xyz)
        )
        obj_rotation = R.from_euler(seq='z', angles=obj_rot_angles, degrees=True).as_quat()
        success_obj_xyz_idxs = []
        
        # check if these grids are reachable from current arm pose
        for o_idx, o_xyz in enumerate(tqdm(obj_xyz, desc='Checking object positions...')):
            ik_success = True
            for object_arm, object_frame, object_start_frame in object_arm_keyframe_pairs:
                used_arm_index = arm_index[object_arm]
                cur_joint_states = source_demo["joint_state"][object_start_frame].copy()
                end_pos = source_demo["state"][object_frame, arm_index[object_arm], :3]
                end_quat = source_demo["state"][object_frame, arm_index[object_arm], 3:7]
                end_pos, end_quat = rotate_pose_around_point(end_pos, end_quat, obj_center, obj_rotation[o_idx])
                end_pos += o_xyz.copy()
                solved_joint, _, _, _, _, _ = self.generate_continuous_path(
                                end_pos,
                                end_quat,
                                cur_joint_states[(used_arm_index)*7:(used_arm_index+1)*7], # 7 dim
                                arm_name=object_arm,
                                check_mode=True,
                )
                ik_success = ik_success & (solved_joint is not None)
            if ik_success:
                success_obj_xyz_idxs.append(True)
            else:
                success_obj_xyz_idxs.append(False)
        success_obj_xyz_idxs = np.array(success_obj_xyz_idxs)
        success_both_xyz_idxs = success_obj_xyz_idxs.copy()
            
        if self.task_n_object >= 2:
            if gen_mode == "full_random":
                targ_xyz = self.generate_trans_vectors(self.target_trans_range, self.cfg.generation.n_gen_per_source * self.check_multiply, mode=gen_mode)
            else:
                targ_xyz = self.generate_trans_vectors(self.target_trans_range, self.cfg.generation.n_grid_targ ** 2, mode=gen_mode)
            tar_rot_mu = (self.target_rot_range[0] + self.target_rot_range[1]) / 2.0
            tar_rot_sigma = (self.target_rot_range[1] - self.target_rot_range[0]) / 6.0
            tar_rot_angles = np.random.normal(
                loc=tar_rot_mu, 
                scale=tar_rot_sigma, 
                size=len(targ_xyz)
            )
            tar_rotation = R.from_euler(seq='z', angles=tar_rot_angles, degrees=True).as_quat()
            success_targ_xyz_idxs = []
            for t_idx, t_xyz in enumerate(tqdm(targ_xyz, desc='Checking target positions...')):
                ik_success = True
                for target_arm, target_frame, target_start_frame in target_arm_keyframe_pairs:
                    used_arm_index = arm_index[object_arm]
                    cur_joint_states = source_demo["joint_state"][target_start_frame].copy()
                    end_pos = source_demo["state"][target_frame, arm_index[target_arm], :3]
                    end_quat = source_demo["state"][target_frame, arm_index[target_arm], 3:7]
                    end_pos, end_quat = rotate_pose_around_point(end_pos, end_quat, tar_center, tar_rotation[t_idx])
                    end_pos += t_xyz.copy()
                    solved_joint, _, _, _, _, _ = self.generate_continuous_path(
                                end_pos,
                                end_quat,
                                cur_joint_states[(used_arm_index)*7:(used_arm_index+1)*7], # 14 dim
                                arm_name=target_arm,
                                check_mode=True,
                    )
                    ik_success = ik_success & (solved_joint is not None)
                if ik_success:
                    success_targ_xyz_idxs.append(True)
                else:
                    success_targ_xyz_idxs.append(False)
            success_targ_xyz_idxs = np.array(success_targ_xyz_idxs)
            success_both_xyz_idxs = np.logical_and(success_obj_xyz_idxs, success_targ_xyz_idxs)
            
        if gen_mode == "full_random":
            success_obj_xyzs = obj_xyz[success_both_xyz_idxs]
            success_obj_rot_angles = obj_rot_angles[success_both_xyz_idxs]
            success_obj_rots = obj_rotation[success_both_xyz_idxs]
            print(f"Filter object positions: {len(obj_xyz)} --> {len(success_obj_xyzs)}")
            if self.task_n_object >= 2:
                success_targ_xyzs = targ_xyz[success_both_xyz_idxs]
                success_tar_rot_angles = tar_rot_angles[success_both_xyz_idxs]
                success_tar_rots = tar_rotation[success_both_xyz_idxs]
                print(f"Filter target positions: {len(targ_xyz)} --> {len(success_targ_xyzs)}")
        else:
            success_obj_xyzs = obj_xyz[success_obj_xyz_idxs]
            success_obj_rot_angles = obj_rot_angles[success_obj_xyz_idxs]
            success_obj_rots = obj_rotation[success_obj_xyz_idxs]
            print(f"Filter object positions: {len(obj_xyz)} --> {len(success_obj_xyzs)}")
            if self.task_n_object >= 2:
                success_targ_xyzs = targ_xyz[success_targ_xyz_idxs]
                success_tar_rot_angles = tar_rot_angles[success_targ_xyz_idxs]
                success_tar_rots = tar_rotation[success_targ_xyz_idxs]
                print(f"Filter target positions: {len(targ_xyz)} --> {len(success_targ_xyzs)}")
        
        if self.task_n_object >= 2:
            checked_obj_bbox = obj_bbox
            obj_bboxes = np.tile(np.expand_dims(checked_obj_bbox, axis=0), (len(success_obj_xyzs), 1, 1))
            tar_bboxes = np.tile(np.expand_dims(tar_bbox, axis=0), (len(success_targ_xyzs), 1, 1))
            obj_transed_bboxes = rotate_n_aabbs_with_n_angles_around_center_z(obj_bboxes, success_obj_rots) + np.expand_dims(success_obj_xyzs,axis=1)
            targ_transed_bboxes = rotate_n_aabbs_with_n_angles_around_center_z(tar_bboxes, success_tar_rots) + np.expand_dims(success_targ_xyzs,axis=1)
            is_intersect = check_bbox_intersection(obj_transed_bboxes, targ_transed_bboxes, expand_dim=not gen_mode == "full_random")
            is_good_relative = check_bbox_y_disjoint_relative(obj_transed_bboxes, targ_transed_bboxes, target_to_object=self.target_to_object, expand_dim=not gen_mode == "full_random")
            valid_mask = np.logical_and(np.logical_not(is_intersect), is_good_relative)
            if gen_mode == "full_random":
                obj_idxs = np.arange(len(success_obj_xyzs))[valid_mask]
                targ_idxs = obj_idxs.copy()
                print(f"Filter intersection or bad relative position: {len(success_obj_xyzs)} --> {len(obj_idxs)}")
            else:
                obj_targ_meshgrid = np.meshgrid(np.arange(len(success_targ_xyzs)), np.arange(len(success_obj_xyzs)))
                obj_idxs = obj_targ_meshgrid[1][valid_mask]
                targ_idxs = obj_targ_meshgrid[0][valid_mask]
                print(f"Filter intersection or bad relative position: {len(success_targ_xyzs) * len(success_obj_xyzs)} --> {len(obj_idxs)}")
        else:
            obj_idxs = np.arange(len(success_obj_xyzs))
            
        if n_demos < len(obj_idxs):
            print(f"Filter layouts to n_demos: {len(obj_idxs)} --> {n_demos}")
            rand_idxs = np.random.choice(len(obj_idxs), n_demos, replace=False)
            obj_idxs_chosen = obj_idxs[rand_idxs]
            success_obj_rots = success_obj_rots[obj_idxs_chosen]
            success_obj_rot_angles = success_obj_rot_angles[obj_idxs_chosen]
            if self.task_n_object >= 2:
                targ_idxs_chosen = targ_idxs[rand_idxs]
                trans_vectors = np.concatenate([success_obj_xyzs[obj_idxs_chosen], success_targ_xyzs[targ_idxs_chosen]], axis=1)
                success_tar_rots = success_tar_rots[targ_idxs_chosen]
                success_tar_rot_angles = success_tar_rot_angles[targ_idxs_chosen]
            else:
                trans_vectors = success_obj_xyzs[obj_idxs_chosen]
        else:
            print(f"Layouts {len(obj_idxs)} <= {n_demos}. Use all.")
            success_obj_rots = success_obj_rots[obj_idxs]
            success_obj_rot_angles = success_obj_rot_angles[obj_idxs]
            if self.task_n_object >= 2:
                trans_vectors = np.concatenate([success_obj_xyzs[obj_idxs], success_targ_xyzs[targ_idxs]], axis=1)
                success_tar_rots = success_tar_rots[targ_idxs]
                success_tar_rot_angles = success_tar_rot_angles[targ_idxs]
            else:
                trans_vectors = success_obj_xyzs[obj_idxs]
                
        return trans_vectors, success_obj_rots, success_obj_rot_angles, success_tar_rots, success_tar_rot_angles
    
    def save_output_file(self, current_frame, output_path, cur_depth_list, cur_pcd=None):
        if self.render_file:
            for camera_i in range(len(self.replay_buffer.target_cameras)):
                depth_map = cur_depth_list[camera_i]
                canny_edge = depth_canny(depth_map, threshold1=0, threshold2=self.canny_thres2)
                depth_vis = visualize_depth(depth_map, max_depth=5.0)
                depth_save_path = os.path.join(output_path, f'{self.replay_buffer.target_cameras[camera_i]}_depth_ori')
                canny_save_path = os.path.join(output_path, f'{self.replay_buffer.target_cameras[camera_i]}_depth_canny')
                depth_render_save_path = os.path.join(output_path, f'{self.replay_buffer.target_cameras[camera_i]}_depth_render')
                os.makedirs(depth_save_path, exist_ok=True)
                os.makedirs(canny_save_path, exist_ok=True)
                os.makedirs(depth_render_save_path, exist_ok=True)
                depth_map_uint16 = (depth_map * 1000).astype(np.uint16)
                imageio.imwrite(os.path.join(depth_save_path, f"{current_frame}.png"), depth_map_uint16)
                imageio.imwrite(os.path.join(canny_save_path, f"{current_frame}.png"), canny_edge)
                cv2.imwrite(os.path.join(depth_render_save_path, f"{current_frame}.png"), depth_vis)
        if self.save_pcd and cur_pcd is not None:
            pcd_save_path = os.path.join(output_path, f'edited_pcd')
            os.makedirs(pcd_save_path, exist_ok=True)
            pcd_points = cur_pcd[..., :3]
            if self.cfg.get("with_color", False):
                pcd_colors = cur_pcd[..., 3:6]
            saved_pcd = o3d.geometry.PointCloud()
            saved_pcd.points = o3d.utility.Vector3dVector(pcd_points)
            saved_pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
            o3d.io.write_point_cloud(os.path.join(pcd_save_path, f"{current_frame}.ply"), saved_pcd)
    
    def save_camera(self, output_path):
        # ----------------------------------------------
        # Save camera parameters
        # ----------------------------------------------
        with open(os.path.join(output_path, 'head_extrinsic_params_aligned.json'), 'w') as json_file:
            extrinsic_dict = extrinsic_to_json(self.replay_buffer.buffer["head_extrinsics_c2w"])
            json.dump([extrinsic_dict], json_file, indent=4)
        hand_left_extrinsic_list = []
        hand_right_extrinsic_list = []
        for traj_c2w in self.traj_c2w_list:
            hand_left_extrinsic_list.append(traj_c2w[1])
            hand_right_extrinsic_list.append(traj_c2w[2])
        with open(os.path.join(output_path, 'hand_left_extrinsic_params_aligned.json'), 'w') as json_file:
            json_list = []
            for extrinsic in hand_left_extrinsic_list:
                json_list.append(extrinsic_to_json(extrinsic))
            json.dump(json_list, json_file, indent=4)
        with open(os.path.join(output_path, 'hand_right_extrinsic_params_aligned.json'), 'w') as json_file:
            json_list = []
            for extrinsic in hand_right_extrinsic_list:
                json_list.append(extrinsic_to_json(extrinsic))
            json.dump(json_list, json_file, indent=4)

        for camera_name in self.replay_buffer.target_cameras:
            with open(os.path.join(self.preprocess_root, '0', f'{camera_name}_adjusted_intrinsic.json'), 'r') as json_file:
                intrinsic_dict = json.load(json_file)
                new_intrinsic_dict = {"intrinsic":{}}
                new_intrinsic_dict["intrinsic"]['fx'] = intrinsic_dict['fx']
                new_intrinsic_dict["intrinsic"]['fy'] = intrinsic_dict['fy']
                new_intrinsic_dict["intrinsic"]['ppx'] = intrinsic_dict['cx']
                new_intrinsic_dict["intrinsic"]['ppy'] = intrinsic_dict['cy']
            with open(os.path.join(output_path, f'{camera_name}_intrinsic_params.json'), 'w') as json_file:
                json.dump(new_intrinsic_dict, json_file, indent=4)

    def save_states(self, output_path):
        # ----------------------------------------------
        # Save states
        # ----------------------------------------------
        traj_states = np.array(self.traj_states)
        new_all_dict = copy.deepcopy(self.replay_buffer.all_dict)
        new_all_dict['state']['end']['position'] = traj_states[:, :, 0:3].tolist()
        new_all_dict['state']['end']['orientation'] = traj_states[:, :, 3:7].tolist()
        new_all_dict['state']['robot']['position'] = np.zeros((len(traj_states), 3)).tolist()
        new_all_dict['state']['robot']['orientation'] = np.zeros((len(traj_states), 3)).tolist()
        new_all_dict['state']['joint']['position'] = np.array(self.traj_current_joint).tolist()
        new_all_dict['state']['joint']['current_value'] = []
        new_all_dict['state']['waist']['position'] = np.tile(np.array(new_all_dict['state']['waist']['position'][0:1]), (len(traj_states), 1)).tolist()
        new_all_dict['state']['head']['position'] = np.tile(np.array(new_all_dict['state']['head']['position'][0:1]), (len(traj_states), 1)).tolist()
        new_all_dict['action'] = dict()
        new_all_dict['action']['left_effector'] = dict()
        new_all_dict['action']['right_effector'] = dict()
        new_all_dict['action']['left_effector']['position'] = np.array(self.traj_current_lgripper).tolist()
        new_all_dict['action']['right_effector']['position'] = np.array(self.traj_current_rgripper).tolist()
        new_all_dict['timestamp'] = []
        save_dict_to_h5(new_all_dict, os.path.join(output_path, 'aligned_joints.h5'))
    
    def save_at_end(self, trans_vec_idx, output_path, obj_trans_vec, tar_trans_vec=None):
        if not self.render_file or self.render_depth or self.render_canny:
            traj_depth_map = np.stack(self.traj_depth_map, axis=0)  # (n_frames, n_cameras, h, w)
            traj_canny_map = np.zeros_like(traj_depth_map).astype(np.uint8)
            for frame_id in range(traj_depth_map.shape[0]):
                for cam_id in range(traj_depth_map.shape[1]):
                    traj_canny_map[frame_id, cam_id] = depth_canny(traj_depth_map[frame_id, cam_id], threshold1=0, threshold2=self.canny_thres2)
                    
        if not self.render_file:
            cprint(f">>> Demo {trans_vec_idx}: saving depths...", "yellow")
            np.savez(os.path.join(output_path, "depthmap.npz"), (traj_depth_map*1000).astype(np.uint16))
            cprint(f">>> Demo {trans_vec_idx}: saving cannys...", "yellow")
            np.savez(os.path.join(output_path, "depthmap_canny.npz"), traj_canny_map)
        
        if self.render_depth:
            cprint(f">>> Demo {trans_vec_idx}: saving depth video...", "yellow")
            if tar_trans_vec is None:
                video_name = f"{trans_vec_idx}_obj[{np.round(obj_trans_vec[0], 3)},{np.round(obj_trans_vec[1], 3)}]_depth.mp4"
            else:
                video_name = f"{trans_vec_idx}_obj[{np.round(obj_trans_vec[0], 3)},{np.round(obj_trans_vec[1], 3)}]_tar[{np.round(tar_trans_vec[0], 3)},{np.round(tar_trans_vec[1], 3)}]_depth.mp4"
            video_path = os.path.join(output_path, video_name)
            visualize_depth_video(traj_depth_map, video_path)
                                
        if self.render_canny:
            cprint(f">>> Demo {trans_vec_idx}: saving depth canny video...", "yellow")
            if tar_trans_vec is None:
                video_name = f"{trans_vec_idx}_obj[{np.round(obj_trans_vec[0], 3)},{np.round(obj_trans_vec[1], 3)}]_depth_canny.mp4"
            else:
                video_name = f"{trans_vec_idx}_obj[{np.round(obj_trans_vec[0], 3)},{np.round(obj_trans_vec[1], 3)}]_tar[{np.round(tar_trans_vec[0], 3)},{np.round(tar_trans_vec[1], 3)}]_depth_canny.mp4"
            video_path = os.path.join(output_path, video_name)
            visualize_pre_computed_canny_video(traj_canny_map, video_path)
            
    def edit_single_arm_multi_objects(self, n_demos, render_video=False, gen_mode='random'):
        pcds, pcd_obj, pcd_tar, pcd_support_obj, \
        obj_bbox, obj_center, tar_bbox, tar_center, support_obj_bbox = self.prepare_pcd_and_bboxes()
        source_demo = self.replay_buffer.get_episode(0)
        motion_1_frame = self.parsing_frames["motion-1"]
        skill_1_frame = self.parsing_frames["skill-1"]
        motion_2_frame = self.parsing_frames["motion-2"]
        skill_2_frame = self.parsing_frames["skill-2"]
        end_frame = self.parsing_frames.get("end", len(source_demo["state"])-1)
        prepare_frame = self.parsing_frames.get("prepare", 30)
        fixed_arms = self.cfg.get("fixed_arms", [])
        single_arm_name = list(set(["left", "right"]) - set(fixed_arms))
        used_arm_index = arm_index[single_arm_name[0]]
        
        # Prepare translation vectors
        if gen_mode == "list":
            trans_list = np.array(self.cfg.get("trans_list", []))
            success_obj_xyzs = trans_list[:, 0, :]
            success_targ_xyzs = trans_list[:, 1, :]
            trans_vectors = np.concatenate([success_obj_xyzs, success_targ_xyzs], axis=1)
            rot_list = np.array(self.cfg.get("rot_list", []))
            success_obj_rot_angles = rot_list[:, 0]
            success_tar_rot_angles = rot_list[:, 1]
            success_obj_rots = R.from_euler(seq='z', angles=success_obj_rot_angles, degrees=True).as_quat()
            success_tar_rots = R.from_euler(seq='z', angles=success_tar_rot_angles, degrees=True).as_quat()
        else:
            trans_vectors, \
            success_obj_rots, success_obj_rot_angles, \
            success_tar_rots, success_tar_rot_angles = self.generate_random_layouts(
                gen_mode, n_demos,
                object_arm_keyframe_pairs=[
                    (single_arm_name[0], skill_1_frame-1, 0),
                    (single_arm_name[0], motion_2_frame-1, 0),
                ],
                obj_bbox=obj_bbox,
                obj_center=obj_center,
                target_arm_keyframe_pairs=[
                    (single_arm_name[0], skill_2_frame-1, 0),
                    (single_arm_name[0], end_frame-1, 0),
                ],
                tar_bbox=tar_bbox,
                tar_center=tar_center,
            )
                    
        # For every source demo
        for i in range(self.n_source_episodes):
            cprint(f"Generating demos for source demo {i}", "blue")
            assert ("left" in fixed_arms and not "right" in fixed_arms) or ("right" in fixed_arms and not "left" in fixed_arms), "Only support one dynamic arm."
            cprint(f"Motion-1: {motion_1_frame}, Skill-1: {skill_1_frame}, Motion-2: {motion_2_frame}, Skill-2: {skill_2_frame}", "red")

            # Generate demos according to translation vectors
            for trans_vec_idx, trans_vec in enumerate(tqdm(trans_vectors, desc=f"Generating {n_demos} demonstrations...")):
                # init urdf_processor and ik_sovler
                cprint(f"start generating {trans_vec_idx}", "green")
                obj_trans_vec = trans_vec[:3]
                tar_trans_vec = trans_vec[3:6]
                obj_rot = success_obj_rots[trans_vec_idx]
                obj_rot_angle = success_obj_rot_angles[trans_vec_idx] 
                tar_rot = success_tar_rots[trans_vec_idx]
                tar_rot_angle = success_tar_rot_angles[trans_vec_idx]
                cprint(f"*****************************", "red")
                cprint(f"obj_trans_vec {obj_trans_vec}", "red")
                cprint(f"obj_rot_angle {obj_rot_angle}", "red")
                cprint(f"tar_trans_vec {tar_trans_vec}", "red")
                cprint(f"tar_rot_angle {tar_rot_angle}", "red")
                cprint(f"*****************************", "red")
                if self.demogen_rank is not None:
                    target_vec_idx = n_demos * self.demogen_rank + trans_vec_idx
                else:
                    target_vec_idx = trans_vec_idx
                output_path = self.output_root + '_trans_{:04d}'.format(target_vec_idx)
                os.makedirs(output_path, exist_ok=True)
                ############## start of generating one episode ##############
                
                self.clean_buffer()
                
                for camera_name in self.replay_buffer.target_cameras:
                    shutil.copy(os.path.join(self.preprocess_root, str(0), f"{camera_name}_preprocessed_image.png"), os.path.join(output_path, f"{camera_name}_frame.png"))

                self.stage_prepare_initial_layout(obj_bbox, tar_bbox, support_obj_bbox,
                                                  prepare_frame, 
                                                  obj_rot_angle, obj_trans_vec, tar_rot_angle, tar_trans_vec, 
                                                  output_path=output_path,
                                                  fixed_arms=["left", "right"])
                        
                if not self.prepare_only:
                    last_transform_mat_motion1, last_joint, motion1_len = self.stage_motion(
                        motion_1_frame,
                        skill_1_frame,
                        obj_trans_vec,
                        tar_trans_vec,
                        obj_center,
                        obj_rot,
                        tar_center,
                        tar_rot,
                        obj_bbox,
                        tar_bbox,
                        support_obj_bbox,
                        cur_joint_states=source_demo["joint_state"][0][(used_arm_index)*7:(used_arm_index+1)*7].copy(),
                        output_frameid_offset=prepare_frame,
                        output_path=output_path,
                        skill_type="motion-1",
                    )
                    if motion1_len is None:
                        cprint(f"generating {trans_vec_idx}: motion-1 ik failed, continue.", "red")
                        shutil.rmtree(output_path)
                        continue
                    last_joint, skill1_len = self.stage_skill(
                        skill_1_frame,
                        motion_2_frame,
                        obj_trans_vec,
                        tar_trans_vec,
                        obj_center,
                        obj_rot,
                        tar_center,
                        tar_rot,
                        obj_bbox,
                        tar_bbox,
                        support_obj_bbox,
                        last_transform_mat_motion=last_transform_mat_motion1,
                        last_joint=last_joint,
                        output_frameid_offset=prepare_frame+motion1_len,
                        output_path=output_path,
                        skill_type="skill-1"
                    )
                    if skill1_len is None:
                        cprint(f"generating {trans_vec_idx}: skill-1 ik failed, continue.", "red")
                        shutil.rmtree(output_path)
                        continue
                    last_transform_mat_motion2, last_joint, motion2_len = self.stage_motion(
                        motion_2_frame,
                        skill_2_frame,
                        obj_trans_vec,
                        tar_trans_vec,
                        obj_center,
                        obj_rot,
                        tar_center,
                        tar_rot,
                        obj_bbox,
                        tar_bbox,
                        support_obj_bbox,
                        cur_joint_states=last_joint[-1],
                        output_frameid_offset=prepare_frame+motion1_len+skill1_len,
                        output_path=output_path,
                        skill_type="motion-2",
                    )
                    if motion2_len is None:
                        cprint(f"generating {trans_vec_idx}: motion-2 ik failed, continue.", "red")
                        shutil.rmtree(output_path)
                        continue
                    last_joint, skill2_len = self.stage_skill(
                        skill_2_frame,
                        end_frame+1,
                        obj_trans_vec,
                        tar_trans_vec,
                        obj_center,
                        obj_rot,
                        tar_center,
                        tar_rot,
                        obj_bbox,
                        tar_bbox,
                        support_obj_bbox,
                        last_transform_mat_motion=last_transform_mat_motion2,
                        last_joint=last_joint,
                        output_frameid_offset=prepare_frame+motion1_len+skill1_len+motion2_len,
                        output_path=output_path,
                        skill_type="skill-2"
                    )
                    if skill2_len is None:
                        cprint(f"generating {trans_vec_idx}: skill-2 ik failed, continue.", "red")
                        shutil.rmtree(output_path)
                        continue
                    
                cprint(f">>> Demo {trans_vec_idx}: saving camera...", "yellow")
                self.save_camera(output_path)
                cprint(f">>> Demo {trans_vec_idx}: saving states...", "yellow")
                self.save_states(output_path)
                
                if not self.prepare_only:
                    meta_info = {
                        "motion-1": prepare_frame, 
                        "skill-1": prepare_frame+motion1_len, 
                        "motion-2": prepare_frame+motion1_len+skill1_len, 
                        "skill-2": prepare_frame+motion1_len+skill1_len+motion2_len,
                        "end": prepare_frame+motion1_len+skill1_len+motion2_len+skill2_len,
                        "obj_center": obj_center.tolist(),
                        "obj_trans_vec": obj_trans_vec.tolist(),
                        "obj_rot_angle": obj_rot_angle,
                        "tar_center": tar_center.tolist(),
                        "tar_trans_vec": tar_trans_vec.tolist(),
                        "tar_rot_angle": tar_rot_angle,
                    }
                    json.dump(meta_info, open(os.path.join(output_path, "meta_info.json"), "w"), indent=2)
                    self.save_at_end(trans_vec_idx, output_path, obj_trans_vec, tar_trans_vec)
                ############## end of generating one episode ##############

    def edit_single_arm_single_object(self, n_demos, render_video=False, gen_mode='random'):
        pcds, pcd_obj, _, _, \
        obj_bbox, obj_center, _, _, _, = self.prepare_pcd_and_bboxes()
        source_demo = self.replay_buffer.get_episode(0)
        motion_1_frame = self.parsing_frames["motion-1"]
        skill_1_frame = self.parsing_frames["skill-1"]
        end_frame = self.parsing_frames.get("end", len(source_demo["state"])-1)
        prepare_frame = self.parsing_frames.get("prepare", 30)
        fixed_arms = self.cfg.get("fixed_arms", [])
        single_arm_name = list(set(["left", "right"]) - set(fixed_arms))
        used_arm_index = arm_index[single_arm_name[0]]
        trans_vectors, success_obj_rots, success_obj_rot_angles, _, _ = self.generate_random_layouts(
            gen_mode, n_demos,
            object_arm_keyframe_pairs=[
                (single_arm_name[0], skill_1_frame-1, 0)
            ],
            obj_bbox=obj_bbox,
            obj_center=obj_center,
        )
        
        # For every source demo
        for i in range(self.n_source_episodes):
            cprint(f"Generating demos for source demo {i}", "blue")
            assert ("left" in fixed_arms and not "right" in fixed_arms) or ("right" in fixed_arms and not "left" in fixed_arms), "Only support one dynamic arm."
            cprint(f"Motion-1: {motion_1_frame}, Skill-1: {skill_1_frame}", "red")
            for trans_vec_idx, trans_vec in enumerate(tqdm(trans_vectors, desc=f"Generating {n_demos} demonstrations...")):
                # init urdf_processor and ik_sovler
                cprint(f"start generating {trans_vec_idx}", "green")
                obj_trans_vec = trans_vec[:3]
                obj_rot = success_obj_rots[trans_vec_idx]
                obj_rot_angle = success_obj_rot_angles[trans_vec_idx] 
                cprint(f"*****************************", "red")
                cprint(f"obj_trans_vec {obj_trans_vec}", "red")
                cprint(f"obj_rot_angle {obj_rot_angle}", "red")
                cprint(f"*****************************", "red")
                if self.demogen_rank is not None:
                    target_vec_idx = n_demos * self.demogen_rank + trans_vec_idx
                else:
                    target_vec_idx = trans_vec_idx
                output_path = self.output_root + '_trans_{:04d}'.format(target_vec_idx)
                os.makedirs(output_path, exist_ok=True)
                ############## start of generating one episode ##############
                
                self.clean_buffer()
                for camera_name in self.replay_buffer.target_cameras:
                    shutil.copy(os.path.join(self.preprocess_root, str(0), f"{camera_name}_preprocessed_image.png"), os.path.join(output_path, f"{camera_name}_frame.png"))
                    
                self.stage_prepare_initial_layout(obj_bbox, None, None,
                                                  prepare_frame, 
                                                  obj_rot_angle, obj_trans_vec, None, None, 
                                                  output_path=output_path,
                                                  fixed_arms=["left", "right"])
                        
                if not self.prepare_only:
                    last_transform_mat_motion1, last_joint, motion1_len = self.stage_motion(
                        motion_1_frame,
                        skill_1_frame,
                        obj_trans_vec,
                        None,
                        obj_center,
                        obj_rot,
                        None,
                        None,
                        obj_bbox,
                        None,
                        None,
                        cur_joint_states=source_demo["joint_state"][0][(used_arm_index)*7:(used_arm_index+1)*7].copy(),
                        output_frameid_offset=prepare_frame,
                        output_path=output_path,
                        skill_type="motion-1",
                    )
                    if motion1_len is None:
                        cprint(f"generating {trans_vec_idx}: motion-1 ik failed, continue.", "red")
                        shutil.rmtree(output_path)
                        continue
                    last_joint, skill1_len = self.stage_skill(
                        skill_1_frame,
                        end_frame,
                        obj_trans_vec,
                        None,
                        obj_center,
                        obj_rot,
                        None,
                        None,
                        obj_bbox,
                        None,
                        None,
                        last_transform_mat_motion=last_transform_mat_motion1,
                        last_joint=last_joint,
                        output_frameid_offset=prepare_frame+motion1_len,
                        output_path=output_path,
                        skill_type="skill-1"
                    )
                    if skill1_len is None:
                        cprint(f"generating {trans_vec_idx}: skill-1 ik failed, continue.", "red")
                        shutil.rmtree(output_path)
                        continue
                    
                cprint(f">>> Demo {trans_vec_idx}: saving camera...", "yellow")
                self.save_camera(output_path)
                cprint(f">>> Demo {trans_vec_idx}: saving states...", "yellow")
                self.save_states(output_path)
                
                if not self.prepare_only:
                    meta_info = {
                        "motion-1": prepare_frame, 
                        "skill-1": prepare_frame+motion1_len, 
                        "end": prepare_frame+motion1_len+skill1_len,
                        "obj_center": obj_center.tolist(),
                        "obj_trans_vec": obj_trans_vec.tolist(),
                        "obj_rot_angle": obj_rot_angle,
                    }
                    json.dump(meta_info, open(os.path.join(output_path, "meta_info.json"), "w"), indent=2)
                    self.save_at_end(trans_vec_idx, output_path, obj_trans_vec)

    def edit_dual_arm_single_object(self, n_demos, render_video=False, gen_mode='random'):
        pcds, pcd_obj, _, _, \
        obj_bbox, obj_center, _, _, _, = self.prepare_pcd_and_bboxes()
        source_demo = self.replay_buffer.get_episode(0)
        left_motion_1_frame = self.parsing_frames["left-motion-1"]
        left_skill_1_frame = self.parsing_frames["left-skill-1"]
        right_motion_1_frame = self.parsing_frames["right-motion-1"]
        right_skill_1_frame = self.parsing_frames["right-skill-1"]
        end_frame = self.parsing_frames.get("end", len(source_demo["state"])-1)
        prepare_frame = self.parsing_frames.get("prepare", 30)
        fixed_arms = self.cfg.get("fixed_arms", [])
        trans_vectors, success_obj_rots, success_obj_rot_angles, _, _ = self.generate_random_layouts(
            gen_mode, n_demos,
            object_arm_keyframe_pairs=[
                ("left", left_skill_1_frame-1, 0),
                ("right", right_skill_1_frame-1, 0)
            ],
            obj_bbox=obj_bbox,
            obj_center=obj_center,
        )
        # For every source demo
        for i in range(self.n_source_episodes):
            cprint(f"Generating demos for source demo {i}", "blue")
            cprint(f"Right-Motion-1: {right_motion_1_frame}, Right-Skill-1: {right_skill_1_frame}, Left-Motion-1: {left_motion_1_frame}, Left-Skill-1: {left_skill_1_frame}", "red")
            for trans_vec_idx, trans_vec in enumerate(tqdm(trans_vectors, desc=f"Generating {n_demos} demonstrations...")):
                # init urdf_processor and ik_sovler
                cprint(f"start generating {trans_vec_idx}", "green")
                obj_trans_vec = trans_vec[:3]
                obj_rot = success_obj_rots[trans_vec_idx]
                obj_rot_angle = success_obj_rot_angles[trans_vec_idx] 
                cprint(f"*****************************", "red")
                cprint(f"obj_trans_vec {obj_trans_vec}", "red")
                cprint(f"obj_rot_angle {obj_rot_angle}", "red")
                cprint(f"*****************************", "red")
                if self.demogen_rank is not None:
                    target_vec_idx = n_demos * self.demogen_rank + trans_vec_idx
                else:
                    target_vec_idx = trans_vec_idx
                output_path = self.output_root + '_trans_{:04d}'.format(target_vec_idx)
                os.makedirs(output_path, exist_ok=True)
                ############## start of generating one episode ##############
                
                self.clean_buffer()
                for camera_name in self.replay_buffer.target_cameras:
                    shutil.copy(os.path.join(self.preprocess_root, str(0), f"{camera_name}_preprocessed_image.png"), os.path.join(output_path, f"{camera_name}_frame.png"))
                    
                self.stage_prepare_initial_layout(obj_bbox, None, None,
                                                  prepare_frame, 
                                                  obj_rot_angle, obj_trans_vec, None, None, 
                                                  output_path=output_path,
                                                  fixed_arms=["left", "right"])
                        
                if not self.prepare_only:
                    used_arm_index = arm_index["right"]
                    last_transform_mat_right_motion1, right_last_joint, right_motion1_len = self.stage_motion(
                        right_motion_1_frame,
                        right_skill_1_frame,
                        obj_trans_vec,
                        None,
                        obj_center,
                        obj_rot,
                        None,
                        None,
                        obj_bbox,
                        None,
                        None,
                        cur_joint_states=source_demo["joint_state"][0][(used_arm_index)*7:(used_arm_index+1)*7].copy(),
                        output_frameid_offset=prepare_frame,
                        output_path=output_path,
                        skill_type="motion-1",
                        fixed_arms=["left"],
                    )
                    if right_motion1_len is None:
                        cprint(f"generating {trans_vec_idx}: right-motion-1 ik failed, continue.", "red")
                        shutil.rmtree(output_path)
                        continue
                    right_last_joint, right_skill1_len = self.stage_skill(
                        right_skill_1_frame,
                        left_motion_1_frame,
                        obj_trans_vec,
                        None,
                        obj_center,
                        obj_rot,
                        None,
                        None,
                        obj_bbox,
                        None,
                        None,
                        last_transform_mat_motion=last_transform_mat_right_motion1,
                        last_joint=right_last_joint,
                        output_frameid_offset=prepare_frame+right_motion1_len,
                        output_path=output_path,
                        skill_type="skill-1",
                        fixed_arms=["left"],
                    )
                    if right_skill1_len is None:
                        cprint(f"generating {trans_vec_idx}: right-skill-1 ik failed, continue.", "red")
                        shutil.rmtree(output_path)
                        continue
                    used_arm_index = arm_index["left"]
                    last_transform_mat_left_motion1, left_last_joint, left_motion1_len, right_last_joint = self.stage_motion(
                        left_motion_1_frame,
                        left_skill_1_frame,
                        obj_trans_vec,
                        None,
                        obj_center,
                        obj_rot,
                        None,
                        None,
                        obj_bbox,
                        None,
                        None,
                        cur_joint_states=source_demo["joint_state"][0][(used_arm_index)*7:(used_arm_index+1)*7].copy(),
                        output_frameid_offset=prepare_frame+right_motion1_len+right_skill1_len,
                        output_path=output_path,
                        skill_type="motion-1-dual",
                        fixed_arms=["right"],
                        dual_arm_follow=True,
                        dual_arm_last_transform_mat_motion=last_transform_mat_right_motion1,
                        dual_joint=right_last_joint,
                    )
                    if left_motion1_len is None:
                        cprint(f"generating {trans_vec_idx}: left-motion-1 ik failed, continue.", "red")
                        shutil.rmtree(output_path)
                        continue
                    left_last_joint, left_skill1_len, right_last_joint = self.stage_skill(
                        left_skill_1_frame,
                        end_frame,
                        obj_trans_vec,
                        None,
                        obj_center,
                        obj_rot,
                        None,
                        None,
                        obj_bbox,
                        None,
                        None,
                        last_transform_mat_motion=last_transform_mat_left_motion1,
                        last_joint=left_last_joint,
                        output_frameid_offset=prepare_frame+right_motion1_len+right_skill1_len+left_motion1_len,
                        output_path=output_path,
                        skill_type="skill-1-dual",
                        fixed_arms=["right"],
                        dual_arm_follow=True,
                        dual_arm_last_transform_mat_motion=last_transform_mat_right_motion1,
                        dual_joint=right_last_joint,
                    )
                    if left_skill1_len is None:
                        cprint(f"generating {trans_vec_idx}: left-skill-1 ik failed, continue.", "red")
                        shutil.rmtree(output_path)
                        continue
                cprint(f">>> Demo {trans_vec_idx}: saving camera...", "yellow")
                self.save_camera(output_path)
                cprint(f">>> Demo {trans_vec_idx}: saving states...", "yellow")
                self.save_states(output_path)
                
                if not self.prepare_only:
                    meta_info = {
                        "right-motion-1": prepare_frame, 
                        "right-skill-1": prepare_frame+right_motion1_len, 
                        "left-motion-1": prepare_frame+right_motion1_len+right_skill1_len,
                        "left-skill-1": prepare_frame+right_motion1_len+right_skill1_len+left_motion1_len,
                        "end": prepare_frame+right_motion1_len+right_skill1_len+left_motion1_len+left_skill1_len,
                        "obj_center": obj_center.tolist(),
                        "obj_trans_vec": obj_trans_vec.tolist(),
                        "obj_rot_angle": obj_rot_angle,
                    }
                    json.dump(meta_info, open(os.path.join(output_path, "meta_info.json"), "w"), indent=2)
                    self.save_at_end(trans_vec_idx, output_path, obj_trans_vec)
    
    def edit_dual_arm_multi_objects(self, n_demos, render_video=False, gen_mode='random'):
        pcds, pcd_obj, pcd_tar, pcd_support_obj, \
        obj_bbox, obj_center, tar_bbox, tar_center, _ = self.prepare_pcd_and_bboxes()
        obj_bbox_ori = obj_bbox.copy()
        source_demo = self.replay_buffer.get_episode(0)
        left_motion_1_frame = self.parsing_frames["left-motion-1"]
        left_skill_1_frame = self.parsing_frames["left-skill-1"]
        left_motion_2_frame = self.parsing_frames["left-motion-2"]
        right_motion_1_frame = self.parsing_frames["right-motion-1"]
        right_skill_1_frame = self.parsing_frames["right-skill-1"]
        right_motion_2_frame = self.parsing_frames["right-motion-2"]
        skill_2_frame = self.parsing_frames["skill-2"]
        end_frame = self.parsing_frames.get("end", len(source_demo["state"])-1)
        prepare_frame = self.parsing_frames.get("prepare", 30)
        fixed_arms = self.cfg.get("fixed_arms", [])
        
        trans_vectors, \
        success_obj_rots, success_obj_rot_angles, \
        success_tar_rots, success_tar_rot_angles = self.generate_random_layouts(
            gen_mode, n_demos,
            object_arm_keyframe_pairs=[
                ("left", left_skill_1_frame-1, left_motion_1_frame),
                ("left", left_motion_2_frame-1, left_motion_1_frame),
            ],
            obj_bbox=obj_bbox,
            obj_center=obj_center,
            target_arm_keyframe_pairs=[
                ("right", right_skill_1_frame-1, right_motion_1_frame),
                ("right", right_motion_2_frame-1, right_motion_1_frame),
            ],
            tar_bbox=tar_bbox,
            tar_center=tar_center,
        )
        
        # For every source demo
        for i in range(self.n_source_episodes):
            cprint(f"Generating demos for source demo {i}", "blue")
            cprint(f"Left-Motion-1: {left_motion_1_frame}, Left-Skill-1: {left_skill_1_frame}, Left-Motion-2: {left_motion_2_frame}", "red")
            cprint(f"Right-Motion-1: {right_motion_1_frame}, Right-Skill-1: {right_skill_1_frame}, Right-Motion-2: {right_motion_2_frame}", "red")
            cprint(f"Skill-2: {skill_2_frame}", "red")

            # Generate demos according to translation vectors
            for trans_vec_idx, trans_vec in enumerate(tqdm(trans_vectors, desc=f"Generating {n_demos} demonstrations...")):
                # init urdf_processor and ik_sovler
                cprint(f"start generating {trans_vec_idx}", "green")
                obj_trans_vec = trans_vec[:3]
                tar_trans_vec = trans_vec[3:6]
                obj_rot = success_obj_rots[trans_vec_idx]
                obj_rot_angle = success_obj_rot_angles[trans_vec_idx] 
                tar_rot = success_tar_rots[trans_vec_idx]
                tar_rot_angle = success_tar_rot_angles[trans_vec_idx]
                cprint(f"*****************************", "red")
                cprint(f"obj_trans_vec {obj_trans_vec}", "red")
                cprint(f"obj_rot_angle {obj_rot_angle}", "red")
                cprint(f"tar_trans_vec {tar_trans_vec}", "red")
                cprint(f"tar_rot_angle {tar_rot_angle}", "red")
                cprint(f"*****************************", "red")
                if self.demogen_rank is not None:
                    target_vec_idx = n_demos * self.demogen_rank + trans_vec_idx
                else:
                    target_vec_idx = trans_vec_idx
                output_path = self.output_root + '_trans_{:04d}'.format(target_vec_idx)
                os.makedirs(output_path, exist_ok=True)
                ############## start of generating one episode ##############
                
                self.clean_buffer()
                
                for camera_name in self.replay_buffer.target_cameras:
                    shutil.copy(os.path.join(self.preprocess_root, str(0), f"{camera_name}_preprocessed_image.png"), os.path.join(output_path, f"{camera_name}_frame.png"))

                self.stage_prepare_initial_layout(obj_bbox, tar_bbox, None,
                                                  prepare_frame, 
                                                  obj_rot_angle, obj_trans_vec, tar_rot_angle, tar_trans_vec, 
                                                  output_path=output_path,
                                                  fixed_arms=["left", "right"])
                if not self.prepare_only:
                    used_arm_index = arm_index["left"]
                    last_transform_mat_left_motion1, left_last_joint, left_motion1_len = self.stage_motion(
                        left_motion_1_frame,
                        left_skill_1_frame,
                        obj_trans_vec,
                        tar_trans_vec,
                        obj_center,
                        obj_rot,
                        tar_center,
                        tar_rot,
                        obj_bbox,
                        tar_bbox,
                        None,
                        fixed_arms=["right"],
                        cur_joint_states=source_demo["joint_state"][0][(used_arm_index)*7:(used_arm_index+1)*7].copy(),
                        output_frameid_offset=prepare_frame,
                        output_path=output_path,
                        skill_type="motion-1",
                    )
                    if left_motion1_len is None:
                        cprint(f"generating {trans_vec_idx}: left-motion-1 ik failed, continue.", "red")
                        shutil.rmtree(output_path)
                        continue
                    left_last_joint, left_skill1_len = self.stage_skill(
                        left_skill_1_frame,
                        left_motion_2_frame,
                        obj_trans_vec,
                        tar_trans_vec,
                        obj_center,
                        obj_rot,
                        tar_center,
                        tar_rot,
                        obj_bbox,
                        tar_bbox,
                        None,
                        fixed_arms=["right"],
                        last_transform_mat_motion=last_transform_mat_left_motion1,
                        last_joint=left_last_joint,
                        output_frameid_offset=prepare_frame+left_motion1_len,
                        output_path=output_path,
                        skill_type="skill-1"
                    )
                    if left_skill1_len is None:
                        cprint(f"generating {trans_vec_idx}: left-skill-1 ik failed, continue.", "red")
                        shutil.rmtree(output_path)
                        continue
                    last_transform_mat_left_motion2, left_last_joint, left_motion2_len = self.stage_motion(
                        left_motion_2_frame,
                        right_motion_1_frame,
                        obj_trans_vec,
                        tar_trans_vec,
                        obj_center,
                        obj_rot,
                        tar_center,
                        tar_rot,
                        obj_bbox,
                        tar_bbox,
                        None,
                        fixed_arms=["right"],
                        cur_joint_states=left_last_joint[-1],
                        output_frameid_offset=prepare_frame+left_motion1_len+left_skill1_len,
                        output_path=output_path,
                        skill_type="motion-2",
                        is_barcode_motion2=True,
                    )
                    if left_motion2_len is None:
                        cprint(f"generating {trans_vec_idx}: left-motion-2 ik failed, continue.", "red")
                        shutil.rmtree(output_path)
                        continue
                    used_arm_index = arm_index["right"]
                    last_transform_mat_right_motion1, right_last_joint, right_motion1_len, left_last_joint = self.stage_motion(
                        right_motion_1_frame,
                        right_skill_1_frame,
                        obj_trans_vec,
                        tar_trans_vec,
                        obj_center,
                        obj_rot,
                        tar_center,
                        tar_rot,
                        obj_bbox,
                        tar_bbox,
                        None,
                        fixed_arms=["left"],
                        cur_joint_states=source_demo["joint_state"][right_motion_1_frame][(used_arm_index)*7:(used_arm_index+1)*7].copy(),
                        output_frameid_offset=prepare_frame+left_motion1_len+left_skill1_len+left_motion2_len,
                        output_path=output_path,
                        skill_type="motion-1-dual",
                        dual_arm_follow=True,
                        dual_arm_last_transform_mat_motion=last_transform_mat_left_motion2,
                        dual_joint=left_last_joint,
                    )
                    if right_motion1_len is None:
                        cprint(f"generating {trans_vec_idx}: right-motion-1 ik failed, continue.", "red")
                        shutil.rmtree(output_path)
                        continue
                    right_last_joint, right_skill1_len, left_last_joint = self.stage_skill(
                        right_skill_1_frame,
                        right_motion_2_frame,
                        obj_trans_vec,
                        tar_trans_vec,
                        obj_center,
                        obj_rot,
                        tar_center,
                        tar_rot,
                        obj_bbox,
                        tar_bbox,
                        None,
                        fixed_arms=["left"],
                        last_transform_mat_motion=last_transform_mat_right_motion1,
                        last_joint=right_last_joint,
                        output_frameid_offset=prepare_frame+left_motion1_len+left_skill1_len+left_motion2_len+right_motion1_len,
                        output_path=output_path,
                        skill_type="skill-1-dual",
                        dual_arm_follow=True,
                        dual_arm_last_transform_mat_motion=last_transform_mat_left_motion2,
                        dual_joint=left_last_joint,
                    )
                    if right_skill1_len is None:
                        cprint(f"generating {trans_vec_idx}: right-skill-1 ik failed, continue.", "red")
                        shutil.rmtree(output_path)
                        continue
                    last_transform_mat_motion2, right_last_joint, right_motion2_len, left_last_joint = self.stage_motion(
                        right_motion_2_frame,
                        skill_2_frame,
                        obj_trans_vec,
                        tar_trans_vec,
                        obj_center,
                        obj_rot,
                        tar_center,
                        tar_rot,
                        obj_bbox,
                        tar_bbox,
                        None,
                        fixed_arms=["left"],
                        cur_joint_states=right_last_joint[-1],
                        output_frameid_offset=prepare_frame+left_motion1_len+left_skill1_len+left_motion2_len+right_motion1_len+right_skill1_len,
                        output_path=output_path,
                        skill_type="motion-2-dual",
                        dual_arm_follow=True,
                        dual_arm_last_transform_mat_motion=last_transform_mat_left_motion2,
                        dual_joint=left_last_joint,
                    )
                    if right_motion2_len is None:
                        cprint(f"generating {trans_vec_idx}: right-motion-2 ik failed, continue.", "red")
                        shutil.rmtree(output_path)
                        continue
                    right_last_joint, right_skill2_len, left_last_joint = self.stage_skill(
                        skill_2_frame,
                        end_frame,
                        obj_trans_vec,
                        tar_trans_vec,
                        obj_center,
                        obj_rot,
                        tar_center,
                        tar_rot,
                        obj_bbox,
                        tar_bbox,
                        None,
                        fixed_arms=["left"],
                        last_transform_mat_motion=last_transform_mat_motion2,
                        last_joint=right_last_joint,
                        output_frameid_offset=prepare_frame+left_motion1_len+left_skill1_len+left_motion2_len+right_motion1_len+right_skill1_len+right_motion2_len,
                        output_path=output_path,
                        skill_type="copy",
                        dual_arm_follow=True,
                        dual_arm_last_transform_mat_motion=last_transform_mat_left_motion2,
                        dual_joint=left_last_joint,
                    )
                cprint(f">>> Demo {trans_vec_idx}: saving camera...", "yellow")
                self.save_camera(output_path)
                cprint(f">>> Demo {trans_vec_idx}: saving states...", "yellow")
                self.save_states(output_path)
                
                if not self.prepare_only:
                    meta_info = {
                        "left-motion-1":  prepare_frame,
                        "left-skill-1":   prepare_frame+left_motion1_len,
                        "left-motion-2":  prepare_frame+left_motion1_len+left_skill1_len,
                        "right-motion-1": prepare_frame+left_motion1_len+left_skill1_len+left_motion2_len,
                        "right-skill-1":  prepare_frame+left_motion1_len+left_skill1_len+left_motion2_len+right_motion1_len,
                        "right-motion-2": prepare_frame+left_motion1_len+left_skill1_len+left_motion2_len+right_motion1_len+right_skill1_len,
                        "skill-2":        prepare_frame+left_motion1_len+left_skill1_len+left_motion2_len+right_motion1_len+right_skill1_len+right_motion2_len,
                        "end":            prepare_frame+left_motion1_len+left_skill1_len+left_motion2_len+right_motion1_len+right_skill1_len+right_motion2_len+right_skill2_len,
                        "obj_center": obj_center.tolist(),
                        "obj_trans_vec": obj_trans_vec.tolist(),
                        "obj_rot_angle": obj_rot_angle,
                        "tar_center": tar_center.tolist(),
                        "tar_trans_vec": tar_trans_vec.tolist(),
                        "tar_rot_angle": tar_rot_angle,
                    }
                    json.dump(meta_info, open(os.path.join(output_path, "meta_info.json"), "w"), indent=2)
                    self.save_at_end(trans_vec_idx, output_path, obj_trans_vec, tar_trans_vec)
    
    def pcd_divide(self, pcd_in: np.ndarray, bbox_list: list, debug: bool = False, debug_bboxes: list = None):
        """
        Segment the point cloud based on a list of bounding boxes (PyTorch version).

        Args:
            pcd: torch.Tensor, shape (N, D), where point cloud coordinates are in the first 3 columns (N, 3).
            bbox_list: list of torch.Tensor or list of list/np.ndarray, 
                       Each element shape should be (2, 3), representing [[x_min, y_min, z_min], [x_max, y_max, z_max]].
            debug: bool, whether to enable debug mode (the call to visualize_pcd_and_bbox will be ignored in the PyTorch version).
            debug_bboxes: list, the list of bounding boxes used during debugging.

        Returns:
            selected_pcds: list of torch.Tensor, the segmented point cloud list.
            masks: list of torch.Tensor (optional), the segmented boolean mask list.
        """

        pcd = torch.tensor(pcd_in).float().to(self.device)
        device = pcd.device
        dtype = pcd.dtype
        N = pcd.shape[0]
        
        if debug:
            if debug_bboxes is None:
                debug_bboxes = bbox_list
            visualize_pcd_and_bbox(pcd_in, debug_bboxes)
            print("Debug mode active, visualization skipped in PyTorch implementation.")
            
        masks = []
        
        # Extract the coordinate part of the point cloud (N, 3)
        pcd_coords = pcd[:, :3]

        for bbox_item in bbox_list:
            # Ensure bbox is a PyTorch Tensor and on the correct device
            if not isinstance(bbox_item, torch.Tensor):
                bbox = torch.tensor(bbox_item, dtype=dtype, device=device)
            else:
                bbox = bbox_item.to(dtype=dtype, device=device)
                
            assert bbox.shape == (2, 3), f"Bounding box shape must be (2, 3), but got {bbox.shape}"

            # 1. Check if points are greater than min_vals: pcd_coords > bbox[0] (N, 3)
            min_check = (pcd_coords > bbox[0]).all(dim=1)  # (N,)
            
            # 2. Check if points are less than max_vals: pcd_coords < bbox[1] (N, 3)
            max_check = (pcd_coords < bbox[1]).all(dim=1)  # (N,)
            
            # 3. Combine masks: min_check AND max_check
            current_mask = min_check & max_check  # (N,)
            masks.append(current_mask)

        # 4. Calculate the mask for the remaining points (added to the last mask)
        # Logic: NOT ( ANY(masks, axis=0) )
        if masks:
            # Stack all segmented masks into (num_bboxes, N)
            stacked_masks = torch.stack(masks, dim=0)
            
            # Find points selected by at least one bounding box
            any_selected = torch.any(stacked_masks, dim=0) # (N,)
            
            # The remaining points are those that were not selected
            rest_mask = torch.logical_not(any_selected)
        else:
            # If bbox_list is empty, all points are remaining points
            rest_mask = torch.ones(N, dtype=torch.bool, device=device)
            
        masks.append(rest_mask.cpu().numpy())

        # 5. Extract point cloud
        selected_pcds = [pcd[mask].cpu().numpy() for mask in masks]

        # 6. Validate total point count
        # total_points_sum = sum(len(p) for p in selected_pcds)
        
        # Use torch.sum(mask) instead of len(pcd[mask]) for validation, which is more efficient
        # total_points_check = torch.sum(torch.stack(masks, dim=0)).item()

        # assert total_points_sum == N, f"BBox overlapping or missing points. Total points: {N}, Sum of segmented points: {total_points_sum}"
        
        return selected_pcds

        
    def pcd_translate(self, pcd, trans_vec):
        """
        Translate the points with trans_vec
        pcd: (n, 3)
        trans_vec (3,)
        """
        return (torch.tensor(pcd).float().to(self.device)[:, :3] + torch.tensor(trans_vec).float().to(self.device).unsqueeze(0)).cpu().numpy()
    
    def pcd_transform(self, point_cloud: torch.Tensor, 
                    rotation_center_xyz: torch.Tensor, 
                    trans_vec: torch.Tensor, 
                    quat_type: str = "xyzw") -> torch.Tensor:
        """
        Rotate the point cloud around a given 3D center and apply a global translation.
        Use PyTorch3D's quaternion_to_matrix function for rotation calculation.

        Args:
            point_cloud (torch.Tensor): Point cloud array with shape (M, 3) or (N, M, 3).
            rotation_center_xyz (torch.Tensor): Rotation center point [x_c, y_c, z_c] with shape (3,) or (N, 3).
            trans_vec (torch.Tensor): A 7-element vector containing [tx, ty, tz, qx, qy, qz, qw].
                                    Shape is (7,) or (N, 7).
            quat_type (str): The order of the quaternion, "xyzw" or "wxyz".

        Returns:
            torch.Tensor: Rotated and translated point cloud with the same shape as point_cloud.
        """
        
        # ----------------- 1. Parse and prepare input -----------------
        if self.cfg.get("with_color", False):
            point_cloud_color = point_cloud[..., 3:6]
            point_cloud = point_cloud[..., :3]
        # Extract translation and quaternion
        trans_xyz = trans_vec[..., :3]    # Shape: (..., 3)
        rot_quat = trans_vec[..., 3:]     # Shape: (..., 4)
        
        # Normalize quaternion format: convert to [w, x, y, z] (format required by PyTorch3D's quaternion_to_matrix)
        if quat_type == "xyzw":
            # [x, y, z, w] -> [w, x, y, z]
            rot_wxyz = torch.cat([rot_quat[..., 3:], rot_quat[..., :3]], dim=-1)
        elif quat_type == "wxyz":
            # Already [w, x, y, z]
            rot_wxyz = rot_quat
        else:
            raise ValueError("quat_type must be 'xyzw' or 'wxyz'")
            
        # Convert rotation quaternion to rotation matrix (R)
        # R_matrix shape: (3, 3) or (N, 3, 3)
        R_matrix = quaternion_to_matrix(rot_wxyz)
        
        # Determine if it is batched
        is_batched = (point_cloud.dim() == 3)
        
        # ----------------- 2. Core transformation logic (translation-rotation-translation) -----------------
        
        if is_batched:
            # --- Batched mode (N, M, 3) ---
            
            # 1. Translate to origin: P_centered = P - P_c
            center = rotation_center_xyz.unsqueeze(1) # (N, 1, 3)
            points_centered = point_cloud - center    # (N, M, 3)
            
            # 2. Rotation: P_rotated_centered = R @ P_centered
            # Use einsum for batched matrix multiplication effect: (N, M, 3) point cloud row vectors @ R^T (N, 3, 3)
            # R[n, i, j] * P_c[n, k, j] -> P_r[n, k, i] 
            # Here we use einsum to match PyTorch3D's rotation matrix application convention: P @ R.T
            points_rotated_centered = torch.einsum('nij, nkj->nki', R_matrix, points_centered) 
            
            # 3. Translate back to original position: P_rotated = P_rotated_centered + P_c
            points_rotated = points_rotated_centered + center
            
        else:
            # --- Non-batched mode (M, 3) ---
            center = rotation_center_xyz # (3,)
            points_centered = point_cloud - center # (M, 3)
            
            # 2. Rotation: P_rotated_centered = P_centered @ R.T
            points_rotated_centered = torch.matmul(points_centered, R_matrix.T) # (M, 3) @ (3, 3)
            
            # 3. Translate back to original position: P_rotated = P_rotated_centered + P_c
            points_rotated = points_rotated_centered + center
            
        # ----------------- 3. Apply global translation -----------------
        
        # Global Translation: P_final = P_rotated + T_vec
        if is_batched:
            # trans_xyz (N, 3) broadcasted to (N, M, 3)
            trans_vector_for_pcd = trans_xyz.unsqueeze(1)
        else:
            # trans_xyz (3,) broadcasted to (M, 3)
            trans_vector_for_pcd = trans_xyz

        points_final = points_rotated + trans_vector_for_pcd
        if self.cfg.get("with_color", False):
            points_final = torch.cat([points_final, point_cloud_color], dim=-1)
        return points_final

    @staticmethod
    def pcd_bbox(pcd, relax=True, z_min_zero=False, 
                 x_upper=None, x_lower=None, y_upper=None, y_lower=None, z_upper=None, z_lower=None):
        if pcd.shape[0] == 0:
            return np.array([[0, 0, 0], [0, 0, 0]])
        min_vals = np.min(pcd[:, :3], axis=0)
        max_vals = np.max(pcd[:, :3], axis=0)
        if relax:
            if z_min_zero:
                min_vals[2] = 0.0
            if x_upper:
                max_vals[0] += x_upper
            if x_lower:
                min_vals[0] -= x_lower
            if y_upper:
                max_vals[1] += y_upper
            if y_lower:
                min_vals[1] -= y_lower
            if z_upper:
                max_vals[2] += z_upper
            if z_lower:
                min_vals[2] -= z_lower
        return np.array([min_vals, max_vals])
    
    @staticmethod
    def relax_bbox(bbox, relax_val):
        bbox_ = bbox.copy()
        bbox_[0] -= relax_val
        bbox_[1] += relax_val
        return bbox_