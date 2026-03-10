#---------------------------------------------------
# Implementation of the AgiBot World dataset
#---------------------------------------------------

import gzip
import json
import os.path as osp
import os
import logging

import cv2
import random
import numpy as np
from tqdm import tqdm


from vggt.utils.geometry import closed_form_inverse_se3
from training.data.dataset_util import *
from training.data.base_dataset import BaseDataset

class AgiBotWorldDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        AGIBOT_WORLD_DIR: str = None,
        dataset_name: str = "agibot_world",
    ):
        """
        Initialize the AgiBotWorldDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            AGIBOT_WORLD_DIR (str): Directory path to AgiBot Wolrd data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If AGIBOT_WORLD_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)
        
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random

        if AGIBOT_WORLD_DIR is None:
            raise ValueError("AGIBOT_WORLD_DIR must be specified.")
        
        self.data_path_list = []
        split_ratio = 0.9 if "all" not in split else 1.0
        for task in tqdm(sorted(os.listdir(AGIBOT_WORLD_DIR)), desc="Loading AgiBot World Data"):
            if dataset_name == 'agibot_world':
                continue
            episode_list = sorted(os.listdir(osp.join(AGIBOT_WORLD_DIR, task)))
            if split == "train":
                split_episode_list = episode_list[:int(len(episode_list) * split_ratio)]
            elif split == "test":
                split_episode_list = episode_list[int(len(episode_list) * split_ratio):]
            elif split == "all_train":
                split_episode_list = episode_list
            elif split == "all_test":
                split_episode_list = episode_list

            for episode in split_episode_list:
                episode_path = osp.join(AGIBOT_WORLD_DIR, task, episode)
                if not osp.isdir(episode_path):
                    continue
                frame_list = sorted(os.listdir(episode_path))
                for frame in frame_list:
                    frame_path = osp.join(episode_path, frame)
                    self.data_path_list.append((frame, frame_path))
        
        self.dataset_name = dataset_name
        self.len_train = len(self.data_path_list)
        self.training = (split == 'train') or (split == 'all_train')
        print(f"AGIBOT_WORLD_DIR is {AGIBOT_WORLD_DIR}")
        status = "Training" if split == 'train' or split == 'all_train' else "Test"
        print(f"{status}: AgiBot World Data size: {self.len_train}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random:
            seq_index = random.randint(0, self.len_train - 1)

        frame, frame_path = self.data_path_list[seq_index]

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []

        world_to_first_view = None
        view_total_list = ['head', 'hand_left', 'hand_right']
        view_list = view_total_list[:img_per_seq]
        for camera_idx, camera_view in enumerate(view_list):
            image_path = osp.join(frame_path, f"{camera_view}_color.jpg")
            image = read_image_cv2(image_path)
            original_size = np.array(image.shape[:2])
            depth_path = image_path.replace("_color.jpg", "_depth.png")
            if camera_view == 'head':
                depth_scale = 0.001
                depth_threshold = 6 if self.dataset_name == 'agibot_world' else 6
            else:
                depth_scale = 0.0001 if self.dataset_name == 'agibot_world' else 0.001
                depth_threshold = 6
            depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) * depth_scale  # Convert to meters
            depth_map = np.clip(depth_map, 0, 6)
            depth_map[depth_map > depth_threshold] = 0  # Mask out depth values greater than depth threshold

            with open(osp.join(frame_path, f"{camera_view}_intrinsic.json"), 'r') as f:
                if self.dataset_name == 'agibot_world':
                    intrinsic_dict = json.load(f)['intrinsic']
                elif self.dataset_name == 'agibot_sim':
                    intrinsic_dict = json.load(f)
            intri_opencv = np.array([
                [intrinsic_dict['fx'], 0, intrinsic_dict['ppx']],
                [0, intrinsic_dict['fy'], intrinsic_dict['ppy']],
                [0, 0, 1]
            ])

            with open(osp.join(frame_path, f"{camera_view}_extrinsic.json"), 'r') as f:
                extrinsic_dict = json.load(f)
            rotation_matrix = np.array(extrinsic_dict[int(frame)]['extrinsic']['rotation_matrix'])
            translation_matrix = np.array(extrinsic_dict[int(frame)]['extrinsic']['translation_vector']).reshape(3, 1)
            extrinsic = np.concatenate((rotation_matrix, translation_matrix), axis=1)
            extrinsic = np.concatenate((extrinsic, np.array([[0, 0, 0, 1]])), axis=0)
            if self.dataset_name == 'agibot_sim':
                extrinsic = isaac_to_colmap_extrinsic(extrinsic)  # Convert to COLMAP format
            # if camera_view == 'head':
            if camera_idx == 0:
                world_to_first_view = closed_form_inverse_se3(extrinsic[None])[0]
            extrinsic = world_to_first_view @ extrinsic
            extri_opencv = closed_form_inverse_se3(extrinsic[None])[0, :3, :] # camera-from-world [3, 4]
            
            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)

        batch = {
            "dataset_name": self.dataset_name,
            "seq_name": frame_path,
            "ids": np.array(list(range(len(extrinsics)))),
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        return batch


def isaac_to_colmap_extrinsic(c2w_isaac):
    """
    Converts Isaac Sim c2w (4x4 matrix) to COLMAP format.

    Input:
        c2w_isaac: Camera-to-World matrix (Right-handed, Z-forward).
    Output:
        c2w_colmap: Camera-to-World matrix (Right-handed, Y-down, Z-forward).
    """
    axis_convert = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])

    c2w_colmap = c2w_isaac @ axis_convert
    return c2w_colmap