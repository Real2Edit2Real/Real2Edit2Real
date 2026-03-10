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
import torch
from tqdm import tqdm

from vggt.utils.geometry import closed_form_inverse_se3
from training.data.dataset_util import *
from training.data.base_dataset import BaseDataset


class AgiBotSimDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        AGIBOT_SIM_DIR: str = None,
        dataset_name: str = "agibot_sim",
    ):
        """
        Initialize the AgiBotSIMDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            AGIBOT_SIM_DIR (str): Directory path to AgiBot Sim data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If AGIBOT_SIM_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.training = common_conf.training
        self.inside_random = common_conf.inside_random

        if AGIBOT_SIM_DIR is None:
            raise ValueError("AGIBOT_SIM_DIR must be specified.")
        
        self.data_path_list = []
        split_ratio = 0.9 if "all" not in split else 1.0
        for task in tqdm(sorted(os.listdir(AGIBOT_SIM_DIR)), desc="Loading AgiBot Sim Data"):
            episode_list = sorted(os.listdir(osp.join(AGIBOT_SIM_DIR, task)))
            if split == "train":
                split_episode_list = episode_list[:int(len(episode_list) * split_ratio)]
            elif split == "test":
                split_episode_list = episode_list[int(len(episode_list) * split_ratio):]
            elif split == "all_train":
                split_episode_list = episode_list
            elif split == "all_test":
                split_episode_list = episode_list

            for episode in split_episode_list:
                episode_path = osp.join(AGIBOT_SIM_DIR, task, episode)
                if not osp.isdir(episode_path):
                    continue
                frame_list = sorted(os.listdir(episode_path))
                for frame in frame_list:
                    frame_path = osp.join(episode_path, frame)
                    self.data_path_list.append((frame, frame_path))
        
        self.dataset_name = dataset_name
        self.len_train = len(self.data_path_list)
        self.training = (split == 'train') or (split == 'all_train')
        print(f"AGIBOT_SIM_DIR is {AGIBOT_SIM_DIR}")
        status = "Training" if split == 'train' or split == "all_train" else "Test"
        print(f"{status}: AgiBot Sim Data size: {self.len_train}")

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
            image_path = osp.join(frame_path, f"{camera_view}.jpg")
            image = read_image_cv2(image_path)
            original_size = np.array(image.shape[:2])
            if camera_view == 'head':
                depth_threshold = 6
            else:
                depth_threshold = 6
            depth_path = image_path.replace(".jpg", "_depth.png")
            depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # Convert to meters
            depth_map = np.clip(depth_map, 0, 6)
            depth_map[depth_map > depth_threshold] = 0  # Mask out depth values greater than depth_threshold

            intrinsic = torch.load(osp.join(frame_path, f"{camera_view}_intrinsic.pt"), weights_only=False).data.cpu().numpy()
            intri_opencv = isaac_to_colmap_intrinsic(intrinsic, original_size[0])  # Convert to COLMAP format
            extrinsic = torch.load(osp.join(frame_path, f"{camera_view}_extrinsic.pt"), weights_only=False).data.cpu().numpy()
            extrinsic = isaac_to_colmap_extrinsic(extrinsic[int(frame)])  # Convert to COLMAP format
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

def isaac_to_colmap_intrinsic(K_isaac, image_height):
    """
    Input:
        K_isaac: 3x3 intrinsic matrix
        image_height: Image height (used for flipping the principal point's Y-axis)
    Output:
        K_colmap
    """
    K_colmap = K_isaac.copy()
    K_colmap[1, 2] = image_height - K_isaac[1, 2]
    return K_colmap
