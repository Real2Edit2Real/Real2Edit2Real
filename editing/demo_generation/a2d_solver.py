import time
import torch
import os
import random
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig, JointState
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.util.logger import setup_logger
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from termcolor import cprint

import numpy as np
import trimesh
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from yourdfpy import URDF

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointLights,
    TexturesVertex,
)
from pytorch3d.utils import cameras_from_opencv_projection
from demo_generation.pose_utils import transform_points
from tqdm import tqdm
setup_logger("warn")

# Enable PyTorch performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Constants

ARM_JS_NAMES = {
    "left": ["Joint1_l", "Joint2_l", "Joint3_l", "Joint4_l", "Joint5_l", "Joint6_l", "Joint7_l"], 
    "right": ["Joint1_r", "Joint2_r", "Joint3_r", "Joint4_r", "Joint5_r", "Joint6_r", "Joint7_r"]
}
"""
ARM_JS_NAMES = {
    "left": ["left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4", "left_arm_joint5", "left_arm_joint6", "left_arm_joint7"], 
    "right": ["right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4", "right_arm_joint5", "right_arm_joint6", "right_arm_joint7"]
}
"""
WHOLE_JS_NAMES = ["joint_lift_body", "joint_body_pitch"] + ARM_JS_NAMES["left"] + ARM_JS_NAMES["right"]
LINK_NAMES = ["Link1_l", "Link2_l", "Link3_l", "Link4_l", "Link5_l", "Link6_l", "Link7_l", 
              "Link1_r", "Link2_r", "Link3_r", "Link4_r", "Link5_r", "Link6_r", "Link7_r"]

LEFT_ARM_LINK_NAMES = ["Link1_l", "Link2_l", "Link3_l", "Link4_l", "Link5_l", "Link6_l", "Link7_l"]
RIGHT_ARM_LINK_NAMES = ["Link1_r", "Link2_r", "Link3_r", "Link4_r", "Link5_r", "Link6_r", "Link7_r"]
LEFT_ARM_BASE_LINK_NAMES = ["Link1_l", "Link2_l", "Link3_l", "Link4_l", "Link5_l", "Link6_l", "Link7_l", "left_base_link"]
RIGHT_ARM_BASE_LINK_NAMES = ["Link1_r", "Link2_r", "Link3_r", "Link4_r", "Link5_r", "Link6_r", "Link7_r", "right_base_link"]
LEFT_EE_LINK_NAMES = ["left_Left_00_Link",  "left_Left_01_Link", "left_Left_2_Link", "left_Left_Pad_Link", "left_Left_Support_Link", "left_Right_00_Link", "left_Right_01_Link", "left_Right_2_Link", "left_Right_Pad_Link", "left_Right_Support_Link", "left_base_link"]
RIGHT_EE_LINK_NAMES = ["right_Left_00_Link",  "right_Left_01_Link", "right_Left_2_Link", "right_Left_Pad_Link", "right_Left_Support_Link", "right_Right_00_Link", "right_Right_01_Link", "right_Right_2_Link", "right_Right_Pad_Link", "right_Right_Support_Link", "right_base_link"]



class DualArmA2DSolver:
    """Inverse Kinematics solver for Dual-Arm A2D robot."""

    def __init__(self, ik_type="motion_gen", disturbance=False, device=torch.device("cuda:0")):
        """
        Initialize the Dual-Arm A2D IK Solver.

        Args:
            ik_type (str): Type of IK solver to use. Options: "ik_solver" or "motion_gen"
        """
        self.tensor_args = TensorDeviceType(device)
        self._initialize_robot_config()
        self._initialize_solver(ik_type, disturbance=disturbance)

        # Pre-allocate tensors for efficiency
        self.pos_tensor_buffer = torch.zeros(
            3, device=self.tensor_args.device, dtype=torch.float32
        )
        self.quat_tensor_buffer = torch.zeros(
            4, device=self.tensor_args.device, dtype=torch.float32
        )

    def _initialize_robot_config(self):
        """Load and initialize robot configuration."""
        config_file = load_yaml(join_path(os.path.dirname(__file__), "a2d.yaml"))
        urdf_file = os.path.abspath(config_file["robot_cfg"]["kinematics"]["urdf_path"])
        left_base_link = config_file["robot_cfg"]["kinematics"]["left_base_link"]
        left_ee_link = config_file["robot_cfg"]["kinematics"]["left_ee_link"]
        right_base_link = config_file["robot_cfg"]["kinematics"]["right_base_link"]
        right_ee_link = config_file["robot_cfg"]["kinematics"]["right_ee_link"]

        self.left_arm_cfg = RobotConfig.from_basic(
            urdf_file, left_base_link, left_ee_link, self.tensor_args
        )
        self.left_arm_kin_model = CudaRobotModel(self.left_arm_cfg.kinematics)

        self.right_arm_cfg = RobotConfig.from_basic(
            urdf_file, right_base_link, right_ee_link, self.tensor_args
        )
        self.right_arm_kin_model = CudaRobotModel(self.right_arm_cfg.kinematics)

        self.kin_model = {
            "left": self.left_arm_kin_model,
            "right": self.right_arm_kin_model
        }


    def _initialize_solver(self, ik_type, disturbance=False):
        """Initialize the appropriate solver based on ik_type."""
        if ik_type == "motion_gen":
            self._initialize_motion_gen(disturbance=disturbance)
        else:
            raise ValueError(f"Unsupported IK type: {ik_type}")
        
    def _initialize_motion_gen(self, disturbance=False):
        """Initialize the motion generator."""
        config_file = load_yaml(join_path(os.path.dirname(__file__), "a2d.yaml"))
        left_ee_link = config_file["robot_cfg"]["kinematics"]["left_ee_link"]
        self.left_plan_config = MotionGenConfig.load_from_robot_config(
            self.left_arm_cfg,
            None,
            filter_robot_command=True,
            tensor_args=self.tensor_args,
            ee_link_name=left_ee_link,
            use_gradient_descent=True,
            interpolation_dt = 1. / 30,
            num_trajopt_noisy_seeds = 20 if disturbance else 1,
            trajopt_seed_ratio = {"linear": 0, "bias": 1} if disturbance else {"linear": 1.0, "bias": 0.0}, # add disturbance
        )
        self.left_motion_gen = MotionGen(self.left_plan_config)
        cprint("Left arm: warming up motion gen solver", "green")
        self.left_motion_gen.warmup()

        right_ee_link = config_file["robot_cfg"]["kinematics"]["right_ee_link"]
        self.right_plan_config = MotionGenConfig.load_from_robot_config(
            self.right_arm_cfg,
            None,
            filter_robot_command=True,
            tensor_args=self.tensor_args,
            ee_link_name=right_ee_link,
            use_gradient_descent=False,
            interpolation_dt = 1. / 30,
            num_trajopt_noisy_seeds = 20 if disturbance else 1,
            trajopt_seed_ratio = {"linear": 0, "bias": 1} if disturbance else {"linear": 1.0, "bias": 0.0}, # add disturbance
        )
        self.right_motion_gen = MotionGen(self.right_plan_config)
        cprint("Right arm: warming up motion gen solver", "green")
        self.right_motion_gen.warmup()

        self.plan_config_temp = MotionGenPlanConfig(
            max_attempts=2,
        )
        self.motion_gen = {
            "left": self.left_motion_gen,
            "right": self.right_motion_gen
        }

    def solve_ik_by_motion_gen_single_arm(self, curr_joint_state, target_trans, target_quat, steps, arm="right", quat_type="wxyz",
                                          debug=False):
        """
        Solve IK using motion generation.

        Args:
            curr_joint_state (list): Current joint state [[q1, q2, ..., q7], [q1, q2, ..., q7]]
            target_trans (list): Target position [[x, y, z], [x, y, z]]
            target_quat (list): Target quaternion [[w, x, y, z], [w, x, y, z]]

        Returns:
            list or None: Joint solution if successful, None otherwise
        """
        cu_js = JointState(
            position=self.tensor_args.to_device(curr_joint_state),
            velocity=self.tensor_args.to_device(curr_joint_state) * 0.0,
            acceleration=self.tensor_args.to_device(curr_joint_state) * 0.0,
            jerk=self.tensor_args.to_device(curr_joint_state) * 0.0,
            joint_names=ARM_JS_NAMES[arm],
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen[arm].kinematics.joint_names)

        if quat_type == "xyzw":
            target_quat = quat_xyzw_to_wxyz(target_quat)
        else:
            target_quat = target_quat
        ik_goal = Pose(
            position=self.tensor_args.to_device(target_trans),
            quaternion=self.tensor_args.to_device(target_quat),
        )
        result = self.motion_gen[arm].plan_single(
            cu_js.unsqueeze(0), ik_goal, self.plan_config_temp
        )

        if result.success.item():
            if steps is not None:
                time_dilation_factor = len(result.get_interpolated_plan().position) / steps
                if time_dilation_factor >= 1:
                    pass
                else:
                    try:
                        result.retime_trajectory(
                            time_dilation_factor=time_dilation_factor,
                        )
                    except Exception as e:
                        if debug:
                            print(f"retime error, {e}")
                        return None
                
            try:
                motion_plan = result.get_interpolated_plan()
                motion_plan = self.motion_gen[arm].get_full_js(motion_plan)
                motion_plan = motion_plan.get_ordered_joint_state(ARM_JS_NAMES[arm])
                motion_plan = motion_plan.position.cpu().numpy()
            except Exception as e:
                if debug:
                    print(f"motion plan error, {e}")
                return None
            
            if steps is not None:
                try:
                    if len(motion_plan) < steps:
                        motion_plan = np.concatenate([motion_plan, np.tile(motion_plan[-1:], (steps - len(motion_plan), 1))], axis=0)
                    elif len(motion_plan) > steps:
                        sampled_indices_float = np.linspace(0, len(motion_plan) - 1, num=steps)
                        sampled_indices_int = np.round(sampled_indices_float).astype(int)
                        motion_plan = motion_plan[sampled_indices_int]
                    assert len(motion_plan) == steps, "len(motion_plan) != steps. Please check."
                except Exception as e:
                    if debug:
                        print(f"motion plan interpolation error, {e}")
                    return None
            return motion_plan
        else:
            if debug:
                print(f"result is not success")
            return None

    def compute_fk_single_arm(self, joint_angles, arm="right", quat_type="xyzw"):
        """
        Compute the forward kinematics for the Dual-Arm A2D robot.

        Args:
            joint_angles (list): Joint angles [q1, q2, ..., q7]

        Returns:
            list: (position, quaternion) where position is [x, y, z] and quaternion is [w, x, y, z]
        """
        joint_angles_gpu = torch.tensor(
            joint_angles, device=self.tensor_args.device, dtype=torch.float32
        )
        out = self.kin_model[arm].get_state(joint_angles_gpu)
        position = out.ee_position.cpu().numpy()
        quaternion = out.ee_quaternion.cpu().numpy()
        if quat_type == "xyzw":
            quaternion = quaternion[:, [1, 2, 3, 0]]
        return position, quaternion
    
def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from xyzw to wxyz
    """
    q = np.asarray(q)
    assert q.shape[-1] == 4
    return np.concatenate([q[..., 3:], q[..., :3]], axis=-1)

class A2D_URDF_Processor():
    """
    Rotate mesh and render by URDF
    """
    def __init__(self, urdf_path, device=torch.device("cuda:0")):
        self.robot = URDF.load(urdf_path)
        self.device = device
        self.urdf_dir = os.path.dirname(urdf_path)
        self.mesh_buffer = dict()
        self.link_names = [l.name for l in self.robot.robot.links]
        self.preload_meshes()
        
    def preload_meshes(self):
        for link_name in tqdm(self.link_names, desc="Preloading URDF parts..."):
            if link_name not in self.robot.link_map:
                raise ValueError(f"Link '{link_name}' not found in the URDF model.")

            link = self.robot.link_map[link_name]
            geometries = link.collisions
            if len(geometries) == 0:
                continue
            
            for g in geometries:
                mesh_data = g.geometry.mesh
                trimesh_mesh = trimesh.load_mesh(mesh_data.filename)#.simplify_quadric_decimation(percent=0.25)
                self.mesh_buffer[mesh_data.filename] = {
                    'vertices_gpu': torch.from_numpy(np.asarray(trimesh_mesh.vertices)).to(self.device),
                    'faces_gpu': torch.from_numpy(np.asarray(trimesh_mesh.faces)).to(self.device),
                }
    
    def fk_meshes(self):
        self.link_meshes = dict()
        # ---- load meshes ----
        for link_name in self.link_names:
            link_meshes_list = []
            if link_name not in self.robot.link_map:
                raise ValueError(f"Link '{link_name}' not found in the URDF model.")

            link = self.robot.link_map[link_name]
            geometries = link.collisions
            if len(geometries) == 0:
                continue
            
            for g in geometries:
                if g.geometry.mesh is None:
                    continue
                mesh_data = g.geometry.mesh
                pose = self.robot._scene.graph.get(frame_to=link_name, frame_from=None)[0]
                # scaling
                if mesh_data.scale is not None:
                    S = np.eye(4)
                    S[:3, :3] = np.diag(mesh_data.scale if isinstance(mesh_data.scale, np.ndarray)
                                        else [mesh_data.scale]*3)
                    pose = pose @ S

                mesh_buffer = self.mesh_buffer[mesh_data.filename]
                mesh_buffer_new_vertices = transform_points(torch.from_numpy(pose).to(self.device), mesh_buffer['vertices_gpu'])
                link_meshes_list.append({
                    "vertices_gpu": mesh_buffer_new_vertices.float(),
                    "faces_gpu": mesh_buffer['faces_gpu'],
                })
            self.link_meshes[link_name] = link_meshes_list
        
    def render_link_depth_and_mask(self, link_names='left_right_arm',
                               camera_intrinsics: torch.Tensor=None, camera_extrinsics: torch.Tensor =None,
                               image_size=(640, 480), concat_meshes=False):
        """Render depth map and mask for specific robot links using PyTorch3D on GPU."""

        device = self.device

        # ---- select link names ----
        if link_names == 'left_arm':
            link_names = LEFT_ARM_LINK_NAMES
        elif link_names == 'right_arm':
            link_names = RIGHT_ARM_LINK_NAMES
        elif link_names == "left_right_arm":
            link_names = LEFT_ARM_LINK_NAMES + RIGHT_ARM_LINK_NAMES
        elif link_names == 'arm_base':
            link_names = LEFT_ARM_BASE_LINK_NAMES + RIGHT_ARM_BASE_LINK_NAMES
        elif link_names == 'left_arm_base':
            link_names = LEFT_ARM_BASE_LINK_NAMES
        elif link_names == 'right_arm_base':
            link_names = RIGHT_ARM_BASE_LINK_NAMES
        elif link_names == 'left_arm_ee':
            link_names = LEFT_ARM_LINK_NAMES + LEFT_EE_LINK_NAMES
        elif link_names == 'right_arm_ee':
            link_names = RIGHT_ARM_LINK_NAMES + RIGHT_EE_LINK_NAMES
        elif link_names == "left_right_arm_leftbase":
            link_names = LEFT_ARM_BASE_LINK_NAMES + RIGHT_ARM_LINK_NAMES
        elif link_names == "left_right_arm_rightbase":
            link_names = LEFT_ARM_LINK_NAMES + RIGHT_ARM_BASE_LINK_NAMES
        elif isinstance(link_names, str):
            link_names = [link_names]
        all_verts = []
        all_faces = []
        if concat_meshes:
            vert_offset = 0
            for link_name in link_names:
                for mesh in self.link_meshes[link_name]:
                    verts = mesh["vertices_gpu"]
                    faces = mesh["faces_gpu"]
                    faces_saved = faces + vert_offset
                    vert_offset += verts.shape[0]
                    all_verts.append(verts)
                    all_faces.append(faces_saved)
            all_verts = torch.cat(all_verts, dim=0).unsqueeze(0).float()
            all_faces = torch.cat(all_faces, dim=0).unsqueeze(0).float()
        else:
            for link_name in link_names:
                for mesh in self.link_meshes[link_name]:
                    verts = mesh["vertices_gpu"]
                    faces = mesh["faces_gpu"]
                    all_verts.append(verts)
                    all_faces.append(faces)
        
        

        # ---- setup camera ----
        if camera_intrinsics is None:
            assert RuntimeError("render_link_depth_and_mask: camera_intrinsics is None")

        if camera_extrinsics is None:
            assert RuntimeError("render_link_depth_and_mask: camera_extrinsics is None")

        H, W = image_size[1], image_size[0]
        # convert extrinsics to R, T
        camera_extrinsics = torch.linalg.inv(camera_extrinsics)
        assert camera_extrinsics.dim() == camera_intrinsics.dim()
        if camera_extrinsics.dim() == 2:
            R = camera_extrinsics[:3,:3].unsqueeze(0).float()
            T = camera_extrinsics[:3,3].unsqueeze(0).float()
            camera_matrix = camera_intrinsics.unsqueeze(0).float()
            image_size = torch.tensor([[H, W]], dtype=torch.float32, device=device)
        elif camera_extrinsics.dim() == 3:
            R = camera_extrinsics[:, :3,:3].float()
            T = camera_extrinsics[:, :3,3].float()
            camera_matrix = camera_intrinsics.float()
            image_size = torch.tensor([[H, W]], dtype=torch.float32, device=device).repeat(len(camera_extrinsics), 1)
            if len(all_verts) != len(camera_extrinsics):
                assert len(all_verts) == 1
                all_verts = all_verts.repeat((len(camera_extrinsics), 1, 1))
                all_faces = all_faces.repeat((len(camera_extrinsics), 1, 1))
        
        cameras = cameras_from_opencv_projection(R, T, camera_matrix, image_size)
        mesh = Meshes(verts=all_verts, faces=all_faces)
        
        # ---- rasterization and rendering ----
        raster_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=32,
            max_faces_per_bin=80000,
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(mesh)
        depth_map = fragments.zbuf[..., 0]
        mask = depth_map > 0
        return depth_map, mask, None
        
    def get_link_bbox(self, link_names):
        """
        Compute the axis-aligned bounding box (AABB) of the given links in world coordinates.

        Args:
            link_names (list or str): List of link names (or 'left_arm', 'right_arm').
        Returns:
            np.ndarray: Bounding box [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        """
        if link_names == 'left_arm':
            link_names = LEFT_ARM_LINK_NAMES
        elif link_names == 'right_arm':
            link_names = RIGHT_ARM_LINK_NAMES
        elif link_names == 'left_arm_ee':
            link_names = LEFT_ARM_LINK_NAMES + LEFT_EE_LINK_NAMES
        elif link_names == 'right_arm_ee':
            link_names = RIGHT_ARM_LINK_NAMES + RIGHT_EE_LINK_NAMES
        elif link_names == 'left_ee':
            link_names = LEFT_EE_LINK_NAMES
        elif link_names == "right_ee":
            link_names = RIGHT_EE_LINK_NAMES
        elif isinstance(link_names, str):
            link_names = [link_names]

        bboxes = []
        for link_name in link_names:
            for mesh in self.link_meshes[link_name]:
                mesh_min = mesh["vertices_gpu"].min(0).values.cpu().numpy()
                mesh_max = mesh["vertices_gpu"].max(0).values.cpu().numpy()
                bboxes.append(np.array([mesh_min, mesh_max]))

        if not bboxes:
            raise ValueError("No valid meshes found for the given links.")

        # Merge all bounding boxes
        # TODO: rotated box for each link
        all_mins = np.min([b[0] for b in bboxes], axis=0)
        all_maxs = np.max([b[1] for b in bboxes], axis=0)

        return np.array([all_mins, all_maxs])
    
    def get_joint_angles_dict(self, joint_angles, joint_names='arm'):
        """
        Get joint dict from names and angles.

        Args:
            joint_angles (list): A list of joint angles.
            joint_names (list):  A list of joint names or defined str
        """
        if joint_names == "arm":
            joint_names = ARM_JS_NAMES["left"] + ARM_JS_NAMES["right"]
        elif joint_names == "whole":
            joint_names = WHOLE_JS_NAMES
        assert len(joint_angles) == len(joint_names), "Length of angles and names must be same"
        joint_angles_dict = {}
        for i in range(len(joint_angles)):
            joint_angles_dict[joint_names[i]] = joint_angles[i]

        return joint_angles_dict
    
    def update_robot(self, joint_angles_dict):
        """
        Update the robot's scene based on the provided joint angles.

        Args:
            joint_angles_dict (dict): A dict of joint angles for A2D Robot.
        """
        if not isinstance(joint_angles_dict, (dict)):
            raise TypeError("joint_angles must be a dict.")
        # Update the robot's configuration using the provided joint angles
        self.robot.update_cfg(joint_angles_dict)
        # Update mesh cache
        self.fk_meshes()
        
