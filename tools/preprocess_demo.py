import torch
import json
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from termcolor import cprint
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms as TF
import warnings
import open3d as o3d
import shutil

# Ignore all warnings
warnings.filterwarnings("ignore")

# VGGT
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor

# Edit
from inpaint_utils import initialize_client, edit_image_list
import argparse
from omegaconf import OmegaConf

GROUNDING_BOX_THRESHOLD = 0.2
GROUNDING_TEXT_THRESHOLD = 0.15
CAMERAS = ["head", "hand_left", "hand_right"]
ARK_API_KEY = "<seededit-api-key>"

def estimate_plane_ransac(points, threshold=0.001):
    """RANSAC plane fitting"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=threshold,
        ransac_n=3,
        num_iterations=1000
    )
    [a, b, c, d] = plane_model
    return plane_model, points[inliers]


def build_groundingdino(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.text_encoder_type = os.path.join("checkpoints", "bert-base-uncased")
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    pred_logits = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            pred_logits.append(logit.max().item())
        else:
            pred_phrases.append(pred_phrase)
            pred_logits.append(1.0)

    return boxes_filt, pred_phrases, pred_logits

def do_grounding(image_path, model, text_prompt, box_threshold, text_threshold, device):
    image_pil, image = load_image(image_path)
    size = image_pil.size
    H, W = size[1], size[0]
    all_boxes_filt = []
    
    if len(text_prompt) > 0:
        for subtext in text_prompt.split("_"):
            boxes_filt, pred_phrases, pred_logits = get_grounding_output(
                model, image, subtext, box_threshold, text_threshold, device=device
            )
            if len(pred_logits) > 0:
                max_box_idx = np.argmax(pred_logits)
                boxes_filt = boxes_filt[max_box_idx:max_box_idx+1]
                boxes_filt = boxes_filt * torch.tensor([W, H, W, H], dtype=boxes_filt.dtype, device=boxes_filt.device).unsqueeze(0)
                boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
                boxes_filt[:, 2:] += boxes_filt[:, :2]
                boxes_filt = boxes_filt.cpu()
                all_boxes_filt.append(boxes_filt)
    
    if len(all_boxes_filt) > 0:
        all_boxes_filt = torch.cat(all_boxes_filt, dim=0)
    else:
        all_boxes_filt = torch.empty((0, 4))
    
    ori_image = cv2.imread(image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    return all_boxes_filt, image, ori_image


class Preprocessor:
    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cameras = CAMERAS
        self.vggt_checkpoint_path = args.vggt_checkpoint_path
        self.config = OmegaConf.load(args.config_path)
        self.steps = args.steps
        if self.config.task_n_object == 2:
            self.care_obj_names = [self.config.mask_names.object, self.config.mask_names.target]
        elif self.config.task_n_object == 1:
            self.care_obj_names = [self.config.mask_names.object]
        elif self.config.task_n_object == 3:
            self.care_obj_names = [self.config.mask_names.object, self.config.mask_names.target, self.config.mask_names.support_object]
        self.vggt_model = None
        
    def load_intrinsic_json(self, json_path):
        with open(json_path, 'r') as f:
            intrinsic_dict = json.load(f)['intrinsic']
        intrinsic = np.array([[intrinsic_dict['fx'], 0, intrinsic_dict['ppx']],
                            [0, intrinsic_dict['fy'], intrinsic_dict['ppy']],
                            [0, 0, 1]])
        return intrinsic
        
    def load_intrinsic(self):
        """
        Load intrinsic parameters from the episode path.
        """
        episode_path = self.config.data_root
        head_intrinsic_path = f'{episode_path}/parameters/camera/head_intrinsic_params.json'
        hand_left_intrinsic_path = f'{episode_path}/parameters/camera/hand_left_intrinsic_params.json'
        hand_right_intrinsic_path = f'{episode_path}/parameters/camera/hand_right_intrinsic_params.json'
        return_dict = {
            "head": self.load_intrinsic_json(head_intrinsic_path),  
            "hand_left": self.load_intrinsic_json(hand_left_intrinsic_path),
            "hand_right": self.load_intrinsic_json(hand_right_intrinsic_path)
        }
        return return_dict
    
    def init_vggt_model(self):
        """
        Initialize the VGGT model with the given checkpoint.
        """
        if self.vggt_model is None:
            checkpoint_path = self.vggt_checkpoint_path
            model = VGGT()
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            cprint(f"Loaded VGGT model from {checkpoint_path}", "green")
            self.vggt_model = model.to(self.device)
    
    def init_grounded_sam2_model(self):
        """
        Initialize the Grounded SAM2 model.
        """
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2_checkpoint = os.path.join("checkpoints", "sam2_hiera_large.pt")
        sam2_model_cfg = "sam2_hiera_l.yaml"
        config_file = os.path.join("third-party", "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinB_cfg.py")
        grounded_checkpoint = os.path.join("checkpoints", "groundingdino_swinb_cogcoor.pth")
    
        # Load GroundingDINO model
        self.groundingdino_model = build_groundingdino(config_file, grounded_checkpoint, device=device)
        cprint("GroundingDINO model loaded successfully", "green")
        
        # Load SAM2 model
        self.sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        cprint("SAM2 model loaded successfully", "green")

    def preprocess_raw_images(self):
        episode_path = self.config.data_root
        preprocessed_episode_path = self.config.preprocess_root
        intrinsics = self.load_intrinsic()
        frame_ids = sorted([int(x) for x in os.listdir(f'{episode_path}/camera') if os.path.isdir(os.path.join(f'{episode_path}/camera', x))])
        for camera_name in self.cameras:
            image_name = f"{camera_name}_color.jpg"
            image_paths = [f'{episode_path}/camera/{frame_id}/{image_name}' for frame_id in frame_ids]
            intrinsic_list = [intrinsics[camera_name] for _ in frame_ids]
            for image_path, intrinsic, frame_id in tqdm(zip(image_paths, intrinsic_list, frame_ids), total=len(frame_ids), desc=f"Processing {camera_name} images"):
                saved_dir = os.path.join(preprocessed_episode_path, str(frame_id))
                os.makedirs(saved_dir, exist_ok=True)
                
                images_tensor, intrinsics_tensor = load_and_preprocess_images([image_path], intrinsics_list=[intrinsic])
                images_np = images_tensor.numpy()[0]
                
                # Save image data (3, H, W) -> (H, W, 3)
                image_data = images_np.transpose(1, 2, 0)  # CHW -> HWC
                
                # Convert range from [0,1] to [0,255]
                if image_data.max() <= 1.0:
                    image_data = (image_data * 255).astype(np.uint8)
                else:
                    image_data = image_data.astype(np.uint8)
                
                # Use PIL to save as PNG
                img = Image.fromarray(image_data)
                img.save(os.path.join(saved_dir, f"{camera_name}_preprocessed_image.png"))
        
                # Save adjusted intrinsics as JSON
                intrinsics_np = intrinsics_tensor.numpy()[0]

                # Convert to Python list and save as JSON
                intrinsic_data = {
                    "intrinsic_matrix": intrinsics_np.tolist(),
                    "fx": float(intrinsics_np[0, 0]),
                    "fy": float(intrinsics_np[1, 1]),
                    "cx": float(intrinsics_np[0, 2]),
                    "cy": float(intrinsics_np[1, 2]),
                    "camera_name": camera_name,
                    "frame_id": frame_id
                }
                
                with open(os.path.join(saved_dir, f"{camera_name}_adjusted_intrinsic.json"), 'w') as f:
                    json.dump(intrinsic_data, f, indent=2)
    
    def mask_images(self):
        self.init_grounded_sam2_model()
        episode_path = self.config.data_root
        preprocessed_episode_path = self.config.preprocess_root
        care_obj_names = self.care_obj_names
        desktop_name = self.config.mask_names.desktop
        frame_ids = sorted([int(x) for x in os.listdir(f'{episode_path}/camera') if os.path.isdir(os.path.join(f'{episode_path}/camera', x))])
        
        for camera_name in self.cameras:
            if camera_name == "head":
                care_obj_names_w_desk = care_obj_names + [desktop_name]
            else:
                care_obj_names_w_desk = [desktop_name]
            if not self.config.get("sam_for_all_frames", False) or camera_name != "head":
                # Mask first frame.
                frame_ids = frame_ids[:1]
            for frame_id in tqdm(frame_ids, total=len(frame_ids), desc=f"Masking {camera_name} images"):
                saved_dir = os.path.join(preprocessed_episode_path, str(frame_id))
                image_path = os.path.join(saved_dir, f"{camera_name}_preprocessed_image.png")
                os.makedirs(saved_dir, exist_ok=True)
                for care_obj_name in care_obj_names_w_desk:
                    if "arm" in care_obj_name:
                        if frame_id != frame_ids[0]:
                            continue
                    # Grounding DINO
                    boxes_filt, image, ori_image = do_grounding(image_path, self.groundingdino_model, "" if "arm" in care_obj_name else care_obj_name, GROUNDING_BOX_THRESHOLD, GROUNDING_TEXT_THRESHOLD, self.device)
                    # Set image
                    self.sam2_predictor.set_image(image)
                    with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                        if len(boxes_filt) == 0:
                            if "arm" not in care_obj_name:
                                raise RuntimeError(f"Model did not find target object '{care_obj_name}' in image {os.path.basename(image_path)}, enabling interactive annotation")
                        else:
                            # Target object found, auto-generate mask
                            masks, _, _ = self.sam2_predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=boxes_filt,
                                multimask_output=False,
                            )
                            if len(boxes_filt) == 1:
                                mask = masks[0]
                            else:
                                mask = masks.sum(0).sum(0).astype(bool).astype(np.float32)
                        # Save mask image as JPG, use camera view as prefix
                        mask_uint8 = (mask * 255).astype(np.uint8)
                        mask_filename = f"{camera_name}_{care_obj_name}.jpg"
                        mask_path = os.path.join(saved_dir, mask_filename)
                        cv2.imwrite(mask_path, mask_uint8)

        del self.groundingdino_model
        del self.sam2_predictor
        del self.sam2_model
        torch.cuda.empty_cache()
    
    def edit_images_with_prompt(self, save_dir, image_path_list, prompt=[''], image_shape=(518, 294), base_name=None):
        client = initialize_client(ARK_API_KEY)
        if isinstance(prompt, str):
            prompt = [prompt]
        for prompt_i in prompt:
            print(f"Prompting SeedEdit: {prompt_i}")
            edit_image_list(client, image_path_list, prompt_i, save_dir, image_shape=image_shape, basename=base_name)
        return
    
    def preprocess_seededit(self):
        episode_path = self.config.data_root
        preprocessed_episode_path = self.config.preprocess_root
        care_obj_names = self.care_obj_names
        desktop_name = self.config.mask_names.desktop
        task_name = self.config.task_name
        #----------------------------------
        # Edit first frame
        #----------------------------------
        first_frame_dir = os.path.join(preprocessed_episode_path, str(0))
        saved_dir = os.path.join(preprocessed_episode_path, 'background')
        shutil.copytree(first_frame_dir, saved_dir, dirs_exist_ok=True)
        image_paths = []
        head_image = cv2.imread(os.path.join(saved_dir, "head_preprocessed_image.png"))
        h, w = head_image.shape[:2]
        for camera_name in self.cameras:
            image_paths.append(os.path.join(saved_dir, f"{camera_name}_preprocessed_image.png"))
        if task_name == "mug_to_box":
            if self.config.task_n_object == 3:
                seededit_prompts = [
                    "Without changing anything else in the image, remove the black gripper and the black wire.",
                    "Without changing anything else in the image, remove the black gripper and the black wire.",
                    "While keeping everything else in the image unchanged, remove the white robotic arm.",
                    f"Remove all objects on the table, while keeping everything else, especially the white table in the image unchanged."
                ]
            else:
                seededit_prompts = [
                    "Without changing anything else in the image, remove the black gripper and the black wire.",
                    "Without changing anything else in the image, remove the black gripper and the black wire.",
                    "While keeping everything else in the image unchanged, remove the white robotic arm.",
                    f"Remove all objects on the table, especially {','.join(care_obj_names)}, while keeping everything else in the image unchanged."
                ]
        elif task_name == "pour_water":
            seededit_prompts = [
                "Without changing anything else in the image, remove the black gripper and the black wire.",
                "Without changing anything else in the image, remove the black gripper and the black wire.",
                "While keeping everything else in the image unchanged, remove the white robotic arm.",
                f"Remove all objects on the table, while keeping everything else in the image unchanged."
            ]
        elif task_name == "lift_box":
            seededit_prompts = [
                "Without changing anything else in the image, remove the black gripper and the black wire.",
                "Without changing anything else in the image, remove the black gripper and the black wire.",
                "While keeping everything else in the image unchanged, remove the white robotic arm.",
                f"Remove all objects on the table, especially {','.join(care_obj_names)}, while keeping everything else in the image unchanged."
            ]
        elif task_name == "scan_barcode":
            seededit_prompts = [
                "Without changing anything else in the image, remove the black gripper and the black wire.",
                "Without changing anything else in the image, remove the black gripper and the black wire.",
                "While keeping everything else in the image unchanged, remove the white robotic arm.",
                f"Remove all objects on the table, especially {','.join(care_obj_names)}, while keeping everything else in the image unchanged."
            ]
        elif task_name == "open_drawer":
            seededit_prompts = [
                "Without changing anything else in the image, remove the black gripper and the black wire.",
                "Without changing anything else in the image, remove the black gripper and the black wire.",
                "While keeping everything else in the image unchanged, remove the white robotic arm.",
                f"Remove all objects on the table, especially {','.join(care_obj_names)}, while keeping everything else in the image unchanged."
            ]
        else:
            raise NotImplementedError(f"Unknown task name {task_name}")
        for seededit_prompt in seededit_prompts:
            self.edit_images_with_prompt(saved_dir, image_paths, 
                                        prompt=[seededit_prompt], 
                                        image_shape=(w, h))
        
    def preprocess_vggt(self, use_point_map=False, use_original_intrinsics=True, is_background=False):
        to_tensor = TF.ToTensor()
        episode_path = self.config.data_root
        preprocessed_episode_path = self.config.preprocess_root
        care_obj_names = self.care_obj_names
        desktop_name = self.config.mask_names.desktop
        task_name = self.config.task_name
        confidence_thres = str(self.config.confidence_thres)
        #-----------------------------------------------------------------------------------------------------
        self.init_vggt_model()
        if is_background:
            frame_ids = ['background']
        else:
            frame_ids = sorted([int(x) for x in os.listdir(f'{episode_path}/camera') if os.path.isdir(os.path.join(f'{episode_path}/camera', x))])
            
        for frame_id in tqdm(frame_ids, desc="Doing VGGT..."):
            saved_dir = os.path.join(preprocessed_episode_path, str(frame_id))
            os.makedirs(saved_dir, exist_ok=True)
            image_paths = []
            
            for camera_name in self.cameras:
                image_paths.append(os.path.join(saved_dir, f"{camera_name}_preprocessed_image.png"))
            images = [to_tensor(Image.open(image_path).convert("RGB")) for image_path in image_paths]
            images = torch.stack(images).to(self.device)
            
            # Run VGGT inference
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = self.vggt_model(images)
            
            # Convert pose encoding
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
            
            # Convert to numpy
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].cpu().numpy().squeeze(0)
                        
            # Unpack prediction results
            images = predictions["images"]  # (S, 3, H, W)
            world_points_map = predictions["world_points"]  # (S, H, W, 3)
            conf_map = predictions["world_points_conf"]  # (S, H, W)
            depth_map = predictions["depth"]  # (S, H, W, 1)
            depth_conf = predictions["depth_conf"]  # (S, H, W)
            extrinsics_cam = predictions["extrinsic"]  # (S, 3, 4)
            
            if use_original_intrinsics:
                intrinsics_cam = []
                for camera_name in self.cameras:
                    intrinsic_json = json.load(open(os.path.join(saved_dir, f"{camera_name}_adjusted_intrinsic.json")))
                    intrinsics_cam.append(intrinsic_json["intrinsic_matrix"])
                intrinsics_cam = np.array(intrinsics_cam)
            else:
                intrinsics_cam = predictions["intrinsic"]  # (S, 3, 3)

            # Choose to use depth map or precomputed point map
            if not use_point_map:
                world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
                conf = depth_conf
            else:
                world_points = world_points_map
                conf = conf_map
            
            # Convert image format (S, 3, H, W) -> (S, H, W, 3)
            colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
            confidence_thres_list = [int(x) for x in confidence_thres.split(",")]
            for i, camera_name in enumerate(self.cameras):
                if len(confidence_thres_list) == 1:
                    percent = confidence_thres_list[0]
                else:
                    percent = confidence_thres_list[i]
                points = world_points[i].reshape(-1, 3)
                colors_flat = colors[i].reshape(-1, 3)
                conf_flat = conf[i].reshape(-1)
                threshold_val = np.percentile(conf_flat, percent)
                conf_mask = (conf_flat >= threshold_val) & (conf_flat > 0.1)
                mask = np.asarray(conf_mask).astype(bool)
                points = points[mask]
                colors_flat = colors_flat[mask]
                pointcloud_o3d = o3d.geometry.PointCloud()
                pointcloud_o3d.points = o3d.utility.Vector3dVector(points)
                pointcloud_o3d.colors = o3d.utility.Vector3dVector(colors_flat)
                o3d.io.write_point_cloud(os.path.join(saved_dir, f"{camera_name}_vggt_ori.ply"), pointcloud_o3d)
                mask_img = mask.reshape(conf[i].shape).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(saved_dir, f"{camera_name}_conf_mask.jpg"), mask_img)
            cam_params_filename = f"camera_params.npz"
            cam_params_path = os.path.join(saved_dir, cam_params_filename)
            np.savez(cam_params_path, extrinsics=extrinsics_cam, intrinsics=intrinsics_cam)
            
            # Only keep head camera view point cloud, S=1; range(S) changed to range(1)
            for i in range(len(self.cameras)):
                camera_name = self.cameras[i]
                points = world_points[i].reshape(-1, 3)
                depth_flat = depth_map[i].reshape(-1)
                colors_flat = colors[i].reshape(-1, 3)
                
                # Save depth map
                depth_filename = f"{camera_name}_depth.npz"
                depth_path = os.path.join(saved_dir, depth_filename)
                np.savez(depth_path, data=depth_map[i])  # Use 'data' as the key name
                if (frame_id == 'background' or frame_id == 0):
                    np.savez(os.path.join(saved_dir, f"{camera_name}_pointmap.npz"), data=world_points[i])
                    
        torch.cuda.empty_cache()
        
    def preprocess_vggt_background(self):
        #-------------------------------
        # fix the background depth
        #-------------------------------
        preprocessed_episode_path = self.config.preprocess_root
        desktop_name = self.config.mask_names.desktop
        bg_dir = os.path.join(preprocessed_episode_path, 'background')
        first_dir = os.path.join(preprocessed_episode_path, '0')
        cam_params_filename = f"camera_params.npz"
        cam_params_path = os.path.join(first_dir, cam_params_filename)
        cam_params = np.load(cam_params_path)
        extrinsics_cam = cam_params["extrinsics"]
        intrinsics_cam = cam_params["intrinsics"]
        scale_dict = dict()
        first_plane_head = None
        for camera_name in self.cameras:
            desktop_mask = cv2.imread(os.path.join(first_dir, f"{camera_name}_{desktop_name}.jpg"), cv2.IMREAD_GRAYSCALE)
            desktop_mask_bool = desktop_mask > 127  # Desktop region True
            bg_points = np.load(os.path.join(bg_dir, f"{camera_name}_pointmap.npz"))["data"]
            bg_points = bg_points[desktop_mask_bool]
            bg_plane, _ = estimate_plane_ransac(bg_points)
            first_points = np.load(os.path.join(first_dir, f"{camera_name}_pointmap.npz"))["data"]
            first_points = first_points[desktop_mask_bool]
            first_plane, _ = estimate_plane_ransac(first_points)
            if camera_name == "head":
                first_plane_head = np.array(first_plane).copy()
            # Plane equation: ax + by + cz + d = 0
            bg_d = np.abs(bg_plane[3])
            first_d = np.abs(first_plane[3])

            # Scale ratio
            scale = first_d / bg_d
            # if "left" in camera_name:
            #     scale = 1.25
            # if "right" in camera_name:
            #     scale = 1.12
            scale_dict[camera_name] = scale
            
            print(f"Camera name: {camera_name}")
            print(f"    Estimated scale:", scale)
            print(f"    Desktop plane:")
            print(f"        First frame: {first_plane}")
            print(f"        Background : {bg_plane}")
        assert first_plane_head is not None
        desktop_plane = {
            "a": first_plane_head[0],
            "b": first_plane_head[1],
            "c": first_plane_head[2],
            "d": first_plane_head[3],
        }
        with open(os.path.join(first_dir, "desktop_plane.json"), "w") as f:
            json.dump(desktop_plane, f, indent=2)
        
        depth_map_list = []
        for i in range(len(self.cameras)):
            camera_name = self.cameras[i]
            depth_map = np.load(os.path.join(bg_dir, f"{camera_name}_depth.npz"))["data"]
            depth_map = depth_map * scale_dict[camera_name]
            depth_map_list.append(depth_map)
            np.savez(os.path.join(bg_dir, f"{camera_name}_depth_scaled.npz"), data=depth_map)
        depth_map = np.stack(depth_map_list, axis=0)  # (S, H, W, 1)
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        for i, camera_name in enumerate(self.cameras):
            points = world_points[i].reshape(-1, 3)
            pointcloud_o3d = o3d.geometry.PointCloud()
            pointcloud_o3d.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(os.path.join(bg_dir, f"{camera_name}_vggt_ori.ply"), pointcloud_o3d)
        self.preprocess_background_frame_world()
        
    def preprocess_first_frame_world(self):
        episode_path = self.config.data_root
        preprocessed_episode_path = self.config.preprocess_root
        head_extrinsics_path = os.path.join(episode_path, "parameters", "camera", "head_extrinsic_params_aligned.json")
        head_extrinsics_dict = json.load(open(head_extrinsics_path, 'r'))
        head_extrinsics_c2w = np.eye(4)
        head_extrinsics_c2w[:3, :3] = head_extrinsics_dict[0]["extrinsic"]["rotation_matrix"]
        head_extrinsics_c2w[:3,  3] = head_extrinsics_dict[0]["extrinsic"]["translation_vector"]
        
        first_dir = os.path.join(preprocessed_episode_path, '0')
        cam_params_filename = f"camera_params.npz"
        cam_params_path = os.path.join(first_dir, cam_params_filename)
        cam_params = np.load(cam_params_path)
        extrinsics_cam = cam_params["extrinsics"]
        intrinsics_cam = cam_params["intrinsics"]
        first_pcd = None
        for i in range(len(self.cameras)):
            camera_name = self.cameras[i]
            if first_pcd is None:
                first_pcd = o3d.io.read_point_cloud(os.path.join(first_dir, f"{camera_name}_vggt_ori.ply"))
            else:
                first_pcd = first_pcd + o3d.io.read_point_cloud(os.path.join(first_dir, f"{camera_name}_vggt_ori.ply"))
        first_pcd.transform(head_extrinsics_c2w)
        o3d.io.write_point_cloud(os.path.join(first_dir, f"vggt_c2w.ply"), first_pcd)
        
    def preprocess_background_frame_world(self):
        episode_path = self.config.data_root
        preprocessed_episode_path = self.config.preprocess_root
        head_extrinsics_path = os.path.join(episode_path, "parameters", "camera", "head_extrinsic_params_aligned.json")
        head_extrinsics_dict = json.load(open(head_extrinsics_path, 'r'))
        head_extrinsics_c2w = np.eye(4)
        head_extrinsics_c2w[:3, :3] = head_extrinsics_dict[0]["extrinsic"]["rotation_matrix"]
        head_extrinsics_c2w[:3,  3] = head_extrinsics_dict[0]["extrinsic"]["translation_vector"]
        
        first_dir = os.path.join(preprocessed_episode_path, 'background')
        cam_params_filename = f"camera_params.npz"
        cam_params_path = os.path.join(first_dir, cam_params_filename)
        cam_params = np.load(cam_params_path)
        extrinsics_cam = cam_params["extrinsics"]
        intrinsics_cam = cam_params["intrinsics"]
        first_pcd = None
        for i in range(len(self.cameras)):
            camera_name = self.cameras[i]
            if first_pcd is None:
                first_pcd = o3d.io.read_point_cloud(os.path.join(first_dir, f"{camera_name}_vggt_ori.ply"))
            else:
                first_pcd = first_pcd + o3d.io.read_point_cloud(os.path.join(first_dir, f"{camera_name}_vggt_ori.ply"))
        first_pcd.transform(head_extrinsics_c2w)
        o3d.io.write_point_cloud(os.path.join(first_dir, f"vggt_c2w.ply"), first_pcd)
        
    def preprocess_all_frame_world(self):
        episode_path = self.config.data_root
        preprocessed_episode_path = self.config.preprocess_root
        head_extrinsics_path = os.path.join(episode_path, "parameters", "camera", "head_extrinsic_params_aligned.json")
        head_extrinsics_dict = json.load(open(head_extrinsics_path, 'r'))
        head_extrinsics_c2w = np.eye(4)
        head_extrinsics_c2w[:3, :3] = head_extrinsics_dict[0]["extrinsic"]["rotation_matrix"]
        head_extrinsics_c2w[:3,  3] = head_extrinsics_dict[0]["extrinsic"]["translation_vector"]

        for subdir in tqdm(os.listdir(preprocessed_episode_path), desc="all frame vggt c2w"):
            first_dir = os.path.join(preprocessed_episode_path, subdir)
            cam_params_filename = f"camera_params.npz"
            cam_params_path = os.path.join(first_dir, cam_params_filename)
            cam_params = np.load(cam_params_path)
            extrinsics_cam = cam_params["extrinsics"]
            intrinsics_cam = cam_params["intrinsics"]
            first_pcd = None
            for i in range(len(self.cameras)):
                camera_name = self.cameras[i]
                if first_pcd is None:
                    first_pcd = o3d.io.read_point_cloud(os.path.join(first_dir, f"{camera_name}_vggt_ori.ply"))
                else:
                    first_pcd = first_pcd + o3d.io.read_point_cloud(os.path.join(first_dir, f"{camera_name}_vggt_ori.ply"))
            first_pcd.transform(head_extrinsics_c2w)
            o3d.io.write_point_cloud(os.path.join(first_dir, f"vggt_c2w.ply"), first_pcd)
        
    def preprocess(self):
        steps = self.steps
        # -----------------------------------
        # (1) Process images and intrinsics
        # -----------------------------------
        if '1' in steps:
            cprint(">>> Step 1: Preprocessing raw images and intrinsics...", "yellow")
            self.preprocess_raw_images()
        
        # -----------------------------------
        # (2) Masking cared objects
        # -----------------------------------
        if '2' in steps:
            cprint(">>> Step 2: Masking images...", "yellow")
            # only mask first frame here
            self.mask_images()

        # -----------------------------------
        # (3) Inpainting first frame with seededit
        # -----------------------------------
        if '3' in steps:
            cprint(">>> Step 3: Inpainting first frame with seededit", "yellow")
            self.preprocess_seededit()
        
        # -----------------------------------
        # (4) Reconstructing with VGGT
        # -----------------------------------
        if '4' in steps:
            cprint(">>> Step 4: Reconstruct background", "yellow")
            self.preprocess_vggt(is_background=True)
            
        if '5' in steps:
            cprint(">>> Step 5: Reconstructing with VGGT...", "yellow")
            self.preprocess_vggt(is_background=False)
            
        # -----------------------------------
        # (5) Post-processing background
        # -----------------------------------
        if '6' in steps:
            cprint(">>> Step 6: Estimating desktop...", "yellow")
            self.preprocess_vggt_background()

        if '7' in steps:
            cprint(">>> Step 7: Merging first frame in world axes...", "yellow")
            self.preprocess_first_frame_world()
        
        if '8' in steps:
            cprint(">>> Step 8: Merging all frames in world axes...", "yellow")
            self.preprocess_all_frame_world()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vggt_checkpoint_path', type=str, default="checkpoints/metric_vggt_cotrain.pth")
    parser.add_argument("--steps", type=str, default='1234567')
    parser.add_argument("--config_path", type=str, default="configs/mug_to_box_1654490.yaml")
    args = parser.parse_args()
    preprocessor = Preprocessor(args)
    preprocessor.preprocess()
