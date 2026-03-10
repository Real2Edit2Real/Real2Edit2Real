import os
import os.path as osp
import numpy as np
import json
import cv2
import torch
import imageio.v2 as imageio

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

"""
Reference data processing script for annotate dataset with metric-VGGT
"""

"""
Task ID we sampled from AgibotWorld-Beta:
327  352  356  358  360  362  365  367  369  373  375  377  380  385  389  392  410  421  424  428  431  434  440  445  451  453  455  462  464  466  470  474  478  483  486  491
351  354  357  359  361  363  366  368  372  374  376  378  384  388  390  398  414  422  425  429  433  438  444  446  452  454  460  463  465  468  471  477  480  485  487  492
"""

def extract_frames(video_path, every_n_frames=1, resize=None, max_frames=None):
    try:
        reader = imageio.get_reader(video_path)
        frames = []
        for i, frame in enumerate(reader):
            if i % every_n_frames != 0:
                continue
            if resize:
                frame = cv2.resize(frame, resize)
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        reader.close()
        return np.stack(frames) if frames else np.array([])
    except Exception as e:
        print(f"[ERROR] Failed to extract frames from {video_path}: {e}")
        return None

def depth_canny(depth):
    """
    Compute Canny edge from depth map
    """
    depth_valid = np.where((depth > 0) & np.isfinite(depth), depth, 0)
    depth_norm = cv2.normalize(depth_valid, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)

    # blurred = cv2.GaussianBlur(depth_uint8, (5, 5), sigmaX=1.0)
    edges = cv2.Canny(depth_uint8, threshold1=0, threshold2=50)

    return edges, depth_uint8

def preprocess_vggt_result(pred_dict: dict,
    output_path: str = None,
    frame: int = 0):
    
    #Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)

    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)

    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)
    
    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points_map.shape

    os.makedirs(osp.join(output_path, "head_depth_ori"), exist_ok=True)
    os.makedirs(osp.join(output_path, "head_depth_canny"), exist_ok=True)
    os.makedirs(osp.join(output_path, "hand_left_depth_ori"), exist_ok=True)
    os.makedirs(osp.join(output_path, "hand_left_depth_canny"), exist_ok=True)
    os.makedirs(osp.join(output_path, "hand_right_depth_ori"), exist_ok=True)
    os.makedirs(osp.join(output_path, "hand_right_depth_canny"), exist_ok=True)
    depth_output_list = [osp.join(output_path, "head_depth_ori"), osp.join(output_path, "hand_left_depth_ori"), osp.join(output_path, "hand_right_depth_ori")]
    canny_output_list = [osp.join(output_path, "head_depth_canny"), osp.join(output_path, "hand_left_depth_canny"), osp.join(output_path, "hand_right_depth_canny")]
    # Flatten and save
    for i in range(3):
        depth_ori = depth_map[i].squeeze()
        canny_edge, depth_norm = depth_canny(depth_ori)
        depth_ori_uint16 = (depth_ori * 1000).astype(np.uint16)

        imageio.imwrite(osp.join(depth_output_list[i], f"{frame}.png"), depth_ori_uint16)
        imageio.imwrite(osp.join(canny_output_list[i], f"{frame}.png"), canny_edge)



def process_single_video(ckpt_path, episode_path, episode_id, task_id, param_path, output_path):
    """
    Preprocesses a single video episode from the AgiBotWorld dataset using the metric-VGGT model.

    This function loads a pretrained VGGT model, extracts frames from multi-view videos 
    (head, left hand, right hand), and performs frame-by-frame inference to predict 
    camera extrinsics, depth maps and Canny edges.

    Args:
        episode_path (str): Path to the directory containing the episode's video files.
        episode_id (str): Unique identifier for the specific episode.
        task_id (str): Identifier for the task category.
        param_path (str): Path to the directory containing camera parameter JSON files 
                          (intrinsics and extrinsics).
        output_path (str): Directory where the inference results (extrinsics) will be saved.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT()
    print(f"Loading pretrained weights from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model_state_dict"])
    model.eval()
    model = model.to(device)

    video_list = [os.path.join(episode_path, 'videos', f"{view}_color.mp4") for view in ['head', 'hand_right', 'hand_left']]
    head_frames = extract_frames(video_list[0])
    hand_right_frames = extract_frames(video_list[1])
    hand_left_frames = extract_frames(video_list[2])
    if head_frames is None or hand_right_frames is None or hand_left_frames is None:
        return
    frame_num = head_frames.shape[0]
    os.makedirs(osp.join(output_path, task_id, episode_id), exist_ok=True)
    
    with open(os.path.join(param_path, task_id, episode_id, 'parameters/camera/head_intrinsic_params.json'), 'r') as f:
        head_intrinsic_dict = json.load(f)['intrinsic']
        head_intrinsic = np.array([[head_intrinsic_dict['fx'], 0, head_intrinsic_dict['ppx']],
                            [0, head_intrinsic_dict['fy'], head_intrinsic_dict['ppy']],
                            [0, 0, 1]])
    with open(os.path.join(param_path, task_id, episode_id, 'parameters/camera/hand_left_intrinsic_params.json'), 'r') as f:
        hand_left_intrinsic_dict = json.load(f)['intrinsic']
        hand_left_intrinsic = np.array([[hand_left_intrinsic_dict['fx'], 0, hand_left_intrinsic_dict['ppx']],
                                [0, hand_left_intrinsic_dict['fy'], hand_left_intrinsic_dict['ppy']],
                                [0, 0, 1]])
    with open(os.path.join(param_path, task_id, episode_id, 'parameters/camera/hand_right_intrinsic_params.json'), 'r') as f:
        hand_right_intrinsic_dict = json.load(f)['intrinsic']
        hand_right_intrinsic = np.array([[hand_right_intrinsic_dict['fx'], 0, hand_right_intrinsic_dict['ppx']],
                                [0, hand_right_intrinsic_dict['fy'], hand_right_intrinsic_dict['ppy']],
                                [0, 0, 1]])

    with open(os.path.join(param_path, task_id, episode_id, 'parameters/camera/head_extrinsic_params_aligned.json'), 'r') as f:
        head_extrinsic_dict_list = json.load(f)

    head_extrinsic_list = []
    hand_left_extrinsic_list = []
    hand_right_extrinsic_list = []
    for frame in range(frame_num):
        images_list = [head_frames[frame], hand_left_frames[frame], hand_right_frames[frame]]
        images = load_and_preprocess_images(images_list).to(device)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        # print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy
        # save results
        preprocess_vggt_result(predictions, os.path.join(output_path, task_id, episode_id), frame)
        head_extrinsic_list.append(predictions["extrinsic"][0])
        hand_left_extrinsic_list.append(predictions["extrinsic"][1])
        hand_right_extrinsic_list.append(predictions["extrinsic"][2])
    np.save(os.path.join(output_path, task_id, episode_id, "head_extrinsic.npy"), np.stack(head_extrinsic_list, axis=0))
    np.save(os.path.join(output_path, task_id, episode_id, "hand_left_extrinsic.npy"), np.stack(hand_left_extrinsic_list, axis=0))
    np.save(os.path.join(output_path, task_id, episode_id, "hand_right_extrinsic.npy"), np.stack(hand_right_extrinsic_list, axis=0))
