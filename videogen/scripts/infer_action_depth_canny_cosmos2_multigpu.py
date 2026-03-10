import sys
import os
import torch
import numpy as np
import math
import glob
import argparse
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import json
import h5py
import imageio
import torchvision.transforms as transforms
from tqdm import tqdm
import time

sys.path.append('.')
import os.path as osp
from pathlib import Path
import warnings
warnings.simplefilter("once", category=FutureWarning) # Apply to FutureWarning specifically
from einops import rearrange
from lib.trainers.action_depth_canny_cond_trainer import ActionDepthCannyWMTrainer
from lib.utils.torch_utils import seed_everything, save_video
from infer_utils import transform_video, normalize_video, transform_extrinsic_sequence, closed_form_inverse_se3, get_traj
from lib.data.utils.get_actions import EEF2CamLeft, EEF2CamRight, Rotation
from termcolor import cprint
import cv2
import shutil

CAMERA_VIEWS = ['head', 'hand_left', 'hand_right']

def prepare_model_and_data(config_file="videogen/configs/action_depth_canny_cosmos2.yaml", 
                           save_dir='checkpoints/video_generation_checkpoint',
                           model_id="checkpoints/Cosmos-Predict2-2B-Video2World",
                           chunk=25):
    model_path = osp.join(save_dir, 'diffusion_pytorch_model.safetensors')
    trainer = ActionDepthCannyWMTrainer(config_file, val_only=True)
    trainer.args.transformer['model_path'] = model_path
    trainer.args.output_dir = save_dir
    trainer.save_folder = save_dir
    chunk_size = chunk # 25  # defaults to 25  # 57  # has to equal 4n+1
    assert (chunk-1) % 4 == 0, "chunk must be 4n+1"
    trainer.args.data['train']['chunk'] = chunk_size
    trainer.args.data['val']['chunk'] = chunk_size
    trainer.args.data['train']['action_chunk'] = chunk_size
    trainer.args.data['val']['action_chunk']= chunk_size
    n_prev = 4
    sample_size = [384, 512]
    caption = "best quality, consistent and smooth motion, realistic, clear and distinct,"
    trainer.args.data['val']['ignore_seek'] = True  # True to only load memory; False to load full video
    trainer.args.wo_hand_cond = False
    fps = 30 # defaults to 16 # change to 30
    trainer.args.data['train']['fps'] = fps
    trainer.args.data['val']['fps'] = fps
    SEP = 1
    
    # prepare models
    trainer.prepare_models(model_id=model_id)

    accelerator = trainer.state.accelerator
    return trainer, chunk_size, n_prev, SEP, accelerator, save_dir, fps, sample_size, caption

def get_actions(left_gripper, right_gripper, all_ends_p=None, all_ends_o=None, n_previous=4):
    ### the first frame is repeated to fill memory
    n = all_ends_p.shape[0]-1+n_previous
    slices = np.concatenate([np.zeros(n_previous-1).astype(int), np.arange(all_ends_p.shape[0])])
    all_left_quat = []
    all_right_quat = []

    ### cam eef 30...CAM_ANGLE...
    cvt_vis_l = Rotation.from_euler("xyz", np.array(EEF2CamLeft))
    cvt_vis_r = Rotation.from_euler("xyz", np.array(EEF2CamRight))
    for i in slices:
        # xyzw
        rot_l = Rotation.from_quat(all_ends_o[i, 0])
        rot_vis_l = rot_l*cvt_vis_l
        left_vis_quat = np.concatenate((all_ends_p[i,0], rot_vis_l.as_quat()), axis=0)
        
        rot_r = Rotation.from_quat(all_ends_o[i, 1])
        rot_vis_r = rot_r*cvt_vis_r
        right_vis_quat = np.concatenate((all_ends_p[i,1], rot_vis_r.as_quat()), axis=0)
        all_left_quat.append(left_vis_quat)
        all_right_quat.append(right_vis_quat)

    ### xyz, xyzw
    all_left_quat = np.stack(all_left_quat)
    all_right_quat = np.stack(all_right_quat)

    ### xyz, xyzw, gripper
    all_abs_actions = np.zeros([n, 16])
    
    for i in range(0, n):
        all_abs_actions[i, 0:7] = all_left_quat[i, :7]
        all_abs_actions[i, 7] = left_gripper[slices[i]]
        all_abs_actions[i, 8:15] = all_right_quat[i, :7]
        all_abs_actions[i, 15] = right_gripper[slices[i]]
        
    return all_abs_actions, slices

def parse_h5(h5_file, n_previous=4):
    """
    read and parse .h5 file, and obtain the absolute actions and the action differences
    """
    with h5py.File(h5_file, "r") as fid:
        # action gripper (0-1) --> state gripper (35-120)
        all_abs_gripper_l = np.array(fid[f"action/left_effector/position"], dtype=np.float32) * 85 + 35
        all_abs_gripper_r = np.array(fid[f"action/right_effector/position"], dtype=np.float32) * 85 + 35
        all_ends_p = np.array(fid["state/end/position"], dtype=np.float32)
        all_ends_o = np.array(fid["state/end/orientation"], dtype=np.float32)

    all_abs_actions, slices = get_actions(
        left_gripper=all_abs_gripper_l,
        right_gripper=all_abs_gripper_r,
        all_ends_p=all_ends_p,
        all_ends_o=all_ends_o,
        n_previous=n_previous,
    )
    return all_abs_actions, slices

def get_image(img_path, n_previous=4):
    img = np.array(Image.open(img_path))
    img = torch.from_numpy(img).float().permute(2,0,1)/255.0
    img = img.unsqueeze(1).repeat(1, n_previous, 1, 1)
    return img

def get_action_h5(
    action_path, n_previous=4
):  
    ### repeat first frame
    action, slices = parse_h5(action_path, n_previous=n_previous) 
    action = torch.FloatTensor(action)
    return action, slices

def get_caminfo_json(extrinsic_path, intrinsic_path, n):
    info = json.load(open(extrinsic_path, "r"))
    if len(info) == 1:
        info = info[0]
        c2w = np.eye(4)
        c2w[:3,:3] = np.array(info["extrinsic"]["rotation_matrix"])
        c2w[:3, 3] = np.array(info["extrinsic"]["translation_vector"])
        c2w = torch.from_numpy(c2w).float()
        w2c = torch.linalg.inv(c2w).float()
        w2c = w2c.unsqueeze(0).repeat(n,1,1)
        c2w = c2w.unsqueeze(0).repeat(n,1,1)
    else:
        c2ws = np.zeros((len(info), 4, 4))
        for info_idx in range(len(info)):
            c2ws[info_idx, :3, :3] = np.array(info[info_idx]["extrinsic"]["rotation_matrix"])
            c2ws[info_idx, :3, 3] = np.array(info[info_idx]["extrinsic"]["translation_vector"])
        c2ws[:, 3, 3] = 1.
        c2ws = torch.from_numpy(c2ws).float()
        c2w = torch.cat([c2ws[0:1].repeat(n-len(c2ws), 1, 1), c2ws], dim=0)
        w2c = torch.linalg.inv(c2w).float()
        
    info = json.load(open(intrinsic_path, "r"))["intrinsic"]
    intrinsic = np.eye(3)
    intrinsic[0,0] = info["fx"]
    intrinsic[0,2] = info["ppx"]
    intrinsic[1,1] = info["fy"]
    intrinsic[1,2] = info["ppy"]
    intrinsic = torch.from_numpy(intrinsic).float()
    return c2w, w2c, intrinsic

def get_depth(preprocess_episode, cam_name, slices, norm_depth=True):
    """
    get depth video frames according to the input slices;
    output video shape: (c,t,h,w)
    """
    depth_root = os.path.join(preprocess_episode, cam_name+"_depth_ori")
    depth = []
    for idx in tqdm(slices, desc=f"{cam_name} depth..."):
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

def normalize_depth_list_global(depth_list, specific_transforms_norm):
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

def get_canny(preprocess_episode, cam_name, slices):
    """
    get canny video frames according to the input slices;
    output video shape: (c,t,h,w)
    """
    canny_root = os.path.join(preprocess_episode, cam_name+"_depth_canny")
    canny = []
    for idx in tqdm(slices, desc=f"{cam_name} canny..."):
        canny_path = os.path.join(canny_root, f"{idx}.png")
        canny_edge = imageio.imread(canny_path).astype(np.uint8)
        canny.append(canny_edge)
    canny =  torch.from_numpy(np.stack(canny ,axis=0)).unsqueeze(0).contiguous()
    canny = canny.float() / 255.0
    return canny.repeat(3, 1, 1, 1)

def load_depth_and_canny(data_path, camera_view, slices):
    head_depth = get_depth(data_path, camera_view, slices, norm_depth=False)
    head_canny = get_canny(data_path, camera_view, slices)
    return head_depth, head_canny

def load_depth_and_canny_npz(data_path, slices):
    all_depth = np.load(os.path.join(data_path, "depthmap.npz"))["arr_0"][slices] / 1000.0 # t x v x h x w
    all_canny = np.load(os.path.join(data_path, "depthmap_canny.npz"))["arr_0"][slices] / 255.0 # t x v x h x w
    all_depth = all_depth.swapaxes(0, 1) # v x t x h x w
    all_canny = all_canny.swapaxes(0, 1) # v x t x h x w
    all_depth = torch.from_numpy(all_depth)
    all_canny = torch.from_numpy(all_canny)
    all_depth = all_depth.unsqueeze(1).repeat(1, 3, 1, 1, 1).float().contiguous() # v x c x t x h x w
    all_canny = all_canny.unsqueeze(1).repeat(1, 3, 1, 1, 1).float().contiguous() # v x c x t x h x w
    return all_depth, all_canny

def main(args):
    seed_everything(args.seed)
    cprint(">>> Preparing model ...", "yellow")
    stime = time.time()
    trainer, chunk, n_previous, SEP, accelerator, save_dir, fps, sample_size, caption = prepare_model_and_data(
        save_dir=args.save_dir,
        model_id=args.model_id,
        chunk=args.chunk,
    )
    prepare_time = time.time() - stime
    cprint(">>> Model ready {:.2f} s...".format(prepare_time), "yellow")
    specific_transforms_resize = transforms.Resize(sample_size)
    specific_transforms_norm = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    scalestr = "{:.1f}".format(args.scale)
    debugstr="_debug" if args.debug else ""
    clips = sorted([
        f.name for f in Path(args.input_root).iterdir() if f.is_dir()
    ], key=lambda x: int(x.split("_")[-1]))[:args.demo_num]
    # clips = [x for x in clips if not os.path.exists(os.path.join(args.input_root, x, f"{args.suffix}rgb_step{args.steps}_chunk{chunk}_scale{scalestr}{debugstr}.mp4"))]
    clips = np.array(clips)
    split_clips = np.array_split(clips, args.num_gpus)
    clips = split_clips[args.rank]
    print(f"rank {args.rank}", clips)
    print(f"no_depth: {args.no_depth}, no_canny: {args.no_canny}, no_action: {args.no_action}, no_raymap: {args.no_raymap}")
    print(f"checkpoint: {args.save_dir}")
    print(f"save_video: {args.save_video}")
    all_inputs_warning_str = "" if not args.save_video_for_all_inputs else "This will cost large memory. Ensure that you have enough CPU memory (~128GB) before you open this option."
    print(f"save_video_for_all_inputs: {args.save_video_for_all_inputs}. {all_inputs_warning_str}")
    for clip in tqdm(clips, desc=f"clips on rank {args.rank}"):
        try:
            path = os.path.join(args.input_root, clip)
            if not os.path.isdir(path):
                continue
            save_path = os.path.join(args.input_root, clip, f"{args.suffix}camera_step{args.steps}_chunk{chunk}_scale{scalestr}" if args.debug else "camera")
            save_name = os.path.join(args.input_root, clip, f"{args.suffix}rgb_step{args.steps}_chunk{chunk}_scale{scalestr}{debugstr}.mp4" if args.debug else f"{clip}_generated_demo_video.mp4")
            os.makedirs(save_path, exist_ok=True)
            depth_mv = []
            canny_mv = []
            img_mv = []
            intrinsics_mv = []
            c2ws_mv = []
            w2cs_mv = []
            traj_mv = []
            cprint(">>> Preloading first chunk ...", "yellow")
            stime = time.time()
            for view_index in range(len(CAMERA_VIEWS)):
                if "ablation" in args.suffix:
                    image_path = os.path.join(path, f"{CAMERA_VIEWS[view_index]}_frame.png")
                else:
                    image_path = os.path.join(path, f"{args.suffix}{CAMERA_VIEWS[view_index]}_frame.png")
                save_dir = os.path.join(save_path, str(0))
                os.makedirs(save_dir, exist_ok=True)
                shutil.copy(image_path, os.path.join(save_dir, f"{CAMERA_VIEWS[view_index]}_color.jpg"))
            action_path = os.path.join(path, "aligned_joints.h5")
            if not os.path.exists(action_path):
                action_path = os.path.join(path, "proprio_stats.h5")
            action, slices = get_action_h5(
                action_path, n_previous
            )
            n = action.shape[0]
                
            if args.depth_canny_type == "npz":
                depth_all, canny_all = load_depth_and_canny_npz(path, slices)
                
                # depth, canny, slices = load_depth_and_canny(path, camera_view, slices)
            for camera_view_idx, camera_view in enumerate(CAMERA_VIEWS):
                if "ablation" in args.suffix:
                    image_path = os.path.join(path, f"{CAMERA_VIEWS[camera_view_idx]}_frame.png")
                else:
                    image_path = os.path.join(path, f"{args.suffix}{CAMERA_VIEWS[camera_view_idx]}_frame.png")
                intrinsic_path = os.path.join(path, f"{camera_view}_intrinsic_params.json")
                extrinsic_path = os.path.join(path, f"{camera_view}_extrinsic_params_aligned.json")

                ### read and repeat image to fille memory frames
                # image: first frame only, n_previous + demo length
                img = get_image(
                    image_path, n_previous
                )
                
                # action, depth, canny, c2w, w2c: all, n_previous + demo length - 1
                if args.depth_canny_type == "png":
                    depth, canny = load_depth_and_canny(path, camera_view, slices)
                else:
                    depth, canny = depth_all[camera_view_idx], canny_all[camera_view_idx]
                
                c2w, w2c, intrinsic = get_caminfo_json(
                    extrinsic_path,
                    intrinsic_path,
                    n
                )
                
                img, intrinsic, depth, canny = transform_video(
                    img, specific_transforms_resize, intrinsic, sample_size, depth, canny
                )
                traj_map = get_traj(sample_size, action, w2c, c2w, intrinsic)

                img = normalize_video(img, specific_transforms_norm)
                traj_map = normalize_video(traj_map, specific_transforms_norm)
                canny = normalize_video(canny, specific_transforms_norm)
                

                traj_mv.append(traj_map)
                depth_mv.append(depth)
                canny_mv.append(canny)
                img_mv.append(img)
                intrinsics_mv.append(intrinsic)
                c2ws_mv.append(c2w)
                w2cs_mv.append(w2c)

            depth_mv = normalize_depth_list_global(depth_mv, specific_transforms_norm)
            all_traj = torch.stack(traj_mv, dim=1).unsqueeze(0) # (b, c, v, t, h, w)
            all_depth = torch.stack(depth_mv, dim=1).unsqueeze(0) # (b, c, v, t, h, w)
            all_canny = torch.stack(canny_mv, dim=1).unsqueeze(0) # (b, c, v, t, h, w)
            video = torch.stack(img_mv, dim=1).unsqueeze(0) # (b, c, v, t, h, w)
            all_intrinsic = torch.stack(intrinsics_mv, dim=0).unsqueeze(0)  # (b, v, 3, 3)
            all_c2w = torch.stack(c2ws_mv, dim=0).unsqueeze(0) # (b, v, t, 4, 4)
            all_w2c = torch.stack(w2cs_mv, dim=0).unsqueeze(0) # (b, v, t, 4, 4)
            video_path = [path]
            n_chunk_to_pred = int(math.ceil((float(n)-n_previous)/chunk))
            prepare_time = time.time() - stime
            video = rearrange(video, 'b c v t h w -> (b v) c t h w')
            # start chunk loop
            video_list = None
            traj_list = None
            depth_list = None
            canny_list = None
            max_frame = all_traj.shape[3]
            cprint(">>> First chunk ready {:.2f} s...".format(prepare_time), "yellow")
            frame_id = 1
            for i_chunk in tqdm(range(n_chunk_to_pred), desc=f'video generation for clip {clip}'):
                if i_chunk == 0:
                    traj = all_traj[:, :, :, :n_previous+chunk].clone()
                    depth = all_depth[:, :, :, :n_previous+chunk].clone()
                    canny = all_canny[:, :, :, :n_previous+chunk].clone()
                    c2w = all_c2w[:, :, :n_previous+chunk].clone()
                    missing = 0
                else:
                    # sample memory from previous chunks
                    select_mem = torch.linspace(0, video_list.shape[2]-1, n_previous).long()  # pyright: ignore
                    video = video_list[:, :, select_mem].clone()  # pyright: ignore
                    start = n_previous + chunk * i_chunk # max 504
                    end = min(n_previous + chunk * (i_chunk + 1), max_frame) # 514
                    cur_len = end - start # 10
                    if cur_len < chunk:
                        missing = chunk - cur_len # 15
                        all_traj = torch.cat((all_traj, all_traj[:, :, :, end-1:end].repeat(1, 1, 1, missing, 1, 1)), dim=3)
                        all_depth = torch.cat((all_depth, all_depth[:, :, :, end-1:end].repeat(1, 1, 1, missing, 1, 1)), dim=3)
                        all_canny = torch.cat((all_canny, all_canny[:, :, :, end-1:end].repeat(1, 1, 1, missing, 1, 1)), dim=3)
                        all_c2w = torch.cat((all_c2w, all_c2w[:, :, end-1:end].repeat(1, 1, missing, 1, 1)), dim=2)
                        end += missing
                    else:
                        missing = 0
                    traj = torch.cat((
                        all_traj[:,:,:,select_mem],
                        all_traj[:,:,:,start:end]
                    ), dim=3)
                    depth = torch.cat((
                        all_depth[:,:,:,select_mem],
                        all_depth[:,:,:,start:end]
                    ), dim=3)
                    canny = torch.cat((
                        all_canny[:,:,:,select_mem],
                        all_canny[:,:,:,start:end]
                    ), dim=3)
                    c2w = torch.cat((
                        all_c2w[:,:,select_mem],
                        all_c2w[:,:,start:end]
                    ), dim=2)
                if args.no_depth:
                    depth = torch.zeros_like(depth)
                if args.no_canny:
                    canny = torch.zeros_like(canny)
                if args.no_action:
                    traj = torch.zeros_like(traj)
                
                # reshape for input
                if len(c2w.shape) == 5:
                    c2w = rearrange(c2w, "b v t i j -> (b v t) i j")
                video = rearrange(video, '(b v) c t h w -> b c v t h w', v=3)
                if i_chunk == n_chunk_to_pred-1:
                    intrinsic = all_intrinsic.unsqueeze(dim=2).repeat(1,1,n_previous+end-start,1,1)  # repeat for frame number times
                else:
                    intrinsic = all_intrinsic.unsqueeze(dim=2).repeat(1,1,n_previous+chunk,1,1)  # repeat for frame number times
                intrinsic = rearrange(intrinsic, "b v t i j -> (b v t) i j")
                # diffusion loop
                # TODO: should we clamp video to (-1, 1) for each chunk?
                preds = trainer.validate(accelerator, save_dir, 
                                        dataloader=None, video=video, prompt=caption,
                                        traj=traj, depth=depth, canny=canny, 
                                        intrinsic=intrinsic, extrinsic=c2w,
                                        n_prev=n_previous, n_view=3, chunk_size=chunk,
                                        merge_view_into_width=False, fps=fps, video_path=video_path,
                                        vis_cat_traj=False, vis_cat_depth=False, vis_cat_canny=False, 
                                        t=traj.shape[3], 
                                        guidance_scale=args.scale, write_video_to_disk=False, pipeline_progress=False,
                                        num_inference_steps=args.steps, no_raymap=args.no_raymap)
                preds = preds.data.cpu()
                preds = preds[:, :, :(traj.shape[3]-n_previous)]
                if video_list is None:
                    video_list = preds.clone()  # preds don't contain n_prev memory
                    if args.save_video and args.save_video_for_all_inputs:
                        traj_list = traj[:, :, :, n_previous:].clone()
                        depth_list = depth[:, :, :, n_previous:].clone()
                        canny_list = canny[:, :, :, n_previous:].clone()
                else:
                    video_list = torch.cat((video_list, preds), dim=2)  # preds don't contain n_prev memory
                    if args.save_video and args.save_video_for_all_inputs:
                        traj_list = torch.cat((traj_list, traj[:,:,:,n_previous:]), dim=3)
                        depth_list = torch.cat((depth_list, depth[:,:,:,n_previous:]), dim=3)
                        canny_list = torch.cat((canny_list, canny[:,:,:,n_previous:]), dim=3)
                    
                preds = preds[:, :, :(chunk-missing)] #:10 # 25 * 20 + 10 = 510
                preds = torch.clamp(preds, min=-1, max=1)
                preds = ((preds+1)/2*255).to(torch.uint8)
                preds = rearrange(preds, 'v c t h w -> v t h w c').numpy()
                preds = preds[:, :, :, :, ::-1] # rgb --> bgr
                for time_index in range(preds.shape[1]):
                    save_dir = os.path.join(save_path, str(frame_id))
                    os.makedirs(save_dir, exist_ok=True)
                    for view_index in range(preds.shape[0]):
                        saved_image = cv2.resize(preds[view_index, time_index], (512, 294))
                        cv2.imwrite(os.path.join(save_dir, f"{CAMERA_VIEWS[view_index]}_color.jpg"), saved_image)
                    frame_id += 1
            if args.save_video:
                video_to_save = rearrange(video_list, '(b v) c t h w -> b c t h (v w)', v=3)[0]
                video_to_save = torch.clamp(video_to_save, min=-1, max=1)
                if args.save_video_for_all_inputs:
                    traj_list = rearrange(traj_list, 'b c v t h w -> b c t h (v w)')[0]
                    depth_list = rearrange(depth_list, 'b c v t h w -> b c t h (v w)')[0]
                    canny_list = rearrange(canny_list, 'b c v t h w -> b c t h (v w)')[0]
                    video_to_save = torch.cat((video_to_save, traj_list, depth_list, canny_list), dim=2)
                video_to_save = video_to_save[:, 1:, :, :]
                cprint(">>> Saving visualization video...".format(prepare_time), "yellow") # 20G
                stime = time.time()
                save_video(video_to_save, save_name, fps=fps)
                video_time = time.time() - stime
                cprint(">>> Saved visualization video {:.2f} s...".format(video_time), "yellow")
        except Exception as e:
            print("Error:", e)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="help document")

    parser.add_argument(
        "--input_root", "-i", type=str,
        help="Path to the input directory"
    )
    parser.add_argument(
        "--save_dir", "-s", type=str,
        default='checkpoints/video_generation_checkpoint',
        help="transformer directory"
    )
    parser.add_argument(
        "--model_id", "-m", type=str,
        default="checkpoints/Cosmos-Predict2-2B-Video2World",
        help="pretrained weights directory"
    )
    parser.add_argument(
        "--save_video", "-sv", action='store_true'
    )
    parser.add_argument(
        "--save_video_for_all_inputs", "-svi", action='store_true'
    )
    parser.add_argument(
        "--debug", "-d", action="store_true"
    )
    parser.add_argument(
        "--steps", "-stp", type=int, default=4
    )
    parser.add_argument(
        "--scale", type=float, default=1.0
    )
    parser.add_argument(
        "--chunk", "-c", type=int, default=25
    )
    parser.add_argument(
        "--rank", "-r", type=int, default=0
    )
    parser.add_argument(
        "--num_gpus", "-ng", type=int, default=8
    )
    parser.add_argument(
        "--demo_num", "-dn", type=int, default=200
    )
    parser.add_argument(
        "--depth_canny_type", type=str, default="npz"
    )
    parser.add_argument(
        "--suffix", type=str, default=""
    ) # edited_
    parser.add_argument(
        "--no_prepare", action="store_true"
    )
    parser.add_argument(
        "--no_depth", action="store_true"
    )
    parser.add_argument(
        "--no_canny", action="store_true"
    )
    parser.add_argument(
        "--no_action", action="store_true"
    )
    parser.add_argument(
        "--no_raymap", action="store_true"
    )
    args = parser.parse_args()
    args.seed = 1
    args.gid = 1
    main(args)

