import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from vggt.models.vggt import VGGT
from training.data.dynamic_dataloader import build_dynamic_dataloader
from training.loss import camera_loss, metric_depth_loss, metric_point_loss
from hydra import initialize, compose
from train_util import MetricLogger
import train_util as misc

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Training for VGGT")
    parser.add_argument("--config_path", type=str, required=True, help="Root Path to the configuration file")
    parser.add_argument("--train_config_path", type=str, required=True, help="Path to the train configuration file")
    parser.add_argument("--max_image_per_gpu", type=int, default=48, help="Max number of images per GPU")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of data loading workers")
    parser.add_argument("--max_iterations", type=int, default=1e4, help="Number of training iterations")
    parser.add_argument("--accu_step", type=int, default=2, help="Iteration interval for accumulating gradient")
    parser.add_argument("--log_step", type=int, default=100, help="Iteration interval for logging")
    parser.add_argument("--eval_step", type=int, default=1000, help="Iteration interval for evaluation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for the backbone")
    parser.add_argument("--frozen_backbone", action='store_true', help="Whether store the backbone")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for optimizer")
    parser.add_argument("--grad_loss_weight", type=float, default=1, help="Weight for gradient matching loss")
    parser.add_argument("--camera_loss_weight", type=float, default=1, help="Weight for camera loss")
    parser.add_argument("--depth_loss_weight", type=float, default=1, help="Weight for depth loss")
    parser.add_argument("--point_loss_weight", type=float, default=1, help="Weight for pointmap loss")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model weights")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save logs")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument("--world_size", type=int, default=1, help="Number of GPUs for distributed training")
    parser.add_argument("--dist_url", type=str, default="env://", help="URL for distributed training setup")
    return parser.parse_args()


def train(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(args.log_dir, args.save_dir)
    # Initialize model
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.load("../checkpoints/vggt_base_model.pt", map_location="cpu"))
    if args.checkpoint:
        print(f"Loading pretrained weights from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model_state_dict"])
    model = model.to(args.gpu)

    # Set different learning rates for different parts of the model
    optimizer = AdamW([
        {"params": model.aggregator.parameters(), "lr": args.lr_backbone if not args.frozen_backbone else 0},
        {"params": model.camera_head.parameters(), "lr": args.lr},
        {"params": model.point_head.parameters(), "lr": args.lr},
        {"params": model.depth_head.parameters(), "lr": args.lr},
        {"params": model.track_head.parameters(), "lr": args.lr}
    ], weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_iterations // args.accu_step, eta_min=1e-6)

    # Wrap model with DistributedDataParallel
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # Dataset and DataLoader
    initialize(config_path=args.config_path)
    train_cfg = compose(config_name=args.train_config_path)
    eval_real_cfg = compose(config_name="eval_real_config.yaml")
    eval_sim_cfg = compose(config_name="eval_sim_config.yaml")
    train_dataloader = build_dynamic_dataloader(
        dataset=train_cfg.dataset,
        common_config=train_cfg.common_config,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        max_img_per_gpu=args.max_image_per_gpu,
        seed=seed
    )
    eval_real_dataloader = build_dynamic_dataloader(
        dataset=eval_real_cfg.dataset,
        common_config=eval_real_cfg.common_config,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        max_img_per_gpu=args.max_image_per_gpu,
        seed=seed,
    )
    eval_sim_dataloader = build_dynamic_dataloader(
        dataset=eval_sim_cfg.dataset,
        common_config=eval_sim_cfg.common_config,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        max_img_per_gpu=args.max_image_per_gpu,
        seed=seed,
    )
    # prepare for logging
    if args.rank == 0:
        log_file_path = os.path.join(args.log_dir, "log.txt")
        os.makedirs(args.log_dir, exist_ok=True)

    train_metricLogger = MetricLogger(window_size=int(args.log_step))
    total_iterations = 0
    total_epochs = 0

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler(enabled=True)

    while total_iterations < args.max_iterations:
        total_epochs += 1
        train_dataloader.batch_sampler.set_epoch(total_epochs)
        for batch in train_dataloader:
            total_iterations += 1
            model.train()
            batch["images"] = batch["images"].to(args.gpu)
            batch["depths"] = batch["depths"].to(args.gpu)
            batch["extrinsics"] = batch["extrinsics"].to(args.gpu)
            batch["intrinsics"] = batch["intrinsics"].to(args.gpu)
            batch["world_points"] = batch["world_points"].to(args.gpu)
            batch["cam_points"] = batch["cam_points"].to(args.gpu)
            batch["point_masks"] = batch["point_masks"].to(args.gpu)

            with autocast(dtype=torch.bfloat16):
                outputs = model(batch["images"])
                depth_loss_dict = metric_depth_loss(outputs["depth"], outputs["depth_conf"], batch, gradient_loss='grad')
                point_loss_dict = metric_point_loss(outputs["world_points"], outputs["world_points_conf"], batch, gradient_loss='grad')
                camera_loss_dict = camera_loss(outputs["pose_enc"], batch, loss_type="l1")
                loss_dict = camera_loss_dict | depth_loss_dict | point_loss_dict
                loss = args.camera_loss_weight * loss_dict['loss_camera'] + args.point_loss_weight * loss_dict['loss_conf'] + \
                        args.depth_loss_weight * loss_dict['loss_conf_depth'] + args.grad_loss_weight * (loss_dict['loss_grad_depth'] + loss_dict['loss_grad'])

            # Scale the loss and backpropagate
            scaler.scale(loss).backward()
            if total_iterations % args.accu_step == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            train_metricLogger.update(loss=loss.item(), **loss_dict)
            # Log training loss
            if total_iterations % args.log_step == 0:
                train_metricLogger.synchronize_between_processes()
                if args.rank == 0:
                    log_message = train_metricLogger.log(total_iterations, args.rank, log_interval=args.log_step)
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"{log_message}\n")
            
            # Evaluate model
            if total_iterations % args.eval_step == 0:
                #---------------------------------------------------
                # Evaluation on Real data
                eval_metrciLogger = MetricLogger()
                model.eval()
                for eval_batch in eval_real_dataloader:
                    eval_batch["images"] = eval_batch["images"].to(args.gpu)
                    eval_batch["depths"] = eval_batch["depths"].to(args.gpu)
                    eval_batch["extrinsics"] = eval_batch["extrinsics"].to(args.gpu)
                    eval_batch["intrinsics"] = eval_batch["intrinsics"].to(args.gpu)
                    eval_batch["world_points"] = eval_batch["world_points"].to(args.gpu)
                    eval_batch["cam_points"] = eval_batch["cam_points"].to(args.gpu)
                    eval_batch["point_masks"] = eval_batch["point_masks"].to(args.gpu)

                    with torch.no_grad(), autocast(dtype=torch.bfloat16):
                        outputs = model(eval_batch["images"])
                        depth_loss_dict = metric_depth_loss(outputs["depth"], outputs["depth_conf"], eval_batch, gradient_loss='grad')
                        point_loss_dict = metric_point_loss(outputs["world_points"], outputs["world_points_conf"], eval_batch, gradient_loss='grad')
                        camera_loss_dict = camera_loss(outputs["pose_enc"], eval_batch, loss_type="l1")
                        loss_dict = camera_loss_dict | depth_loss_dict | point_loss_dict
                        loss = args.camera_loss_weight * loss_dict['loss_camera'] + args.point_loss_weight * loss_dict['loss_conf'] + \
                                args.depth_loss_weight * loss_dict['loss_conf_depth'] + args.grad_loss_weight * (loss_dict['loss_grad_depth'] + loss_dict['loss_grad'])
                        eval_metrciLogger.update(loss=loss.item(), **loss_dict)
                eval_metrciLogger.synchronize_between_processes()
                if args.rank == 0:
                    avg_loss = eval_metrciLogger.global_average()
                    log_message = f"Real Eval Iteration {total_iterations}: " + \
                                  ", ".join([f"{k}: {v:.4f}" for k, v in avg_loss.items()])
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"{log_message}\n")
                #-------------------------------------------------
                # Evaluation on Sim data
                eval_metrciLogger = MetricLogger()
                model.eval()
                for eval_batch in eval_sim_dataloader:
                    eval_batch["images"] = eval_batch["images"].to(args.gpu)
                    eval_batch["depths"] = eval_batch["depths"].to(args.gpu)
                    eval_batch["extrinsics"] = eval_batch["extrinsics"].to(args.gpu)
                    eval_batch["intrinsics"] = eval_batch["intrinsics"].to(args.gpu)
                    eval_batch["world_points"] = eval_batch["world_points"].to(args.gpu)
                    eval_batch["cam_points"] = eval_batch["cam_points"].to(args.gpu)
                    eval_batch["point_masks"] = eval_batch["point_masks"].to(args.gpu)

                    with torch.no_grad(), autocast(dtype=torch.bfloat16):
                        outputs = model(eval_batch["images"])
                        depth_loss_dict = metric_depth_loss(outputs["depth"], outputs["depth_conf"], eval_batch, gradient_loss='grad')
                        point_loss_dict = metric_point_loss(outputs["world_points"], outputs["world_points_conf"], eval_batch, gradient_loss='grad')
                        camera_loss_dict = camera_loss(outputs["pose_enc"], eval_batch, loss_type="l1")
                        loss_dict = camera_loss_dict | depth_loss_dict | point_loss_dict
                        loss = args.camera_loss_weight * loss_dict['loss_camera'] + args.point_loss_weight * loss_dict['loss_conf'] + \
                                args.depth_loss_weight * loss_dict['loss_conf_depth'] + args.grad_loss_weight * (loss_dict['loss_grad_depth'] + loss_dict['loss_grad'])
                        eval_metrciLogger.update(loss=loss.item(), **loss_dict)
                eval_metrciLogger.synchronize_between_processes()
                if args.rank == 0:
                    avg_loss = eval_metrciLogger.global_average()
                    log_message = f"Sim Eval Iteration {total_iterations}: " + \
                                  ", ".join([f"{k}: {v:.4f}" for k, v in avg_loss.items()])
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"{log_message}\n")

                if args.rank == 0:
                    # Save checkpoint
                    checkpoint_path = os.path.join(args.save_dir, f"checkpoint_{total_iterations}.pth")
                    os.makedirs(args.save_dir, exist_ok=True)
                    torch.save({
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "iteration": total_iterations,
                        "epoch": total_epochs
                    }, checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")
                
            if total_iterations >= args.max_iterations:
                break

if __name__ == "__main__":
    args = parse_args()
    train(args)