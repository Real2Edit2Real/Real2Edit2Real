from yaml import load, Loader
import argparse
import os
import os.path as osp
import random
import json

import torch
from torch import nn
from diffusers.utils import export_to_video
from diffusers.utils.torch_utils import randn_tensor
from diffusers.training_utils import compute_density_for_timestep_sampling, \
    compute_loss_weighting_for_sd3
from accelerate.logging import get_logger
from accelerate import DistributedType
from transformers import T5EncoderModel, T5TokenizerFast
from einops import rearrange
import numpy as np

from lib.data.agibotworld_dataset import AgiBotWorld
from lib.trainers.base_trainer import BaseTrainer, State
from lib.models.transformers.transformer_cosmos_multiview import MultiViewCosmosTransformer3DModel
from lib.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from lib.pipelines.pipeline_cosmos2_acwm import ACWMCosmos2Pipeline
from lib.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from lib.utils.memory_utils import get_memory_statistics, free_memory
from lib.utils.misc import ProgressTracker
from lib.utils.geometry_utils import resize_traj_and_ray
from lib.utils.torch_utils import load_state_dict, unwrap_model, apply_color_jitter_to_video, \
                                  get_latents, save_video

logger = get_logger(__name__)
logger.setLevel('INFO')

SPATIAL_DOWN_RATIO = 8
TEMPORAL_DOWN_RATIO = 4


class ActionDepthCannyWMTrainer(BaseTrainer):

    def __init__(self, config_file, checkpoint_root=None, val_only=False):

        cd = load(open(config_file, 'r'), Loader=Loader)

        cd.setdefault('noisy_video', False)
        cd.setdefault('load_weights', True)
        cd.setdefault('state_label', False)
        cd.setdefault('num_features', 1)

        cd.setdefault('cat_traj', False)
        cd.setdefault('cat_rays', False)
        cd.setdefault('cat_depth', False)
        cd.setdefault('cat_canny', False)

        args = argparse.Namespace(**cd)

        if checkpoint_root is not None:
            args.output_dir = checkpoint_root

        args.lr = float(args.lr)
        args.epsilon = float(args.epsilon)
        args.weight_decay = float(args.weight_decay)

        self.args = args
        self.state = State

        # Tokenizers
        self.tokenizer = None

        # Text encoders
        self.text_encoder = None

        # Denoisers
        self.transformer = None

        # Autoencoders
        self.vae = None

        # Scheduler
        self.scheduler = None

        self._init_distributed()
        if not val_only:
            self._init_logging()
            self._init_directories_and_repositories()

        self.state.model_name = self.args.model_name

    def prepare_dataset(self):
        logger.info("Initializing AgiBot dataset and dataloader")

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dataset_type = getattr(self.args, "dataset_type", "processed")
        if dataset_type == "processed":
            dataset_func = AgiBotWorld
        # elif dataset_type == "all":
            # dataset_func= MixAll
        else:
            raise NotImplementedError

        # dataset_func = MixAC

        dataset_config = self.args.data['train']
        if getattr(self.args, "multi_resolution", False):
            dataset_config['sample_size'] = self.args.resolution_list[local_rank % len(self.args.resolution_list)]
            self.cur_batch_size = self.args.batch_size_list[local_rank % len(self.args.resolution_list)]
        else:
            self.cur_batch_size = self.args.batch_size
        self.train_dataset = dataset_func(gpuid=local_rank, **dataset_config)
        
        if self.train_dataset.decode_type == 'gpu':
            import multiprocessing as mp
            spawn_ctx = mp.get_context('spawn')
        else:
            spawn_ctx = None
        logger.info(f'>>>>>>>>>>>>>>>>>Video decode mode:{self.train_dataset.decode_type}<<<<<<<<<<<<<<<<<<<<')
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.cur_batch_size,
            num_workers=self.args.dataloader_num_workers,
            multiprocessing_context=spawn_ctx,
        )

        logger.info(f">>>>>>>>>>>>>Total eps:{len(self.train_dataset)}<<<<<<<<<<<<<<<<<<")
        if 'val' in self.args.data and getattr(self.args, "load_val", True):
            self.val_dataset = dataset_func(**self.args.data['val'])

            self.val_index = []
            for _ in range(self.args.batch_size):
                self.val_index.append(random.randint(0, len(self.val_dataset)-1))
            if self.state.accelerator.is_main_process:
                with open(os.path.join(self.save_folder, 'idx.txt'), "w") as file:
                    file.write(", ".join(map(str, self.val_index)))

            subset = torch.utils.data.Subset(self.val_dataset, self.val_index)

            # DataLoader
            self.val_dataloader = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size, shuffle=getattr(self.args, "val_shuffle", False))

    def prepare_models(self, model_id="../checkpoints/Cosmos-Predict2-2B-Video2World", from_hf=False):
        device = self.state.accelerator.device
        torch_dtype = self.state.weight_dtype
        # self.text_encoder = T5EncoderModel.from_pretrained(osp.join(model_id, 'text_encoder'), torch_dtype=torch_dtype).to(device)
        # self.tokenizer = T5TokenizerFast.from_pretrained(osp.join(model_id, 'tokenizer'))
        # self.vae = AutoencoderKLWan.from_pretrained(osp.join(model_id, 'vae'), torch_dtype=torch_dtype).to(device)
        self.text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder='text_encoder', torch_dtype=torch_dtype).to(device)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder='tokenizer')
        self.vae = AutoencoderKLWan.from_pretrained(model_id, subfolder='vae', torch_dtype=torch_dtype).to(device)
        if self.vae is not None:
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder='scheduler')

        if from_hf:
            self.transformer = MultiViewCosmosTransformer3DModel.from_pretrained(model_id, subfolder='transformer', torch_dtype=torch_dtype).to(device)
            patch_size = self.transformer.patch_embed.proj.patch_size
            in_channels = 16 + 3 + 6 # TODO
            out_channels = self.transformer.hidden_size
            self.transformer.patch_embed.proj = nn.Linear(in_channels * patch_size[0] * patch_size[1] * patch_size[2], out_channels, bias=False)
        else:
            self.transformer = MultiViewCosmosTransformer3DModel(**self.args.transformer['config']).to(device, dtype=torch_dtype)
            if self.args.transformer['model_path'] is not None:
                self.transformer = load_state_dict(self.transformer, self.args.transformer['model_path'])

    @torch.no_grad()
    def prepare_ray_map(self, intrinsic, c2w, H, W):
        ###
        ### intrinsic: b, 3, 3
        ### c2w:       b, 4, 4
        ### rays:      b, H, W, 3 and b, H, W, 3
        ### 
        # print(intrinsic.shape, c2w.shape)
        batch_size = intrinsic.shape[0]
        fx, fy, cx, cy = intrinsic[:,0,0].unsqueeze(1).unsqueeze(2), intrinsic[:,1,1].unsqueeze(1).unsqueeze(2), intrinsic[:,0,2].unsqueeze(1).unsqueeze(2), intrinsic[:,1,2].unsqueeze(1).unsqueeze(2)
        i, j = torch.meshgrid(torch.linspace(0.5, W-0.5, W, device=c2w.device), torch.linspace(0.5, H-0.5, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        i = i.unsqueeze(0).repeat(batch_size,1,1)
        j = j.unsqueeze(0).repeat(batch_size,1,1)
        dirs = torch.stack([(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:,np.newaxis,np.newaxis, :3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        rays_o = c2w[:, :3,-1].unsqueeze(1).unsqueeze(2).repeat(1,H,W,1)
        viewdir = rays_d/torch.norm(rays_d, dim=-1, keepdim=True)
        return rays_o, viewdir

    def train(self):
        logger.info("Starting training")
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.train_batch_size = (
            self.cur_batch_size * self.state.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.train_dataset),
            "train epochs": self.state.train_epochs,
            "train steps": self.state.train_steps,
            "batches per device": self.cur_batch_size,
            "total batches observed per epoch": len(self.train_dataloader),
            "train batch size": self.state.train_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        accelerator = self.state.accelerator
        weight_dtype = self.state.weight_dtype
        scheduler_sigmas = self.scheduler.sigmas.clone().to(device=accelerator.device, dtype=weight_dtype)  # pyright: ignore
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        cat_traj = getattr(self.args, "cat_traj", False)
        cat_rays = getattr(self.args, "cat_rays", False)
        cat_depth = getattr(self.args, "cat_depth", False)
        cat_canny = getattr(self.args, "cat_canny", False)
        cat_mems = getattr(self.args, "cat_mems", False)
        frame_wise_noise = getattr(self.args, "frame_wise_noise", False)

        # loss spikes
        anomalies = []

        # null prompt for classifier-free guidance
        negative_prompt = ""
        negative_prompt_embed = self.encode_prompt(negative_prompt)

        # ??? Acquired by cosmos transformers. looks like it's always 0
        h, w = self.args.data['train']['sample_size']  # 384, 512
        padding_mask = torch.zeros(1, 1, h, w, device=accelerator.device, dtype=weight_dtype)  # ???

        tracker = ProgressTracker(self.state.train_steps, description='Training Iterations')
        tracker.start()
        for epoch in range(first_epoch, self.state.train_epochs):
            logger.info(f"Starting epoch ({epoch + 1}/{self.state.train_epochs})")
            
            self.transformer.train()  # pyright: ignore

            running_loss = 0.0

            for step, batch in enumerate(self.train_dataloader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}
                model_list = [self.transformer, ]

                with accelerator.accumulate(model_list):
                    video = batch['video']

                    # shape b, c, v, t, h, w ranging from -1 to 1
                    video = video.to(accelerator.device, dtype=weight_dtype).contiguous()
                    batch_size, c, n_view, t, h, w = video.shape

                    cond_to_concat = None
                    all_dropout = random.random() < self.args.all_dropout
                    if cat_traj:
                        traj = batch['trajs']
                        traj = traj.to(accelerator.device, dtype=weight_dtype).contiguous()
                        if cond_to_concat is None:
                            cond_to_concat = traj
                        else:
                            cond_to_concat = torch.cat((cond_to_concat, traj), dim=1)

                    if cat_rays:
                        intrinsics = rearrange(batch['intrinsic'].unsqueeze(dim=2).repeat(1,1,t,1,1), "b v t i j -> (b v t) i j")
                        extrinsics = rearrange(batch['extrinsic'], "b v t i j -> (b v t) i j")
                        rays_o, rays_d = self.prepare_ray_map(intrinsics, extrinsics, H=h, W=w)
                        ### (b v t) h w c -> b c v t h w
                        rays = rearrange(torch.cat((rays_o, rays_d), dim=-1), "(b v t) h w c -> b c v t h w", v=n_view, t=t)
                        rays = rays.to(accelerator.device, dtype=weight_dtype).contiguous()
                        if cond_to_concat is None:
                            cond_to_concat = rays
                        else:
                            cond_to_concat = torch.cat((cond_to_concat, rays), dim=1)

                    if cat_depth:
                        depth = batch['depths']
                        depth = torch.zeros_like(depth) if (all_dropout or random.random() < self.args.depth_dropout) else depth
                        depth = depth.to(accelerator.device, dtype=weight_dtype).contiguous()
                        if cond_to_concat is None:
                            cond_to_concat = depth
                        else:
                            cond_to_concat = torch.cat((cond_to_concat, depth), dim=1)
                    
                    if cat_canny:
                        canny = batch['cannys']
                        canny = torch.zeros_like(canny) if (all_dropout or random.random() < self.args.canny_dropout) else canny
                        canny = canny.to(accelerator.device, dtype=weight_dtype).contiguous()
                        if cond_to_concat is None:
                            cond_to_concat = canny
                        else:
                            cond_to_concat = torch.cat((cond_to_concat, canny), dim=1)

                    video = rearrange(video, 'b c v t h w -> (b v) c t h w')

                    if random.random() <= self.args.use_color_jitter:
                        video = apply_color_jitter_to_video(video, same_jitter_within_view=True, n_view=n_view)

                    # slice out memory & future
                    mem_size = self.args.data['train']['n_previous']
                    fut_size = video.shape[2] - mem_size
                    mem = video[:, :, :mem_size]
                    future_video = video[:, :, mem_size:]

                    if self.train_dataset.ignore_seek:  # in this case the future frame of video is not provided
                        future_video = future_video.repeat(1,1,self.args.data['train']['chunk'],1,1)

                    # get the shape params
                    _, _, raw_frames, raw_height, raw_width = future_video.shape
                    latent_channels = self.vae.z_dim  # pyright: ignore
                    latent_frames = raw_frames // TEMPORAL_DOWN_RATIO + 1  # future only
                    latent_height = raw_height // SPATIAL_DOWN_RATIO
                    latent_width = raw_width // SPATIAL_DOWN_RATIO

                    with torch.no_grad():
                        # vae encode && reshape
                        # time shrink for future video, but not for mem
                        mem_latents, future_video_latents = get_latents(self.vae, mem, future_video)[:2]  # pyright: ignore

                    mem_latents = rearrange(mem_latents, '(b v m) (h w) c -> (b v) c m h w', b=batch_size, m=mem_size, h=latent_height)
                    future_video_latents = rearrange(future_video_latents, '(b v) (f h w) c -> (b v) c f h w',b=batch_size,h=latent_height,w=latent_width)

                    # concat memory and future video
                    latents = torch.cat((mem_latents, future_video_latents), dim=2)  # bv c m+f h w
                    # latents = rearrange(latents, 'bv c f h w -> bv (f h w) c')

                    # resize cond_to_concat (traj & ray map & depth & canny)
                    cond_to_concat = resize_traj_and_ray(cond_to_concat, 
                        mem_size=mem_size, future_size=latent_frames,
                        height=latent_height, width=latent_width)
                    cond_to_concat = rearrange(cond_to_concat, 'b c v t h w -> (b v) c t h w')

                    # gen noise
                    noise, conditioning_mask, cond_indicator = self.gen_noise_from_condition_frame_latent(
                            mem_latents, latent_frames, latent_height, latent_width, 
                            noise_to_condition_frames=self.args.noise_to_first_frame,
                            device=accelerator.device, dtype=weight_dtype
                        )  # bv c m+f h w

                    # encode text prompt
                    prompt_embeds = self.encode_prompt(batch['caption'])
                    if self.args.train_w_cfg:
                        # dropout a portion of the prompts for classifier-free guidance
                        dropout_factor = torch.rand(batch_size).to(accelerator.device, dtype=weight_dtype)
                        dropout_mask_prompt = dropout_factor < self.args.caption_dropout_p
                        dropout_mask_prompt = dropout_mask_prompt.unsqueeze(1).unsqueeze(2)
                        prompt_embeds = negative_prompt_embed.repeat(batch_size,1,1) * dropout_mask_prompt + \
                                        prompt_embeds * ~dropout_mask_prompt

                    # choose timesteps
                    timestep_weights = compute_density_for_timestep_sampling(
                        weighting_scheme=self.args.flow_weighting_scheme,
                        batch_size=batch_size,
                        logit_mean=self.args.flow_logit_mean,
                        logit_std=self.args.flow_logit_std,
                        mode_scale=self.args.flow_mode_scale,
                    )
                    timestep_weights = timestep_weights.unsqueeze(1).repeat(1,n_view)
                    timestep_weights = timestep_weights.reshape(-1)
                    indices = (timestep_weights * self.scheduler.config.num_train_timesteps).long()  # pyright: ignore
                    sigmas = scheduler_sigmas[indices]  # ranges from 0 to 1
                    # ATTN! cosmos-predict2 uses a different timestep conditioning
                    # timesteps = (sigmas * 1000.0).long()  # LTX: ranges from 0 to 1000
                    timesteps = sigmas  # Cosmos2: ranges from 0 to 1
                    # The sigma value used for scaling conditioning latents. Ideally, it should not be changed or should be
                    # set to a small value close to zero.
                    sigma_conditioning = torch.tensor(0.0001, dtype=torch.float32, device=accelerator.device)
                    t_conditioning = sigma_conditioning / (sigma_conditioning + 1)  # a very small number
                    cond_timestep = cond_indicator * t_conditioning + (1 - cond_indicator) * timesteps.view(-1, 1, 1, 1, 1)

                    # adding noise to latents
                    ss = sigmas.reshape(-1, 1, 1, 1, 1)
                    noisy_latents = (1.0 - ss) * latents + ss * noise  # bv c m+f h w

                    # loss weight
                    loss_weights = compute_loss_weighting_for_sd3(
                        weighting_scheme=self.args.flow_weighting_scheme, sigmas=sigmas).reshape(-1, 1, 1, 1, 1)

                    if cond_to_concat is not None:
                        noisy_latents = torch.cat([noisy_latents, cond_to_concat], dim=1)

                    pred = self.transformer(  # pyright: ignore
                        hidden_states=noisy_latents,
                        timestep=cond_timestep,
                        encoder_hidden_states=prompt_embeds,
                        fps=self.args.data['train']['fps'],
                        condition_mask=conditioning_mask,
                        padding_mask=padding_mask,
                        return_dict=False,
                        n_view=n_view
                    )[0]  # bv vae_z_dim m+f h w

                    # TODO: we may need to linearly interpolate noise_pred with noisy_latents,
                    # but assume the interpolation as None before NVIDIA releases training code
                    # noise_pred = noise_pred + noisy_latents
                    target = noise - latents

                    loss_video = loss_weights.float() * (pred.float() - target.float()).pow(2)
                    loss_video = loss_video * (1 - conditioning_mask)

                    # # Average loss across channel dimension
                    loss_video = loss_video.mean(list(range(1, loss_video.ndim)))
                    # Average loss across batch dimension
                    loss_video = loss_video.mean()

                    assert torch.isnan(loss_video) == False, "NaN loss detected"
                    accelerator.backward(loss_video)
                    if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                        grad_norm = accelerator.clip_grad_norm_(self.transformer.parameters(), self.args.max_grad_norm)  # pyright: ignore
                        logs["grad_norm"] = grad_norm

                    self.optimizer.step()  # pyright: ignore
                    self.lr_scheduler.step()  # pyright: ignore
                    self.optimizer.zero_grad()  # pyright: ignore

                # gather loss info outside of accelerator.accumulate
                loss_video = accelerator.reduce(loss_video.detach(), reduction='mean')

                running_loss += loss_video.item()
                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    tracker.update()
                    global_step += 1

                logs = {
                    "loss": loss_video.detach().item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],  # pyright: ignore
                }
                accelerator.log(logs, step=global_step)

                if global_step >= self.state.train_steps:
                    logger.info(">>> max train step reached")
                    break

                if global_step % self.args.steps_to_log == 0:
                    if accelerator.is_main_process:
                        self.writer.add_scalar("Training Loss", loss_video.item(), global_step)
                        print(f'loss: {logs["loss"]:.6f} lr: {logs["lr"]:.6f} | {tracker.get_progress_string()}')

                if self.args.load_val and global_step % self.args.steps_to_val == 0 or global_step == 1:
                    accelerator.wait_for_everyone()
                    # if accelerator.is_main_process:
                    with torch.no_grad():
                        model_save_dir = os.path.join(self.save_folder, f'Validation_step_{global_step}')

                        _ = self.validate(
                            accelerator, model_save_dir, 
                            dataloader=self.train_dataloader, n_view=n_view,
                            fps=self.args.data['train']['fps']
                        )
                        _ = self.validate(
                            accelerator, model_save_dir, 
                            dataloader=self.val_dataloader, n_view=n_view,
                            fps=self.args.data['val']['fps']
                        )
                    accelerator.wait_for_everyone()

                if global_step % self.args.steps_to_save == 0:
                    
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        model_to_save = unwrap_model(accelerator, self.transformer)
                        model_save_dir = osp.join(self.save_folder,f'step_{global_step}')
                        os.makedirs(model_save_dir, exist_ok=True)
                        model_to_save.save_pretrained(model_save_dir, safe_serialization=True)
                        del  model_to_save

            # get mem info after each epoch
            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

            if accelerator.is_main_process:
                avg_loss = running_loss / len(self.train_dataloader)
                self.writer.add_scalar("Average Training Loss", avg_loss, epoch)

        # training finished, save final model
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.transformer = unwrap_model(accelerator, self.transformer)
            model_save_dir = os.path.join(self.save_folder,f'step_{global_step}')
            self.transformer.save_pretrained(model_save_dir, safe_serialization=True)

        del self.transformer, self.scheduler
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()
                       

    def gen_noise_from_condition_frame_latent(self, mem_latents, future_latent_frames,
                                              latent_height, latent_width,
                                              noise_to_condition_frames=0.2,
                                              noise=None, generator=None,
                                              device='cuda', dtype=torch.bfloat16
    ):
        '''
        mem_latents: (b v) c m h w
        future_latent_frames: number of future latent frames
        '''
        mem_size = mem_latents.shape[2]
        num_channels_latents = mem_latents.shape[1]
        batch_size = mem_latents.shape[0]   # bv

        # noise with latent shape
        shape = (batch_size, num_channels_latents, mem_size+future_latent_frames, latent_height, latent_width)
        if noise is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        ### mem:1, pred: 0
        mask_shape = (batch_size, 1, mem_size+future_latent_frames, latent_height, latent_width)
        conditioning_mask = torch.zeros(mask_shape, device=device, dtype=dtype)
        conditioning_mask[:, :, :mem_size] = 1.0

        # similar to conditioning mask but useful to timesteps
        cond_indicator = torch.zeros((1, 1, mem_size+future_latent_frames, 1, 1), device=device, dtype=dtype)
        cond_indicator[:, :, :mem_size] = 1.0

        init_latents = mem_latents[:,:,-1:].repeat(1, 1, mem_size+future_latent_frames, 1, 1)
        init_latents[:,:,:mem_size] = mem_latents

        if noise_to_condition_frames > 0: 
            rand_noise_ff_s = torch.rand(batch_size) * noise_to_condition_frames
            rand_noise_ff_e = torch.rand(batch_size) * noise_to_condition_frames
            rand_noise_ff_s, rand_noise_ff_e = torch.minimum(rand_noise_ff_s, rand_noise_ff_e), torch.maximum(rand_noise_ff_s, rand_noise_ff_e)
            rand_noise_ff = torch.stack([torch.linspace(rand_noise_ff_s[_], rand_noise_ff_e[_], mem_size) for _ in range(batch_size)], dim=0)
            rand_noise_ff = rand_noise_ff.reshape(batch_size, 1, mem_size, 1, 1).to(dtype=dtype, device=device)        
            first_frame_mask = conditioning_mask.clone() 
            first_frame_mask[:, :, :mem_size] = 1.0 - rand_noise_ff
        else:
            first_frame_mask = conditioning_mask.clone()

        # before mem_size it's memory info; after mem_size it's noise
        noise_mask = 1 - first_frame_mask
        latents = init_latents * first_frame_mask + noise * noise_mask

        return latents, conditioning_mask, cond_indicator


    @torch.no_grad()
    def encode_prompt(self, prompt, max_sequence_length=512):
        device = self.state.accelerator.device
        dtype = self.state.weight_dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
            return_length=True,
            return_offsets_mapping=False,
        )  # pyright: ignore

        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask.bool().to(device)

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids  # pyright: ignore
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])  # pyright: ignore
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), attention_mask=prompt_attention_mask
        ).last_hidden_state  # pyright: ignore
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        lengths = prompt_attention_mask.sum(dim=1).cpu()
        for i, length in enumerate(lengths):
            prompt_embeds[i, length:] = 0

        return prompt_embeds


    @torch.no_grad()
    def validate(self, accelerator, model_save_dir, 
                 dataloader=None, video=None, prompt=None,
                 traj=None, depth=None, canny=None, intrinsic=None, extrinsic=None,
                 n_prev=None, n_view=3, chunk_size=None,
                 merge_view_into_width=True, fps=16, video_path=None,
                 vis_cat_traj=True, vis_cat_depth=True, vis_cat_canny=True, 
                 write_video_to_disk=True, t=None, 
                 guidance_scale=1.0, save_tag='', pipeline_progress=True,
                 num_inference_steps=35, no_raymap=False,
    ):
        os.makedirs(model_save_dir, exist_ok=True)
        pipe = ACWMCosmos2Pipeline(
            self.text_encoder, self.tokenizer, unwrap_model(accelerator, self.transformer),
            self.vae, self.scheduler)  # pyright: ignore

        if isinstance(dataloader, torch.utils.data.dataloader.DataLoader):
            dataloader = iter(dataloader)  # pyright: ignore

        if video is None:
            assert dataloader is not None
            batch = next(dataloader)
            video = batch['video']
            video = video[:,:,:,:self.args.data['train']['n_previous']]  # shape b,c,v,t,h,w -> 1,c,h,w
            prompt = batch['caption']
            video_path = batch['path']
            gt_video = batch['video']
        else:
            gt_video = video.clone()
            assert prompt is not None
            assert traj is not None
            assert depth is not None
            assert canny is not None
            assert intrinsic is not None
            assert extrinsic is not None
            assert video_path is not None

        b, c, n_view, _, h, w = video.shape  # t in video.shape might only be the mem size. using a different t
        n_prev = n_prev if n_prev is not None else self.args.data['val']['n_previous']
        chunk_size = chunk_size if chunk_size is not None else self.args.data['train']['chunk']
        if t is None:
            t = n_prev + chunk_size
        batch_size = 1

        cond_to_concat = None
        if self.args.cat_traj:
            if traj is None:
                traj = batch['trajs'][:, :, :, :t]  # pyright: ignore  # b c v t h w
            traj = traj.to(accelerator.device, dtype=self.state.weight_dtype).contiguous()
            if cond_to_concat is None:
                cond_to_concat = traj
            else:
                cond_to_concat = torch.cat((cond_to_concat, traj), dim=1)
        
        if self.args.cat_rays:
            if intrinsic is None:
                intrinsic = rearrange(batch['intrinsic'].unsqueeze(dim=2).repeat(1,1,t,1,1), "b v t i j -> (b v t) i j")  # pyright: ignore
            if extrinsic is None:
                extrinsic = batch['extrinsic'][:, :, :t]
                extrinsic = rearrange(extrinsic, "b v t i j -> (b v t) i j")
            
            rays_o, rays_d = self.prepare_ray_map(intrinsic, extrinsic, H=h, W=w)
            ### (b v t) h w c -> b c v t h w
            rays = rearrange(torch.cat((rays_o, rays_d), dim=-1), "(b v t) h w c -> b c v t h w", v=n_view, t=t)
            rays = rays.to(accelerator.device, dtype=self.state.weight_dtype).contiguous()
            if no_raymap:
                rays = torch.zeros_like(rays)
            if cond_to_concat is None:
                cond_to_concat = rays
            else:
                cond_to_concat = torch.cat((cond_to_concat, rays), dim=1)
        else:
            intrinsic, extrinsic = None, None
        
        if self.args.cat_depth:
            if depth is None:
                depth = batch['depths'][:, :, :, :t]  # pyright: ignore  # b c v t h w
            depth = depth.to(accelerator.device, dtype=self.state.weight_dtype).contiguous()
            if cond_to_concat is None:
                cond_to_concat = depth
            else:
                cond_to_concat = torch.cat((cond_to_concat, depth), dim=1)
        
        if self.args.cat_canny:
            if canny is None:
                canny = batch['cannys'][:, :, :, :t]  # pyright: ignore  # b c v t h w
            canny = canny.to(accelerator.device, dtype=self.state.weight_dtype).contiguous()
            if cond_to_concat is None:
                cond_to_concat = canny
            else:
                cond_to_concat = torch.cat((cond_to_concat, canny), dim=1)

        negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."

        video = rearrange(video, 'b c v t h w -> (b v) t c h w')  # * 5D Torch tensors: expected shape for each array `(batch_size, num_frames, num_channels, height, width)`.
        # resize cond_to_concat (traj & ray map)
        cond_to_concat = resize_traj_and_ray(cond_to_concat, 
            mem_size=n_prev, future_size=(t-n_prev)//TEMPORAL_DOWN_RATIO+1,
            height=h//SPATIAL_DOWN_RATIO, width=w//SPATIAL_DOWN_RATIO)
        cond_to_concat = rearrange(cond_to_concat, 'b c v t h w -> (b v) c t h w')
 
        preds = pipe(
            video=video,
            cond_to_concat=cond_to_concat,  # pyright: ignore
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=384, width=512, num_frames=(t-n_prev),
            n_view=n_view, n_prev=n_prev,
            guidance_scale=guidance_scale,
            merge_view_into_width=merge_view_into_width, fps=fps, 
            postprocess_video=False,
            show_progress=pipeline_progress,
            num_inference_steps=num_inference_steps,
            )['frames']  # (b v) c t h w or b c t h (v w), range -1 to 1 (could exceed range)

        if vis_cat_traj:
            # assert postprocess_video == False
            if merge_view_into_width:
                traj = rearrange(traj, 'b c v t h w -> b c t h (v w)')[:, :, n_prev:]
                preds = torch.cat([preds, traj], dim=3)  # concat on height
            else:
                # TODO
                raise NotImplementedError
        if vis_cat_depth:
            # assert postprocess_video == False
            if merge_view_into_width:
                depth = rearrange(depth, 'b c v t h w -> b c t h (v w)')[:, :, n_prev:]
                preds = torch.cat([preds, depth], dim=3)  # concat on height
            else:
                # TODO
                raise NotImplementedError
        if vis_cat_canny:
            # assert postprocess_video == False
            if merge_view_into_width:
                canny = rearrange(canny, 'b c v t h w -> b c t h (v w)')[:, :, n_prev:]
                preds = torch.cat([preds, canny], dim=3)  # concat on height
            else:
                # TODO
                raise NotImplementedError
        
        if write_video_to_disk:
            save_tag = f'_{save_tag}_'
            if merge_view_into_width == True:
                save_name = osp.join(model_save_dir, video_path[0].split('/')[-1] + save_tag + '.mp4')
                save_video(preds[0], save_name)
                print(f'Result saved to {save_name}')
            else:
                for iv in range(len(preds)):  # (b v) c t h w
                    save_name = osp.join(model_save_dir, 
                                         video_path[0].split('/')[-1] + save_tag + f'_view{iv}.mp4')
                    save_video(preds[iv], save_name, fps=fps)
                    print(f'Result saved to {save_name}')

            # if save_gt:  # TODO
            #     save_video(rearrange(gt_video[0], 'c v t h w -> c t h (v w)', v=n_view), os.path.join(model_save_dir, f'{cap}_gt.mp4'), fps=fps)  # pyright: ignore

        return preds
