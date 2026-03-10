from yaml import load, Loader
import argparse
from datetime import datetime
from datetime import timedelta
import os
import os.path as osp
from copy import deepcopy
import json
import yaml
from pathlib import Path
import math

import torch
from torch.utils.tensorboard import SummaryWriter
from torch import distributed as dist

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
    gather_object,
)

import transformers
import diffusers
from diffusers.training_utils import cast_training_params
from diffusers.optimization import get_scheduler

from lib.utils.logging import print_and_save_logging
from lib.utils.optimizer_utils import get_optimizer, gradient_norm
from lib.utils.misc import Dict


logger = get_logger('acwm_cosmos2_trainer')
logger.setLevel('INFO')


class State:
    # Training state
    seed: int = None
    model_name: str = None
    accelerator: Accelerator = None
    weight_dtype: torch.dtype = None
    train_epochs: int = None
    train_steps: int = None
    overwrote_max_train_steps: bool = False
    num_trainable_parameters: int = 0
    learning_rate: float = None
    train_batch_size: int = None
    generator: torch.Generator = None

    # Hub state
    repo_id: str = None
    # Artifacts state
    output_dir: str = None


class BaseTrainer(object):
    """docstring for BaseTrainer"""
    def __init__(self, config_file, val_only=False):
        cd = load(open(config_file, 'r'), Loader=Loader)

        args = argparse.Namespace(**cd)
        args.lr = float(args.lr)
        args.epsilon = float(args.epsilon)
        args.weight_decay = float(args.weight_decay)

        self.args = args

        self.state = State

        self._init_distributed()
        if not val_only:
            self._init_logging()
            self._init_directories_and_repositories()

        self.state.model_name = self.args.model_name

    def _init_logging(self):
        if self.state.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        current_time = datetime.now()
        start_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")

        if self.state.accelerator.is_main_process:
            self.save_folder = osp.join(self.args.output_dir, start_time)
            if getattr(self.args, "sub_folder", False):   # if there is a subfolder, then save dir iscreated according to subfolder
                self.save_folder = os.path.join(self.args.output_dir, self.args.sub_folder)
            os.makedirs(self.save_folder, exist_ok=True)
     
            # Save as YAML file
            if isinstance(self.args, Dict):
                args_dict = self.args.to_dict()
            else:
                args_dict = vars(deepcopy(self.args))
                for k, v in args_dict.items():
                    args_dict[k] = str(v)
            # with open(os.path.join(self.save_folder, 'config.json'), "w") as file:
            #     json.dump(args_dict, file, indent=4, sort_keys=False)
            with open(osp.join(self.save_folder, 'config.yaml'), "w") as file:
                yaml.dump(args_dict, file, sort_keys=False, indent=2)

            self.writer = SummaryWriter(log_dir=self.save_folder)

            save_folder_bytes = self.save_folder.encode()
            folder_len_tensor = torch.tensor([len(save_folder_bytes)], device=self.state.accelerator.device)
            if dist.is_initialized():
                dist.broadcast(folder_len_tensor, src=0)
                folder_tensor = torch.ByteTensor(list(save_folder_bytes)).to(self.state.accelerator.device)
                dist.broadcast(folder_tensor, src=0)
        else:
            if dist.is_initialized():
                # Receive path from the main process
                folder_len_tensor = torch.tensor([0], device=self.state.accelerator.device)
                dist.broadcast(folder_len_tensor, src=0)
                folder_tensor = torch.empty(folder_len_tensor.item(), dtype=torch.uint8, device=self.state.accelerator.device)
                dist.broadcast(folder_tensor, src=0)
                self.save_folder = bytes(folder_tensor.tolist()).decode()

        # Initialize logging for all processes
        print_and_save_logging(self.save_folder, rank=self.state.accelerator.process_index)

    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        if getattr(self.args, "use_deepspeed", False):
            per_device_bs = self.args.batch_size
            world_size = int(os.environ.get("WORLD_SIZE", 1))  # or self.args.world_size
            grad_accum = self.args.gradient_accumulation_steps

            train_batch_size = per_device_bs * world_size * grad_accum
            self.args.deepspeed["train_batch_size"] = train_batch_size
            ds_plugin = DeepSpeedPlugin(
                hf_ds_config=self.args.deepspeed,
                gradient_accumulation_steps=grad_accum
            )
        else:
            ds_plugin = None

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
            deepspeed_plugin=ds_plugin,
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.state.accelerator = accelerator

        if self.args.seed is not None:
            self.state.seed = self.args.seed
            set_seed(self.args.seed)

        weight_dtype = torch.float32
        if self.state.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.state.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            
        self.state.weight_dtype = weight_dtype

    def _init_directories_and_repositories(self):
        if self.state.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            self.state.output_dir = self.args.output_dir


    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        
        if self.args.train_type == "lora":
            components_to_disable_grads = [ self.transformer ]
        else:
            components_to_disable_grads = []
            
        for component in components_to_disable_grads:
            if component is not None:
                component.requires_grad_(False)

        if torch.backends.mps.is_available() and self.state.weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        if self.args.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
                
        if self.args.prev_checkpoint is not None:
            raise NotImplementedError

    def prepare_optimizer(self):
        logger.info("Initializing optimizer and lr scheduler")

        train_mode = self.args.train_mode

        self.state.train_epochs = self.args.train_epochs
        self.state.train_steps = self.args.train_steps

        # Make sure the trainable params are in float32
        if self.args.mixed_precision == "fp16":
        # if self.args.mixed_precision == "fp16" or self.args.mixed_precision == "bf16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params([self.transformer], dtype=torch.float32)

        self.state.learning_rate = self.args.lr
        if self.args.scale_lr:
            self.state.learning_rate = (
                self.state.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.batch_size
                * self.state.accelerator.num_processes
            )

        transformer_lora_parameters = []
        if train_mode == 'action_only':
            for name, param in self.transformer.named_parameters():
                if 'action_' in name:
                    param.requires_grad = True
                    transformer_lora_parameters.append(param)
                else:
                    param.requires_grad = False

        elif train_mode == "video_only":
            for name, param in self.transformer.named_parameters():
                if 'action_' not in name and "lang_" not in name:
                    param.requires_grad = True
                    transformer_lora_parameters.append(param)
                else:
                    param.requires_grad = False

        elif train_mode == "all" or train_mode == 'action_full':
            for name, param in self.transformer.named_parameters():
                param.requires_grad = True
                transformer_lora_parameters.append(param)

        else:
            raise NotImplementedError

        num_trainable_params = sum(p.numel() for p in transformer_lora_parameters)
        logger.info(f'Total trainable parameters: {num_trainable_params}')

        transformer_parameters_with_lr = {
            "params": transformer_lora_parameters,
            "lr": self.state.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in transformer_lora_parameters)

        # TODO(aryan): add deepspeed support
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_8bit = self.args.optimizer_8bit,
            use_torchao = self.args.optimizer_torchao,
        )

        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.state.train_steps is None:
            self.state.train_steps = self.state.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.state.accelerator.num_processes,
            num_training_steps=self.state.train_steps * self.state.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler



    def prepare_for_training(self):
        self.transformer, self.optimizer, self.train_dataloader, self.lr_scheduler = self.state.accelerator.prepare(
            self.transformer, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

    def prepare_trackers(self):
        logger.info("Initializing trackers")

        tracker_name = self.args.tracker_name or "acwm_cosmos2"
        self.state.accelerator.init_trackers(tracker_name, config=self.args.__dict__)