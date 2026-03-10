import torch.distributed as dist
from collections import deque
import numpy as np
import torch

import torch
import torch.distributed as dist
import builtins
import datetime
import os

class MetricLogger:
    def __init__(self, window_size=100):
        """
        Tracks and aggregates training metrics (e.g., loss).

        Args:
            window_size (int): Size of the smoothing window.
        """
        self.values = {}
        self.deques = {}
        self.totals = {} 
        self.counts = {}
        self.window_size = window_size

    def update(self, **kwargs):
        """
        Updates metric values.

        Args:
            kwargs: Metric names and values to be updated.
        """
        for k, v in kwargs.items():
            if k not in self.values:
                self.values[k] = 0.0
                self.deques[k] = deque(maxlen=self.window_size)
                self.totals[k] = 0.0
                self.counts[k] = 0
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            self.deques[k].append(v)
            self.values[k] = np.mean(self.deques[k])
            self.totals[k] += v
            self.counts[k] += 1

    def synchronize_between_processes(self):
        """
        Synchronize metric values across all processes during multi-GPU training.
        """
        for k in self.values.keys():
            tensor = torch.tensor(self.values[k], dtype=torch.float32, device="cuda")
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            self.values[k] = tensor.item() / dist.get_world_size()

            total_tensor = torch.tensor(self.totals[k], dtype=torch.float32, device="cuda")
            count_tensor = torch.tensor(self.counts[k], dtype=torch.float32, device="cuda")
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            self.totals[k] = total_tensor.item()
            self.counts[k] = count_tensor.item()

    def global_average(self):
        """
        Computes the global average of all accumulated values (without using a sliding window).

        Returns:
            dict: A dictionary containing the global average of all metrics.
        """
        global_avg = {}
        for k in self.totals.keys():
            if self.counts[k] > 0:
                global_avg[k] = self.totals[k] / self.counts[k]
            else:
                global_avg[k] = 0.0
        return global_avg

    def log(self, iteration, rank, log_interval=100):
        """
        Prints log information (main process only).

        Args:
            iteration (int): The current iteration number.
            rank (int): The rank of the current process.
            log_interval (int): The interval at which to print logs.
        """
        if rank == 0 and iteration % log_interval == 0:
            log_message = f"Iteration {iteration}: "
            log_message += ", ".join([f"{k}: {v:.4f}" for k, v in self.values.items()])
            print(log_message)
            return log_message


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    nodist = args.nodist if hasattr(args,'nodist') else False 
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and not nodist:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)