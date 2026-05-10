"""Distributed training utilities for multi-GPU DDP."""
import os
import torch
import torch.distributed as dist


def init_dist():
    """Initialize distributed training. Auto-detect torchrun env vars.

    Returns:
        (rank, local_rank, world_size). Falls back to (0, 0, 1) for single-GPU.
    """
    if 'RANK' not in os.environ:
        return 0, 0, 1

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()
    return rank, local_rank, world_size


def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_value(value, average=True):
    """All-reduce a scalar value across processes."""
    world_size = get_world_size()
    if world_size < 2:
        return value
    with torch.no_grad():
        t = torch.tensor(value, dtype=torch.float32, device='cuda')
        dist.all_reduce(t)
        if average:
            t /= world_size
    return t.item()
