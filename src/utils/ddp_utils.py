import os
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
from datetime import timedelta
import torch.distributed as dist
import itertools

def seed_everything(seed: int = 0):
    """Set random seed for reproducibility across all libraries."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        

def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"]) 
    world_size = int(os.environ['WORLD_SIZE'])
    print(f"RANK and WORLD_SIZE in environ: {global_rank}/{world_size}")

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(hours=2))
    torch.distributed.barrier()

    return local_rank, global_rank


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def gather_texts(texts):
    output = [None for _ in range(dist.get_world_size())]
    gather_objects = texts
    dist.all_gather_object(output, gather_objects)
    gathered_texts = list(itertools.chain(*output))
    return gathered_texts
