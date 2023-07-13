import os
import torch
import builtins

def setup_distributed():
    """
        Distributed training using Pytorch and slurm
    """
    world_size = int(os.environ['WORLD_SIZE'])
    if 'SLURM_JOB_ID' in os.environ:  # for slurm schedule jobs
        rank = int(os.environ['SLURM_PROCID'])
        device = int(os.environ['SLURM_LOCALID'])
    else: # local
        device = rank = int(os.environ['LOCAL_RANK'])

    # init number of processes
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    print('WORLD_SIZE: {}, RANK: {}, LOCAL_RANK: {}, MASTER: {}:{}'.format(
        world_size, rank, device,
        os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
    )

    if rank != 0:  # print only if on master gpu (0)
        def print_pass(*args):
            pass
        builtins.print = print_pass

    return device, rank, world_size