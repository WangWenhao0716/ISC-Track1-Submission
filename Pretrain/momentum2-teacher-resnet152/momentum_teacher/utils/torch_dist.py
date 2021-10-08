# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""
import subprocess
import os
from torch import distributed as dist


def reduce_tensor_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def configure_nccl():
    """Configure multi-machine environment variables.

    It is required for multi-machine training.
    """
    os.environ["NCCL_SOCKET_IFNAME"] = "ib0"
    os.environ["NCCL_IB_DISABLE"] = "1"

    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
    os.environ["NCCL_IB_HCA"] = subprocess.getoutput(
        "cd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; "
        "do cat $i/ports/1/gid_attrs/types/* 2>/dev/null "
        "| grep v >/dev/null && echo $i ; done; > /dev/null"
    )
    os.environ["NCCL_IB_GID_INDEX"] = "3"
    os.environ["NCCL_IB_TC"] = "106"


def synchronize():
    """Helper function to synchronize (barrier) among all processes when using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    current_world_size = dist.get_world_size()
    if current_world_size == 1:
        return
    dist.barrier()
