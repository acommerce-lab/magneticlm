import os
import multiprocessing as mp
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class Resources:
    num_cpus: int
    num_gpus: int
    gpu_ids: List[int]
    primary_device: torch.device
    multi_gpu: bool

    def __str__(self):
        gpu_str = ",".join(str(i) for i in self.gpu_ids) if self.gpu_ids else "none"
        return (
            f"CPUs={self.num_cpus}  GPUs={self.num_gpus}[{gpu_str}]  "
            f"primary={self.primary_device}  multi_gpu={self.multi_gpu}"
        )


def detect(cfg) -> Resources:
    cpu_count = os.cpu_count() or 1
    if cfg.num_workers > 0:
        num_cpus = min(cfg.num_workers, cpu_count)
    else:
        num_cpus = cpu_count

    cuda_available = torch.cuda.is_available()
    if cfg.device == "cpu" or not cuda_available:
        num_gpus = 0
        gpu_ids: List[int] = []
        primary = torch.device("cpu")
    else:
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))
        primary = torch.device("cuda:0")

    multi_gpu = cfg.multi_gpu and num_gpus > 1
    return Resources(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        gpu_ids=gpu_ids,
        primary_device=primary,
        multi_gpu=multi_gpu,
    )


def pool_context(resources: Resources, workers: int = -1):
    n = resources.num_cpus if workers <= 0 else min(workers, resources.num_cpus)
    return mp.get_context("fork").Pool(processes=n)


def shard_indices(n_items: int, n_shards: int) -> List[range]:
    if n_shards <= 1:
        return [range(0, n_items)]
    step = (n_items + n_shards - 1) // n_shards
    return [range(i, min(i + step, n_items)) for i in range(0, n_items, step)]
