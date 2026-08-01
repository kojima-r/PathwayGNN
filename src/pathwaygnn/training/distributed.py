from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0


def initialize(device_name: str = "auto", timeout_minutes: int = 30) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    use_cuda = device_name != "cpu" and torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    if distributed and not dist.is_initialized():
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))
    rank = dist.get_rank() if distributed else 0
    return DistributedContext(rank, local_rank, world_size, distributed, device)


def finalize(context: DistributedContext) -> None:
    if context.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

