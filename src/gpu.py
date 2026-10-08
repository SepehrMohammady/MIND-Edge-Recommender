"""GPU memory setting for the training runs on this laptop (Windows, 8 GB RTX 5070 Laptop GPU).

On Windows the CUDA driver backs allocations beyond the card's memory with system RAM ("sysmem
fallback") instead of failing them. PyTorch's caching allocator frees its cached blocks only when
an allocation fails, so with steps of varying size (the number of distinct titles per batch
changes) its reserved pool grows without bound into system RAM. On 2026-10-08 the click training
of 256-4-384 reserved 19.24 GiB for 3.30 GiB of tensors, ran at half speed (17 min per epoch) and
left 1.5 GB of the 31.5 GB of RAM free. Capping the process's share of the card makes the
allocator release cached blocks instead: same run 34 s instead of 170 s, 6.68 GiB reserved. The
computation itself is unchanged.
"""
from __future__ import annotations

import os

import torch


def cap_gpu_memory(fraction: float | None = None) -> None:
    """Limit the caching allocator to ``fraction`` of the card (default 0.85, env MIND_GPU_FRACTION)."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(fraction or os.environ.get("MIND_GPU_FRACTION", 0.85)))
