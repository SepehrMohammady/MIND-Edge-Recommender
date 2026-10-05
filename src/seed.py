"""Seeding helper: every run script calls this once so that weight
initialisation, data order and sampling are repeatable for a given seed.

GPU convolution kernels can still differ in the last bits between runs, so a
seed fixes a run up to that numerical noise, not bit for bit.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
