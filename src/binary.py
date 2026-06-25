"""Improved binary (1-bit) byte-CNN encoder, ReActNet/Bi-Real style.

Naive sign-binarization collapses ranking accuracy. This module adds the
standard tricks that recover most of it while keeping the model 1-bit:
  * RSign       - sign with a learnable per-channel threshold (ReActNet)
  * RPReLU      - learnable per-channel shifted PReLU (ReActNet)
  * Bi-Real     - real-valued shortcut around each binary conv
  * binary weights = sign x per-output-channel mean|w| (XNOR-Net scaling)
Following XNOR-Net the byte embedding, the first projection, and the linear head
stay full precision; only the inner conv weights+activations are binarized.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quantize import _SignSTE


class RSign(nn.Module):
    """Sign activation with a learnable per-channel threshold."""

    def __init__(self, ch: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, ch, 1))

    def forward(self, x):
        return _SignSTE.apply(x - self.bias)


class RPReLU(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.b1 = nn.Parameter(torch.zeros(1, ch, 1))
        self.b2 = nn.Parameter(torch.zeros(1, ch, 1))
        self.a = nn.Parameter(torch.full((1, ch, 1), 0.25))

    def forward(self, x):
        x = x - self.b1
        return torch.where(x >= 0, x, self.a * x) + self.b2


def _binary_weight(w):
    scale = w.abs().mean(dim=tuple(range(1, w.dim())), keepdim=True)
    return _SignSTE.apply(w) * scale


class BinConv1d(nn.Conv1d):
    def forward(self, x):
        return F.conv1d(x, _binary_weight(self.weight), self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class BiRealBlock(nn.Module):
    """Binary-activated, binary-weight conv with a real shortcut + RPReLU."""

    def __init__(self, ch: int):
        super().__init__()
        self.sign = RSign(ch)
        self.conv = BinConv1d(ch, ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(ch)
        self.act = RPReLU(ch)

    def forward(self, x):
        out = self.bn(self.conv(self.sign(x)))
        return self.act(out) + x          # Bi-Real real-valued shortcut


class BinaryByteCNNEncoder(nn.Module):
    """Same interface/shape as ByteCNNEncoder, with binary inner blocks."""

    def __init__(self, byte_embed_dim=64, channels=128, depth=3, out_dim=384):
        super().__init__()
        self.embed = nn.Embedding(257, byte_embed_dim, padding_idx=0)   # FP
        self.proj_in = nn.Conv1d(byte_embed_dim, channels, 1)           # FP (first)
        self.blocks = nn.ModuleList(BiRealBlock(channels) for _ in range(depth))
        self.head = nn.Linear(channels, out_dim)                        # FP (last)
        self.out_dim = out_dim

    def forward(self, ids):
        mask = (ids != 0).float().unsqueeze(1)
        x = self.proj_in(self.embed(ids).transpose(1, 2))
        for blk in self.blocks:
            x = blk(x)
        x = (x * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        return self.head(x)
