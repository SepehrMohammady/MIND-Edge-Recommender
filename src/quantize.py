"""Precision axis: FP32 -> INT8 -> Binary, via simulated quantization (QAT/PTQ).

We swap Conv1d/Linear for quant-aware versions that fake-quantize their weights
in the forward pass with a straight-through estimator. The SAME model then
serves both:
  * PTQ  -> ``convert_to_quant`` then evaluate (no retraining)
  * QAT  -> ``convert_to_quant`` then ``qat_finetune`` on the ranking task

Following XNOR-Net practice the byte EMBEDDING and (optionally) the first/last
projection stay full precision; only the cheap inner conv/linear weights are
quantized. INT8 uses bits=8; Binary uses bits=1 (sign x per-channel scale).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

BITS = {"fp32": 32, "int8": 8, "binary": 1}


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


class _SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, 1.0, -1.0)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1).float()      # clip grad outside [-1, 1]


def fake_quant_weight(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-output-channel symmetric fake quantization with STE."""
    if bits >= 32:
        return w
    dims = tuple(range(1, w.dim()))
    if bits == 1:                              # binary: sign * mean|w|
        scale = w.abs().mean(dim=dims, keepdim=True).clamp(min=1e-8)
        return _SignSTE.apply(w) * scale
    qmax = 2 ** (bits - 1) - 1
    scale = (w.abs().amax(dim=dims, keepdim=True) / qmax).clamp(min=1e-8)
    return _RoundSTE.apply((w / scale).clamp(-qmax - 1, qmax)) * scale


class QuantConv1d(nn.Conv1d):
    bits = 8

    def forward(self, x):
        return F.conv1d(x, fake_quant_weight(self.weight, self.bits), self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class QuantLinear(nn.Linear):
    bits = 8

    def forward(self, x):
        return F.linear(x, fake_quant_weight(self.weight, self.bits), self.bias)


def _swap(parent, name, child, bits):
    if isinstance(child, nn.Conv1d):
        q = QuantConv1d(child.in_channels, child.out_channels, child.kernel_size[0],
                        stride=child.stride, padding=child.padding,
                        dilation=child.dilation, groups=child.groups,
                        bias=child.bias is not None)
    else:
        q = QuantLinear(child.in_features, child.out_features, bias=child.bias is not None)
    q.weight.data = child.weight.data.clone()
    if child.bias is not None:
        q.bias.data = child.bias.data.clone()
    q.bits = bits
    setattr(parent, name, q)


def convert_to_quant(model: nn.Module, precision: str, keep_first_last=True) -> nn.Module:
    """Deep-copy ``model`` and quantize inner Conv1d/Linear to ``precision``.

    Embedding stays FP32 always; ``proj_in`` (first) and ``head`` (last) stay
    FP32 when ``keep_first_last`` (standard for binary nets)."""
    bits = BITS[precision]
    m = copy.deepcopy(model)
    if bits >= 32:
        return m
    skip = {"news_encoder.proj_in", "proj_in", "news_encoder.head", "head"} if keep_first_last else set()
    for full_name, module in list(m.named_modules()):
        if isinstance(module, (nn.Conv1d, nn.Linear)) and full_name not in skip:
            parent = m.get_submodule(full_name.rsplit(".", 1)[0]) if "." in full_name else m
            _swap(parent, full_name.rsplit(".", 1)[-1], module, bits)
    return m


def quant_fp_fraction(model: nn.Module) -> float:
    """Fraction of parameters kept full precision (embedding + skipped layers)."""
    total = sum(p.numel() for p in model.parameters())
    quant = sum(m.weight.numel() for m in model.modules()
                if isinstance(m, (QuantConv1d, QuantLinear)))
    return 1.0 - quant / total if total else 1.0


def qat_finetune(quant_model, cfg, train_fn, epochs: int | None = None):
    """Fine-tune an already-converted quant model on the ranking task.
    ``train_fn(model, epochs)`` is supplied by the caller (recommender loop)."""
    return train_fn(quant_model, epochs or cfg["train"].get("qat_epochs", 3))
