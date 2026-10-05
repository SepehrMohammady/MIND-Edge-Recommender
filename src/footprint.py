"""Model footprint + energy proxy for the cost axis of the results matrix.

Costs are counted per layer on the network that is actually built, with forward
hooks on every ``nn.Conv1d`` / ``nn.Linear`` instance. Subclasses are included,
so the simulated-quantized layers (``QuantConv1d``, ``QuantLinear``,
``BinConv1d``) are counted like any other layer. An earlier version relied on
``thop``, which looks layers up by exact type and therefore skipped them.

Conventions
  * MACs: one multiply-accumulate per weight per output position. Batch
    normalisation is folded into the preceding convolution at deployment and
    adds none.
  * Size: every stored parameter. Weights of a quantized layer cost ``bits/8``
    bytes each; everything else (embedding, biases, normalisation, layers kept
    in full precision) costs 4 bytes.
  * Energy: per-layer MACs x per-operation energy at that layer's weight
    precision (Horowitz, ISSCC 2014, 45 nm). A relative proxy only: it ignores
    memory traffic and assumes the deployed layer runs at its weight precision.
"""
from __future__ import annotations

import torch
import torch.nn as nn

BYTES_PER_WEIGHT = {32: 4.0, 8: 1.0, 1: 0.125}   # 1 bit = 1/8 byte


def layer_bits(module: nn.Module) -> int:
    """Weight precision of a layer: 32 unless it is a simulated-quantized one."""
    from src.binary import BinConv1d
    from src.quantize import QuantConv1d, QuantLinear

    if isinstance(module, (QuantConv1d, QuantLinear)):
        return int(module.bits)
    if isinstance(module, BinConv1d):
        return 1
    return 32


@torch.no_grad()
def layer_costs(model: nn.Module, example_input: torch.Tensor) -> list[dict]:
    """One row per Conv1d/Linear layer: weights, weight precision and MACs for a
    single input (batch size 1)."""
    rows: list[dict] = []
    hooks = []

    def make_hook(name: str):
        def hook(module, _inputs, output):
            weights = module.weight.numel()
            if isinstance(module, nn.Conv1d):
                positions = output.shape[-1]
            else:                                   # Linear: once per input vector
                positions = output.numel() // (output.shape[0] * module.out_features)
            rows.append({"layer": name, "type": type(module).__name__,
                         "bits": layer_bits(module), "weights": weights,
                         "macs": int(positions * weights)})
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = example_input.device
    model(example_input[:1].to(device))
    for h in hooks:
        h.remove()
    model.train(was_training)
    return rows


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model: nn.Module, example_input: torch.Tensor) -> int:
    """Total MACs of one inference (all Conv1d/Linear layers, any precision)."""
    return sum(r["macs"] for r in layer_costs(model, example_input))


def model_size_bytes(model: nn.Module) -> float:
    """Stored size: quantized layer weights at their precision, the rest FP32."""
    quantized = 0.0
    quantized_weights = 0
    for module in model.modules():
        bits = layer_bits(module) if isinstance(module, (nn.Conv1d, nn.Linear)) else 32
        if bits < 32:
            quantized += module.weight.numel() * BYTES_PER_WEIGHT[bits]
            quantized_weights += module.weight.numel()
    return quantized + (count_params(model) - quantized_weights) * BYTES_PER_WEIGHT[32]


def energy_pj(costs: list[dict], cfg: dict) -> float:
    """Energy proxy in picojoules: per-layer MACs x per-op energy at the layer's
    weight precision."""
    e = cfg["energy_pj"]
    per_op = {32: e["fp32_mac"], 8: e["int8_mac"], 1: e["binary_op"]}
    return sum(r["macs"] * per_op[r["bits"]] for r in costs)


def summarize(model: nn.Module, example_input: torch.Tensor, cfg: dict,
              precision: str | None = None) -> dict:
    """One row of the footprint table. ``precision`` is only a label; the costs
    are read from the layers of ``model`` itself."""
    costs = layer_costs(model, example_input)
    macs = sum(r["macs"] for r in costs)
    macs_fp32 = sum(r["macs"] for r in costs if r["bits"] == 32)
    size_b = model_size_bytes(model)
    row = {
        "params": count_params(model),
        "macs": macs,
        "macs_fp32": macs_fp32,
        "macs_quantized": macs - macs_fp32,
        "size_kb": round(size_b / 1024, 2),
        "size_mb": round(size_b / 1024 / 1024, 4),
        "energy_uj_per_inf": round(energy_pj(costs, cfg) / 1e6, 4),
    }
    if precision is not None:
        row = {"precision": precision, **row}
    return row
