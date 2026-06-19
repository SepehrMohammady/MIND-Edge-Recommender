"""Model footprint + energy proxies for the cost axis of the results matrix.

We report, per (architecture x precision) cell:
  * parameter count
  * MACs (thop, with fvcore as a cross-check)
  * model size in bytes at the given precision
  * an architecture-agnostic ENERGY proxy using Horowitz (ISSCC 2014, 45nm)

The Horowitz figure is a *relative* proxy and is labelled as such in the paper;
measured energy (NVML on the laptop, INA219 on the Pi 5) is reported separately.
"""
from __future__ import annotations

BYTES_PER_PARAM = {"fp32": 4.0, "int8": 1.0, "binary": 0.125}  # 1 bit = 1/8 byte


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model, example_input) -> int:
    """MACs via thop. Returns 0 if thop is unavailable."""
    try:
        from thop import profile
    except ImportError:
        return 0
    try:
        device = next(model.parameters()).device
        example_input = example_input.to(device)
    except StopIteration:
        pass
    macs, _ = profile(model, inputs=(example_input,), verbose=False)
    return int(macs)


def model_size_bytes(num_params: int, precision: str,
                     fp_fraction: float = 0.0) -> float:
    """Size at ``precision``. ``fp_fraction`` = share of params kept FP32
    (e.g. first/last layers + embedding in a binary net)."""
    bpp = BYTES_PER_PARAM[precision]
    fp_params = num_params * fp_fraction
    q_params = num_params - fp_params
    return q_params * bpp + fp_params * BYTES_PER_PARAM["fp32"]


def energy_pj(macs: int, precision: str, cfg: dict,
              fp_fraction: float = 0.0) -> float:
    """Energy proxy in picojoules = MACs x per-op energy (Horowitz 45nm)."""
    e = cfg["energy_pj"]
    per_op = {"fp32": e["fp32_mac"], "int8": e["int8_mac"],
              "binary": e["binary_op"]}[precision]
    fp_macs = macs * fp_fraction
    q_macs = macs - fp_macs
    return q_macs * per_op + fp_macs * e["fp32_mac"]


def summarize(model, example_input, precision: str, cfg: dict,
              fp_fraction: float = 0.0) -> dict:
    """One row of the footprint table for a model at a given precision."""
    params = count_params(model)
    macs = count_macs(model, example_input)
    size_b = model_size_bytes(params, precision, fp_fraction)
    return {
        "precision": precision,
        "params": params,
        "macs": macs,
        "size_kb": round(size_b / 1024, 2),
        "size_mb": round(size_b / 1024 / 1024, 4),
        "energy_uj_per_inf": round(energy_pj(macs, precision, cfg, fp_fraction) / 1e6, 4),
    }
