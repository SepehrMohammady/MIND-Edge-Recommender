"""Measured latency + energy (complements the Horowitz proxy in footprint.py).

  * Laptop GPU (RTX 5070): latency via CUDA-synced timing; energy via NVML power
    sampling (nvidia-ml-py) integrated over the run.
  * Raspberry Pi 5 / CPU: latency via onnxruntime on the exported ONNX (the same
    code path runs on the Pi5 aarch64 wheel; pair with an INA219 for real power).

Report mJ/inference and ms/inference with the idle baseline subtracted.
"""
from __future__ import annotations

import time

import numpy as np
import torch


@torch.no_grad()
def latency_torch(model, example, n: int = 200, warmup: int = 20) -> float:
    """Mean ms/inference on the model's device."""
    device = next(model.parameters()).device
    example = example.to(device)
    model.eval()
    for _ in range(warmup):
        model(example)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(example)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


@torch.no_grad()
def energy_nvml(model, example, n: int = 500) -> dict:
    """Energy/inference (mJ) via NVML power sampling on the GPU."""
    try:
        import pynvml
    except ImportError:
        return {"available": False}
    device = next(model.parameters()).device
    if device.type != "cuda":
        return {"available": False}
    example = example.to(device)
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    idle = np.mean([pynvml.nvmlDeviceGetPowerUsage(h) for _ in range(20)]) / 1000  # W

    model.eval()
    for _ in range(20):
        model(example)
    torch.cuda.synchronize()
    powers, t0 = [], time.perf_counter()
    for _ in range(n):
        model(example)
        powers.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    pynvml.nvmlShutdown()

    avg_w = float(np.mean(powers))
    per_inf_ms = dt / n * 1000
    return {"available": True, "avg_power_w": round(avg_w, 2),
            "idle_w": round(float(idle), 2),
            "energy_mj_per_inf": round((avg_w - idle) * dt / n * 1000, 4),
            "latency_ms": round(per_inf_ms, 4)}


def latency_onnx(onnx_path: str, max_len: int, n: int = 200, warmup: int = 20) -> float:
    """Mean ms/inference via onnxruntime CPU (same path used on the Pi 5)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.zeros((1, max_len), dtype=np.int64)
    for _ in range(warmup):
        sess.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(None, {name: x})
    return (time.perf_counter() - t0) / n * 1000
