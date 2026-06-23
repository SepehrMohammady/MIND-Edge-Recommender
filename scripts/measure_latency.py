"""Measure REAL single-inference latency of each native architecture on this
laptop (GPU via torch, CPU via onnxruntime). Latency is weight-independent, so
random-init encoders of the searched shapes give valid timings. Saves
artifacts/latency.json. Run: python -m scripts.measure_latency
"""
import json
from pathlib import Path

import torch

from src.config import load_config
from src import export, measure_energy
from src.student import ByteCNNEncoder

cfg = load_config()
be, L = cfg["student"]["byte_embed_dim"], cfg["data"]["max_title_bytes"]
ARCHS = {"NAS": (256, 4, 384), "Micro-NAS": (64, 5, 384), "Bin.uNAS": (96, 2, 384)}

rows = []
for name, (c, d, o) in ARCHS.items():
    enc = ByteCNNEncoder(be, c, d, o)
    onnx = export.export_encoder_onnx(enc, cfg, name=f"_lat_{name}.onnx")
    cpu_ms = measure_energy.latency_onnx(onnx, L, n=200)
    dev_enc = enc.to("cuda") if torch.cuda.is_available() else enc
    gpu_ms = measure_energy.latency_torch(dev_enc, torch.zeros(1, L, dtype=torch.long), n=500)
    rows.append({"arch": name, "gpu_ms": round(gpu_ms, 4), "cpu_onnx_ms": round(cpu_ms, 4)})
    print(rows[-1], flush=True)
    Path(onnx).unlink(missing_ok=True)

Path(cfg["paths"]["artifacts_dir"], "latency.json").write_text(json.dumps(rows, indent=2))
print("LATENCY DONE")
