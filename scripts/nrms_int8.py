"""Measure NRMS at FP32 and INT8 (real numbers) so the baseline can be reported
as two models. INT8 = per-tensor symmetric weight quantization of the trained
FP32 model (embedding included; that table is 92% of the size). MINDsmall-dev.
Run: python -m scripts.nrms_int8
"""
import copy
import json
from pathlib import Path

import torch

from src.config import load_config
from src import baseline_nrms

cfg = load_config()
EP = 8  # match the full-run baseline


def q8(t):
    s = (t.abs().max() / 127).clamp(min=1e-8)
    return (t / s).round().clamp(-128, 127) * s


m = baseline_nrms.train_nrms(cfg, epochs=EP)
r_fp32 = baseline_nrms.evaluate_nrms(cfg, m, split="dev")

mq = copy.deepcopy(m)
with torch.no_grad():
    for p in mq.parameters():
        p.copy_(q8(p.data))
mq._vocab = m._vocab
r_int8 = baseline_nrms.evaluate_nrms(cfg, mq, split="dev")

out = {"epochs": EP,
       "fp32": {k: round(v, 4) for k, v in r_fp32.items()},
       "int8": {k: round(v, 4) for k, v in r_int8.items()}}
Path(cfg["paths"]["artifacts_dir"], "nrms_int8.json").write_text(json.dumps(out, indent=2))
print("NRMS:", json.dumps(out, indent=2))
