"""Check that the PyTorch build of the chosen µNAS encoders (src/unas_encoder.py) computes the
same function as the fork's Keras build.

Input: artifacts/unas/port/<name>_keras.npz from unas/dump_keras.py (seeded random weights,
BatchNorm statistics not trivial, 64 validation rows). The Keras weights are copied into the
PyTorch build, both run in inference mode on the same byte ids, and the outputs are compared.
Also compares the cost counts: this repository counts multiply-accumulates of convolution and
dense layers (src/footprint.py); the fork's resource model also counts pooling operations.
Writes paper/results/unas/port_check.json. CPU only.

    python -m scripts.check_unas_port mind_h7 mind_f401
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src import footprint
from src.config import load_config
from src.unas_encoder import UnasEncoder, load_keras_weights

ROOT = Path(__file__).resolve().parents[1]
cfg = load_config()
out, ok = {}, True
for name in sys.argv[1:] or ["mind_h7", "mind_f401"]:
    z = np.load(ROOT / f"artifacts/unas/port/{name}_keras.npz")
    arch = json.loads(str(z["arch"]))
    model = UnasEncoder(arch).eval()
    load_keras_weights(model, z)
    ids = torch.from_numpy(z["ids"]).long()
    with torch.no_grad():
        emb = torch.cat([model.embed(ids), (ids != 0).float().unsqueeze(-1)], -1).numpy()
        y = model(ids).numpy()
    yk = z["y"]
    cos = (y * yk).sum(-1) / (np.linalg.norm(y, axis=-1) * np.linalg.norm(yk, axis=-1))
    hist = json.loads((ROOT / f"paper/results/unas/{name}_history.json").read_text(encoding="utf-8"))
    cand = next(c for c in hist["candidates"] if c["arch"] == arch)
    macs = footprint.count_macs(model, ids[:1])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    r = {"candidate": cand["index"], "rows": int(len(ids)),
         "input_max_abs_diff": float(np.abs(emb - z["x"]).max()),
         "output_max_abs_diff": float(np.abs(y - yk).max()),
         "output_max_abs_keras": float(np.abs(yk).max()),
         "cosine_min": float(cos.min()),
         "trainable_params": trainable, "byte_table_params": model.embed.weight.numel(),
         "macs_conv_dense": macs, "macs_fork": int(cand["macs"]),
         "fork_minus_ours": int(cand["macs"]) - macs}
    r["pass"] = r["input_max_abs_diff"] == 0.0 and r["output_max_abs_diff"] <= 1e-4 * max(1.0, r["output_max_abs_keras"])
    ok &= r["pass"]
    out[name] = r
    print(name, json.dumps(r))
(ROOT / "paper/results/unas/port_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
