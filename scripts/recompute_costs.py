"""Recompute the cost columns of the June 2026 results matrix.

The ranking metrics of that run stay as measured (``results_summary.json``).
Size, MACs, RAM proxy and the energy proxy depend only on the architecture and
the precision of each layer, so they are recomputed here with the corrected
per-layer counter of ``src/footprint.py`` and written to
``paper/results/results_matrix.csv``, the file the paper's table and figure read.
The same costs are added for the ReActNet/Bi-Real binary encoder of
``scripts/run_binary.py``.

Run: python -m scripts.recompute_costs
"""
import json
from pathlib import Path

import pandas as pd
import torch

from src import footprint, quantize
from src.binary import BinaryByteCNNEncoder
from src.config import load_config
from src.nas.search import arch_cost, build_encoder, estimate_ram_kb

cfg = load_config()
results = Path(cfg["paths"]["results_dir"])
summary = json.loads((results / "results_summary.json").read_text(encoding="utf-8"))

rows = []
for r in summary["matrix"]:
    arch = {k: r[k] for k in ("channels", "depth", "out_dim")}
    cost = arch_cost(cfg, arch, r["precision"])
    rows.append({"arm": r["arm"], "precision": r["precision"], **arch,
                 "auc": r["auc"], "mrr": r["mrr"], "ndcg@10": r["ndcg@10"],
                 "size_kb": cost["size_kb"], "ram_kb": cost["ram_kb"],
                 "macs": cost["macs"], "macs_fp32": cost["macs_fp32"],
                 "energy_uj": cost["energy_uj_per_inf"]})
matrix = pd.DataFrame(rows)
matrix.to_csv(results / "results_matrix.csv", index=False)
print(matrix.to_string())

# Improved binary encoder (full k=3 binary convolutions, RSign/RPReLU, Bi-Real shortcut).
improved = json.loads((results / "binary_improved.json").read_text(encoding="utf-8"))
a = improved["arch"]
enc = BinaryByteCNNEncoder(**a)
ex = torch.zeros(1, cfg["data"]["max_title_bytes"], dtype=torch.long)
cost = footprint.summarize(enc, ex, cfg, precision="binary")
cost["ram_kb"] = estimate_ram_kb(a, cfg, "binary")
improved["cost"] = cost
(results / "binary_improved.json").write_text(json.dumps(improved, indent=2), encoding="utf-8")
print("improved binary:", cost)

ratios = {}
for arm, grp in matrix.groupby("arm"):
    e = grp.set_index("precision")["energy_uj"]
    ratios[arm] = round(float(e["fp32"] / e["int8"]), 2)
print("FP32 / INT8 energy ratio per arm:", ratios)
