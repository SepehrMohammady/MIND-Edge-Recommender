"""Choose the final architecture of a MIND µNAS search with several seeds (validation data only).

One training run per candidate scores a model with some seed noise, so the search's own
ranking is only a shortlist. For each search given:

  1. take the K in-budget candidates with the lowest validation error from
     paper/results/unas/<name>_history.json (unas/harvest_search.py);
  2. re-train each with seeds 42, 1 and 2 under the search recipe (configs/mind_config.py);
  3. choose the candidate with the best mean validation cosine ("best"), and the one with the
     fewest MACs whose mean lies within one standard error of the best ("smallest").

Test rows are never used for the choice. Writes paper/results/unas/<name>_selection.json.

Run in WSL inside the MIND fork copy:
    cd ~/uNAS_mind && ~/dmir_nas/bin/python /mnt/c/Projects/PhD/MIND/unas/select_by_seeds.py mind_h7 [K]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
import keras  # noqa: E402

from configs.cnn1d_gap import FaithfulGapCnn1DArchitecture  # noqa: E402
from configs.mind_config import training_config  # noqa: E402
from dataset.mind_dataset import MIND_Embedding_Dataset  # noqa: E402
from uNAS.model_trainer import ModelTrainer  # noqa: E402

RES = Path("/mnt/c/Projects/PhD/MIND/paper/results/unas")
SEEDS = (42, 1, 2)

name = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
hist = json.loads((RES / f"{name}_history.json").read_text())
ok = [c for c in hist["candidates"] if c["in_budget"] and not c["failed"]]
short = sorted(ok, key=lambda c: c["val_error"])[:K]
out = RES / f"{name}_selection.json"
done = json.loads(out.read_text()) if out.exists() else {"runs": []}
have = {(r["index"], r["seed"]) for r in done["runs"]}

data = MIND_Embedding_Dataset()
trainer = ModelTrainer(training_config(data))
for c in short:
    for seed in SEEDS:
        if (c["index"], seed) in have:
            continue
        keras.utils.set_random_seed(seed)
        t0 = time.time()
        model = FaithfulGapCnn1DArchitecture(c["arch"]).to_keras_model(data.input_shape, data.num_classes)
        r = trainer.train_and_eval(model)
        done["runs"].append({"index": c["index"], "seed": seed, "val_cos": 1 - float(r["val_error"]),
                             "test_cos": 1 - float(r["test_error"]), "minutes": round((time.time() - t0) / 60, 2)})
        out.write_text(json.dumps(done, indent=1))
        print(f"{name} candidate {c['index']} seed {seed}: val cos {done['runs'][-1]['val_cos']:.4f}", flush=True)
        keras.backend.clear_session()

rows = []
for c in short:
    v = [r["val_cos"] for r in done["runs"] if r["index"] == c["index"]]
    rows.append({"index": c["index"], "search_val_cos": c["val_cos"], "mean_val_cos": float(np.mean(v)),
                 "se": float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None, "n": len(v),
                 "macs": c["macs"], "model_size_bytes": c["model_size_bytes"], "peak_mem_bytes": c["peak_mem_bytes"],
                 "summary": c["summary"]})
best = max(rows, key=lambda r: r["mean_val_cos"])
within = [r for r in rows if r["mean_val_cos"] >= best["mean_val_cos"] - (best["se"] or 0.0)]
smallest = min(within, key=lambda r: r["macs"])
done.update({"shortlist": rows, "best": best["index"], "smallest_within_one_se": smallest["index"],
             "seeds": list(SEEDS), "k": K})
out.write_text(json.dumps(done, indent=1))
print(f"{name}: best candidate {best['index']} (mean val cos {best['mean_val_cos']:.4f}, {best['macs'] / 1e6:.2f} M MACs); "
      f"smallest within one SE {smallest['index']} ({smallest['mean_val_cos']:.4f}, {smallest['macs'] / 1e6:.2f} M MACs)")
