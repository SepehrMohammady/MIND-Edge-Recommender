"""Second choice among the best µNAS candidates of each search, after full distillation.

The first choice (unas/select_by_seeds.py) re-trained the eight best in-budget candidates of each
search with three seeds under the search recipe (96,000 rows, at most 15 epochs) and kept the best
mean validation cosine. After full training that recipe proved a weak predictor (LOGBOOK, 2026-10-08:
the H7 choice was 0.012 below 64-5-384 under the recipe and 0.067 below after full distillation).
Here the same eight candidates per search go through the full distillation of scripts/run_unas_full.py
(15 epochs on all 769k training rows, Matryoshka cosine loss) and are scored on the validation rows the
search used (artifacts/unas/data/mind_unas_val.npz: 4,000 rows in 15 languages from one half of the dev
articles; mean cosine to the teacher's anchor, 384-d). Click data and test rows are not used.

  stage 1  all eight candidates of each search, seed 42
  stage 2  the three best of each search, seeds 12 and 1
  choice   best mean over the three seeds, and the one with the fewest MACs within one standard error
           of it (the rule of unas/select_by_seeds.py)
  stage 3  click training (English clicks, seeds 42, 12, 1) of a choice that differs from the first one

Every training step is a subprocess of scripts/run_unas_full.py (one training process at a time; the
distillation records go to paper/results/unas_full.json). Writes paper/results/unas/rechoice.json.

    python -m scripts.rechoose_unas [--no-train]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.unas_encoder import UnasEncoder, model_arch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "paper/results"
CKPT = ROOT / "artifacts/runs/unas_full"
OUT = RES / "unas" / "rechoice.json"
SEARCHES = {"mind_h7": 144, "mind_f401": 134}            # search -> first choice
parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--no-train", action="store_true", help="score and choose only; no subprocess")
args = parser.parse_args()
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:7.1f}m] {msg}", flush=True)


def train(models, seeds, protocols=("distill",)):
    cmd = [sys.executable, "-u", "-m", "scripts.run_unas_full", "--models", *models, "--protocols", *protocols,
           "--seeds", *map(str, seeds)]
    log("run: " + " ".join(cmd[3:]))
    subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, check=True)


val = np.load(ROOT / "artifacts/unas/data/mind_unas_val.npz")
ids, target = torch.from_numpy(val["ids"].astype(np.int64)), torch.from_numpy(val["y"].astype(np.float32))


@torch.no_grad()
def val_cos(name, seed):
    path = CKPT / f"{name}_distilled_seed{seed}.pt"
    enc = UnasEncoder(model_arch(name)[1])
    enc.load_state_dict(torch.load(path, map_location="cpu"))
    enc.eval()
    out = torch.cat([enc(ids[i:i + 500]) for i in range(0, len(ids), 500)])
    return float(torch.nn.functional.cosine_similarity(out, target, dim=-1).mean())


def dev_cos(name, seed):
    rec = json.loads((RES / "unas_full.json").read_text(encoding="utf-8")).get(f"{name}/distill/seed{seed}")
    return rec["results"]["teacher_cosine_dev"]["384"] if rec else None


short = {s: [r for r in json.loads((RES / "unas" / f"{s}_selection.json").read_text(encoding="utf-8"))["shortlist"]]
         for s in SEARCHES}
names = {s: [f"{s}_c{r['index']}" for r in short[s]] for s in SEARCHES}
result = {"searches": {}, "seeds": [42, 12, 1], "criterion": "mean cosine to the teacher anchor on the "
          "search's validation rows (mind_unas_val.npz, 4,000 rows, 15 languages) after full distillation"}

if not args.no_train:
    train([n for s in SEARCHES for n in names[s]], [42])
stage1 = {s: sorted(({"name": n, "index": r["index"], "macs": r["macs"], "search_recipe_val_cos": r["mean_val_cos"],
                      "val_cos_seed42": val_cos(n, 42), "dev_cos_seed42": dev_cos(n, 42)}
                     for n, r in zip(names[s], short[s])), key=lambda r: -r["val_cos_seed42"]) for s in SEARCHES}
for s in SEARCHES:
    for r in stage1[s]:
        log(f"{s} c{r['index']:>3}: val cos {r['val_cos_seed42']:.4f}, dev cos {r['dev_cos_seed42']}, "
            f"{r['macs'] / 1e6:.2f} M MACs (search recipe {r['search_recipe_val_cos']:.4f})")
top = {s: [r["name"] for r in stage1[s][:3]] for s in SEARCHES}
if not args.no_train:
    train([n for s in SEARCHES for n in top[s]], [12, 1])

choice = {}
for s in SEARCHES:
    rows = []
    for r in stage1[s][:3]:
        v = [val_cos(r["name"], seed) for seed in (42, 12, 1)]
        rows.append({**r, "val_cos": v, "mean_val_cos": statistics.fmean(v),
                     "se": float(np.std(v, ddof=1) / np.sqrt(len(v))),
                     "dev_cos": [dev_cos(r["name"], seed) for seed in (42, 12, 1)]})
    best = max(rows, key=lambda r: r["mean_val_cos"])
    smallest = min((r for r in rows if r["mean_val_cos"] >= best["mean_val_cos"] - best["se"]), key=lambda r: r["macs"])
    choice[s] = {"best": best["index"], "smallest_within_one_se": smallest["index"],
                 "first_choice": SEARCHES[s], "changed": best["index"] != SEARCHES[s]}
    result["searches"][s] = {"stage1": stage1[s], "stage2": rows, **choice[s]}
    log(f"{s}: best c{best['index']} ({best['mean_val_cos']:.4f} ± {best['se']:.4f}), smallest within one SE "
        f"c{smallest['index']}; first choice c{SEARCHES[s]}")
OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")

new = [f"{s}_c{c['best']}" for s, c in choice.items() if c["changed"]]
if new and not args.no_train:
    train(new, [42, 12, 1], protocols=("distill_ft_en",))
log(f"RECHOICE DONE; newly chosen: {new or 'none'}")
