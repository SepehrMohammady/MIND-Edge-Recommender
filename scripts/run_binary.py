"""One-bit byte-CNN with ReActNet / Bi-Real blocks (src/binary.py), seeded runs under the October loop.

The June run (paper/results/binary_improved.json: one unseeded run, June training loop) scored
0.572 AUC, against 0.521 for the naive binary cell of the June matrix. The matrix rerun
(scripts/run_matrix.py) gives the naive binary cells three seeds; this script gives the ReActNet
encoder the same: seeds 42, 1 and 2, distillation of the binary 64-5-384 encoder to the teacher
anchors (15 epochs, all training rows in 15 languages), then English clicks for 8 epochs, full
dev set. The programme is the June one; only the loop (negatives redrawn at every step) and the
seeding differ.

Results: paper/results/binary_reactnet.json and paper/results/experiments.jsonl, one record per
seed; checkpoints in artifacts/runs/binary. Finished seeds are skipped on restart.

Run:  python -m scripts.run_binary [--seeds 42 1 2] [--smoke]
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from src import footprint, recommender, runlog, student
from src.binary import BinaryByteCNNEncoder
from src.config import load_config, use_run_dir
from src.gpu import cap_gpu_memory
from src.seed import seed_everything

ARCH = dict(byte_embed_dim=64, channels=64, depth=5, out_dim=384)   # the Micro-NAS shape
DISTILL_EP, TRAIN_EP = 15, 8
TRAIN_IMPRESSIONS = EVAL_IMPRESSIONS = DISTILL_TITLES = None

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
parser.add_argument("--smoke", action="store_true", help="tiny settings, results kept out of paper/results")
args = parser.parse_args()

cfg = load_config()
use_run_dir(cfg, "binary_smoke" if args.smoke else "binary")
cap_gpu_memory()
CKPT = Path(cfg["paths"]["artifacts_dir"])
if args.smoke:
    DISTILL_EP, TRAIN_EP = 1, 1
    TRAIN_IMPRESSIONS, EVAL_IMPRESSIONS, DISTILL_TITLES = 2000, 1000, 20000
    cfg["paths"]["results_dir"] = str(CKPT / "results")
    Path(cfg["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)
OUT = Path(cfg["paths"]["results_dir"]) / "binary_reactnet.json"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{(time.time() - T0) / 60:7.1f}m] {msg}", flush=True)


done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"runs": {}}
ex = torch.zeros(1, cfg["data"]["max_title_bytes"], dtype=torch.long)
cost = footprint.summarize(BinaryByteCNNEncoder(**ARCH), ex, cfg, precision="binary")
for seed in args.seeds:
    key = f"seed{seed}"
    if key in done["runs"]:
        continue
    started = time.time()
    cfg["seed"] = seed
    seed_everything(seed)
    enc = student.distill_encoder(cfg, BinaryByteCNNEncoder(**ARCH), epochs=DISTILL_EP, max_titles=DISTILL_TITLES,
                                  tag=f"bin-distill seed {seed}")
    teacher_cos = student.teacher_similarity(cfg, enc)
    seed_everything(seed)
    model = recommender.train_recommender(cfg, model=recommender.NewsRecommender(enc), epochs=TRAIN_EP,
                                          max_train_impressions=TRAIN_IMPRESSIONS)
    torch.save(model.state_dict(), CKPT / f"reactnet_64-5-384_seed{seed}.pt")
    res = recommender.evaluate(cfg, model, split="dev", max_impressions=EVAL_IMPRESSIONS)
    results = {**{k: round(v, 4) for k, v in res.items() if k != "n_impressions"},
               "n_impressions": res.get("n_impressions"), "teacher_cosine_dev": teacher_cos}
    settings = {"arch": ARCH, "blocks": "ReActNet RSign/RPReLU, Bi-Real shortcuts, binary weights and activations",
                "init": "distilled", "distill_epochs": DISTILL_EP, "epochs": TRAIN_EP, "train_langs": ["en"],
                "encoder_cost": cost}
    record = runlog.append(cfg, f"binary/reactnet_64-5-384/{key}", settings, results, started)
    done["runs"][key] = {k: record[k] for k in ("seed", "minutes", "settings", "results")}
    vals = [r["results"]["auc"] for r in done["runs"].values()]
    done["summary"] = {"n": len(vals), "auc": round(statistics.fmean(vals), 4),
                       "auc_sd": round(statistics.pstdev(vals), 4) if len(vals) > 1 else None,
                       "seeds": sorted(r["seed"] for r in done["runs"].values()), "cost": cost}
    OUT.write_text(json.dumps(done, indent=1), encoding="utf-8")
    log(f"{key}: AUC {results['auc']:.4f}, teacher cosine {teacher_cos['384']:.4f}, {record['minutes']} min")
    del model, enc
    torch.cuda.empty_cache()
log("BINARY DONE")
