"""Rerun of the June 2026 architecture x precision matrix (Table 1 of the paper)
under the October training loop, with three seeds.

The three architectures are the June search winners, one per arm:

  nas                  256-4-384  FP32 search, no constraints
  micro_nas             64-5-384  INT8 search under the June constraints
  binarized_micro_nas   96-2-384  binary search under the June constraints

The protocol is the one of the June matrix (scripts/run_full.py): each architecture
is trained from a random initialisation on English clicks for 8 epochs (FP32); the
INT8 and binary variants are the trained FP32 model with simulated weight
quantization of the inner layers (src/quantize.py), fine-tuned for 2 more epochs;
evaluation on the full MINDsmall dev set. Changes against June: seeded runs,
negatives redrawn at every step, the corrected cost counter, checkpoints kept.
The architectures are not searched again; the µNAS search in the lab fork
(unas/) replaced the June search.

Every (arm, precision, seed) cell is written when it finishes to
``paper/results/matrix_runs.json`` and ``paper/results/experiments.jsonl``;
finished cells are skipped on restart. Checkpoints go to ``artifacts/runs/matrix``,
so the quantized cells of an interrupted arm start from the trained FP32 model.
Seed means and population standard deviations (as in Table 2) are written to
``paper/results/matrix_summary.json`` at the end.

Only one training process runs on this laptop at a time. A file
``artifacts/runs/matrix/PAUSE`` makes the run stop before its next cell, so that
another job can use the GPU; delete the file and start the script again to continue.

Run:  python -m scripts.run_matrix [--seeds 42 1 2] [--arms micro_nas ...] [--smoke]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from src import footprint, quantize, recommender, runlog
from src.config import load_config, use_run_dir
from src.gpu import cap_gpu_memory
from src.nas.search import build_encoder, estimate_ram_kb
from src.seed import seed_everything

# small arms first: a PAUSE then waits at most one small cell
ARCHS = {
    "micro_nas": {"channels": 64, "depth": 5, "out_dim": 384},
    "binarized_micro_nas": {"channels": 96, "depth": 2, "out_dim": 384},
    "nas": {"channels": 256, "depth": 4, "out_dim": 384},
}
PRECISIONS = ("fp32", "int8", "binary")
EPOCHS, QAT_EPOCHS = 8, 2
TRAIN_IMPRESSIONS = EVAL_IMPRESSIONS = None          # None = everything

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
parser.add_argument("--arms", nargs="+", default=list(ARCHS), choices=list(ARCHS))
parser.add_argument("--smoke", action="store_true",
                    help="tiny settings, results kept out of paper/results")
args = parser.parse_args()

cfg = load_config()
use_run_dir(cfg, "matrix_smoke" if args.smoke else "matrix")
cap_gpu_memory()                    # src/gpu.py: keeps the allocator's cache inside the card
CKPT = Path(cfg["paths"]["artifacts_dir"])
if args.smoke:
    EPOCHS, QAT_EPOCHS = 1, 1
    TRAIN_IMPRESSIONS, EVAL_IMPRESSIONS = 2000, 1000
    cfg["paths"]["results_dir"] = str(CKPT / "results")
    Path(cfg["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)
RESULTS = Path(cfg["paths"]["results_dir"])
OUT, SUMMARY, PAUSE = RESULTS / "matrix_runs.json", RESULTS / "matrix_summary.json", CKPT / "PAUSE"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{(time.time() - T0) / 60:7.1f}m] {msg}", flush=True)


def load_done() -> dict:
    return json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}


def save_cell(key: str, settings: dict, results: dict, started: float) -> None:
    record = runlog.append(cfg, f"matrix/{key}", settings, results, started)
    done = load_done()
    done[key] = {k: record[k] for k in ("seed", "minutes", "settings", "results")}
    OUT.write_text(json.dumps(done, indent=1), encoding="utf-8")
    log(f"{key}: AUC {results['auc']:.4f}, {results['size_kb']} KB, {results['macs'] / 1e6:.2f} M MACs, "
        f"{record['minutes']} min")


def check_pause() -> None:
    if PAUSE.exists():
        log(f"{PAUSE} found: stopping before the next cell")
        sys.exit(0)


def run_cell(arm: str, prec: str, seed: int, model, started: float) -> None:
    """Evaluate a trained model of one cell and record it."""
    arch = ARCHS[arm]
    res = recommender.evaluate(cfg, model, split="dev", max_impressions=EVAL_IMPRESSIONS)
    ex = torch.zeros(1, cfg["data"]["max_title_bytes"], dtype=torch.long)
    foot = footprint.summarize(model.news_encoder, ex, cfg, precision=prec)
    results = {**{k: round(v, 4) for k, v in res.items() if k != "n_impressions"},
               "n_impressions": res.get("n_impressions"),
               **{k: foot[k] for k in ("params", "macs", "macs_fp32", "macs_quantized", "size_kb")},
               "energy_uj": foot["energy_uj_per_inf"], "ram_kb": estimate_ram_kb(arch, cfg, prec),
               "fp_param_fraction": round(quantize.quant_fp_fraction(model.news_encoder), 4)}
    settings = {"arm": arm, "precision": prec, "arch": arch, "init": "random", "train_langs": ["en"],
                "epochs": EPOCHS, "qat_epochs": 0 if prec == "fp32" else QAT_EPOCHS,
                "lr": cfg["train"]["lr"], "batch_size": cfg["train"]["batch_size"],
                "train_impressions": TRAIN_IMPRESSIONS, "eval_impressions": EVAL_IMPRESSIONS}
    save_cell(f"{arm}/{prec}/seed{seed}", settings, results, started)


def run_arm(arm: str, seed: int) -> None:
    done = load_done()
    todo = [p for p in PRECISIONS if f"{arm}/{p}/seed{seed}" not in done]
    if not todo:
        return
    cfg["seed"] = seed
    path = CKPT / f"{arm}_fp32_seed{seed}.pt"
    started = time.time()
    if path.exists():
        model = recommender.NewsRecommender(build_encoder(cfg, ARCHS[arm]))
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        log(f"{arm} seed {seed}: FP32 model loaded from {path.name}")
    else:
        check_pause()
        seed_everything(seed)
        model = recommender.train_recommender(cfg, news_encoder=build_encoder(cfg, ARCHS[arm]), epochs=EPOCHS,
                                              max_train_impressions=TRAIN_IMPRESSIONS)
        torch.save(model.state_dict(), path)
    if f"{arm}/fp32/seed{seed}" not in load_done():
        run_cell(arm, "fp32", seed, model, started)
    for prec in ("int8", "binary"):
        if prec not in todo:
            continue
        check_pause()
        started = time.time()
        seed_everything(seed)
        qm = quantize.convert_to_quant(model, prec)
        qm = recommender.train_recommender(cfg, model=qm, epochs=QAT_EPOCHS,
                                           max_train_impressions=TRAIN_IMPRESSIONS)
        torch.save(qm.state_dict(), CKPT / f"{arm}_{prec}_seed{seed}.pt")
        run_cell(arm, prec, seed, qm, started)
        del qm
    del model
    torch.cuda.empty_cache()


def summarize() -> None:
    done = load_done()
    rows = []
    for arm in ARCHS:
        for prec in PRECISIONS:
            cells = [v for k, v in done.items() if k.startswith(f"{arm}/{prec}/seed")]
            if not cells:
                continue
            r = {"arm": arm, "precision": prec, **ARCHS[arm], "n": len(cells),
                 "seeds": sorted(c["seed"] for c in cells),
                 "minutes": round(sum(c["minutes"] for c in cells), 1)}
            for m in ("auc", "mrr", "ndcg@10"):
                vals = [c["results"][m] for c in cells]
                r[m] = round(statistics.fmean(vals), 4)
                r[f"{m}_sd"] = round(statistics.pstdev(vals), 4) if len(vals) > 1 else None   # population SD, as tab_p1
            for m in ("size_kb", "macs", "macs_fp32", "energy_uj", "ram_kb", "params"):
                r[m] = cells[0]["results"][m]
            rows.append(r)
    SUMMARY.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for r in rows:
        log(f"{r['arm']:>20s} {r['precision']:>6s}  n={r['n']}  AUC {r['auc']:.4f}"
            + (f" +- {r['auc_sd']:.4f}" if r["auc_sd"] is not None else ""))


if __name__ == "__main__":
    log(f"arms={args.arms} seeds={args.seeds} smoke={args.smoke} device="
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    for arm in [a for a in ARCHS if a in args.arms]:
        for seed in args.seeds:
            run_arm(arm, seed)
    summarize()
    log("MATRIX DONE")
