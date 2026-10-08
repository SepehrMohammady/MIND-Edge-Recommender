"""Collect a finished (or running) MIND µNAS search into result files.

Reads the search state (every evaluated candidate: validation error, peak activation memory,
INT8 weight size, MACs, architecture) and writes, for each search name given:

  paper/results/unas/<name>_history.json   all candidates in evaluation order
  paper/results/unas/<name>_front.json     candidates within the board budgets that no other
                                           in-budget candidate beats on validation error and
                                           all three costs at once (Pareto set)

Test error is copied from the state for the record and never used for any choice. A candidate
whose validation error is not a finite number (training stopped on a NaN loss) counts as failed:
NaN breaks sorting and dominance tests (first harvest of mind_h7, 2026-10-07: one NaN candidate
moved the best in-budget candidate out of the selection shortlist).

Run in WSL inside the MIND fork copy:
    cd ~/uNAS_mind && ~/dmir_nas/bin/python /mnt/c/Projects/PhD/MIND/unas/harvest_search.py mind_h7 [mind_f401]
"""
import json
import math
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from configs.mind_config import BUDGETS  # noqa: E402

OUT = Path("/mnt/c/Projects/PhD/MIND/paper/results/unas")
OUT.mkdir(parents=True, exist_ok=True)


def summary(arch: dict) -> dict:
    layers = [l for b in arch["conv_blocks"] for l in b["layers"]]
    return {"conv_blocks": len(arch["conv_blocks"]), "conv_layers": len(layers),
            "branches": sum(1 for b in arch["conv_blocks"] if b.get("is_branch")),
            "strided": sum(1 for l in layers if l.get("1x_stride")),
            "prepool": sum(1 for l in layers if l.get("has_prepool")),
            "max_filters": max([l.get("filters", 0) for l in layers] or [0]),
            "head": (arch.get("pooling") or {}).get("type", "flatten"),
            "dense": [d["units"] for d in arch.get("dense_blocks", [])]}


for name in sys.argv[1:]:
    board = name.split("_")[1]
    b = BUDGETS[board]
    state = Path("artifacts") / name / f"{name}_agingevosearch_state.pickle"
    history = pickle.load(open(state, "rb"))
    rows = []
    for i, e in enumerate(history):
        pm, ms, macs = (float(x) for x in e.resource_features)
        rows.append({"index": i, "val_error": float(e.val_error), "val_cos": 1 - float(e.val_error),
                     "test_error": float(e.test_error), "peak_mem_bytes": pm, "model_size_bytes": ms, "macs": macs,
                     "in_budget": pm <= b["peak_mem"] and ms <= b["model_size"] and macs <= b["macs"],
                     "failed": not math.isfinite(float(e.val_error)) or float(e.val_error) >= 1.0,
                     "summary": summary(e.point.arch.architecture),
                     "arch": e.point.arch.architecture})
    ok = [r for r in rows if r["in_budget"] and not r["failed"]]

    def dominated(r):
        keys = ("val_error", "peak_mem_bytes", "model_size_bytes", "macs")
        return any(all(o[k] <= r[k] for k in keys) and any(o[k] < r[k] for k in keys) for o in ok if o is not r)

    front = sorted((r for r in ok if not dominated(r)), key=lambda r: r["val_error"])
    (OUT / f"{name}_history.json").write_text(json.dumps({"budget": b, "candidates": rows}, indent=1))
    (OUT / f"{name}_front.json").write_text(json.dumps({"budget": b, "front": front}, indent=1))
    best = min(ok, key=lambda r: r["val_error"]) if ok else None
    print(f"{name}: {len(rows)} candidates, {sum(r['failed'] for r in rows)} failed, {len(ok)} in budget, "
          f"front {len(front)}; best in budget val_cos "
          f"{best['val_cos']:.4f} at {best['macs'] / 1e6:.2f} M MACs, {best['model_size_bytes'] / 1024:.1f} KiB"
          if best else f"{name}: {len(rows)} candidates, none in budget")
