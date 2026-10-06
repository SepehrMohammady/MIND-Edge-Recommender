"""Aggregate the P1 stage files over seeds: mean and standard deviation of the
English AUC (with history buckets) and of the mean AUC over the 14 xMIND
languages, per condition. Writes paper/results/p1_summary.json and prints a
table. The paper quotes these aggregates; scripts/check_numbers.py reads them.

    python -m scripts.summarize_p1
"""
import json
import statistics
from pathlib import Path

from src.config import load_config

cfg = load_config()
RES = Path(cfg["paths"]["results_dir"])
XLANGS = ["zho", "fin", "grn", "hat", "ind", "jpn", "kat", "ron", "som", "swh", "tam", "tha", "tur", "vie"]


def stat(values):
    values = [float(v) for v in values]
    return {"mean": round(statistics.fmean(values), 4),
            "std": round(statistics.pstdev(values), 4) if len(values) > 1 else None,
            "n": len(values), "values": values}


summary = {}
for stage in ("teacher", "nrms", "student"):
    path = RES / f"p1_{stage}.json"
    if not path.exists():
        continue
    runs = json.loads(path.read_text(encoding="utf-8"))
    by_condition = {}
    for key, rec in runs.items():
        condition, seed = key.split("/seed")
        by_condition.setdefault(condition, []).append((int(seed), rec))
    for condition, items in by_condition.items():
        items.sort()
        en = [r["results"]["en"]["all"] for _, r in items]
        entry = {"seeds": [s for s, _ in items],
                 "minutes": stat([r["minutes"] for _, r in items]),
                 "en_auc": stat([e["auc"] for e in en]),
                 "en_mrr": stat([e["mrr"] for e in en]),
                 "en_ndcg10": stat([e["ndcg@10"] for e in en]),
                 "buckets": {b: stat([r["results"]["en"][b]["auc"] for _, r in items])
                             for b in ("0", "1-5", "6-20", ">20")}}
        if "mean_xlang_auc" in items[0][1]["results"]:
            entry["xlang_auc"] = stat([r["results"]["mean_xlang_auc"] for _, r in items])
            entry["per_language"] = {lang: stat([r["results"][lang]["auc"] for _, r in items])
                                     for lang in XLANGS}
        settings = items[0][1]["settings"]
        entry["settings"] = {k: v for k, v in settings.items() if k != "encoder_cost"}
        if "encoder_cost" in settings:
            entry["encoder_cost"] = settings["encoder_cost"]
        summary[f"{stage}/{condition}"] = entry

(RES / "p1_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(f"{'condition':34s} {'seeds':>8s} {'EN AUC':>16s} {'14-lang AUC':>16s} {'min/run':>8s}")
for name, e in summary.items():
    def fmt(s):
        return f"{s['mean']:.4f}±{s['std']:.4f}" if s and s["std"] is not None else (f"{s['mean']:.4f}" if s else "-")
    print(f"{name:34s} {str(e['seeds']):>8s} {fmt(e['en_auc']):>16s} {fmt(e.get('xlang_auc')):>16s} {e['minutes']['mean']:8.1f}")
