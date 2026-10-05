"""Summarise the run records of paper/results/experiments.jsonl: what ran, when
it finished, how long it took and its headline result, plus total compute time.
With a GPU sample file (logs/gpu_monitor.csv from nvidia-smi) it also reports
mean GPU utilisation while each run was active.

    python -m scripts.log_report                      # every record
    python -m scripts.log_report p1/student           # records whose name starts so
    python -m scripts.log_report p1/ --logbook "P1 batch finished"   # also append to LOGBOOK.md
"""
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

from src import runlog
from src.config import load_config

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("prefix", nargs="?", default="")
parser.add_argument("--logbook", metavar="TITLE", help="append the table to LOGBOOK.md under this title")
parser.add_argument("--gpu", default="logs/gpu_monitor.csv", help="nvidia-smi sample file")
args = parser.parse_args()

cfg = load_config()
records = [r for r in runlog.read(cfg) if r["run"].startswith(args.prefix)]

samples = []                                  # (local time, utilisation %)
gpu_file = Path(args.gpu)
if gpu_file.exists():
    with open(gpu_file, encoding="utf-8", errors="ignore") as fh:
        for row in csv.reader(fh):
            try:
                samples.append((datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f"),
                                float(row[1].strip().rstrip(" %"))))
            except (ValueError, IndexError):
                continue


def gpu_mean(record: dict) -> str:
    end = datetime.fromisoformat(record["finished_utc"]).astimezone().replace(tzinfo=None)
    start = end - timedelta(minutes=record["minutes"])
    vals = [u for t, u in samples if start <= t <= end]
    return f"{sum(vals) / len(vals):.0f}%" if vals else "-"


lines = ["| run | finished (local) | minutes | EN AUC | mean AUC, 14 languages | GPU util. | commit |",
         "|---|---|--:|--:|--:|--:|---|"]
for r in records:
    res = r["results"]
    en = res.get("en", {}).get("all", {}).get("auc") if isinstance(res.get("en"), dict) else None
    local = datetime.fromisoformat(r["finished_utc"]).astimezone().strftime("%Y-%m-%d %H:%M")
    commit = r["env"]["commit"] + ("*" if r["env"]["dirty"] else "")
    lines.append(f"| {r['run']} | {local} | {r['minutes']:.1f} | "
                 f"{en if en is not None else '-'} | {res.get('mean_xlang_auc', '-')} | "
                 f"{gpu_mean(r)} | {commit} |")
total = sum(r["minutes"] for r in records)
lines.append(f"\n{len(records)} runs, {total:.0f} min of compute ({total / 60:.1f} h). "
             "A commit marked * had uncommitted code changes when the process started.")
table = "\n".join(lines)
print(table)
if args.logbook:
    runlog.logbook(args.logbook, table)
    print("\nappended to LOGBOOK.md")
