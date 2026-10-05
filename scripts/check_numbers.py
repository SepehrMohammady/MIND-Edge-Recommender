"""Recompute the numbers quoted in paper/paper.tex from the frozen result files
and check that each formatted value appears in the manuscript.

    python -m scripts.check_numbers

Exit status 1 lists every value that is missing or differs. Numbers taken from
the literature are not checked here; they carry a citation in the text.
"""
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd
import torch

from src import quantize
from src.config import load_config
from src.nas.search import build_encoder

cfg = load_config()
RES = Path(cfg["paths"]["results_dir"])
ROOT = RES.parents[1]
TEX = " ".join((ROOT / "paper" / "paper.tex").read_text(encoding="utf-8").split())
missing: list[str] = []
notes: list[str] = []


def f(x: float, nd: int) -> str:
    """Round half up on the stored decimal value (0.3595 -> 0.360), the rule the
    tables use; Python's float formatting would print 0.359."""
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def expect(text: str, why: str) -> None:
    if text not in TEX:
        missing.append(f"{why}: '{text}' not in paper.tex")


def load(name: str):
    return json.loads((RES / name).read_text(encoding="utf-8"))


matrix = pd.read_csv(RES / "results_matrix.csv")
summary = load("results_summary.json")
cell = {(r.arm, r.precision): r for r in matrix.itertuples(index=False)}

# ---- Table 1: every row, bold and dagger marks ignored
ARM = {"nas": "NAS", "micro_nas": "Micro-NAS", "binarized_micro_nas": r"Bin.\,$\mu$NAS"}
PREC = {"fp32": "FP32", "int8": "INT8", "binary": "Binary"}
plain = re.sub(r"\\textbf\{([^}]*)\}", r"\1", TEX).replace(r"$^\dagger$", "")
for r in matrix.itertuples(index=False):
    energy = f"{f(r.energy_uj, 1)}" if r.energy_uj >= 100 else f"{f(r.energy_uj, 2)}"
    row = (f"{ARM[r.arm]} & {PREC[r.precision]} & {f(r.auc, 3)} & {f(r.mrr, 3)} & "
           f"{f(getattr(r, '_7'), 3)} & {f(r.size_kb, 1)} & {energy}")
    if row not in plain:
        missing.append(f"Table 1 row {r.arm}/{r.precision}: expected '{row}'")

# ---- headline numbers
mi8, ni8 = cell[("micro_nas", "int8")], cell[("nas", "int8")]
nrms = summary["baseline"]["auc"]
expect(f"scores {f(mi8.auc, 3)} AUC at an estimated {f(mi8.size_kb, 0)}\\,KB", "abstract, constrained INT8")
expect(f"{f(ni8.auc, 3)} at {f(ni8.size_kb, 0)}\\,KB", "abstract, unconstrained INT8")
expect(f"word table scores {f(nrms, 3)}", "abstract, reduced NRMS")
expect(f"reaches {f(nrms, 3)};", "results, reduced NRMS")

# ---- search space, run time, costs
space = cfg["nas"]["search_space"]
n_arch = len(space["channels"]) * len(space["depth"]) * len(space["out_dim"])
expect(f"holds {n_arch} architectures", "search-space size")
expect(f"takes {f(summary['runtime_min'] / 60, 1)} hours", "run time of the June study")
archs = {a: summary["best_arch"][a] for a in ARM}
expect(", ".join(f"{a['channels']}-{a['depth']}-{a['out_dim']}" for a in list(archs.values())[:2])
       + f" and {archs['binarized_micro_nas']['channels']}-{archs['binarized_micro_nas']['depth']}-"
         f"{archs['binarized_micro_nas']['out_dim']}", "searched architectures")
macs = [cell[(a, "fp32")].macs / 1e6 for a in ARM]
expect(f"{f(macs[0], 1)}, {f(macs[1], 2)} and {f(macs[2], 2)} million", "MACs per title")
expect(f"({f(macs[1], 2)}\\,M and {f(macs[2], 2)}\\,M)", "constrained winners over the 2 M bound")
if not all(m > 2.0 for m in macs[1:]):
    missing.append("text says both constrained winners exceed 2 M MACs; they do not")
shares = []
for a in ARM:
    enc = quantize.convert_to_quant(build_encoder(cfg, archs[a]), "int8")
    shares.append(round(100 * (1 - quantize.quant_fp_fraction(enc))))
expect(f"cover {shares[0]}\\%, {shares[1]}\\% and {shares[2]}\\%", "quantised share of parameters")
e_ratio = [cell[(a, "fp32")].energy_uj / cell[(a, "int8")].energy_uj for a in ARM]
s_ratio = [cell[(a, "fp32")].size_kb / cell[(a, "int8")].size_kb for a in ARM]
expect(f"falls {f(min(e_ratio), 1)} to {f(max(e_ratio), 1)} times", "FP32/INT8 energy ratio")
expect(f"the size {f(min(s_ratio), 1)} to {f(max(s_ratio), 0)} times", "FP32/INT8 size ratio")
d_auc = max(abs(cell[(a, "int8")].auc - cell[(a, "fp32")].auc) for a in ARM)
expect(f"changes AUC by at most {f(d_auc, 3)}", "FP32 to INT8 AUC change")
drop = [100 * (cell[(a, "int8")].auc - cell[(a, "binary")].auc) for a in ARM]
if not (5 <= min(drop) and max(drop) <= 9):
    missing.append(f"'loses five to nine AUC points': measured {f(min(drop), 1)} to {f(max(drop), 1)}")
expect(f"is {f(mi8.ram_kb, 0)}\\,KB for the constrained model and {f(ni8.ram_kb, 0)}\\,KB", "RAM estimate")
expect(f"{f(mi8.auc, 3)}, {f(mi8.size_kb, 0)}\\,KB, {f(mi8.energy_uj, 1)}\\uJ", "Fig. 2 sentence")
expect(f"({f(ni8.auc, 3)}, {f(ni8.size_kb, 0)}\\,KB)", "Fig. 2 sentence, unconstrained")

# ---- laptop latency
lat = load("latency.json")
cpu = [r["cpu_onnx_ms"] for r in lat]
expect(f"{f(min(cpu), 2)}--{f(max(cpu), 2)}\\,ms on CPU", "CPU latency range")
if max(r["gpu_ms"] for r in lat) >= 0.7:
    missing.append("'under 0.7 ms on GPU' does not hold")

# ---- heuristics (P1)
heur = load("p1_heuristics.json")["heuristics"]["results"]
expect(f"popularity gives {f(heur['popularity']['all']['auc'], 3)}", "popularity AUC")
expect(f"category labels {f(heur['category']['all']['auc'], 3)}", "category histogram AUC")
expect(f"network {f(heur['category']['all']['auc'], 3)}", "abstract, category histogram")
expect(f"or {f(heur['subcategory']['all']['auc'], 3)} with the finer", "subcategory histogram AUC")
cold = heur["category"]["0"]["n_impressions"] / heur["category"]["all"]["n_impressions"]
expect(f"For the {f(100 * cold, 0)}\\% of impressions", "share of impressions without history")
expect(f"popularity gives {f(heur['popularity']['0']['auc'], 3)} there", "popularity, no history")
expect(f"category distribution {f(heur['category']['0']['auc'], 3)}", "category prior, no history")
if not mi8.auc - heur["category"]["all"]["auc"] < 0.02:
    missing.append("'within 0.02 of a label histogram' does not hold")

# ---- languages (June matrix model)
lang = pd.read_csv(RES / "lang_matrix.csv").set_index("lang")["auc"]
x = lang.drop("en")
expect(f"between {f(x.min(), 3)} (Georgian) and {f(x.max(), 3)} (Haitian Creole), {f(x.mean(), 3)} on average",
       "per-language range")
expect(f"against {f(cell[('micro_nas', 'fp32')].auc, 3)} in English", "English AUC of that model")
expect(f"({f(x.mean(), 3)} on average)", "abstract, mean over translations")
if x.idxmin() != "kat" or x.idxmax() != "hat":
    missing.append("lowest/highest language is not Georgian/Haitian Creole")

# ---- ablation, seeds, binary
abl = load("ablation.json")
expect(f"from {f(abl['scratch']['auc'], 3)} to {f(abl['distilled_init']['auc'], 3)}", "distillation ablation")
expect(f"adds {f(100 * (abl['distilled_init']['auc'] - abl['scratch']['auc']), 1)} AUC points",
       "abstract, distillation gain")
log = (ROOT / "docs" / "run_log.md").read_text(encoding="utf-8")
m = re.search(r"seeds 42/1/2 AUC: \*\*([\d.]+), ([\d.]+), ([\d.]+)\*\*", log)
if m:
    seeds = [float(v) for v in m.groups()]
    expect(f"score {f(seeds[0], 3)}, {f(seeds[1], 3)} and {f(seeds[2], 3)}", "seeded reruns (docs/run_log.md)")
    if not max(seeds) - mi8.auc <= 0.0205:
        missing.append("'up to 0.02' between seeded reruns and the matrix run does not hold")
else:
    missing.append("seeded rerun values not found in docs/run_log.md")
imp = load("binary_improved.json")
naive = cell[("micro_nas", "binary")].auc
expect(f"near chance ({f(naive, 3)})", "naive binary")
expect(f"gives {f(imp['result']['auc'], 3)} at an estimated {f(imp['cost']['size_kb'], 0)}\\,KB", "improved binary")
expect(f"brings it to {f(imp['result']['auc'], 3)} at {f(imp['cost']['size_kb'], 0)}\\,KB".replace("brings", "bring"),
       "abstract, improved binary")
expect(f"{f(nrms - imp['result']['auc'], 3)} below the reduced", "improved binary vs NRMS")
expect(f"The {f(imp['result']['auc'] - naive, 2)} gain", "binary gain")
expect(f"standard deviation of {f(load('reviews2.json')['binary_multiseed']['std'], 3)}", "binary seed spread")

# ---- reduced NRMS size: checked once the P1 record exists
p1_nrms = RES / "p1_nrms.json"
reduced = [v for k, v in (load("p1_nrms.json") if p1_nrms.exists() else {}).items()
           if k.startswith("nrms_reduced/")]
if reduced:
    st = reduced[0]["settings"]
    expect(f"{f(100 * st['embedding_params'] / st['params'], 0)}\\% of its {f(st['params'] / 1e6, 2)} million",
           "reduced NRMS parameters")
    expect(f"({f(st['size_mb_fp32'], 0)}\\,MB", "reduced NRMS size")
else:
    notes.append("reduced-NRMS parameter count (92%, 7.08 M, 27 MB) has no result file yet; "
                 "it is checked once p1_nrms.json holds an nrms_reduced run")

for line in notes:
    print("note:", line)
if missing:
    print(f"{len(missing)} mismatch(es):")
    for line in missing:
        print("  -", line)
    sys.exit(1)
print("paper.tex agrees with paper/results")
