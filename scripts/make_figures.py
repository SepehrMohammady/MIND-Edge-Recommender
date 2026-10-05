"""Generate the paper figures from the frozen result files in ``paper/results``
(never from a run folder, so a smoke run cannot leak into a figure).

Run: python -m scripts.make_figures
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from src.config import load_config

cfg = load_config()
res = Path(cfg["paths"]["results_dir"])
figdir = Path("paper/figures")
figdir.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d2", "#ffffff"
ARMS = {"nas": ("NAS", "#2a78d6"),                       # colour = search arm
        "micro_nas": ("Micro-NAS", "#eb6834"),
        "binarized_micro_nas": (r"Bin. $\mu$NAS", "#1baf7a")}
PREC = {"fp32": ("FP32", "o"), "int8": ("INT8", "s"), "binary": ("1-bit", "^")}   # shape = precision

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6})

m = pd.read_csv(res / "results_matrix.csv")
nrms = json.loads((res / "results_summary.json").read_text(encoding="utf-8"))["baseline"]["auc"]
improved = json.loads((res / "binary_improved.json").read_text(encoding="utf-8"))

# Fig. 2: AUC against encoder size and against the energy proxy, per arm and precision.
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.75), sharey=True)
for a, (xcol, xlabel) in zip(ax, (("size_kb", "encoder size (KB)"),
                                  ("energy_uj", r"energy proxy ($\mu$J per title)"))):
    for arm, (_, color) in ARMS.items():
        for prec, (_, marker) in PREC.items():
            r = m[(m["arm"] == arm) & (m["precision"] == prec)].iloc[0]
            a.scatter(r[xcol], r["auc"], s=46, marker=marker, color=color,
                      edgecolor=SURFACE, linewidth=0.9, zorder=3)
    x_imp = improved["cost"]["size_kb"] if xcol == "size_kb" else improved["cost"]["energy_uj_per_inf"]
    a.scatter(x_imp, improved["result"]["auc"], s=62, marker="*", color=ARMS["micro_nas"][1],
              edgecolor=SURFACE, linewidth=0.7, zorder=3)
    a.axhline(nrms, ls="--", color=MUTED, lw=0.8, zorder=1)
    a.set_xscale("log")
    a.set_xticks([200, 400, 800, 1600] if xcol == "size_kb" else [3, 10, 30, 100])
    a.xaxis.set_major_formatter(ScalarFormatter())
    a.minorticks_off()
    a.set_xlabel(xlabel)
    a.grid(color=GRID, lw=0.5, zorder=0)
    a.set_axisbelow(True)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
ax[0].set_ylabel("AUC (MINDsmall dev)")
ax[0].text(ax[0].get_xlim()[1], nrms + 0.002, f"NRMS, 27 MB ({nrms:.3f})", ha="right", va="bottom",
           color=MUTED, fontsize=7)
handles = ([Line2D([], [], marker="o", ls="", color=c, markeredgecolor=SURFACE, markersize=6, label=n)
            for n, c in ARMS.values()]
           + [Line2D([], [], marker=mk, ls="", color=MUTED, markeredgecolor=SURFACE, markersize=6, label=n)
              for n, mk in PREC.values()]
           + [Line2D([], [], marker="*", ls="", color=ARMS["micro_nas"][1], markeredgecolor=SURFACE,
                     markersize=8, label="1-bit, ReActNet blocks")])
fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=7,
           handletextpad=0.2, columnspacing=1.0, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(figdir / "pareto.png", dpi=300, facecolor=SURFACE)
print("wrote", figdir / "pareto.png")

# Per-language AUC of the June full-run model (scratch, English clicks).
lang_file = res / "lang_matrix.csv"
if lang_file.exists():
    lang = pd.read_csv(lang_file).sort_values("auc", ascending=True)
    fig, a = plt.subplots(figsize=(4.2, 3.2))
    a.hlines(lang["lang"], 0.5, lang["auc"], color=GRID, lw=1.2, zorder=1)
    a.scatter(lang["auc"], lang["lang"], s=30, color=ARMS["nas"][1], edgecolor=SURFACE,
              linewidth=0.8, zorder=3)
    a.axvline(0.5, color=MUTED, ls="--", lw=0.8)
    a.set_xlabel("AUC (MINDsmall dev; 0.5 = chance)")
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(figdir / "multilingual.png", dpi=300, facecolor=SURFACE)
    print("wrote", figdir / "multilingual.png")
