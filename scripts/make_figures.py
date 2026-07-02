"""Generate paper figures from the full-run artifacts.
Run: python -m scripts.make_figures
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config

cfg = load_config()
art = Path(cfg["paths"]["artifacts_dir"])
figdir = Path("paper/figures"); figdir.mkdir(parents=True, exist_ok=True)

m = pd.read_csv(art / "results_matrix.csv")

# Fig 1: AUC vs size and AUC vs energy, per arm.
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
for arm in m["arm"].unique():
    s = m[m["arm"] == arm]
    ax[0].scatter(s["size_kb"], s["auc"], s=60, label=arm)
    ax[1].scatter(s["energy_uj"], s["auc"], s=60, label=arm)
    for _, r in s.iterrows():
        ax[0].annotate(r["precision"], (r["size_kb"], r["auc"]), fontsize=7)
for a in ax:                       # NRMS baseline as a dashed AUC reference
    a.axhline(0.607, ls="--", color="grey", lw=1.2)
ax[0].text(ax[0].get_xlim()[1], 0.607, "NRMS 0.607 ", va="bottom", ha="right",
           fontsize=8, color="grey")
ax[0].set(xlabel="encoder size (KB)", ylabel="AUC", title="Accuracy vs flash footprint")
ax[1].set(xlabel="energy (uJ / inference)", ylabel="AUC", title="Accuracy vs energy")
ax[1].set_xscale("log")
for a in ax:
    a.legend(fontsize=8); a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figdir / "pareto.png", dpi=200, bbox_inches="tight")
print("wrote", figdir / "pareto.png")

# Fig 2: per-language AUC bar.
if (art / "lang_matrix.csv").exists():
    lang = pd.read_csv(art / "lang_matrix.csv").sort_values("auc", ascending=False)
    plt.figure(figsize=(9, 3.5))
    plt.bar(lang["lang"], lang["auc"])
    plt.axhline(0.5, color="grey", ls="--", lw=1)
    plt.ylabel("AUC"); plt.title("Cross-lingual transfer (Micro-NAS, all languages)")
    plt.tight_layout()
    plt.savefig(figdir / "multilingual.png", dpi=200, bbox_inches="tight")
    print("wrote", figdir / "multilingual.png")
