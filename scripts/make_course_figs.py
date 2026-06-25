"""Real (matplotlib) size chart for the course, replacing the hand-drawn CSS bars.
Horizontal bars on a true log x-axis (real ticks/gridlines) so the 132x span
between NRMS and the edge models is honestly readable. Transparent background +
mid-gray text so it works in both light and dark course themes.
Run: python -m scripts.make_course_figs
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRAY = "#8a8a8a"
models = ["NRMS (FP32)", "NAS / FP32", "NAS / INT8", "Bin-uNAS / Binary", "Micro-NAS / INT8"]
sizes = [27000, 1566, 790, 240, 204]                 # KB
colors = ["#c0392b", "#e67e22", "#2e86c1", "#7d3c98", "#27ae60"]

fig, ax = plt.subplots(figsize=(7.2, 3.0))
bars = ax.barh(models[::-1], sizes[::-1], color=colors[::-1], height=0.62)
ax.set_xscale("log")
ax.set_xlabel("content-encoder size (KB) — log scale", color=GRAY, fontsize=9)
for b, s in zip(bars, sizes[::-1]):
    label = f"{s/1000:.1f} MB" if s >= 1000 else f"{s} KB"
    ax.text(s * 1.15, b.get_y() + b.get_height() / 2, label, va="center",
            fontsize=8.5, color=GRAY)
ax.set_xlim(120, 70000)
ax.grid(axis="x", which="both", alpha=0.25)
ax.tick_params(colors=GRAY, labelsize=8)
for t in ax.get_yticklabels():
    t.set_color(GRAY)
for sp in ax.spines.values():
    sp.set_color(GRAY)
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
plt.tight_layout()
out = Path("course/assets/size_chart.png")
plt.savefig(out, dpi=150, transparent=True)
print("wrote", out)
