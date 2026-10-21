"""Generate the paper figures from the frozen result files in ``paper/results``
(never from a run folder, so a smoke run cannot leak into a figure).

Figures are drawn at the LNCS text width (122 mm), so their 7-point text prints at that size; each is
written as PDF (included by the paper) and PNG (for previews).

Run: python -m scripts.make_figures
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, ScalarFormatter

from src.config import load_config

cfg = load_config()
res = Path(cfg["paths"]["results_dir"])
figdir = Path("paper/figures")
figdir.mkdir(parents=True, exist_ok=True)

INK, MUTED, FAINT, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#d9d8d2", "#ffffff"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"       # categorical slots 1-3, validated all-pairs on white
ARMS = {"nas": ("NAS", C1),                          # colour = search arm
        "micro_nas": ("Micro-NAS", C2),
        "binarized_micro_nas": (r"Bin. $\mu$NAS", C3)}
PREC = {"fp32": ("FP32", "o"), "int8": ("INT8", "s"), "binary": ("1-bit", "^")}   # shape = precision
TW = 4.8                                             # LNCS text width in inches

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "legend.fontsize": 6.5, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "pdf.fonttype": 42})


def load(name):
    return json.loads((res / name).read_text(encoding="utf-8"))


def despine(a, grid="both"):
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    a.grid(axis=grid, color=GRID, lw=0.5, zorder=0)
    a.set_axisbelow(True)


def save(fig, name):
    fig.savefig(figdir / f"{name}.pdf", facecolor=SURFACE, metadata={"CreationDate": None})
    fig.savefig(figdir / f"{name}.png", dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", figdir / f"{name}.pdf", "and .png")


# The matrix rerun (scripts/run_matrix.py, mean and population SD over seeds) replaces the June single
# run once it exists; the dashed line is the reduced NRMS of Table 2 (three seeds).
summary_file = res / "matrix_summary.json"
if summary_file.exists():
    m = pd.DataFrame(load("matrix_summary.json"))
    nrms = load("p1_summary.json")["nrms/nrms_reduced"]["en_auc"]["mean"]
else:
    m = pd.read_csv(res / "results_matrix.csv")
    nrms = load("results_summary.json")["baseline"]["auc"]
# ReActNet one-bit encoder: three-seed rerun (scripts/run_binary.py) when present, else the June single run
if (res / "binary_reactnet.json").exists():
    _r = load("binary_reactnet.json")["summary"]
    improved = {"cost": _r["cost"], "result": {"auc": _r["auc"], "auc_sd": _r["auc_sd"]}}
else:
    improved = load("binary_improved.json")

# Fig. 3: AUC against encoder size and against the energy proxy, per arm and precision.
fig, ax = plt.subplots(1, 2, figsize=(TW, 2.3), sharey=True)
for a, (xcol, xlabel) in zip(ax, (("size_kb", "encoder size (KB)"),
                                  ("energy_uj", r"energy proxy ($\mu$J per title)"))):
    for arm, (_, color) in ARMS.items():
        for prec, (_, marker) in PREC.items():
            r = m[(m["arm"] == arm) & (m["precision"] == prec)].iloc[0]
            if "auc_sd" in r and pd.notna(r["auc_sd"]):
                a.errorbar(r[xcol], r["auc"], yerr=r["auc_sd"], fmt="none", ecolor=color, elinewidth=0.7,
                           capsize=1.2, zorder=2)
            a.scatter(r[xcol], r["auc"], s=26, marker=marker, color=color,
                      edgecolor=SURFACE, linewidth=0.6, zorder=3)
    x_imp = improved["cost"]["size_kb"] if xcol == "size_kb" else improved["cost"]["energy_uj_per_inf"]
    if improved["result"].get("auc_sd"):
        a.errorbar(x_imp, improved["result"]["auc"], yerr=improved["result"]["auc_sd"], fmt="none",
                   ecolor=ARMS["micro_nas"][1], elinewidth=0.7, capsize=1.2, zorder=2)
    a.scatter(x_imp, improved["result"]["auc"], s=44, marker="*", color=ARMS["micro_nas"][1],
              edgecolor=SURFACE, linewidth=0.5, zorder=3)
    a.axhline(nrms, ls="--", color=MUTED, lw=0.7, zorder=1)
    a.set_xscale("log")
    a.set_xticks([200, 400, 800, 1600] if xcol == "size_kb" else [3, 10, 30, 100])
    a.xaxis.set_major_formatter(ScalarFormatter())
    a.xaxis.set_minor_locator(NullLocator())
    a.set_xlabel(xlabel)
    despine(a)
ax[0].set_ylabel("AUC (MINDsmall dev)")
ax[1].text(ax[1].get_xlim()[1], nrms - 0.002, f"reduced NRMS ({nrms:.3f})", ha="right", va="top",
           color=MUTED, fontsize=6.5)
handles = ([Line2D([], [], marker="o", ls="", color=c, markeredgecolor=SURFACE, markersize=5, label=n)
            for n, c in ARMS.values()]
           + [Line2D([], [], marker=mk, ls="", color=MUTED, markeredgecolor=SURFACE, markersize=5, label=n)
              for n, mk in PREC.values()]
           + [Line2D([], [], marker="*", ls="", color=ARMS["micro_nas"][1], markeredgecolor=SURFACE,
                     markersize=7, label="1-bit, ReActNet blocks")])
fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, handletextpad=0.1,
           columnspacing=0.8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0)
save(fig, "pareto")

# Fig. 4: (a) per-language AUC of the reference byte-CNN under three training conditions, English first,
# then the fourteen xMIND translations by the mixed-language model's AUC; (b) AUC by history length.
# Means over three seeds (scripts/run_p1.py, p1_summary.json).
P = load("p1_summary.json")
LANG = {"zho": "Chinese", "fin": "Finnish", "grn": "Guarani", "hat": "Haitian", "ind": "Indonesian",
        "jpn": "Japanese", "kat": "Georgian", "ron": "Romanian", "som": "Somali", "swh": "Swahili",
        "tam": "Tamil", "tha": "Thai", "tur": "Turkish", "vie": "Vietnamese"}
CONDS = [("student/scratch_en", "random start, English clicks", C1, "o"),
         ("student/distill_ft_en", "distilled, English clicks", C2, "s"),
         ("student/distill_ft_mixed", "distilled, mixed-language", C3, "^")]


def lang_auc(key, lang):
    return P[key]["en_auc"]["mean"] if lang == "en" else P[key]["per_language"][lang]["mean"]


order = ["en"] + sorted(LANG, key=lambda k: -lang_auc("student/distill_ft_mixed", k))
ys = [0] + [i + 1.6 for i in range(len(LANG))]                  # a gap between English and the translations
fig, (a, b) = plt.subplots(1, 2, figsize=(TW, 2.75), gridspec_kw={"width_ratios": [2.3, 1]})
for y, lang in zip(ys, order):
    v = [lang_auc(k, lang) for k, *_ in CONDS]
    a.hlines(y, min(v), max(v), color=GRID, lw=1.8, zorder=1)
for key, label, color, marker in CONDS:
    a.scatter([lang_auc(key, lang) for lang in order], ys, s=16, marker=marker, color=color,
              edgecolor=SURFACE, linewidth=0.5, zorder=3, label=label)
a.axvline(0.5, color=MUTED, ls="--", lw=0.7, zorder=1)
a.text(0.502, 0.8, "chance", ha="left", va="center", color=MUTED, fontsize=6.5)
a.set_yticks(ys)
a.set_yticklabels(["English"] + [LANG[k] for k in order[1:]])
a.set_ylim(ys[-1] + 0.9, -0.7)
a.set_xlim(0.49, 0.64)
a.set_xlabel("AUC (MINDsmall dev)")
a.set_title("(a) per language", loc="left")
despine(a, grid="x")
BUCKETS = [("1-5", "1–5"), ("6-20", "6–20"), (">20", ">20")]
for key, label, color, marker in (("nrms/nrms_glove", "NRMS,\nGloVe", MUTED, "D"),
                                  ("student/distill_ft_en", "byte-\nCNN", C2, "s")):
    y = [P[key]["buckets"][k]["mean"] for k, _ in BUCKETS]
    b.plot(range(3), y, color=color, lw=1.0, zorder=2)
    b.scatter(range(3), y, s=15, marker=marker, color=color, edgecolor=SURFACE, linewidth=0.5, zorder=3)
    b.text(2.15, y[-1], label, ha="left", va="center", fontsize=6.5, color=INK)
b.set_xticks(range(3))
b.set_xticklabels([t for _, t in BUCKETS])
b.set_xlim(-0.3, 2.1)
b.set_xlabel("clicks in history")
b.set_ylabel("AUC")
b.set_title("(b) by history", loc="left")
despine(b, grid="y")
handles = [Line2D([], [], marker=mk, ls="", color=c, markeredgecolor=SURFACE, markersize=5, label=lb)
           for _, lb, c, mk in CONDS]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, handletextpad=0.1, columnspacing=1.0,
           bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 0.94, 1), w_pad=1.5)
save(fig, "languages")

# Fig. 5: the two muNAS searches (unas/<search>_history.json): validation cosine to the teacher under the
# search recipe against MACs per title, for every candidate and for the 30 hand-designed shapes of the June
# space under the same recipe (unas/grid.json, one seed); (c) the eight finalists of each search under the
# search recipe (mean of three seeds) and after full distillation (seed 42), from unas/rechoice.json.
U = res / "unas"
grid_rows = json.loads((U / "grid.json").read_text(encoding="utf-8"))
rechoice = json.loads((U / "rechoice.json").read_text(encoding="utf-8"))["searches"]
SEARCHES = [("mind_h7", "(a) STM32H7B3I-DK budget"), ("mind_f401", "(b) NUCLEO-F401RE budget")]
CHOICE = {"first_choice": "D", "best": "*"}          # recipe choice, final choice


def choice_marker(a, x, y, marker):
    a.scatter(x, y, s=34 if marker == "*" else 15, marker=marker, facecolor=INK if marker == "*" else SURFACE,
              edgecolor=INK, linewidth=0.7 if marker == "D" else 0.3, zorder=5)


fig, ax = plt.subplots(1, 3, figsize=(TW, 2.15), gridspec_kw={"width_ratios": [1, 1, 0.95]})
for a, (name, title) in zip(ax[:2], SEARCHES):
    h = json.loads((U / f"{name}_history.json").read_text(encoding="utf-8"))
    bud = h["budget"]
    cand = {c["index"]: c for c in h["candidates"] if not c["failed"]}
    inb = [c for c in cand.values() if c["in_budget"]]
    out = [c for c in cand.values() if not c["in_budget"]]
    a.scatter([c["macs"] / 1e6 for c in out], [c["val_cos"] for c in out], s=7, facecolor="none",
              edgecolor=FAINT, linewidth=0.5, zorder=2)
    a.scatter([c["macs"] / 1e6 for c in inb], [c["val_cos"] for c in inb], s=7, color=C1, linewidth=0, zorder=3)

    def fits(r):
        return (r["macs"] <= bud["macs"] and r["model_size_bytes"] <= bud["model_size"]
                and r["peak_mem_bytes"] <= bud["peak_mem"])
    g_in = [r for r in grid_rows if fits(r)]
    g_out = [r for r in grid_rows if not fits(r)]
    a.scatter([r["macs"] / 1e6 for r in g_out], [r["val_cos"] for r in g_out], s=9, marker="s",
              facecolor="none", edgecolor=C2, linewidth=0.6, zorder=3)
    a.scatter([r["macs"] / 1e6 for r in g_in], [r["val_cos"] for r in g_in], s=9, marker="s", color=C2,
              linewidth=0, zorder=4)
    for key, marker in CHOICE.items():
        cnd = cand[rechoice[name][key]]
        choice_marker(a, cnd["macs"] / 1e6, cnd["val_cos"], marker)
    a.axvline(bud["macs"] / 1e6, color=MUTED, ls="--", lw=0.7, zorder=1)
    a.text(bud["macs"] / 1e6 / 1.2, 0.512, "MAC\nbudget", ha="right", va="top", color=MUTED, fontsize=6)
    a.set_xscale("log")
    a.set_xlim(0.025, 60)
    a.set_xticks([0.1, 1, 10])
    a.set_xticklabels(["0.1", "1", "10"])
    a.xaxis.set_minor_locator(NullLocator())
    a.set_ylim(0.19, 0.52)
    a.set_xlabel("M MACs per title")
    a.set_title(title, loc="left")
    despine(a)
ax[0].set_ylabel("cosine to teacher, search recipe")
ax[1].set_yticklabels([])
c = ax[2]
for name, marker, label, (tx, ty, ha) in (("mind_h7", "o", "H7 finalists", (0.3205, 0.349, "center")),
                                          ("mind_f401", "^", "F401 finalists", (0.300, 0.309, "left"))):
    st = rechoice[name]["stage1"]
    pick = {rechoice[name][k]: mk for k, mk in CHOICE.items()}
    rest = [r for r in st if r["index"] not in pick]
    c.scatter([r["search_recipe_val_cos"] for r in rest], [r["val_cos_seed42"] for r in rest], s=16,
              marker=marker, color=C1, edgecolor=SURFACE, linewidth=0.5, zorder=3)
    for r in st:
        if r["index"] in pick:
            xy = (r["search_recipe_val_cos"], r["val_cos_seed42"])
            choice_marker(c, *xy, pick[r["index"]])
            if pick[r["index"]] == "*":
                c.annotate(str(r["index"]), xy, xytext=(-4, 3), textcoords="offset points", ha="right", fontsize=6)
            else:
                c.annotate(str(r["index"]), xy, xytext=(0, -5), textcoords="offset points", ha="center",
                           va="top", fontsize=6)
    c.text(tx, ty, label, ha=ha, va="center", fontsize=6, color=MUTED)
c.set_xlabel("search recipe (3 seeds)")
c.set_ylabel("full distillation (seed 42)")
c.set_title("(c) the eight finalists", loc="left")
c.set_xlim(0.265, 0.338)
c.set_ylim(0.298, 0.392)
despine(c)
handles = [Line2D([], [], marker="o", ls="", color=C1, markersize=3.5, label=r"$\mu$NAS candidate"),
           Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=FAINT, markersize=3.5,
                  label="over budget"),
           Line2D([], [], marker="s", ls="", color=C2, markersize=3.5, label="hand-designed"),
           Line2D([], [], marker="D", ls="", markerfacecolor=SURFACE, markeredgecolor=INK, markersize=3.5,
                  label="recipe choice"),
           Line2D([], [], marker="*", ls="", color=INK, markersize=6, label="final choice")]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, handletextpad=0.1, columnspacing=1.0,
           bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.6)
save(fig, "unas_search")

# Fig. 6: (a) milliseconds per title of the 8-bit files on the STM32H7B3I-DK under three generator settings
# (board/STM32H7B3I-DK_sweep.json, scripts/board_sweep.py); (b) energy per title of the four deployments:
# phone (phone_energy.json, battery gauge), boards (board/energy/*_energy.json, USB meter).
sweep = {r["model"]: r for r in load("board/STM32H7B3I-DK_sweep.json")["rows"]}
ROWS = [("unas_h7", r"$\mu$NAS 144 (H7 recipe)"), ("unas_h7_c140", r"$\mu$NAS 140 (H7 final)"),
        ("hand_64-2-384", "64-2-384"), ("unas_f401", r"$\mu$NAS 134 (F401 recipe)"),
        ("unas_f401_c142", r"$\mu$NAS 142 (F401 final)"), ("hand_32-5-384", "32-5-384"),
        ("hand_64-5-384", "64-5-384")]
SETTINGS = [("cloud_ms", "ST's board", "D", "none", MUTED), ("balanced_ms", "ours, default", "o", C2, C2),
            ("time_ms", "ours, -O time", "o", C1, C1)]
bench = sorted((res / "phone" / "edge_bench").glob("latency_*.json"))[-1]
phone_ms = {(r["kind"], r["threads"]): r["medianMs"]
            for r in json.loads(bench.read_text(encoding="utf-8"))["rows"]}[("int8", 1)]
phone = load("phone_energy.json")
E = res / "board" / "energy"
f4 = json.loads((E / "NUCLEO-F401RE__unas_f401_c142_energy.json").read_text(encoding="utf-8"))
h7 = json.loads((E / "STM32H7B3I-DK__unas_h7_c140_Otime_energy.json").read_text(encoding="utf-8"))
bnn = json.loads((E / "STM32H7B3I-DK__reactnet_64-5-384_seed42_lce_energy.json").read_text(encoding="utf-8"))
DEPLOY = [(f"phone, 8-bit 64-5-384, {phone_ms:.2f} ms (battery)", phone["energy_per_title_mj"]),
          (f"H7, 8-bit $\\mu$NAS 140, {h7['network_ms']:.1f} ms (USB)", h7["energy_per_title_usb_mj"]),
          (f"F401, 8-bit $\\mu$NAS 142, {f4['network_ms']:.1f} ms (USB)", f4["energy_per_title_usb_mj"]),
          (f"H7, 1-bit 64-5-384, {bnn['network_ms']:.1f} ms (USB)", bnn["energy_per_title_usb_mj"])]
fig, (a, b) = plt.subplots(1, 2, figsize=(TW, 2.25), gridspec_kw={"width_ratios": [1.1, 1]})
ys = list(range(len(ROWS)))[::-1]
for y, (model, _) in zip(ys, ROWS):
    v = [sweep[model][k] for k, *_ in SETTINGS]
    a.hlines(y, min(v), max(v), color=GRID, lw=1.6, zorder=1)
for k, label, marker, face, edge in SETTINGS:
    a.scatter([sweep[mdl][k] for mdl, _ in ROWS], ys, s=17, marker=marker, facecolor=face, edgecolor=edge,
              linewidth=0.8 if face == "none" else 0.5, zorder=3 if face == "none" else 2, label=label)
for y in (3.5, 0.5):
    a.axhline(y, color=GRID, lw=0.5, ls=":")
a.set_yticks(ys)
a.set_yticklabels([lb for _, lb in ROWS])
a.set_xscale("log")
a.set_xticks([5, 10, 20, 40, 80])
a.xaxis.set_major_formatter(ScalarFormatter())
a.xaxis.set_minor_locator(NullLocator())
a.set_xlim(5, 90)
a.set_xlabel("ms per title, 8-bit files")
a.set_title("(a) STM32H7B3I-DK", loc="left")
despine(a, grid="x")
yb = [3, 2, 1, 0]
b.barh(yb, [e for _, e in DEPLOY], height=0.38, color=C1, zorder=2)
for y, (label, e) in zip(yb, DEPLOY):
    b.text(0.25, y + 0.27, label, va="bottom", ha="left", fontsize=6, color=MUTED)
    b.text(e + 0.3, y, f"{e:.2f}" if e < 1 else f"{e:.1f}", va="center", ha="left", fontsize=6.5, color=INK)
b.set_yticks([])
b.set_ylim(-0.4, 3.75)
b.set_xlim(0, 18.5)
b.set_xlabel("mJ per title")
b.set_title("(b) energy per title", loc="left")
despine(b, grid="x")
fig.legend(handles=a.get_legend_handles_labels()[0], loc="lower center", ncol=3, frameon=False,
           handletextpad=0.1, columnspacing=1.2, bbox_to_anchor=(0.3, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.0)
save(fig, "boards")
