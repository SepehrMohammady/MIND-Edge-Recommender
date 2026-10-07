"""Figures and tables for the MSc thesis, built from the frozen result files
(paper/results) and the data on disk. Written to paper/figures/thesis/ (PNG,
300 dpi) and paper/tables/thesis/ (LaTeX fragments); the thesis project copies
them. Every number comes from a committed result file or from the datasets.

    python -m scripts.make_thesis_assets
"""
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from src import data_mind, data_xmind
from src.config import load_config


def group_sizes_kb(channels, depth, out_dim, prec, byte_embed=64):
    """Stored size (KB) per layer group of the byte-CNN, by parameter formula.
    INT8 quantises the weights of the inner depthwise and pointwise convolutions;
    the embedding, first projection, head, biases and batch-norm stay FP32."""
    emb = 257 * byte_embed * 4
    proj = (byte_embed * channels + channels) * 4
    head = (channels * out_dim + out_dim) * 4
    w_bytes = 1 if prec == "int8" else 4
    blocks = depth * ((channels * 3) * w_bytes + channels * 4 + (channels * channels) * w_bytes + channels * 4 + 2 * channels * 4)
    return {"embedding": emb / 1024, "first projection": proj / 1024, "blocks": blocks / 1024, "head": head / 1024}

cfg = load_config()
RES = Path(cfg["paths"]["results_dir"])
FIG = Path("paper/figures/thesis")
TAB = Path("paper/tables/thesis")
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d2", "#ffffff"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"            # validated categorical slots 1-3
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6})

S = json.loads((RES / "p1_summary.json").read_text(encoding="utf-8"))
H = json.loads((RES / "p1_heuristics.json").read_text(encoding="utf-8"))["heuristics"]["results"]
M = pd.read_csv(RES / "results_matrix.csv")
SUMMARY = json.loads((RES / "results_summary.json").read_text(encoding="utf-8"))
IMPROVED = json.loads((RES / "binary_improved.json").read_text(encoding="utf-8"))
LANG = {"zho": "Chinese", "fin": "Finnish", "grn": "Guarani", "hat": "Haitian Creole", "ind": "Indonesian",
        "jpn": "Japanese", "kat": "Georgian", "ron": "Romanian", "som": "Somali", "swh": "Swahili",
        "tam": "Tamil", "tha": "Thai", "tur": "Turkish", "vie": "Vietnamese"}


def f(x, nd=3):
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def pm(st, nd=3):
    return f(st["mean"], nd) + r"$\pm$" + f(st["std"], nd) if st.get("std") is not None else f(st["mean"], nd)


def despine(a):
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    a.grid(color=GRID, lw=0.5, zorder=0)
    a.set_axisbelow(True)


# ---------------------------------------------------------------- figures
# F1: per-language AUC, three byte-CNN training conditions (3 seeds, means)
conds = [("student/scratch_en", "random start, English clicks", C1, "o"),
         ("student/distill_ft_en", "distilled start, English clicks", C2, "s"),
         ("student/distill_ft_mixed", "distilled start, mixed-language clicks", C3, "^")]
order = sorted(LANG, key=lambda k: S["student/distill_ft_mixed"]["per_language"][k]["mean"])
fig, a = plt.subplots(figsize=(6.4, 4.2))
for i, lang in enumerate(order):
    a.hlines(i, 0.48, 0.63, color=GRID, lw=0.8, zorder=1)
for key, label, color, marker in conds:
    vals = [S[key]["per_language"][l]["mean"] for l in order]
    a.scatter(vals, range(len(order)), s=40, marker=marker, color=color, edgecolor=SURFACE,
              linewidth=0.8, zorder=3, label=label)
    en = S[key]["en_auc"]["mean"]
    a.axvline(en, color=color, lw=0.8, ls=":", zorder=2)
a.axvline(0.5, color=MUTED, lw=0.8, ls="--")
a.set_yticks(range(len(order)))
a.set_yticklabels([f"{LANG[l]} ({l})" for l in order])
a.set_xlabel("AUC on the translated dev set (dotted: the same model on English)")
a.set_xlim(0.48, 0.64)
despine(a)
fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(FIG / "p1_per_language.png", dpi=300, facecolor=SURFACE)

# F2: AUC by history length: GloVe NRMS, distilled byte-CNN, subcategory histogram
buckets = ["0", "1-5", "6-20", ">20"]
series = [("nrms/nrms_glove", "NRMS, GloVe (11.3 M params)", C1, "o"),
          ("student/distill_ft_en", "byte-CNN, distilled start (68 k params)", C2, "s")]
fig, a = plt.subplots(figsize=(5.6, 3.4))
x = range(len(buckets))
for key, label, color, marker in series:
    a.plot(x, [S[key]["buckets"][b]["mean"] for b in buckets], color=color, lw=1.6, zorder=2)
    a.scatter(x, [S[key]["buckets"][b]["mean"] for b in buckets], s=42, marker=marker, color=color,
              edgecolor=SURFACE, linewidth=0.8, zorder=3, label=label)
a.plot(x, [H["subcategory"][b]["auc"] for b in buckets], color=C3, lw=1.6, zorder=2)
a.scatter(x, [H["subcategory"][b]["auc"] for b in buckets], s=42, marker="^", color=C3,
          edgecolor=SURFACE, linewidth=0.8, zorder=3, label="subcategory histogram (no network)")
a.set_xticks(list(x))
a.set_xticklabels(["no history", "1–5 clicks", "6–20 clicks", "> 20 clicks"])
a.set_ylabel("AUC (MINDsmall dev)")
a.set_ylim(0.49, 0.70)
despine(a)
a.legend(loc="upper left", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "p1_history_buckets.png", dpi=300, facecolor=SURFACE)

# F3: reference points: AUC vs parameters of the news encoder (log x)
pts = [("NRMS GloVe", S["nrms/nrms_glove"]["settings"]["params"], S["nrms/nrms_glove"]["en_auc"]["mean"]),
       ("NRMS reduced", S["nrms/nrms_reduced"]["settings"]["params"], S["nrms/nrms_reduced"]["en_auc"]["mean"]),
       ("byte-CNN 64-5-384, distilled start", S["student/scratch_en"]["encoder_cost"]["params"],
        S["student/distill_ft_en"]["en_auc"]["mean"]),
       ("byte-CNN 64-5-384, random start", S["student/scratch_en"]["encoder_cost"]["params"],
        S["student/scratch_en"]["en_auc"]["mean"]),
       ("NAS 256-4-384, INT8 (June run)", 401088, float(M[(M.arm == "nas") & (M.precision == "int8")].auc.iloc[0]))]
fig, a = plt.subplots(figsize=(6.0, 3.4))
a.scatter([p[1] for p in pts], [p[2] for p in pts], s=46, color=C1, edgecolor=SURFACE, linewidth=0.8, zorder=3)
for name, n, auc in pts:
    a.annotate(name, (n, auc), xytext=(6, 4), textcoords="offset points", fontsize=7.5, color=INK)
for name, auc in (("subcategory histogram, no network", H["subcategory"]["all"]["auc"]),
                  ("category histogram, no network", H["category"]["all"]["auc"]),
                  ("popularity", H["popularity"]["all"]["auc"])):
    a.axhline(auc, color=MUTED, lw=0.7, ls="--", zorder=1)
    a.text(4e4, auc + 0.002, name, fontsize=7.5, color=MUTED, va="bottom")
a.set_xscale("log")
a.set_xlabel("news-encoder parameters")
a.set_ylabel("AUC (MINDsmall dev)")
a.set_xlim(4e4, 4e7)
despine(a)
fig.tight_layout()
fig.savefig(FIG / "reference_points.png", dpi=300, facecolor=SURFACE)

# F4: where the bytes go: stored size per layer group, FP32 vs simulated INT8, three architectures
groups = ["embedding", "first projection", "blocks", "head"]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=False)
for a, (name, arch) in zip(axes, (("NAS 256-4-384", (256, 4, 384)), ("Micro-NAS 64-5-384", (64, 5, 384)),
                                   ("Bin. µNAS 96-2-384", (96, 2, 384)))):
    bars = {prec: group_sizes_kb(*arch, prec) for prec in ("fp32", "int8")}
    left = [0.0, 0.0]
    for grp, color in zip(groups, [C1, C2, C3, MUTED]):
        vals = [bars["fp32"][grp], bars["int8"][grp]]
        a.barh([0, 1], vals, left=left, color=color, edgecolor=SURFACE, linewidth=0.6, label=grp)
        left = [left[0] + vals[0], left[1] + vals[1]]
    a.set_yticks([0, 1])
    a.set_yticklabels(["FP32", "INT8 (weights)"])
    a.set_xlabel("stored size (KB)")
    a.set_title(name, fontsize=9)
    despine(a)
axes[0].legend(loc="lower right", frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "size_breakdown.png", dpi=300, facecolor=SURFACE)

# F5: June matrix: AUC vs size and vs energy (same as the paper's Fig. 2, larger)
ARMS = {"nas": ("NAS", C1), "micro_nas": ("Micro-NAS", C2), "binarized_micro_nas": ("Bin. µNAS", C3)}
PREC = {"fp32": ("FP32", "o"), "int8": ("INT8", "s"), "binary": ("1-bit", "^")}
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
for a, (xcol, xlabel) in zip(ax, (("size_kb", "estimated encoder size (KB)"), ("energy_uj", "energy proxy (µJ per title)"))):
    for arm, (_, color) in ARMS.items():
        for prec, (_, marker) in PREC.items():
            r = M[(M.arm == arm) & (M.precision == prec)].iloc[0]
            a.scatter(r[xcol], r.auc, s=46, marker=marker, color=color, edgecolor=SURFACE, linewidth=0.9, zorder=3)
    ximp = IMPROVED["cost"]["size_kb"] if xcol == "size_kb" else IMPROVED["cost"]["energy_uj_per_inf"]
    a.scatter(ximp, IMPROVED["result"]["auc"], s=64, marker="*", color=C2, edgecolor=SURFACE, linewidth=0.7, zorder=3)
    a.axhline(SUMMARY["baseline"]["auc"], ls="--", color=MUTED, lw=0.8)
    a.axhline(S["nrms/nrms_glove"]["en_auc"]["mean"], ls="-.", color=MUTED, lw=0.8)
    a.set_xscale("log")
    a.set_xticks([200, 400, 800, 1600] if xcol == "size_kb" else [3, 10, 30, 100])
    a.xaxis.set_major_formatter(ScalarFormatter())
    a.minorticks_off()
    a.set_xlabel(xlabel)
    despine(a)
ax[0].set_ylabel("AUC (MINDsmall dev)")
ax[0].text(1650, SUMMARY["baseline"]["auc"] + 0.002, "NRMS reduced (June)", ha="right", va="bottom", color=MUTED, fontsize=7)
ax[0].text(1650, S["nrms/nrms_glove"]["en_auc"]["mean"] + 0.002, "NRMS GloVe (3 seeds)", ha="right", va="bottom", color=MUTED, fontsize=7)
handles = ([Line2D([], [], marker="o", ls="", color=c, markeredgecolor=SURFACE, markersize=6, label=n) for n, c in ARMS.values()]
           + [Line2D([], [], marker=mk, ls="", color=MUTED, markeredgecolor=SURFACE, markersize=6, label=n) for n, mk in PREC.values()]
           + [Line2D([], [], marker="*", ls="", color=C2, markeredgecolor=SURFACE, markersize=8, label="1-bit, ReActNet blocks")])
fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=7, handletextpad=0.2, columnspacing=0.9, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIG / "june_matrix.png", dpi=300, facecolor=SURFACE)
print("figures written to", FIG)

# ---------------------------------------------------------------- tables
train_news = data_mind.read_news(cfg, "train")
dev_news = data_mind.read_news(cfg, "dev")
train_beh = data_mind.read_behaviors(cfg, "train")
dev_beh = data_mind.read_behaviors(cfg, "dev")
dev_imps = data_mind.build_eval_impressions(dev_beh, cfg["data"]["max_history"])
users = {b["user"] for b in train_beh} | {b["user"] for b in dev_beh}
cats = sorted({v["category"] for v in train_news.values() if v["category"]})
subs = sorted({v["subcategory"] for v in train_news.values() if v["subcategory"]})
langs = data_xmind.available_langs(cfg)
rows = [("Users (train and dev)", f"{len(users):,}"),
        ("Training impressions", f"{len(train_beh):,}"),
        ("Dev impressions (labelled, used for evaluation)", f"{len(dev_imps):,}"),
        ("Articles in the training split", f"{len(train_news):,}"),
        ("Articles in the dev split", f"{len(dev_news):,}"),
        ("Categories / subcategories (editorial labels)", f"{len(cats)} / {len(subs)}"),
        ("xMIND languages (translated titles)", f"{len(langs)}"),
        ("Clicked history kept per user", f"{cfg['data']['max_history']}"),
        ("Title length (UTF-8 bytes)", f"{cfg['data']['max_title_bytes']}"),
        ("Negatives per positive (redrawn each step)", f"{cfg['data']['neg_ratio']}")]
tex = ["\\begin{tabular}{lr}", "\\toprule", "Quantity & Value\\\\", "\\midrule"] + [f"{k} & {v}\\\\" for k, v in rows] + ["\\bottomrule", "\\end{tabular}"]
(TAB / "dataset_stats.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

# June matrix with corrected costs
tex = ["\\begin{tabular}{llcccrrrr}", "\\toprule",
       "Arm & Prec. & AUC & MRR & nDCG@10 & Size (KB) & RAM (KB) & MMAC & Energy (µJ)\\\\", "\\midrule"]
names = {"nas": "NAS", "micro_nas": "Micro-NAS", "binarized_micro_nas": "Bin.\\,µNAS"}
prev = None
for r in M.itertuples(index=False):
    if prev and r.arm != prev:
        tex.append("\\midrule")
    prev = r.arm
    e = f(r.energy_uj, 1) if r.energy_uj >= 100 else f(r.energy_uj, 2)
    tex.append(f"{names[r.arm]} & {PREC[r.precision][0]} & {f(r.auc)} & {f(r.mrr)} & {f(getattr(r, '_7'))} & "
               f"{f(r.size_kb, 1)} & {f(r.ram_kb, 0)} & {r.macs / 1e6:.2f} & {e}\\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
(TAB / "june_matrix.tex").write_text("\n".join(tex).replace("µ", "$\\mu$") + "\n", encoding="utf-8")

# per-language table: three byte-CNN conditions + teacher-linear (means over seeds)
cols = [("student/scratch_en", "random, EN"), ("student/distill_ft_en", "distilled, EN"),
        ("student/distill_ft_mixed", "distilled, mixed"), ("teacher/teacher_linear", "teacher + linear")]
tex = ["\\begin{tabular}{ll" + "c" * len(cols) + "}", "\\toprule",
       "Code & Language & " + " & ".join(c[1] for c in cols) + "\\\\", "\\midrule",
       "en & English & " + " & ".join(f(S[k]["en_auc"]["mean"]) for k, _ in cols) + "\\\\", "\\midrule"]
for lang in sorted(LANG, key=lambda k: LANG[k]):
    tex.append(f"{lang} & {LANG[lang]} & " + " & ".join(f(S[k]["per_language"][lang]["mean"]) for k, _ in cols) + "\\\\")
tex += ["\\midrule", "& mean of 14 & " + " & ".join(f(S[k]["xlang_auc"]["mean"]) for k, _ in cols) + "\\\\",
        "\\bottomrule", "\\end{tabular}"]
(TAB / "per_language.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

# heuristics by bucket
tex = ["\\begin{tabular}{lrcccc}", "\\toprule", "Scorer & Impressions & no history & 1--5 clicks & 6--20 clicks & $>$20 clicks\\\\", "\\midrule"]
for key, name in (("popularity", "Popularity"), ("category", "Category histogram"), ("subcategory", "Subcategory histogram")):
    r = H[key]
    tex.append(f"{name} & {r['all']['n_impressions']:,} & " + " & ".join(f(r[b]["auc"]) for b in buckets) + "\\\\")
for key, name in (("nrms/nrms_glove", "NRMS, GloVe"), ("student/distill_ft_en", "Byte-CNN, distilled start, EN clicks")):
    tex.append(f"{name} & {H['category']['all']['n_impressions']:,} & " + " & ".join(f(S[key]["buckets"][b]["mean"]) for b in buckets) + "\\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
(TAB / "history_buckets.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

# share of impressions per bucket
n_all = H["category"]["all"]["n_impressions"]
shares = {b: H["category"][b]["n_impressions"] / n_all for b in buckets}
json.dump({"bucket_share": shares, "n_dev": n_all, "n_train_impressions": len(train_beh), "n_users": len(users),
           "n_train_news": len(train_news), "n_dev_news": len(dev_news), "n_cats": len(cats), "n_subcats": len(subs)},
          open(TAB / "dataset_numbers.json", "w"), indent=1)
print("tables written to", TAB, "| buckets", {b: round(v, 4) for b, v in shares.items()})

# ---------------------------------------------------------------- thesis-only additions
# F6: the embedding-table memory wall, with the table sizes of this project's own NRMS runs
glove_emb = S["nrms/nrms_glove"]["settings"]["embedding_params"]        # 34,304 words x 300
reduced_emb = S["nrms/nrms_reduced"]["settings"]["embedding_params"]    # 25,355 words x 256
byte_table = 257 * 64
n = list(range(1, 16))
fig, a = plt.subplots(figsize=(5.8, 3.3))
a.plot(n, [k * glove_emb * 4 / 2**20 for k in n], color=C1, lw=1.6, marker="o", ms=4,
       label=f"GloVe-300d table, {glove_emb // 300:,} words, one table per language")
a.plot(n, [k * reduced_emb * 4 / 2**20 for k in n], color=C2, lw=1.6, marker="s", ms=4,
       label=f"256-d table, {reduced_emb // 256:,} words, one table per language")
a.plot(n, [byte_table * 4 / 2**20] * len(n), color=C3, lw=1.6, marker="^", ms=4,
       label="byte table, 257 rows x 64, shared by every language")
a.set_yscale("log")
a.set_xlabel("languages served")
a.set_ylabel("embedding table, FP32 (MB, log)")
a.set_xticks([1, 5, 10, 15])
despine(a)
a.legend(loc="center left", frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig(FIG / "memory_wall.png", dpi=300, facecolor=SURFACE)

# F7: one-bit weights on the constrained architecture: naive sign, searched binary arm, ReActNet recipe, INT8 reference
def cell(arm, prec, col):
    return float(M[(M.arm == arm) & (M.precision == prec)][col].iloc[0])
bars = [("naive sign\n64-5-384\n" + f(cell("micro_nas", "binary", "size_kb"), 0) + " KB", cell("micro_nas", "binary", "auc"), C3),
        ("searched\nbinary arm, 96-2-384\n" + f(cell("binarized_micro_nas", "binary", "size_kb"), 0) + " KB", cell("binarized_micro_nas", "binary", "auc"), C3),
        ("ReActNet blocks,\ndistilled start, 64-5-384\n" + f(IMPROVED["cost"]["size_kb"], 0) + " KB", IMPROVED["result"]["auc"], C2),
        ("INT8 weights\n64-5-384\n" + f(cell("micro_nas", "int8", "size_kb"), 0) + " KB", cell("micro_nas", "int8", "auc"), C1)]
fig, a = plt.subplots(figsize=(6.2, 3.3))
a.bar(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars], width=0.58, zorder=3)
for i, b in enumerate(bars):
    a.text(i, b[1] + 0.003, f(b[1]), ha="center", va="bottom", fontsize=8)
a.axhline(0.5, color=MUTED, ls="--", lw=0.8)
a.text(len(bars) - 0.6, 0.502, "chance", fontsize=7.5, color=MUTED, ha="right", va="bottom")
a.set_xticks(range(len(bars)))
a.set_xticklabels([b[0] for b in bars], fontsize=7.5)
a.set_ylim(0.48, 0.64)
a.set_ylabel("AUC (MINDsmall dev)")
despine(a)
fig.tight_layout()
fig.savefig(FIG / "binary_recovery.png", dpi=300, facecolor=SURFACE)

# per-seed values of every October condition (appendix)
NAMES = {"teacher/teacher_frozen": "Frozen teacher + user encoder", "teacher/teacher_linear": "Frozen teacher + linear map",
         "nrms/nrms_glove": "NRMS, GloVe reference config.", "nrms/nrms_reduced": "NRMS, reduced (random 256-d table)",
         "student/scratch_en": "Byte-CNN, random start, EN clicks", "student/scratch_mixed": "Byte-CNN, random start, mixed clicks",
         "student/distill_frozen": "Byte-CNN, distilled, frozen", "student/distill_ft_en": "Byte-CNN, distilled start, EN clicks",
         "student/distill_ft_mixed": "Byte-CNN, distilled start, mixed clicks"}
tex = ["\\begin{tabular}{lcccccc}", "\\toprule",
       "Condition & \\multicolumn{3}{c}{English AUC} & \\multicolumn{3}{c}{mean AUC, 14 languages}\\\\",
       "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", " & seed 42 & seed 12 & seed 1 & seed 42 & seed 12 & seed 1\\\\", "\\midrule"]
for key, name in NAMES.items():
    r = S[key]
    idx = {s: i for i, s in enumerate(r["seeds"])}
    en = [f(r["en_auc"]["values"][idx[s]]) for s in (42, 12, 1)]
    xl = [f(r["xlang_auc"]["values"][idx[s]]) if "xlang_auc" in r else "--" for s in (42, 12, 1)]
    tex.append(f"{name} & " + " & ".join(en) + " & " + " & ".join(xl) + "\\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
(TAB / "p1_seeds.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

# wall-clock minutes per run (appendix); the stretched run is reported as it happened
tex = ["\\begin{tabular}{lrrrr}", "\\toprule", "Condition & seed 42 & seed 12 & seed 1 & mean (min)\\\\", "\\midrule"]
total = 0.0
for key, name in NAMES.items():
    r = S[key]
    idx = {s: i for i, s in enumerate(r["seeds"])}
    mins = [r["minutes"]["values"][idx[s]] for s in (42, 12, 1)]
    total += sum(mins)
    tex.append(f"{name} & " + " & ".join(f(m, 1) for m in mins) + f" & {f(r['minutes']['mean'], 1)}\\\\")
heur_min = json.loads((RES / "p1_heuristics.json").read_text(encoding="utf-8"))["heuristics"]["minutes"]
total += heur_min
tex.append(f"Heuristic scorers (one pass) & {f(heur_min, 2)} & -- & -- & {f(heur_min, 2)}\\\\")
tex += ["\\midrule", f"Total & \\multicolumn{{4}}{{r}}{{{f(total, 1)} min ({f(total / 60, 1)} h)}}\\\\", "\\bottomrule", "\\end{tabular}"]
(TAB / "p1_runs.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

# reference points (same content as the paper's Table 2, as a bare tabular for the thesis)
tex = ["\\begin{tabular}{lrcccc}", "\\toprule", "Scorer & Params & AUC & MRR & nDCG@10 & 14-lang.\\ AUC\\\\", "\\midrule"]
for key, name in (("popularity", "Popularity"), ("category", "Category histogram"), ("subcategory", "Subcategory histogram")):
    r = H[key]["all"]
    tex.append(f"{name} & -- & {f(r['auc'])} & {f(r['mrr'])} & {f(r['ndcg@10'])} & -- \\\\")
tex.append("\\midrule")
for key, name, params in (("nrms/nrms_reduced", "NRMS, reduced (random 256-d table)", "7.08\\,M"),
                          ("nrms/nrms_glove", "NRMS, GloVe reference config.", "11.29\\,M")):
    r = S[key]
    tex.append(f"{name} & {params} & {pm(r['en_auc'])} & {pm(r['en_mrr'])} & {pm(r['en_ndcg10'])} & -- \\\\")
tex.append("\\midrule")
for key, name in (("teacher/teacher_frozen", "Frozen teacher + user encoder"), ("teacher/teacher_linear", "\\quad + trainable linear map")):
    r = S[key]
    tex.append(f"{name} & -- & {pm(r['en_auc'])} & {pm(r['en_mrr'])} & {pm(r['en_ndcg10'])} & {pm(r['xlang_auc'])} \\\\")
tex.append("\\midrule")
for key, name in (("student/scratch_en", "Byte-CNN, random start, EN clicks"), ("student/scratch_mixed", "Byte-CNN, random start, mixed clicks"),
                  ("student/distill_frozen", "Byte-CNN, distilled, frozen"), ("student/distill_ft_en", "Byte-CNN, distilled start, EN clicks"),
                  ("student/distill_ft_mixed", "Byte-CNN, distilled start, mixed clicks")):
    r = S[key]
    tex.append(f"{name} & 68\\,k & {pm(r['en_auc'])} & {pm(r['en_mrr'])} & {pm(r['en_ndcg10'])} & {pm(r['xlang_auc'])} \\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
(TAB / "reference_points.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
# June matrix at four decimals with the corrected costs (appendix)
tex = ["\\begin{tabular}{llcccrrrr}", "\\toprule",
       "Arm & Prec. & AUC & MRR & nDCG@10 & Size (KB) & RAM (KB) & MMAC & Energy (µJ)\\\\", "\\midrule"]
prev = None
for row in SUMMARY["matrix"]:
    r = M[(M.arm == row["arm"]) & (M.precision == row["precision"])].iloc[0]
    if prev and row["arm"] != prev:
        tex.append("\\midrule")
    prev = row["arm"]
    e = f(r.energy_uj, 1) if r.energy_uj >= 100 else f(r.energy_uj, 2)
    tex.append(f"{names[row['arm']]} & {PREC[row['precision']][0]} & {f(row['auc'], 4)} & {f(row['mrr'], 4)} & "
               f"{f(row['ndcg@10'], 4)} & {f(r.size_kb, 2)} & {f(r.ram_kb, 0)} & {r.macs / 1e6:.3f} & {e}\\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
(TAB / "june_matrix_full.tex").write_text("\n".join(tex).replace("µ", "$\\mu$") + "\n", encoding="utf-8")

# June single-run per-language evaluation of the constrained full-precision model (appendix)
tex = ["\\begin{tabular}{llcccc}", "\\toprule", "Code & Language & AUC & MRR & nDCG@5 & nDCG@10\\\\", "\\midrule"]
for row in SUMMARY["languages"]:
    name = "English" if row["lang"] == "en" else LANG[row["lang"]]
    tex.append(f"{row['lang']} & {name} & {f(row['auc'])} & {f(row['mrr'])} & {f(row['ndcg@5'])} & {f(row['ndcg@10'])}\\\\")
    if row["lang"] == "en":
        tex.append("\\midrule")
others = [r for r in SUMMARY["languages"] if r["lang"] != "en"]
tex += ["\\midrule", "& mean of 14 & " + " & ".join(f(sum(r[k] for r in others) / len(others)) for k in ("auc", "mrr", "ndcg@5", "ndcg@10")) + "\\\\",
        "\\bottomrule", "\\end{tabular}"]
(TAB / "june_languages.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

# data manifest (appendix)
man = json.loads(Path("docs/data_manifest.json").read_text(encoding="utf-8"))
files = man.get("files", man)
tex = ["\\begin{longtable}{p{5.0cm}p{9.5cm}}", "\\toprule", "File & SHA-256\\\\", "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"]
for k, v in files.items():
    if not isinstance(v, (str, dict)):
        continue
    h = v.get("sha256", "") if isinstance(v, dict) else v
    kk = k.replace("_", "\\_")
    tex.append(f"\\texttt{{\\scriptsize {kk}}} & \\texttt{{\\scriptsize {h[:32]}\\hspace{{0pt}}{h[32:]}}}\\\\")
tex += ["\\end{longtable}"]
(TAB / "data_manifest.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
print("thesis additions written: memory_wall.png, binary_recovery.png, p1_seeds.tex, p1_runs.tex, reference_points.tex, "
      "june_matrix_full.tex, june_languages.tex, data_manifest.tex", "| manifest entries", len(files))

# ---------------------------------------------------------------- board measurements (ST Edge AI Developer Cloud)
runs_file = RES / "stedgeai_cloud" / "runs.jsonl"
dep_file = RES / "deploy_int8.json"
if runs_file.exists():
    runs = [json.loads(l) for l in runs_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    bench = {}
    for r in runs:
        if r.get("cmd") == "benchmark" and r.get("duration_ms", -1) and r.get("duration_ms", -1) > 0:
            bench[(r["model"], r["board"])] = r          # last successful run per model and board
    MODELS = [("micro_nas_64-5-384_body_fp32.onnx", "Micro-NAS 64-5-384", "FP32"),
              ("micro_nas_64-5-384_body_int8qdq.onnx", "Micro-NAS 64-5-384", "INT8"),
              ("bin_unas_96-2-384_body_fp32.onnx", "Binary-aware 96-2-384", "FP32"),
              ("bin_unas_96-2-384_body_int8qdq.onnx", "Binary-aware 96-2-384", "INT8"),
              ("nas_256-4-384_body_fp32.onnx", "NAS 256-4-384", "FP32"),
              ("nas_256-4-384_body_int8qdq.onnx", "NAS 256-4-384", "INT8")]
    BOARDS = [("STM32H7B3I-DK", "H7B3I-DK"), ("NUCLEO-F401RE", "F401RE")]
    tex = ["\\begin{tabular}{llrrrrrr}", "\\toprule",
           "Architecture & Precision & Board & Weights (KB) & Flash total (KB) & RAM (KB) & MAC & Latency (ms)\\\\", "\\midrule"]
    prev = None
    for fname, arch, prec in MODELS:
        for board, short in BOARDS:
            r = bench.get((fname, board))
            if r is None:
                continue
            if prev and arch != prev:
                tex.append("\\midrule")
            prev = arch
            tex.append(f"{arch} & {prec} & {short} & {f(r['weights_bytes'] / 1024, 1)} & {f(r['rom_bytes'] / 1024, 1)} & "
                       f"{f(r['ram_bytes'] / 1024, 1)} & {r['macc'] / 1e6:.2f}\\,M & {f(r['duration_ms'], 2)}\\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "board_benchmarks.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("board table rows:", len(bench))

    # F8: latency on the two boards, full precision against 8-bit, per architecture
    ARCHS = [("Micro-NAS\n64-5-384", "micro_nas_64-5-384"), ("Binary-aware\n96-2-384", "bin_unas_96-2-384"),
             ("NAS\n256-4-384", "nas_256-4-384")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False)
    for a, (board, title) in zip(axes, (("STM32H7B3I-DK", "STM32H7B3I-DK, Cortex-M7, 280 MHz"),
                                        ("NUCLEO-F401RE", "NUCLEO-F401RE, Cortex-M4, 84 MHz"))):
        xs = range(len(ARCHS))
        for k, (prec, color, off) in enumerate((("fp32", C1, -0.19), ("int8qdq", C2, 0.19))):
            for x, (_, stem) in zip(xs, ARCHS):
                r = bench.get((f"{stem}_body_{prec}.onnx", board))
                if r is None:
                    if k == 0:
                        a.text(x, 3, "does not fit\nin 96 KB RAM", ha="center", va="bottom", fontsize=7, color=MUTED)
                    continue
                a.bar(x + off, r["duration_ms"], width=0.36, color=color, zorder=3,
                      label=("FP32" if prec == "fp32" else "INT8 (QDQ)") if x == 0 else None)
                a.text(x + off, r["duration_ms"] * 1.08, f"{r['duration_ms']:.0f}", ha="center", va="bottom", fontsize=7)
        a.set_yscale("log")
        a.set_ylim(1, 3000)
        a.set_xlim(-0.6, len(ARCHS) - 0.4)
        a.set_xticks(list(xs))
        a.set_xticklabels([n for n, _ in ARCHS], fontsize=7.5)
        a.set_title(title, fontsize=8.5)
        despine(a)
    axes[0].set_ylabel("latency per title (ms, log)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "board_latency.png", dpi=300, facecolor=SURFACE)
    print("board latency figure written")
if dep_file.exists():
    D = json.loads(dep_file.read_text(encoding="utf-8"))
    tex = ["\\begin{tabular}{lrcccc}", "\\toprule", "Model file & Size (KB) & AUC & MRR & nDCG@5 & nDCG@10\\\\", "\\midrule"]
    for key, name in (("torch_fp32", "PyTorch, full precision (reference)"), ("onnx_fp32", "ONNX, full precision"),
                      ("onnx_int8qdq", "ONNX, 8-bit QDQ (integer kernels)")):
        m = D[key]
        size = f(D["files"][key]["bytes"] / 1024, 1) if key in D["files"] else "--"
        tex.append(f"{name} & {size} & {f(m['auc'])} & {f(m['mrr'])} & {f(m['ndcg@5'])} & {f(m['ndcg@10'])}\\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "deploy_int8.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("deploy table written; cosine", D["cosine_fp32_vs_int8"])
