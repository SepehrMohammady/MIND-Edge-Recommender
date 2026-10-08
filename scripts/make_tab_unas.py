"""Write the table of the encoders under the board budgets (paper/tab_unas.tex) and its numbers
(paper/results/unas_summary.json) from the result files, so no number is typed by hand.

Rows: the µNAS choice and the best hand-designed encoder of the step-3 grid within each board's
budget, and the hand-designed 64-5-384 (outside both MAC budgets), all in the fork's terms (fixed
byte table, padding-mask channel, global average pooling). Columns:

  costs          from the fork's resource model, the one the search's budgets use: MACs per title
                 (pooling operations included), INT8 weight bytes, peak activation bytes
  search cos     validation mean cosine to the teacher under the search recipe (96k rows, at most 15
                 epochs): mean of three seeds for the µNAS choices (unas/select_by_seeds.py) and for
                 64-5-384 (reference.json), one seed for the grid rows (grid.json)
  full programme scripts/run_unas_full.py: teacher cosine on the English dev titles after distillation
                 on all 769k rows, then English clicks: dev AUC and mean AUC over the 14 translations
  INT8           AUC of the integer ONNX file (scripts/export_int8.py)

Means and population standard deviations over the seeds, as in paper/tab_p1.tex. Missing runs
print as "--".

    python -m scripts.make_tab_unas
"""
import json
import statistics
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "paper/results"
U = RES / "unas"


def load(p, default):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


hist = {n: {c["index"]: c for c in load(U / f"{n}_history.json", {"candidates": []})["candidates"]}
        for n in ("mind_h7", "mind_f401")}
sel = {n: load(U / f"{n}_selection.json", {}) for n in ("mind_h7", "mind_f401")}
grid = {r["arch"]: r for r in load(U / "grid.json", [])}
ref = [r for r in load(U / "reference.json", []) if r["arch"] == "64-5-384"]
full = load(RES / "unas_full.json", {})
int8 = load(RES / "int8_export.json", {"runs": {}})["runs"]


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": round(statistics.fmean(values), 4),
            "std": round(statistics.pstdev(values), 4) if len(values) > 1 else None, "n": len(values)}


def full_runs(name, stage):
    return [v for k, v in sorted(full.items()) if k.startswith(f"{name}/{stage}/seed")]


def row(group, label, name, int8_key):
    if name in hist:
        index = sel[name]["best"]
        c = hist[name][index]
        short = next(r for r in sel[name]["shortlist"] if r["index"] == index)
        search_cos = {"mean": round(short["mean_val_cos"], 4), "n": short["n"]}
        cost = (c["macs"], c["model_size_bytes"], c["peak_mem_bytes"])
    else:
        arch = name[len("hand_"):]
        if arch == "64-5-384":
            v = [r["val_cos"] for r in ref]
            search_cos = {"mean": round(statistics.fmean(v), 4), "n": len(v)}
            c = ref[0]
        else:
            c = grid[arch]
            search_cos = {"mean": round(c["val_cos"], 4), "n": 1}
        cost = (c["macs"], c["model_size_bytes"], c["peak_mem_bytes"])
    distill = full_runs(name, "distill")
    en = full_runs(name, "distill_ft_en")
    q = [v for k, v in sorted(int8.items()) if k.startswith(f"{int8_key}/")]
    return {"group": group, "label": label, "name": name,
            "macs": int(cost[0]), "weights_bytes": int(cost[1]), "activation_bytes": int(cost[2]),
            "search_cos": search_cos,
            "teacher_cos_dev": stats([r["results"]["teacher_cosine_dev"]["384"] for r in distill]),
            "en_auc": stats([r["results"]["en"]["all"]["auc"] for r in en]),
            "xlang_auc": stats([r["results"]["mean_xlang_auc"] for r in en]),
            "torch_auc_export": stats([r["torch_fp32"]["auc"] for r in q]),
            "int8_auc": stats([r["onnx_int8qdq"]["auc"] for r in q]),
            "int8_file_bytes": q[0]["onnx_int8qdq"]["file_bytes"] if q else None,
            "seeds_full": [r["seed"] for r in en]}


ROWS = [
    row("h7", r"$\mu$NAS, candidate 144", "mind_h7", "unas_h7"),
    row("h7", "Hand-designed 64-2-384", "hand_64-2-384", "hand_64-2-384"),
    row("f401", r"$\mu$NAS, candidate 134", "mind_f401", "unas_f401"),
    row("f401", "Hand-designed 32-5-384", "hand_32-5-384", "hand_32-5-384"),
    row("none", "Hand-designed 64-5-384", "hand_64-5-384", "hand_64-5-384"),
]
(RES / "unas_summary.json").write_text(json.dumps(ROWS, indent=1), encoding="utf-8")


def f(x, nd=3):
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def pm(st, nd=3):
    if not st:
        return "--"
    return f(st["mean"], nd) + (r"$\pm$" + f(st["std"], nd) if st.get("std") is not None else "")


GROUPS = {"h7": r"STM32H7B3I-DK budget: 256\,KiB weights, 128\,KiB activations, 2\,M MACs",
          "f401": r"NUCLEO-F401RE budget: 128\,KiB weights, 48\,KiB activations, 1\,M MACs",
          "none": r"Outside both MAC budgets"}
lines, group = [], None
for r in ROWS:
    if r["group"] != group:
        if group is not None:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{8}{l}{\textit{" + GROUPS[r["group"]] + r"}}\\")
        group = r["group"]
    cos = f(r["search_cos"]["mean"]) + ("" if r["search_cos"]["n"] > 1 else r"$^\ddagger$")
    lines.append(f"{r['label']} & {r['macs'] / 1e6:.2f} & {r['weights_bytes'] / 1024:.1f} & "
                 f"{r['activation_bytes'] / 1024:.1f} & {cos} & {pm(r['teacher_cos_dev'])} & "
                 f"{pm(r['en_auc'])} & {pm(r['int8_auc'])}" + r" \\")
seeds = sorted({len(r["seeds_full"]) for r in ROWS if r["seeds_full"]})
tex = r"""% Generated by scripts/make_tab_unas.py from paper/results (unas/, unas_full.json, int8_export.json); do not edit.
\begin{table}[t]
\caption{Encoders under the board budgets. Costs from the search's resource model (MACs per title,
eight-bit weights, peak activations). Search cos: validation cosine to the teacher under the search
recipe, mean of three seeds ($^\ddagger$: one seed). Teacher cos: English dev titles after distillation
on all 769k rows; AUC: MINDsmall dev after English clicks, full precision and as the integer file
(INT8). Mean $\pm$ standard deviation over NSEEDS seeds. All rows read the bytes through the same fixed
table with a padding-mask channel and average over all positions.}
\label{tab:unas}\centering\scriptsize\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{\begin{tabular}{lccccccc}
\toprule
Encoder & MACs\,(M) & Weights\,(KiB) & Act.\,(KiB) & Search cos & Teacher cos & AUC & INT8 AUC\\
\midrule
""".replace("NSEEDS", "/".join(str(s) for s in seeds) or "3") + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
(ROOT / "paper/tab_unas.tex").write_text(tex, encoding="utf-8")
for r in ROWS:
    print(f"{r['name']:>15s}  {r['macs'] / 1e6:5.2f} M  {r['weights_bytes'] / 1024:6.1f} KiB  "
          f"{r['activation_bytes'] / 1024:5.1f} KiB  search {r['search_cos']['mean']:.4f} (n={r['search_cos']['n']})  "
          f"teacher {pm(r['teacher_cos_dev'], 4)}  AUC {pm(r['en_auc'], 4)}  14-lang {pm(r['xlang_auc'], 4)}  "
          f"INT8 {pm(r['int8_auc'], 4)}")
