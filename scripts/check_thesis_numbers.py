"""Check the numbers quoted in the MSc thesis chapters against the frozen result
files in paper/results and the data manifest. Every expected string is computed
from a result file (half-up rounding) and must appear in the chapter text.

    python -m scripts.check_thesis_numbers "C:/Projects/PhD/H.E. Thesis/src"
"""
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

RES = Path("paper/results")
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/Projects/PhD/H.E. Thesis/src")


def f(x, nd=3):
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


text = ""
for p in sorted((SRC / "chapters").glob("*.tex")) + sorted((SRC / "tables").glob("*.tex")):
    text += p.read_text(encoding="utf-8") + "\n"
for a, b in (("\\%", "%"), ("\\,", " "), ("~", " "), ("\\KB", " KB"), ("\\MB", " MB"), ("\\uJ", " µJ"), ("{}", "")):
    text = text.replace(a, b)
text = re.sub(r"\s+", " ", text)


def dd(a, b, nd=3):
    """Difference of two displayed (rounded) values, as the reader computes it from a table."""
    return str(Decimal(f(a, nd)) - Decimal(f(b, nd)))

S = json.loads((RES / "p1_summary.json").read_text(encoding="utf-8"))
H = json.loads((RES / "p1_heuristics.json").read_text(encoding="utf-8"))["heuristics"]["results"]
M = pd.read_csv(RES / "results_matrix.csv")
SUM = json.loads((RES / "results_summary.json").read_text(encoding="utf-8"))
IMP = json.loads((RES / "binary_improved.json").read_text(encoding="utf-8"))
LAT = json.loads((RES / "latency.json").read_text(encoding="utf-8"))
NI8 = json.loads((RES / "nrms_int8.json").read_text(encoding="utf-8"))
ABL = json.loads((RES / "ablation.json").read_text(encoding="utf-8"))
DN = json.loads(Path("paper/tables/thesis/dataset_numbers.json").read_text(encoding="utf-8"))
recs = [json.loads(l) for l in (RES / "experiments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
p1 = [r for r in recs if str(r.get("run", "")).startswith("p1/")]

failures = []


def expect(s, label):
    if s not in text:
        failures.append(f"{label}: expected '{s}'")


def cell(arm, prec, col):
    return float(M[(M.arm == arm) & (M.precision == prec)][col].iloc[0])


def en(k):
    return S[k]["en_auc"]["mean"]


def xl(k):
    return S[k]["xlang_auc"]["mean"]


# --- October reference conditions
for k, lab in (("nrms/nrms_glove", "glove"), ("nrms/nrms_reduced", "reduced"), ("student/scratch_en", "scratch"),
               ("student/distill_ft_en", "distilled"), ("student/distill_ft_mixed", "mixed"),
               ("student/distill_frozen", "frozen"), ("teacher/teacher_frozen", "teacher"),
               ("teacher/teacher_linear", "teacher+linear"), ("student/scratch_mixed", "scratch mixed")):
    expect(f(en(k)), f"EN AUC {lab}")
    if "xlang_auc" in S[k]:
        expect(f(xl(k)), f"14-lang AUC {lab}")
expect(f(S["nrms/nrms_reduced"]["en_auc"]["std"]), "reduced std")
expect(f(en("nrms/nrms_glove") - en("student/distill_ft_en")), "gap distilled to GloVe")
expect(f(en("student/distill_ft_en") - en("student/scratch_en")), "distillation gain (October)")
expect(f(en("student/distill_ft_en") - H["subcategory"]["all"]["auc"]), "distilled above subcategory")
expect(f(en("student/distill_ft_en") - en("nrms/nrms_reduced")), "distilled above reduced")
expect(f(en("student/distill_ft_en") - en("student/distill_ft_mixed")), "mixed: cost in English")
expect(f(xl("student/distill_ft_mixed") - xl("student/distill_ft_en")), "mixed: gain across languages")
expect(f(en("student/scratch_en") - en("student/scratch_mixed")), "scratch mixed: cost in English")
expect(f(en("student/distill_ft_en") - xl("student/distill_ft_mixed")), "translations below English-only encoder")
expect(f(en("student/distill_ft_mixed") - xl("student/distill_ft_mixed")), "translations below English (mixed)")
expect(str(round(S["nrms/nrms_glove"]["settings"]["params"] / S["student/scratch_en"]["encoder_cost"]["params"])), "parameter ratio GloVe/byte-CNN")
pl = S["student/distill_ft_mixed"]["per_language"]
expect(f(min(v["mean"] for v in pl.values())), "mixed: weakest language")
expect(f(max(v["mean"] for v in pl.values())), "mixed: strongest language")
n58 = sum(v["mean"] > 0.58 for v in pl.values())
expect({11: "eleven", 12: "twelve", 10: "ten"}[n58], "languages above 0.58")
for b in ("1-5", "6-20", ">20"):
    expect(f(S["nrms/nrms_glove"]["buckets"][b]["mean"]), f"GloVe bucket {b}")
    expect(f(S["student/distill_ft_en"]["buckets"][b]["mean"]), f"distilled bucket {b}")
    expect(f(H["subcategory"][b]["auc"]), f"subcategory bucket {b}")
gaps = [S["nrms/nrms_glove"]["buckets"][b]["mean"] - S["student/distill_ft_en"]["buckets"][b]["mean"] for b in ("1-5", ">20")]
expect(f"from {f(gaps[0])} to {f(gaps[1])}", "bucket gap range")
expect(f(S["student/distill_ft_en"]["buckets"]["1-5"]["mean"] - H["subcategory"]["1-5"]["auc"]), "byte-CNN above subcategory, short histories")
cos = S["student/distill_frozen"]["settings"]["teacher_cosine_dev"]
expect(", ".join(f(cos[d]) for d in ("64", "128", "256")) + " and " + f(cos["384"]), "teacher cosine by prefix")

# --- heuristics and buckets
for k, lab in (("popularity", "popularity"), ("category", "category"), ("subcategory", "subcategory")):
    expect(f(H[k]["all"]["auc"]), f"{lab} AUC")
expect(f(H["popularity"]["0"]["auc"]), "popularity, no history")
expect(f(H["category"]["0"]["auc"]), "category prior, no history")
for b, sh in DN["bucket_share"].items():
    expect(f(100 * sh, 1) + "%", f"bucket share {b}")
expect(f"{DN['n_dev']:,}", "dev impressions")
expect(f"{DN['n_train_impressions']:,}", "train impressions")
expect(f"{DN['n_users']:,}", "users")
expect(f"{DN['n_train_news']:,}", "train articles")
expect(f"{DN['n_dev_news']:,}", "dev articles")
expect(str(DN["n_subcats"]), "subcategories")

# --- June matrix
for arm in ("nas", "micro_nas", "binarized_micro_nas"):
    for prec in ("fp32", "int8", "binary"):
        expect(f(cell(arm, prec, "auc")), f"matrix AUC {arm}/{prec}")
        expect(f(cell(arm, prec, "size_kb"), 1), f"matrix size {arm}/{prec}")
    expect("+" + dd(cell(arm, "int8", "auc"), cell(arm, "fp32", "auc")), f"INT8 delta {arm}")
ratios = [cell(a, "fp32", "energy_uj") / cell(a, "int8", "energy_uj") for a in ("nas", "micro_nas", "binarized_micro_nas")]
expect(f"{f(min(ratios), 1)} to {f(max(ratios), 1)} times", "energy ratio range")
sz = [cell(a, "fp32", "size_kb") / cell(a, "int8", "size_kb") for a in ("nas", "micro_nas", "binarized_micro_nas")]
expect(f"{f(min(sz), 1)} to {f(max(sz), 0)} times", "size ratio range")
loss = [Decimal(dd(cell(a, "fp32", "auc"), cell(a, "binary", "auc"))) for a in ("nas", "micro_nas", "binarized_micro_nas")]
expect(f"{min(loss)} to {max(loss)}", "binary loss range")
expect(dd(cell("nas", "fp32", "auc"), cell("micro_nas", "fp32", "auc")), "budget cost")
expect(f(cell("nas", "fp32", "size_kb") / cell("micro_nas", "fp32", "size_kb"), 1) + " times smaller", "budget size ratio")
expect(f"{cell('nas', 'fp32', 'macs') / 1e6:.1f} to {cell('micro_nas', 'fp32', 'macs') / 1e6:.2f} million", "MAC counts")
expect(str(round(cell("nas", "fp32", "macs") / cell("micro_nas", "fp32", "macs"))) + " times fewer", "MAC ratio")
expect(f(cell("micro_nas", "int8", "size_kb") - cell("micro_nas", "binary", "size_kb"), 0) + " KB", "1-bit saving over INT8")
expect(f(SUM["baseline"]["auc"]), "June baseline")
expect(f(SUM["runtime_min"] / 60, 1) + " hours", "June wall-clock")
langs = [r["auc"] for r in SUM["languages"] if r["lang"] != "en"]
expect(f(sum(langs) / len(langs)), "June per-language mean")

# --- binary recovery
expect(f(IMP["result"]["auc"]), "recovered binary AUC")
expect(f(IMP["cost"]["size_kb"], 0) + " KB", "recovered binary size")
expect(f"{IMP['cost']['params']:,}", "recovered binary params")
expect(f"{IMP['cost']['macs'] / 1e6:.2f} million", "recovered binary ops")
expect(dd(IMP["result"]["auc"], cell("micro_nas", "binary", "auc")), "recovery gain")
expect(dd(cell("micro_nas", "int8", "auc"), IMP["result"]["auc"]), "recovered below INT8")
expect(dd(SUM["baseline"]["auc"], IMP["result"]["auc"]), "recovered below June reduced NRMS")
expect(f(cell("micro_nas", "int8", "size_kb") - IMP["cost"]["size_kb"], 0) + " KB smaller", "recovered vs INT8 size")

# --- latency, NRMS INT8, ablation
cpu = [r["cpu_onnx_ms"] for r in LAT]
expect(", ".join(f(c, 2) for c in cpu[:2]) + " and " + f(cpu[2], 2) + " ms", "CPU latency")
expect(f(NI8["fp32"]["auc"], 4) + " to " + f(NI8["int8"]["auc"], 4), "NRMS INT8")
expect(f(ABL["scratch"]["auc"]) + " against " + f(ABL["distilled_init"]["auc"]), "June ablation")
expect(f(ABL["distilled_init"]["auc"] - ABL["scratch"]["auc"]), "June ablation gain")

# --- October compute
expect(str(len(p1)) + " runs", "October run count")
mins = sum(r["minutes"] for r in p1)
expect(f(mins / 60, 1) + " hours", "October wall-clock")
expect(f(mins, 1) + " min", "October minutes")

# --- NRMS tables and the size table of Chapter 3
g = S["nrms/nrms_glove"]["settings"]
r = S["nrms/nrms_reduced"]["settings"]
expect(f"{g['embedding_params'] // 300:,}", "GloVe vocabulary")
expect(f"{r['embedding_params'] // 256:,}", "reduced vocabulary")
expect(str(round(100 * g["embedding_params"] / g["params"])) + "%", "GloVe table share")
expect(str(round(100 * r["embedding_params"] / r["params"])) + "%", "reduced table share")
expect(str(round(g["size_mb_fp32"] * 1024 / cell("micro_nas", "int8", "size_kb"))) + " times", "GloVe size ratio")
expect(str(round(r["size_mb_fp32"] * 1024 / cell("micro_nas", "int8", "size_kb"))) + " times", "reduced size ratio")


def groups(C, D, out, prec):
    w = 1 if prec == "int8" else 4
    emb = 257 * 64 * 4
    proj = (64 * C + C) * 4
    head = (C * out + out) * 4
    blocks = D * ((C * 3) * w + C * 4 + C * C * w + C * 4 + 2 * C * 4)
    return emb / 1024, proj / 1024, blocks / 1024, head / 1024


for C, D, out in ((256, 4, 384), (64, 5, 384), (96, 2, 384)):
    e, p, b32, h = groups(C, D, out, "fp32")
    b8 = groups(C, D, out, "int8")[2]
    expect(f"{f(e, 1)} & {f(p, 1)} & {f(b32, 1)} & {f(b8, 1)} & {f(h, 1)} & {f(e + p + b32 + h, 1)} / {f(e + p + b8 + h, 1)}", f"size table {C}-{D}-{out}")
q = {}
for C, D, out in ((256, 4, 384), (64, 5, 384), (96, 2, 384)):
    tot = 257 * 64 + 64 * C + C + D * (3 * C + C + C * C + C + 2 * C) + C * out + out
    q[C] = round(100 * D * (3 * C + C * C) / tot)
expect(f"{q[64]}%", "quantisable share constrained")
expect(f"{q[256]}%", "quantisable share unconstrained")

print(f"{len(failures)} failures")
for x in failures:
    print(" ", x)
sys.exit(1 if failures else 0)
