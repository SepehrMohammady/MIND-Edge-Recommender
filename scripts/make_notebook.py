"""Generate the master notebook: end-to-end (A->Z), valid nbformat, one markdown
note per code cell, short code cells that only call src/ functions, and every
knob driven from the notebook. Run: python -m scripts.make_notebook
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

# ── 0. Title / map ────────────────────────────────────────────────────────────
md("""# MIND-Edge-Recommender — end-to-end
**Tiny multilingual news recommender: offline pretraining on MIND + xMIND, with a
NAS × precision study of flash / RAM / energy for the edge.**

This notebook is the single control surface for the whole project (A→Z). Long
code lives in `src/*.py`; every cell here is short and preceded by a note, and
every knob is set from `config.yaml` / this notebook.

**Map:** setup → data → teacher → student/distillation → NAS (3 arms) →
train → precision sweep → baseline → ablation → multilingual → latency/energy →
export → plots. Architecture axis: NAS → Micro-NAS → binarized-Micro-NAS.
Precision axis: FP32 → INT8 → Binary. Targets: RTX 5070 → Raspberry Pi 5 → STM32H7.""")

# ── 1. Setup ──────────────────────────────────────────────────────────────────
md("""## 1. Setup
Add the repo root to the path (works whether the kernel starts in the repo root
or in `notebooks/`), import the modules, and load the central config.""")
code("""import sys, pathlib
root = pathlib.Path.cwd()
root = root if (root / "src").exists() else root.parent
sys.path.insert(0, str(root))
import pandas as pd, torch
from src.config import load_config
from src import experiment, export, measure_energy, data_mind, data_xmind
cfg = load_config()
print("device:", "cuda" if torch.cuda.is_available() else "cpu", "| artifacts:", cfg["paths"]["artifacts_dir"])""")

# ── 2. Config control surface ─────────────────────────────────────────────────
md("""## 2. Configuration (the control surface)
Every experiment knob lives in `config.yaml` and can be overridden here in memory
before running anything. The key groups:""")
code("""{k: cfg[k] for k in ["data", "teacher", "student", "nas", "quant", "train"]}""")

# ── 3. Run scale ──────────────────────────────────────────────────────────────
md("""## 3. Run scale
`QUICK=True` runs a fast smoke pass (few epochs / small NAS / capped impressions)
so the whole notebook executes in minutes. Set `QUICK=False` for the full
paper-grade run. This single switch drives every heavy cell below.""")
code("""QUICK = True
if QUICK:
    cfg["train"]["distill_epochs"], cfg["train"]["epochs"] = 2, 2
    NAS = dict(generations=3, population=8, n_train=8000, n_val=2000, distill_epochs=1)
    EVAL_IMPR, QAT, TRAIN_IMPR, DISTILL_EP = 1000, 0, 2000, 2
else:
    NAS = dict(generations=cfg["nas"]["generations"], population=cfg["nas"]["population"])
    EVAL_IMPR, QAT, TRAIN_IMPR, DISTILL_EP = None, 2, None, 15
QUICK""")

# ── 4. Datasets ───────────────────────────────────────────────────────────────
md("""## 4. Datasets
MIND + xMIND are pre-downloaded (`python -m src.download` to refresh). Show the
SHA256 manifest size and basic statistics.""")
code("""import json
manifest = json.load(open(pathlib.Path(cfg["paths"]["data_dir"]) / "manifest.json"))
news = data_mind.read_news(cfg, "train")
print(len(manifest), "files SHA256-pinned")
print("MIND train: news", len(news), "| categories", len(data_mind.category_vocab(news)))
print("xMIND languages:", data_xmind.available_langs(cfg))""")

# ── 5. Peek one article across languages ──────────────────────────────────────
md("""## 5. One article, many languages
xMIND translates the same article; behaviours stay English. Here is one news id
shown in a few languages (the byte-level model will encode all of them).""")
code("""nid = next(iter(news))
print("[en ]", news[nid]["title"])
for lang in ["ron", "jpn", "zho"]:
    print(f"[{lang}]", data_xmind.read_xmind_text(cfg, lang, "train").get(nid, {}).get("title", "-"))""")

# ── 6. Teacher anchors ────────────────────────────────────────────────────────
md("""## 6. Teacher anchors
Compute (and cache) the frozen multilingual teacher's English-title embeddings —
the shared distillation targets for all 14 languages.""")
code("""experiment.ensure_anchors(cfg)""")

# ── 7. Distil the student ─────────────────────────────────────────────────────
md("""## 7. Distil the byte-CNN student
Distil the language-agnostic byte-level student to the teacher anchors.""")
code("""student_encoder = experiment.distill(cfg)""")

# ── 8. Distillation sanity ────────────────────────────────────────────────────
md("""## 8. Sanity: does the student align languages?
A title and its translation should map to nearby vectors (high cosine).""")
code("""import torch.nn.functional as F
from src.student import text_to_bytes
L = cfg["data"]["max_title_bytes"]; enc_cpu = student_encoder.cpu()
def emb(t):
    with torch.no_grad():
        return F.normalize(enc_cpu(torch.tensor([text_to_bytes(t, L)])), dim=-1)[0]
dev_en = data_mind.read_news(cfg, "dev"); dev_ro = data_xmind.localized_news(cfg, "ron", "dev")
k = next(iter(dev_en))
print("cosine(en, ron):", round(F.cosine_similarity(emb(dev_en[k]["title"]), emb(dev_ro[k]["title"]), dim=0).item(), 3))""")

# ── 9. NAS ────────────────────────────────────────────────────────────────────
md("""## 9. Architecture search (3 arms)
One search space; three arms differ by precision + constraints: `nas` (FP32,
loose), `micro_nas` (INT8, hard MCU budget), `binarized_micro_nas` (Binary).
Fitness = distillation quality under footprint feasibility.""")
code("""arms = experiment.search_arms(cfg, **NAS)
best = {arm: experiment.best_arch(res) for arm, res in arms.items()}
pd.DataFrame([{"arm": a, **best[a], "quality": arms[a][0]["quality"],
               "size_kb": arms[a][0]["size_kb"], "feasible": arms[a][0]["feasible"]} for a in arms])""")

# ── 10. Train per arm ─────────────────────────────────────────────────────────
md("""## 10. Train a recommender per arm
Train the end-to-end recommender (news encoder = each arm's best architecture).""")
code("""models = {arm: experiment.train_for_arch(cfg, best[arm], max_train_impressions=TRAIN_IMPR)
          for arm in best}""")

# ── 11. Precision sweep ───────────────────────────────────────────────────────
md("""## 11. Precision sweep → results matrix
Evaluate each model at FP32 / INT8 / Binary (PTQ + optional QAT) with ranking
metrics and footprint (size, RAM, MACs, energy).""")
code("""rows = []
for arm in models:
    rows += experiment.precision_sweep(cfg, models[arm], best[arm], arm, qat_epochs=QAT, max_eval=EVAL_IMPR)
matrix = experiment.save_matrix(cfg, rows); matrix""")

# ── 12. Baseline ──────────────────────────────────────────────────────────────
md("""## 12. NRMS baseline (FP32 ceiling)
Reproduced NRMS on the **same** MINDsmall-dev split (like-for-like). External
reference: published MINDlarge-test NRMS AUC 0.6776.""")
code("""experiment.baseline_row(cfg, epochs=cfg["train"]["epochs"], max_eval=EVAL_IMPR)""")

# ── 13. Ablation ──────────────────────────────────────────────────────────────
md("""## 13. Ablation: does distillation help?
Same Micro-NAS architecture, news encoder trained from scratch vs initialised
from the distilled student then fine-tuned.""")
code("""abl = experiment.run_ablation(cfg, distill_epochs=DISTILL_EP,
                             train_epochs=cfg["train"]["epochs"], max_train_impressions=TRAIN_IMPR)
pd.DataFrame([{"init": "scratch", **abl["scratch"]},
              {"init": "distilled", **abl["distilled_init"]}])[["init", "auc", "mrr", "ndcg@10"]]""")

# ── 14. Multilingual ──────────────────────────────────────────────────────────
md("""## 14. Multilingual evaluation (all 14 languages)
Cross-lingual transfer: identical English impressions, news text swapped per
language through the SAME byte-level model.""")
code("""deploy = models["micro_nas"]
experiment.eval_languages(cfg, deploy, max_impressions=EVAL_IMPR or 2000)""")

# ── 15. Latency & energy ──────────────────────────────────────────────────────
md("""## 15. Measured latency & energy (laptop)
GPU latency + NVML energy; ONNX CPU latency uses the same path that runs on the
Pi 5 aarch64 wheel (pair with an INA219 for real Pi/STM32 power).""")
code("""ex = torch.zeros(8, cfg["data"]["max_title_bytes"], dtype=torch.long)
print("GPU latency ms:", round(measure_energy.latency_torch(deploy.news_encoder, ex), 4))
print("Energy:", measure_energy.energy_nvml(deploy.news_encoder, ex, n=200))""")

# ── 16. Export single ─────────────────────────────────────────────────────────
md("""## 16. Export the deployment model
ONNX content encoder + cold-start `topicWeights` prior, mapped to the app's
`edgeml_model_state_v1` schema (Micro-NAS / INT8 sweet spot).""")
code("""langs = ["en"] + data_xmind.available_langs(cfg)
print(export.export_artifacts(cfg, deploy.news_encoder.cpu(), langs))""")

# ── 17. Export all arms ───────────────────────────────────────────────────────
md("""## 17. Export every arm + comparison manifest
Export one ONNX per arm and a `models_manifest.json` so the app can pick a
trade-off. (INT8/Binary are *simulated* for measurement; the deployed ONNX is the
FP32 graph that the target runtime — e.g. X-CUBE-AI — quantizes itself.)""")
code("""arm_label = {"nas": "nas", "micro_nas": "micro_nas", "binarized_micro_nas": "bin_unas"}
to_export = {(arm_label[a], "fp32"): m.news_encoder.cpu() for a, m in models.items()}
print(export.export_all(cfg, to_export, langs))""")

# ── 18. Pareto plots ──────────────────────────────────────────────────────────
md("""## 18. Pareto: accuracy vs footprint and energy""")
code("""import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for arm in matrix["arm"].unique():
    s = matrix[matrix["arm"] == arm]
    ax[0].scatter(s["size_kb"], s["auc"], label=arm); ax[1].scatter(s["energy_uj"], s["auc"], label=arm)
ax[0].set(xlabel="size (KB)", ylabel="AUC"); ax[1].set(xlabel="energy (uJ/inf)", ylabel="AUC")
ax[1].set_xscale("log"); ax[0].legend(); [a.grid(alpha=.3) for a in ax]; plt.tight_layout(); plt.show()""")

# ── 19. Reproduce paper numbers ───────────────────────────────────────────────
md("""## 19. Reproduce the paper's headline numbers
Load the cached full-run summary (`scripts/run_full.py`) — the numbers reported
in `paper/`.""")
code("""p = pathlib.Path(cfg["paths"]["artifacts_dir"]) / "results_summary.json"
if p.exists():
    s = json.load(open(p))
    print("best archs:", s["best_arch"]); print("NRMS baseline:", s["baseline"])
    display(pd.DataFrame(s["matrix"])[["arm", "precision", "auc", "size_kb", "energy_uj"]])
else:
    print("Run `python -m scripts.run_full` for the full paper-grade numbers.")""")

# ── 20. Conclusions ───────────────────────────────────────────────────────────
md("""## 20. Conclusions
- One byte-level model fits the STM32H7 budget and serves all 14 languages.
- INT8 Micro-NAS is the measured sweet spot; Binary is footprint-motivated.
- Distillation makes the tiny model competitive (see the ablation).
- Artifacts in `artifacts/`; the write-up in `paper/`.""")

nb["cells"] = cells
out = Path("notebooks"); out.mkdir(exist_ok=True)
path = out / "MIND-Edge-Recommender.ipynb"
nbf.write(nb, path)
print("wrote", path, "with", len(cells), "cells")
