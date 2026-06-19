"""Generate the clean master notebook programmatically (valid nbformat, one
markdown note per code cell, short code cells that only call src/ functions).
Run: python -m scripts.make_notebook
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# FeedWell-Edge — Tiny Multilingual News Recommender for the Edge
**Offline pretraining on MIND + xMIND with a NAS × precision footprint/energy study.**

This notebook is the single control surface. Long code lives in `src/*.py`;
each cell here is short and preceded by a note. Architecture axis:
NAS → Micro-NAS → binarized-Micro-NAS. Precision axis: FP32 → INT8 → Binary.
Targets: laptop (RTX 5070) → Raspberry Pi 5 → STM32H7.""")

md("""## 1. Setup
Import the project modules and load the central configuration (`config.yaml`).""")
code("""import sys, pathlib
root = pathlib.Path.cwd()
root = root if (root / "src").exists() else root.parent   # works from repo root or notebooks/
sys.path.insert(0, str(root))
import pandas as pd
from src.config import load_config
from src import experiment, export, measure_energy
cfg = load_config()
cfg["paths"]["artifacts_dir"]""")

md("""### Run scale
`QUICK=True` runs a fast smoke pass (small epochs / few NAS gens / capped
impressions) so the whole notebook executes in minutes. Set `QUICK=False` for
the full paper-grade run. Everything is driven from here.""")
code("""QUICK = True
if QUICK:
    cfg["train"]["distill_epochs"] = 2
    cfg["train"]["epochs"] = 2
    NAS = dict(generations=3, population=8, n_train=8000, n_val=2000, distill_epochs=1)
    EVAL_IMPR, QAT = 1000, 0
else:
    NAS = dict(generations=cfg["nas"]["generations"], population=cfg["nas"]["population"])
    EVAL_IMPR, QAT = None, 2
QUICK""")

md("""## 2. Datasets
MIND + xMIND are already downloaded (run `python -m src.download` to refresh).
Show the SHA256 manifest and basic statistics.""")
code("""import json, pathlib
manifest = json.load(open(pathlib.Path(cfg["paths"]["data_dir"]) / "manifest.json"))
print(len(manifest), "files pinned")
from src import data_mind, data_xmind
news = data_mind.read_news(cfg, "train")
print("MIND train news:", len(news), "| categories:", len(data_mind.category_vocab(news)))
print("xMIND langs on disk:", data_xmind.available_langs(cfg))""")

md("""## 3. Teacher anchors
Compute (cached) the frozen multilingual teacher's English-title embeddings.
These are the distillation targets shared across all 14 languages.""")
code("""experiment.ensure_anchors(cfg)""")

md("""## 4. Distill the byte-CNN student
Distil a language-agnostic byte-level student to the teacher anchors.
(Optional here — NAS distils per candidate; this trains the default student.)""")
code("""student_encoder = experiment.distill(cfg)""")

md("""## 5. Architecture search (3 arms)
One search space, three arms: `nas` (FP32, loose), `micro_nas` (INT8, STM32H7
constraints), `binarized_micro_nas` (Binary, constraints). Fitness = distillation
quality under feasibility.""")
code("""arms = experiment.search_arms(cfg, **{k: NAS[k] for k in NAS})
best = {arm: experiment.best_arch(res) for arm, res in arms.items()}
pd.DataFrame([{"arm": a, **best[a], "quality": arms[a][0]["quality"],
               "size_kb": arms[a][0]["size_kb"]} for a in arms])""")

md("""## 6. Train a recommender per arm
Train the end-to-end recommender (news encoder = each arm's best architecture)
on MINDsmall click behaviour.""")
code("""models = {arm: experiment.train_for_arch(cfg, best[arm]) for arm in best}""")

md("""## 7. Precision sweep → results matrix
Evaluate each trained model at FP32 / INT8 / Binary (PTQ + optional QAT), with
ranking metrics and footprint (size, RAM, MACs, energy).""")
code("""rows = []
for arm in models:
    rows += experiment.precision_sweep(cfg, models[arm], best[arm], arm,
                                       qat_epochs=QAT, max_eval=EVAL_IMPR)
matrix = experiment.save_matrix(cfg, rows)
matrix""")

md("""## 8. NRMS baseline (FP32 ceiling)
Reproduced MINDsmall-dev NRMS. External reference: ACL'20 MINDlarge-test
AUC 0.6776.""")
code("""baseline = experiment.baseline_row(cfg, epochs=cfg["train"]["epochs"], max_eval=EVAL_IMPR)
baseline""")

md("""## 9. Multilingual evaluation (all 14 languages)
Cross-lingual transfer: identical English impressions, news text swapped per
language through the SAME byte-level model. (Pick the deployment model.)""")
code("""deploy = models["micro_nas"]
lang_table = experiment.eval_languages(cfg, deploy, max_impressions=EVAL_IMPR or 2000)
lang_table""")

md("""## 10. Measured latency & energy (laptop)
GPU latency + NVML energy, and ONNX CPU latency (the same path runs on the Pi 5
aarch64 wheel; pair with an INA219 for real Pi/STM32 power).""")
code("""import torch
ex = torch.zeros(8, cfg["data"]["max_title_bytes"], dtype=torch.long)
enc = deploy.news_encoder
print("GPU latency ms:", round(measure_energy.latency_torch(enc, ex), 4))
print("Energy:", measure_energy.energy_nvml(enc, ex, n=200))""")

md("""## 11. Export deployment artifacts
ONNX content encoder + cold-start `topicWeights` prior, mapped to the app's
`edgeml_model_state_v1` schema.""")
code("""state = export.export_artifacts(cfg, deploy.news_encoder.cpu(),
                                ["en"] + data_xmind.available_langs(cfg))
print(state)""")

md("""## 12. Pareto: accuracy vs footprint
Visualise the accuracy/size and accuracy/energy trade-offs across the matrix.""")
code("""import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for arm in matrix["arm"].unique():
    s = matrix[matrix["arm"] == arm]
    ax[0].scatter(s["size_kb"], s["auc"], label=arm)
    ax[1].scatter(s["energy_uj"], s["auc"], label=arm)
ax[0].set(xlabel="size (KB)", ylabel="AUC"); ax[1].set(xlabel="energy (uJ/inf)", ylabel="AUC")
ax[0].legend(); plt.tight_layout(); plt.show()""")

md("""## 13. Conclusions
- Byte-level student fits the STM32H7 flash/RAM budget while serving all 14 languages.
- INT8 is the measured MCU sweet spot; Binary is footprint-motivated (analytical on MCU).
- See `paper/` for the write-up and `artifacts/` for exported models + matrix.""")

nb["cells"] = cells
out = Path("notebooks"); out.mkdir(exist_ok=True)
path = out / "MIND-Edge-Recommender.ipynb"
nbf.write(nb, path)
print("wrote", path, "with", len(cells), "cells")
