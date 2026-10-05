# MIND-Edge-Recommender

Offline pretraining pipeline for a **privacy-first, on-device news recommender**
on **MIND** (English) + **xMIND** (14 languages), with a **NAS × precision**
study of flash footprint, RAM and energy. Deployment targets are **STM32
microcontrollers** and an **Android phone** (the FeedWell-Edge RSS reader).

> **Scope: offline pretraining only.** This repository produces the model and
> the exportable artifacts: a cold-start topic prior and a content encoder
> (ONNX). The on-device continual-learning loop lives in
> [FeedWell-Edge](https://github.com/SepehrMohammady/FeedWell-Edge).

## Status (2026-10-05)

The June 2026 results were audited against the files on disk and the pipeline was
corrected. What holds today and what is pending:

| Item | State |
|------|-------|
| Ranking metrics of the June matrix (Table 1 of the paper) | measured, single unseeded runs; kept as they are |
| Size, MACs, energy of that matrix | recomputed with a per-layer counter (the first count skipped quantized layers) |
| INT8 and 1-bit rows | weight-only simulation of the inner convolutions; activations and outer layers stay FP32 |
| Integer (INT8) and 1-bit model files | not produced yet |
| Board and phone measurements | not done yet (ST Edge AI Developer Cloud for STM32, then Android) |
| Constrained search | flash and RAM budgets enforced; rerun pending with the corrected MAC count and the lab µNAS |
| Multilingual transfer of the distilled encoder | being measured (`scripts/run_p1.py`) |
| Strong NRMS baseline (GloVe) | being measured (`scripts/run_p1.py`) |

`LOGBOOK.md` records what was done and how long it took;
`paper/results/experiments.jsonl` holds one record per run.

---

## Research design

| Axis | Values |
|------|--------|
| **Architecture** | NAS → Micro-NAS (budget-constrained) → binarized Micro-NAS |
| **Precision** | FP32 → INT8 → 1-bit |
| **Targets** | STM32H7 / STM32F4 boards · Android phone (laptop RTX 5070 for training) |
| **Metrics** | Ranking: AUC, MRR, nDCG@5, nDCG@10 · Cost: params, MACs, size, latency, energy |

**Content encoder (the flash-footprint argument):** a word-embedding table costs
37–120 MB per language. Here a title is read as its UTF-8 bytes, so the
vocabulary is one 256-row table shared by all languages, followed by a small
depthwise-separable 1D CNN. A frozen multilingual sentence-transformer
(`paraphrase-multilingual-MiniLM-L12-v2`) provides distillation targets that
place a translated title near its English original.

**Reference points on MINDsmall dev** (`paper/results/`):

| Scorer | AUC | Storage |
|--------|----:|---------|
| Popularity | 0.525 | – |
| Per-user category histogram, no network | 0.598 | 17 counters |
| Per-user subcategory histogram, no network | 0.617 | – |
| NRMS, reduced (random 256-d word table), June run | 0.607 | 27 MB |
| Byte-CNN, constrained arm, 8-bit weights, June run | 0.610 | 204 KB (estimate) |
| Byte-CNN, unconstrained arm, 8-bit weights, June run | 0.647 | 790 KB (estimate) |
| NRMS with GloVe (literature: DIGAT, Findings of EMNLP 2022) | 0.656 | – |

> **Multilingual caveat:** xMIND ships machine-translated text (NLLB-3.3B) and
> reuses MIND's English click logs. "Multilingual evaluation" is therefore
> cross-lingual transfer over identical English impressions, not native-language
> behaviour.

> **Quantization caveat:** INT8 and 1-bit are simulated on weights with a
> straight-through estimator. Sizes are parameter counts at each layer's
> precision, and the energy column is a per-operation proxy (Horowitz, ISSCC
> 2014), useful as an ordering only.

> **Binary on a microcontroller:** needs a binary runtime. The lab routes are
> CBin-NN and Larq Compute Engine on TensorFlow Lite Micro; ST Edge AI Core 4.0
> marks its Larq import as deprecated. Nothing has been measured on a board yet.

---

## Quick start

```powershell
# 1. One-shot setup: virtual environment + datasets + PyTorch cu130 + ML stack
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

# 2. Open the master notebook (kernel "MIND-Edge-Recommender (.venv)")
#    notebooks/MIND-Edge-Recommender.ipynb
```

Everything is controlled from `config.yaml` and the notebook. Useful commands:

```powershell
python -m src.download                 # MIND + xMIND + NRMS utils, SHA256 manifest
python -m scripts.run_p1               # seeded design experiments (teacher bound, baselines, multilingual)
python -m scripts.recompute_costs      # cost columns of the June matrix
python -m scripts.make_figures         # paper figures from paper/results
python -m scripts.check_numbers        # every number in paper.tex against paper/results
python -m scripts.log_report p1/       # runs, durations, results, GPU use
```

**CPU-only / Linux:** swap the PyTorch wheel in `requirements.txt` for the
appropriate index URL.

---

## Repository layout

```
config.yaml              # single control surface (all knobs)
LOGBOOK.md               # dated journal: what was done, how long it took
src/
  config.py              # config loader, per-run output folders
  download.py            # MIND + xMIND + utils download, SHA256 manifest
  data_mind.py           # parse news.tsv / behaviors.tsv
  data_xmind.py          # join translated text on nid
  teacher.py             # frozen multilingual teacher embeddings (cached)
  student.py             # byte-level char-CNN + distillation
  recommender.py         # recommender, training loop, batched evaluation
  baseline_nrms.py       # NRMS: GloVe reference configuration and reduced variant
  heuristics.py          # popularity and category-histogram scorers
  nas/                   # search space, evolutionary search
  quantize.py            # simulated INT8 / 1-bit weights
  binary.py              # ReActNet / Bi-Real binary encoder
  metrics.py             # AUC, MRR, nDCG@5/10 per impression, history buckets
  footprint.py           # per-layer params, MACs, size, energy proxy
  export.py              # ONNX export + cold-start topic prior
  measure_energy.py      # laptop latency (GPU, onnxruntime CPU)
  runlog.py, seed.py     # run records, logbook, seeding
scripts/                 # run_full, run_p1, run_binary, recompute_costs, make_figures, check_numbers, ...
notebooks/MIND-Edge-Recommender.ipynb
paper/                   # LaTeX source (LLNCS), figures, references.bib
paper/results/           # frozen result files quoted by the paper (versioned)
artifacts/               # run outputs and caches (not versioned)
data/                    # downloaded datasets (not versioned)
docs/                    # June run log, data manifest, binary deployment notes
course/                  # Farsi course on the project
```

## Exported artifacts

`src/export.py` writes, into the folder of the run that calls it:

| File | Description |
|---|---|
| `content_encoder*.onnx` | byte-CNN encoder, FP32 graph (`title_bytes` int64 → 384-d vector) |
| `edgeml_*.json` | state for the app: population category prior, encoder settings, languages |
| `models_manifest.json` | list of the exported encoders |

The ONNX files carry FP32 weights. A full-integer INT8 export for STM32 and
Android, with accuracy measured from the exported file, is the next step; until
then no file in this repository is a deployed INT8 or 1-bit model.

---

## Datasets & licenses

| Dataset | License | Source |
|---------|---------|--------|
| **MIND** | Microsoft Research License (non-commercial research) | [HuggingFace mirror](https://huggingface.co/datasets/Recommenders/MIND) — Wu et al., ACL 2020 |
| **xMIND** | CC-BY-NC-SA-4.0 | [HuggingFace](https://huggingface.co/datasets/aiana94/xMINDsmall) — Iana et al., SIGIR 2024 |

Both datasets are for non-commercial research only; cite both when publishing
results.

## Reproducibility

* Every run script seeds Python, NumPy and PyTorch from `seed` in `config.yaml`.
  The June matrix run did not seed PyTorch; seeded reruns differ from it by up
  to 0.02 AUC.
* `docs/data_manifest.json` pins 31 downloaded files by SHA256.
* `requirements-lock.txt` pins the installed package versions.
* PyTorch cu130 wheels (CUDA 13.0, Blackwell sm_120).
* `scripts/check_numbers.py` fails if a number in the paper differs from
  `paper/results`.

## Citation

```bibtex
@incollection{mind-edge-recommender2026,
  title   = {A Byte-Level Multilingual News Recommender for Microcontrollers},
  note    = {In preparation},
  year    = {2026},
}
```

## Key references

NRMS (Wu et al., EMNLP 2019) · MIND (Wu et al., ACL 2020) · xMIND (Iana et al.,
SIGIR 2024) · multilingual distillation (Reimers & Gurevych, EMNLP 2020) ·
char-CNN (Zhang et al., NeurIPS 2015) · CANINE (Clark et al., TACL 2022) ·
MCUNet (Lin et al., NeurIPS 2020) · NAS-BNN (Lin et al., Pattern Recognition
2025) · NAS for BNNs on WakeVision (Pighetti et al., ApplePies 2025, LNEE 1553) ·
CBin-NN (Sakr et al., Electronics 2024) · Horowitz (ISSCC 2014).
