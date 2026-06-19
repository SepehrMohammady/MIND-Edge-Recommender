# MIND-Edge-Recommender — Tiny Multilingual News Recommender (offline pretraining)

Offline pretraining of a **privacy-first, on-device news recommender** on
**MIND** (English) + **xMIND** (14 languages), with a systematic
**NAS × precision** study of **flash footprint, energy, and RAM** for
deployment on **laptop → Raspberry Pi 5 → STM32H7**.

Part of a PhD project; target venue **ApPLIES / ApplePies** (Springer LNEE,
Scopus/WoS).

> **Scope: offline pretraining only.** This repo produces the model and the
> exportable deployment artifacts — (a) a cold-start topic prior and (b) a
> quantized content encoder (ONNX) — ready to be embedded in a downstream
> on-device reader. The on-device continual-learning loop is out of scope here.

---

## Research design (locked)

| Axis | Values |
|------|--------|
| **Architecture** | NAS → Micro-NAS (µNAS, MCU-constrained) → binarized-µNAS |
| **Precision** | FP32 → INT8 (PTQ/QAT) → Binary (BNN) |
| **Targets** | Laptop (RTX 5070) · Raspberry Pi 5 · STM32H7 |
| **Metrics** | Ranking: AUC, MRR, nDCG@5, nDCG@10 · Cost: params, MACs, size (MB), latency (ms), energy (mJ) |

**Baseline (FP32 ceiling):** NRMS (title-only, easiest to quantize).
Reference ACL'20 MINDlarge-test: AUC 0.6776 / MRR 0.3305 / nDCG@5 0.3594 /
nDCG@10 0.4163. We additionally reproduce our own **MINDsmall-dev** NRMS.

**Content encoder (the flash-footprint solution):** a frozen multilingual
sentence-transformer **teacher** (`paraphrase-multilingual-MiniLM-L12-v2`)
distills into a tiny **byte-level char-CNN student**. The 256-symbol UTF-8
alphabet is shared by all 14 languages → **no per-language vocabulary table**
(which would be 37–120 MB and never fit an MCU), and all 14 xMIND languages run
through one model.

> **Multilingual caveat:** xMIND ships translated text only and reuses MIND's
> English click logs. "Multilingual evaluation" is therefore **cross-lingual
> transfer over identical English impressions**, not native-language behavior.

> **Binary-on-MCU caveat:** there is no PyTorch→1-bit→STM32H7 toolchain
> (X-CUBE-AI's 1-bit path is Larq/TF only). Binary MCU numbers are
> **analytical** (bit-packed weights + Horowitz-2014 energy proxy); real binary
> inference is measured on Pi 5 (Larq Compute Engine) and the laptop. INT8 is
> the honest measured MCU sweet spot.

---

## Quick start

```powershell
# 1. One-shot setup: venv + datasets (first) + torch cu130 + ML stack
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

# 2. Open the master notebook with the "MIND-Edge-Recommender (.venv)" kernel
#    notebooks/MIND-Edge-Recommender.ipynb
```

Everything is controlled from `config.yaml` and the notebook. To re-download or
change the language set, edit `config.yaml` then `python -m src.download`.

---

## Repository layout

```
config.yaml              # single control surface (all knobs)
requirements*.txt        # core (data) / full (ML) / lock (frozen)
scripts/setup_env.ps1    # venv + staged install + dataset download
src/
  config.py              # config loader
  download.py            # MIND + xMIND download, SHA256-pinned
  data_mind.py           # parse news.tsv / behaviors.tsv, build histories
  data_xmind.py          # join translated text on nid
  teacher.py             # frozen multilingual teacher embeddings
  student.py             # byte-level char-CNN + distillation
  baseline_nrms.py       # NRMS FP32 baseline
  nas/                   # search space, supernet, evolutionary search
  quantize.py            # FP32 -> INT8 -> Binary
  metrics.py             # AUC, MRR, nDCG@5/10 (impression-level)
  footprint.py           # params, MACs, size, Horowitz energy proxy
  export.py              # ONNX/TFLite + cold-start topicWeights JSON
notebooks/MIND-Edge-Recommender.ipynb  # short cells, one markdown note per code cell
artifacts/               # exported models, priors, results tables
paper/                   # LNEE draft + figures + references.bib
data/                    # downloaded datasets (gitignored)
```

---

## Datasets & licenses

* **MIND** — Microsoft Research License (non-commercial research). Mirror:
  `huggingface.co/datasets/Recommenders/MIND`. Wu et al., ACL 2020.
* **xMIND** — CC-BY-NC-SA-4.0. `huggingface.co/datasets/aiana94/xMINDsmall`.
  14 langs (NLLB-3.3B translations). Iana et al., SIGIR 2024 (arXiv:2403.17876).

Both are non-commercial research only; cite both. Translations are
machine-generated (quality varies for low-resource langs: grn, hat, som).

---

## Reproducibility

* Fixed `seed` in `config.yaml`; SHA256 manifest in `data/manifest.json`.
* `requirements-lock.txt` pins exact installed versions after setup.
* PyTorch cu130 wheels (Blackwell sm_120); verify `get_device_capability()==(12,0)`.

## Key references

NRMS (Wu, EMNLP'19) · MIND (Wu, ACL'20) · xMIND (Iana, SIGIR'24) ·
multilingual distillation (Reimers & Gurevych, EMNLP'20) · CANINE (Clark'22) ·
NAS-BNN (arXiv:2408.15484) · NAS-for-BNN @ ApplePies (LNEE vol 1553,
DOI 10.1007/978-3-032-17174-0_12) · Horowitz (ISSCC'14).
