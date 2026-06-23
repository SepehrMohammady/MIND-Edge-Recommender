# MIND-Edge-Recommender

Offline pretraining pipeline for a **privacy-first, on-device news recommender**
on **MIND** (English) + **xMIND** (14 languages), with a systematic
**NAS × precision** study of **flash footprint, energy, and RAM** for
deployment across **RTX 5070 (Blackwell) → Raspberry Pi 5 → STM32H7**.

> **Scope: offline pretraining only.** This repository produces the model and
> the exportable deployment artifacts — (a) a cold-start topic prior and (b) a
> quantized content encoder (ONNX) — ready to be embedded in a downstream
> on-device reader. The on-device continual-learning loop is out of scope here.

---

## Research design (locked)

| Axis | Values |
|------|--------|
| **Architecture** | NAS → Micro-NAS (µNAS, MCU-constrained) → binarized-µNAS |
| **Precision** | FP32 → INT8 (PTQ/QAT) → Binary (BNN) |
| **Targets** | RTX 5070 Blackwell GPU · Raspberry Pi 5 (aarch64) · STM32H7 (Cortex-M7) |
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

> **Multilingual data:** xMIND translates MIND's English titles and abstracts to
> 14 languages using **NLLB-3.3B** (Meta's No Language Left Behind, 3.3B-param
> Transformer for 200+ languages). Translation quality varies: languages with
> abundant NLLB training data (Chinese, Japanese) → high-quality translations
> → better performance; low-resource languages (Georgian, Tamil) → lower-quality
> translations → lower AUC. xMIND reuses MIND's English click logs, so
> "multilingual evaluation" is **cross-lingual transfer over identical English
> impressions**, not native-language behavior. To extend xMIND with new languages
> (e.g., Italian, Farsi), run NLLB directly on MIND text and retrain encoder.

> **Multilingual caveat:** xMIND ships translated text only and reuses MIND's
> English click logs. "Multilingual evaluation" is therefore **cross-lingual
> transfer over identical English impressions**, not native-language behavior.

> **Binary-on-MCU caveat:** there is no PyTorch→1-bit→STM32H7 toolchain
> (X-CUBE-AI's 1-bit path is Larq/TF only). Binary MCU numbers are
> **analytical** (bit-packed weights + Horowitz-2014 energy proxy); real binary
> inference is measured on Pi 5 (Larq Compute Engine) and the RTX 5070. INT8 is
> the honest measured MCU sweet spot.

---

## Quick start

```powershell
# 1. One-shot setup: virtual environment + datasets + PyTorch cu130 + ML stack
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

# 2. Open the master notebook
#    notebooks/MIND-Edge-Recommender.ipynb
#    (select the "MIND-Edge-Recommender (.venv)" kernel)
```

Everything is controlled from `config.yaml` and the notebook. To re-download or
change the language set, edit `config.yaml` then `python -m src.download`.

**CPU-only / Linux:** swap the PyTorch wheel in `requirements.txt` for the
appropriate index URL (`cpu`, `rocm`, or the default pip index for aarch64).

---

## Repository layout

```
config.yaml              # single control surface (all knobs)
requirements*.txt        # core (data) / full (ML) / lock (frozen)
scripts/setup_env.ps1    # virtual env + staged install + dataset download
src/
  config.py              # config loader
  download.py            # MIND + xMIND download, SHA256-pinned
  data_mind.py           # parse news.tsv / behaviors.tsv, build histories
  data_xmind.py          # join translated text on nid
  teacher.py             # frozen multilingual teacher embeddings
  student.py             # byte-level char-CNN + distillation loss
  baseline_nrms.py       # NRMS FP32 baseline
  nas/                   # search space, evolutionary search
  quantize.py            # FP32 → INT8 → Binary
  metrics.py             # AUC, MRR, nDCG@5/10 (impression-level)
  footprint.py           # params, MACs, size, Horowitz energy proxy
  export.py              # ONNX export + cold-start topicWeights JSON
  measure_energy.py      # NVML (GPU) / onnxruntime (CPU/Pi 5) energy
notebooks/MIND-Edge-Recommender.ipynb  # end-to-end pipeline notebook
artifacts/               # exported models, priors, results tables
paper/                   # LaTeX source (LLNCS), figures, references.bib
data/                    # downloaded datasets (gitignored)
```

## Deployment artifacts

`src/export.py` produces one set of files per (arm, precision) combination. A full run exports all 9 combinations (3 arms × 3 precisions):

| File pattern | Description |
|---|---|
| `content_encoder_{arm}_{precision}.onnx` | Byte-CNN encoder for that variant |
| `edgeml_{arm}_{precision}.json` | State schema: topic prior, encoder config, metrics |
| `models_manifest.json` | Comparison table for all exported models |

**Recommended on-device sweet spot:** `content_encoder_micro_nas_int8.onnx` — 204 KB, AUC 0.610.

```python
import onnxruntime as ort, numpy as np, json

# Load the recommended model
state = json.load(open("artifacts/edgeml_micro_nas_int8.json"))
sess  = ort.InferenceSession("artifacts/content_encoder_micro_nas_int8.onnx",
                              providers=["CPUExecutionProvider"])
# title_bytes: int64 array of UTF-8 byte ids, shape (1, max_title_bytes)
emb = sess.run(None, {"title_bytes": np.zeros((1, 64), dtype=np.int64)})[0]
# emb.shape → (1, 384)
```

To export all models from the notebook:
```python
from src.export import export_all
manifest = export_all(cfg, models={
    ("nas",       "fp32"):   model_nas_fp32,
    ("nas",       "int8"):   model_nas_int8,
    ("nas",       "binary"): model_nas_bin,
    ("micro_nas", "fp32"):   model_mnas_fp32,
    ("micro_nas", "int8"):   model_mnas_int8,   # ← sweet spot
    ("micro_nas", "binary"): model_mnas_bin,
    ("bin_unas",  "fp32"):   model_bunas_fp32,
    ("bin_unas",  "int8"):   model_bunas_int8,
    ("bin_unas",  "binary"): model_bunas_bin,
}, langs=cfg["languages"], metrics_table=results)
```

`student.pt` contains the PyTorch checkpoint for re-training or re-export.

---

## Datasets & licenses

| Dataset | License | Source |
|---------|---------|--------|
| **MIND** | Microsoft Research License (non-commercial research) | [HuggingFace](https://huggingface.co/datasets/Recommenders/MIND) — Wu et al., ACL 2020 |
| **xMIND** | CC-BY-NC-SA-4.0 | [HuggingFace](https://huggingface.co/datasets/aiana94/xMINDsmall) — Iana et al., SIGIR 2024 |

Both datasets are for non-commercial research only; cite both when publishing
results. Translations are machine-generated (NLLB-3.3B); quality varies for
low-resource languages (grn, hat, som).

---

## Reproducibility

* Fixed `seed` in `config.yaml`; SHA256 manifest in `data/manifest.json`.
* `requirements-lock.txt` pins exact installed package versions after setup.
* PyTorch cu130 wheels (CUDA 13.0, Blackwell sm_120);
  verify with `torch.cuda.get_device_capability() == (12, 0)`.
* CPU-only: replace the PyTorch index URL in `requirements.txt`.

---

## Citation

If you use this code or the artifacts in your research, please cite:

```bibtex
@incollection{mind-edge-recommender2026,
  title   = {A Byte-Level Multilingual News Recommender for Microcontrollers:
             NAS, Micro-NAS and Binarized Micro-NAS across Flash, RAM and Energy},
  note    = {To appear},
  year    = {2026},
}
```

---

## Key references

NRMS (Wu, EMNLP'19) · MIND (Wu, ACL'20) · xMIND (Iana, SIGIR'24) ·
multilingual distillation (Reimers & Gurevych, EMNLP'20) · CANINE (Clark'22) ·
NAS-BNN (arXiv:2408.15484) · NAS-for-BNN @ ApplePies (LNEE vol 1553,
DOI 10.1007/978-3-032-17174-0_12) · Horowitz (ISSCC'14).
