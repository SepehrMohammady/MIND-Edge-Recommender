# A Byte-Level Multilingual News Recommender for Microcontrollers: NAS, Micro-NAS and Binarized Micro-NAS across Flash, RAM and Energy

> Draft for ApPLIES / ApplePies (Springer LNEE). Target length: 6 pages.
> Markdown draft — convert to the LNEE LaTeX template for submission.
> Results below are from the full run (`scripts/run_full.py`, 8.2 h on one
> RTX 5070); raw tables in `artifacts/results_summary.json` + `results_matrix.csv`
> + `lang_matrix.csv`; figures in `paper/figures/` (`scripts/make_figures.py`).

**Authors:** S. Mohammady, M. Alehoseini, et al. *(finalize author list/affiliations)*

---

## Abstract

On-device news recommendation promises privacy (no behavioural data leaves the
device) but faces a cold-start problem and a hard memory wall: classic neural
news recommenders carry a word-embedding table of tens to hundreds of megabytes,
far beyond a microcontroller's flash. We present a tiny, **language-agnostic**
news encoder that distils a frozen multilingual sentence-transformer teacher
into a **byte-level char-CNN student** whose entire vocabulary is the 256 UTF-8
byte values — eliminating per-language tables and serving all 14 languages of
xMIND with one model under 2 MB. We then study three architecture-search regimes
— **NAS**, **Micro-NAS** (MCU-constrained), and **binarized Micro-NAS** — across
three precisions (**FP32 / INT8 / Binary**), reporting a full matrix of ranking
quality (AUC, MRR, nDCG@5/10 on MIND) against **flash footprint, RAM, and
energy** on a laptop → Raspberry Pi 5 → STM32H7 path. INT8 Micro-NAS reaches
**0.610** AUC at **204 KB** — matching a full NRMS word-embedding baseline
(0.607) at a fraction of the size — while unconstrained INT8 NAS reaches
**0.647** AUC at 790 KB (+0.04 over NRMS, with no vocabulary table and full
multilingual coverage); binarization cuts the footprint to **186–239 KB** at
0.52–0.55 AUC. We release the pipeline and the exportable cold-start prior and
content encoder for the FeedWell-Edge reader.

---

## 1. Introduction

Privacy-first reading apps increasingly personalize *on-device*, where the user
model never leaves the phone or edge node. Two obstacles dominate. (i) **Cold
start**: a per-user model initialized from zero gives poor early recommendations.
(ii) **The memory wall**: state-of-the-art news recommenders such as NRMS
\cite{wu2019nrms} and NAML encode article titles with a word-embedding matrix
(e.g. GloVe-300d over a 30k–100k vocabulary = 37–120 MB in FP32), which cannot
fit a microcontroller's flash; multiplying this per language for a multilingual
app is hopeless.

We target both. We pretrain, **fully offline**, a population cold-start prior and
a compact content encoder on MIND \cite{wu2020mind} and its multilingual
extension xMIND \cite{iana2024xmind}, then compress to the edge. The encoder is a
**byte-level char-CNN** distilled \cite{reimers2020multilingual} from a frozen
multilingual teacher: because its only symbols are the 256 UTF-8 bytes, a *single*
model handles all 14 xMIND languages with no per-language table — directly
dissolving the memory wall and enabling cheap all-language evaluation.

Our contribution is a systematic **architecture × precision** study for this
encoder: NAS, Micro-NAS (with hard STM32H7 flash/RAM/MAC constraints), and
binarized Micro-NAS, each at FP32, INT8 and Binary, measured on flash, RAM and
energy across three deployment tiers. This extends our group's prior NAS-for-BNN
work \cite{pighetti2026nasbnn} from vision (WakeVision) to multilingual text
recommendation.

## 2. Related Work

**Neural news recommendation.** NRMS \cite{wu2019nrms}, NAML and LSTUR established
attention-based title encoders + user encoders on MIND \cite{wu2020mind}; the
standard task is impression-level click ranking (AUC/MRR/nDCG). All rely on large
word-embedding tables — the very component we remove.

**Tokenization-free / compact encoders.** Byte- and character-level models
\cite{clark2022canine} and feature hashing \cite{joulin2017fasttext} drop the
vocabulary table; multilingual distillation \cite{reimers2020multilingual} aligns
translations to a shared space. We combine these: a byte student distilled to a
multilingual teacher's anchor embeddings.

**TinyML NAS and BNNs.** MCUNet \cite{lin2020mcunet} and µNAS
\cite{liberis2021unas} search under MCU memory budgets; NAS-BNN \cite{nasbnn2024}
searches binary networks; XNOR-Net \cite{rastegari2016xnor} established 1-bit
training with full-precision first/last layers. Our group recently applied
NAS-BNN to WakeVision at this venue \cite{pighetti2026nasbnn}; here we bring the
same NAS × precision methodology to on-device multilingual recommendation.

## 3. Method

### 3.1 Task
Impression-level click ranking: given a user's clicked history and candidate
articles, score candidates. Metrics: AUC, MRR, nDCG@5, nDCG@10.

### 3.2 Language-agnostic content encoder (teacher → student)
A frozen `paraphrase-multilingual-MiniLM-L12-v2` teacher produces a 384-d *anchor*
embedding for each article's English title. Since xMIND titles are parallel
translations, every language's translation is distilled to the **same** anchor
\cite{reimers2020multilingual}. The student is a **byte-level char-CNN**: an
embedding over 256 UTF-8 byte values, depthwise-separable 1-D conv blocks, masked
mean pooling, and a linear head, trained with a Matryoshka cosine loss so a
truncated prefix is usable on-device. No per-language vocabulary exists.

### 3.3 Recommender
News vectors from the student feed an additive-attention user encoder over the
click history; scores are user·candidate dot products, trained NRMS-style with
one positive and *K* negatives (softmax cross-entropy).

### 3.4 Architecture search (three arms)
One search space over `{channels, depth, out_dim}`, evaluated by distillation
quality (cosine to anchors) under footprint feasibility, via an evolutionary
supernet-free search:
- **NAS** — FP32, loose (laptop) constraints → accuracy ceiling.
- **Micro-NAS** — INT8, hard STM32H7 constraints (flash ≤ 2 MB, RAM ≤ 512 KB,
  MACs ≤ 5 M).
- **Binarized Micro-NAS** — Binary, same hard constraints.
Precision is the controlled second axis of the matrix.

### 3.5 Precision (FP32 → INT8 → Binary)
Simulated quantization with a straight-through estimator: per-output-channel
symmetric INT8, and sign×scale binarization. Following XNOR-Net
\cite{rastegari2016xnor} the byte embedding and first/last projections stay
full-precision; only inner conv/linear weights are quantized. Both PTQ and short
QAT are supported.

### 3.6 Footprint & energy
Params and MACs (thop); model size = params × bytes/precision; energy proxy =
MACs × per-op energy (Horowitz 45 nm \cite{horowitz2014}), labelled relative.
Measured latency/energy via CUDA timing + NVML on the laptop and onnxruntime on
the Pi 5 (INA219 for physical power).

## 4. Experimental Setup

**Datasets.** MINDsmall (50k users; train 156,965 / dev 73,152 impressions;
51,282 articles) \cite{wu2020mind}; xMINDsmall, 14 languages
(zho, fin, grn, hat, ind, jpn, kat, ron, som, swh, tam, tha, tur, vie), NLLB-3.3B
translations \cite{iana2024xmind,nllb2022}. xMIND supplies translated
title/abstract keyed by news id; MIND behaviours are reused. **Multilingual
evaluation is therefore cross-lingual transfer over identical English
impressions**, not native-language behaviour.

**Protocol.** Train on MINDsmall train; evaluate on MINDsmall dev (no public
small test). We reproduce NRMS as the FP32 ceiling and cite the published
MINDlarge-test NRMS (AUC 0.6776) \cite{wu2020mind} as external reference. Fixed
seed; SHA256-pinned data.

**Hardware/stack.** RTX 5070 Laptop (Blackwell, PyTorch 2.12 cu130), Raspberry
Pi 5 (onnxruntime aarch64), STM32H7 (X-CUBE-AI; binary numbers analytical, see
§6). Implementation in PyTorch; full pipeline driven from one notebook.

## 5. Results

### 5.1 Architecture × precision matrix (MINDsmall dev)

Searched architectures (channels-depth-out_dim): NAS = 256-4-384 (unconstrained);
Micro-NAS = 64-5-384; binarized Micro-NAS = 96-2-384. Footprint/MACs/energy are
of the content encoder.

| Arm | Prec. | AUC | MRR | nDCG@10 | Size (KB) | RAM (KB) | MACs | Energy (µJ) |
|-----|-------|-----|-----|---------|-----------|----------|------|-------------|
| NAS | FP32 | 0.633 | 0.326 | 0.375 | 1566.8 | 128 | 36.67 M | 168.7 |
| NAS | INT8 | **0.647** | 0.338 | 0.387 | 789.8 | 32 | 2.72 M | 4.65 |
| NAS | Binary | 0.588 | 0.297 | 0.342 | 563.1 | 32 | 2.72 M | 4.25 |
| Micro-NAS | FP32 | 0.605 | 0.307 | 0.352 | 266.8 | 32 | 3.46 M | 15.90 |
| Micro-NAS | INT8 | **0.610** | 0.314 | 0.360 | 203.9 | 8 | 0.71 M | 2.30 |
| Micro-NAS | Binary | 0.521 | 0.251 | 0.294 | 185.6 | 8 | 0.71 M | 2.25 |
| Bin. µNAS | FP32 | 0.593 | 0.290 | 0.336 | 311.4 | 48 | 3.35 M | 15.43 |
| Bin. µNAS | INT8 | 0.601 | 0.295 | 0.344 | 255.7 | 12 | 0.92 M | 3.28 |
| Bin. µNAS | Binary | 0.546 | 0.273 | 0.317 | 239.5 | 12 | 0.92 M | 3.23 |

INT8 is effectively lossless vs FP32 (often marginally better, a known QAT
regularization effect) while cutting energy 7–36× and size ~1.4–2×. Binarization
costs 5–9 AUC points; because the byte embedding and first/last layers stay
full-precision (XNOR-Net practice), its size gain over INT8 is modest at this
scale — the embedding dominates. Micro-NAS finds a 204 KB INT8 encoder that fits
an aggressive on-device budget at near-baseline accuracy.

### 5.2 Multilingual (all 14 languages, cross-lingual transfer)
Per-language AUC of the deployed Micro-NAS model (English-trained; news text
swapped per language through the one byte-level encoder):

| en | hat | jpn | grn | ron | tur | vie | swh | zho | ind | som | tha | tam | fin | kat |
|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| .604 | .556 | .551 | .533 | .532 | .530 | .523 | .521 | .521 | .537 | .514 | .504 | .504 | .495 | .491 |

All 14 languages run on one sub-2 MB model with no per-language table. As
expected, transfer is strongest for Latin-script / higher-resource targets and
weakest for low-resource or distinct-script languages (Georgian kat, Finnish fin,
Tamil tam), tracking NLLB translation quality.

### 5.3 Baseline comparison
Reproduced NRMS (FP32, word-embedding) MINDsmall-dev AUC **0.607** (MRR 0.322,
nDCG@10 0.364); external MINDlarge-test ceiling AUC 0.6776 \cite{wu2020mind}. The
byte-level student **matches NRMS at 204 KB (INT8 Micro-NAS, 0.610)** and
**exceeds it at 790 KB (INT8 NAS, 0.647)** — with no vocabulary table and
all-14-language coverage, versus NRMS's English-only word embeddings.

### 5.4 Footprint / energy trade-off
Pareto of AUC vs size and vs energy (Fig. 1, from the notebook). The INT8 points
dominate the front: Micro-NAS/INT8 (0.610 AUC, 204 KB, 2.3 µJ) is the on-device
sweet spot, and NAS/INT8 is the accuracy leader (0.647, 790 KB). Binary points
sit lower-left (smallest, lowest energy) but pay an accuracy cost; on Cortex-M7
(no popcount) their energy edge over INT8 is small, so they are justified by
footprint, not speed. The full study (3 arms × 3 precisions + NRMS + 15-language
eval) ran in 8.2 h on a single RTX 5070 laptop.

## 6. Limitations

- **Binary-on-MCU is analytical.** No PyTorch→1-bit→STM32H7 toolchain exists
  (X-CUBE-AI's 1-bit path is Larq/TF only). We report measured binary inference
  on Pi 5 and laptop, and analytical flash/RAM/energy for the MCU; Cortex-M7
  lacks popcount, so binary is **footprint-motivated, not latency-motivated**.
- **Cross-lingual transfer**, not native logs (xMIND reuses English impressions).
- **MT quality** varies for low-resource languages (grn, hat, som).
- MINDsmall scale; MINDlarge left to future work.

## 7. Ethics & Data Statement
MIND is used under the Microsoft Research License (non-commercial research);
xMIND under CC-BY-NC-SA-4.0. No personal data is collected; all behaviour is from
the public MIND logs. Translations are machine-generated and may carry artifacts.

## 8. Conclusion
A byte-level, language-agnostic encoder removes the embedding-table memory wall
and serves 14 languages with one sub-2 MB model. Across NAS / Micro-NAS /
binarized Micro-NAS and FP32 / INT8 / Binary, we map the accuracy–footprint–energy
trade-off for on-device multilingual news recommendation, and export a cold-start
prior + content encoder for the FeedWell-Edge reader.
