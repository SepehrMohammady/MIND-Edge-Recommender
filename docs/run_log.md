# Experiment log — timings + model numbers (for the paper)

Hardware: Intel Core Ultra 9 275HX (24c), NVIDIA RTX 5070 Laptop GPU (8 GB),
32 GB RAM, Windows 11, PyTorch 2.12.1+cu130.

## Full paper run (`scripts/run_full.py`)
- Wall-clock: **8.2 h** (`artifacts/results_summary.json`, runtime_min = 494.4).
- Searched architectures: NAS 256-4-384, Micro-NAS 64-5-384, Bin.µNAS 96-2-384.
- Matrix numbers in `artifacts/results_matrix.csv`; baseline NRMS dev AUC 0.607.

## NRMS FP32 vs INT8 (`scripts/nrms_int8.py`, 8 epochs)
- FP32 AUC 0.6045, INT8 AUC 0.6047 → ΔAUC < 0.001 (same model, 27 MB vs 6.8 MB).

## Improved binary (`scripts/run_binary.py`, ReActNet+Bi-Real, distilled-init)
- Micro-NAS shape (64-5-384): binary AUC **0.572** (vs naive 0.521).

## Multi-seed — Micro-NAS INT8 (FULL protocol: full 156k-train, 8+2 epochs, full 73k-dev eval)
- seeds 42/1/2 AUC: **0.6302, 0.6216, 0.6209** → mean **0.624**, std **0.004**.
- NOTE: ~8 h for these 3 seeds alone (full-dev eval over 73k impressions is the
  bottleneck). Switched to the fast protocol below for the remaining configs.

## Reviewer experiments (`scripts/run_reviews.py`, FAST protocol)
Fast protocol: 40k-impression train subset, 6 train epochs (+1 QAT),
8k-impression dev eval; binary distil on a 120k-title subset, 8 epochs. Used for
variance estimation and the Matryoshka / cold-start checks (not the headline
matrix numbers). Per-stage timing appended below by the script.

<!-- run_reviews appends timed entries here -->
- [7.7m] multiseed micro_int8 seed 42: AUC=0.5828 (7.7m)
- [15.4m] multiseed micro_int8 seed 1: AUC=0.5641 (7.7m)
- [23.1m] multiseed micro_int8 seed 2: AUC=0.5692 (7.7m)
- [23.1m] multiseed micro_int8: mean=0.572 std=0.0079
- [151.2m] multiseed nas_int8 seed 42: AUC=0.6206 (128.1m)
- [279.4m] multiseed nas_int8 seed 1: AUC=0.6 (128.1m)
- [1.0m] binary: distill data built
- [16.4m] binary micro seed 42: AUC=0.5172 (15.4m)
- [31.8m] binary micro seed 1: AUC=0.5398 (15.3m)
- [47.1m] binary micro seed 2: AUC=0.5462 (15.4m)
- [47.1m] binary micro: mean=0.5344 std=0.0124
- [47.2m] matryoshka: {'64': 0.459, '128': 0.4577, '256': 0.4555, '384': 0.4539}
- [54.0m] coldstart: with_history=0.5908 masked=0.5 topic_prior=0.551
- [54.0m] REVIEWS2 DONE
