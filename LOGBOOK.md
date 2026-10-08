# Logbook — MIND-Edge-Recommender

Dated journal of what was done, why, how long it took and what came out. Machine-readable
run records (settings, seed, commit, minutes, metrics) are in
`paper/results/experiments.jsonl`; `python -m scripts.log_report` prints them as a table.
The June 2026 experiments are in `docs/run_log.md`. Times are local (Europe/Rome).

## 2026-10-05 15:30 — Audit of the June work (repos, paper, artifacts)

Work on the project stopped on 2026-07-02. Before restarting, the main repository, the
FeedWell-Edge app, the course project and the thesis outline were read against each other
and against the files on disk. About one hour (15:20 to 16:20), with a 39-minute literature
scan running in parallel (about 300 page fetches; every reference checked on arXiv, Crossref
or the publisher page).

What exists
- Offline pipeline on MINDsmall + xMIND (14 languages): teacher anchors, byte-CNN student,
  three-arm search, simulated FP32 / INT8 / 1-bit, NRMS baseline, ONNX export, laptop latency.
  Real data throughout.
- Paper draft (LLNCS, 6 pages), Farsi course (7 lessons), master notebook.
- FeedWell-Edge v2.0.5 (2026-05-15): local event log, per-topic weights, dual-EMA drift
  flag, ranking boost, telemetry export. No neural model and no inference runtime in the app.

What the audit found (each item checked on disk)
1. Fig. 2 of the paper had been drawn from a QUICK smoke run. The notebook run of
   2026-06-23 overwrote `artifacts/results_matrix.csv`; `make_figures.py` read that file on
   2026-07-02. Table 1 itself comes from the full run of 2026-06-18 (8.2 h).
2. `thop` skipped the custom quantized layers, so INT8 and binary rows reported the MACs of
   their full-precision layers only. Energy ratios in the text (7 to 36 times) were too
   large, and the 2 M MAC bound of the constrained search was checked on wrong counts: the
   two constrained winners have 3.29 M and 3.26 M MACs.
3. The nine models of Table 1 were trained from a random initialisation on English clicks.
   The distilled student was used only as search fitness and in the ablation. The
   per-language numbers come from that scratch FP32 model (mean of 14 languages 0.522).
4. "INT8" and "binary" are weight-only fake quantization of the inner convolutions; 34 to
   76 % of the parameters and all activations stay FP32. Sizes are computed, not file sizes.
5. The ReActNet binary model (0.572) was quoted with the size of the naive model (186 KB);
   its own estimate is 194 KB.
6. `run_full.py` never seeded torch. Three seeded reruns of Micro-NAS INT8 scored 0.630,
   0.622, 0.621 against 0.610 in the matrix run.
7. The NRMS baseline is a reduced variant (random 256-d table, 0.607).
8. The search space has 120 architectures; the search is a small evolutionary loop, not the
   lab µNAS.
9. `binary_op` was 0.0072 pJ while its comment said fp32_mac / 64 (0.0719).
10. `references.bib`: wrong first author for NAS-BNN; the on-device survey entry pointed to
    the tutorial record.
11. Full-run checkpoints were never saved, so Table 1 cannot be re-evaluated without
    retraining.

Literature scan, main points
- No earlier news recommender that reads raw bytes and none on a microcontroller was found
  (arXiv and IEEE Xplore; ACM DL could not be searched). Closest: letter-trigram hashing
  (Elkahky, WWW 2015), char-CNN click prediction (Edizel, SIGIR 2017), PRADO (200 KB).
- NRMS with GloVe on MIND-small: 0.656 AUC (DIGAT, Findings of EMNLP 2022, Table 1).
- Frozen multilingual encoder on xMINDsmall: NaSE 64.19 English, 63.79 mean of 14 languages.
- ApplePies 2026 took place in Bologna on 10–11 September 2026 (deadline was 3 July).
- ST Edge AI Core 4.0.0 marks Larq layers deprecated; quantized ONNX (QDQ) and INT8 TFLite
  are the supported 8-bit routes.

## 2026-10-05 16:20 — Decisions (Sepehr)

- Skip the Raspberry Pi 5 for now. Measure on STM32 boards (ST Edge AI Developer Cloud,
  real boards for what the cloud cannot do) and on an Android phone.
- The paper is not submitted. Target ApplePies 2027 or another conference or journal.
- The thesis built on this work is the MSc thesis of Houriyeh Emadoleslami.
- FeedWell-Edge stays a separate app from FeedWell (Google Play). Rebuild it on the latest
  FeedWell code, keep the 2.x version line.
- Search and deployment move to the lab µNAS fork with INT8 TFLite output.
- Log everything: what was done, how long it took.
- Watch GPU use and adjust the configuration to it.

## 2026-10-05 16:54 — P0: cost model, seeding, run folders (commit f5b7a0a)

Written and validated between 16:23 and 16:54.

- `src/footprint.py`: per-layer MAC count with forward hooks, any Conv1d/Linear subclass
  included; energy summed per layer at its weight precision; batch normalisation folded.
- `config.yaml`: `binary_op` 0.0719 pJ; new `cache_dir` and `results_dir`.
- `src/seed.py` and `seed_everything()` in every run script.
- `src/recommender.py`: negatives redrawn at every step, each distinct title of a batch
  encoded once, index tensors on the GPU, batched evaluation. `src/metrics.py`: rank-based
  AUC (same value as scikit-learn, ties count one half).
- `src/baseline_nrms.py`: reference GloVe configuration (`MINDsmall_utils.zip`) next to the
  reduced June variant.
- New: `src/heuristics.py`, `src/runlog.py`, `scripts/run_p1.py`.
- Run outputs go to `artifacts/runs/<name>`, teacher anchors to `artifacts/cache`. The June
  files were moved to `artifacts/legacy/` (full run of 06-18, QUICK run of 06-23).
- Data manifest now pins 31 files (3 MIND archives, 28 xMIND parquet files); tracked copy
  in `docs/data_manifest.json`.

Checks and timings (RTX 5070 Laptop GPU)

| check | result |
|---|---|
| fast AUC against scikit-learn, 2000 random cases with ties | largest difference 2e-16 |
| new evaluation against the June implementation, same weights | largest metric difference 2.4e-6 |
| full dev evaluation, 73,152 impressions | 12.9 s |
| one training step, 64-5-384, batch 64 | 29.9 ms, 1.8 min per epoch (236,344 instances) |
| one training step, 256-4-384, batch 64 | 149.9 ms, 9.2 min per epoch |
| distinct titles encoded per step | 1,499 of 3,520 slots |
| 15-epoch distillation of 64-5-384 (769k titles, 15 languages) | 4.9 min |
| student-to-teacher cosine on dev after distillation | en 0.468, ron 0.464, ind 0.466, fin 0.444, tur 0.455, jpn 0.416, zho 0.400, tha 0.381, tam 0.353, kat 0.345 |

Windows Smart App Control blocked a pandas DLL in `.venv` once during the audit
("An Application Control policy has blocked this file"); later imports worked. If a run
dies at import with that message, start it again.

## 2026-10-05 17:03 — Cost columns recomputed, Fig. 2 redrawn (commit c5957fa)

- `scripts/recompute_costs.py` rewrote size, MACs, RAM proxy and energy of the June matrix
  (ranking metrics untouched). FP32 to INT8 energy ratio: 3.4 to 9.3 times. ReActNet binary
  encoder: 109,248 parameters, 8.41 M MACs (7.86 M one-bit), 194.25 KB, 3.09 µJ.
- `scripts/make_figures.py` reads `paper/results` only.
- `references.bib` corrected and extended (7 entries, each checked on Crossref or arXiv).

First P1 records

| scorer, full MINDsmall dev | AUC | MRR | nDCG@10 |
|---|--:|--:|--:|
| popularity | 0.5254 | 0.2553 | 0.2991 |
| per-user category histogram | 0.5976 | 0.3070 | 0.3536 |
| per-user subcategory histogram, category back-off | 0.6172 | 0.3412 | 0.3840 |
| frozen teacher vectors + attention user encoder, seed 42 | 0.5888 | – | – |

The subcategory histogram, which has no network, is above the reduced NRMS (0.607) and the
June Micro-NAS INT8 model (0.610) on all three metrics. The frozen teacher scores 0.5748 on
average over the 14 translations, 0.014 below its English score.

## 2026-10-05 17:04 — P1 batch running in two processes

- Process A: seeds 42 then 2. Process B: seed 1. Stages per seed: teacher bound, NRMS (GloVe
  and reduced), student (scratch / distilled, English / mixed-language clicks), all evaluated
  in 15 languages. Logs in `logs/p1_seed*.log`.
- A first single-process launch at 16:54 was stopped at 17:03 to split the work; its two
  finished records (heuristics, frozen teacher seed 42) are kept.
- Memory limits the parallelism: each process holds about 4.3 GB and the machine was at 81 %
  with two. GPU memory is not the limit (2.6 of 8 GB).
- `logs/gpu_monitor.csv`: nvidia-smi sample every 30 s from 17:08.
- Expected duration: about 1.9 h per seed and process.

## 2026-10-05 17:12 — Paper synchronised with the result files (commit 9440b80)

- `paper/paper.tex` rewritten where it disagreed with the artifacts: abstract, method
  (weight-only quantization, what distillation is used for), setup (budgets enforced, seeds),
  Table 1 energy column, baselines paragraph (category and subcategory histograms, GloVe
  figure from the literature), language paragraph, limitations. Pending items carry red TODO
  markers. The Pi 5 is no longer named as a target.
- `scripts/check_numbers.py` recomputes about 50 quoted values from `paper/results` and fails on
  any mismatch. It caught two rounding slips and one wrong ratio in the first draft of the
  new text.
- The PDF is 8 pages (body 6, references 2). It will be cut to the page limit of the chosen
  venue at the final pass.
- README rewritten around a status table; `docs/binary_deployment.md` now lists the three
  routes to a 1-bit model on a board; `docs/run_log.md` says which June numbers still stand.
- Course: energy values of lesson 1 and the binary size of lesson 5 corrected, review notice
  added to the index and to lesson 1. Lessons 6 and 7 (cold start, Pi 5 quiz, "14 languages"
  badges) wait for the P1 results.
- Not done yet: the master notebook was not re-executed (two training processes hold the
  RAM); `scripts/make_notebook.py` is updated and the notebook will be regenerated when the
  batch ends.
- More P1 records while writing: frozen teacher with a trainable linear map, seed 42: 0.6203
  English, 0.5923 mean of 14 languages. Frozen teacher, seed 1: 0.5890 / 0.5747 (seed 42:
  0.5888 / 0.5748).
- GPU samples at 17:10 during NRMS training in both processes: 98 % utilisation, 5.3 GB,
  80 to 82 °C, 93 W.

## 2026-10-05 17:15 — FeedWell-Edge moved to the FeedWell 1.18.2 code base (branch, not built)

Repository `FeedWell-Edge`, branch `rebuild-on-feedwell-1.18`, commit 145b3b3, pushed.

- Merged upstream FeedWell main (v1.18.2, 49 commits since the 1.6.22 base). Eight files
  conflicted: the four version files, `AppSettingsContext.js`, `ArticleReaderScreen.js`,
  `FeedListScreen.js` (7 hunks) and `SettingsScreen.js` (4 hunks).
- The on-device learning layer of v2.0.5 is carried over unchanged. In the feed list,
  upstream's deterministic date sort and priority feeds stay; with learning enabled and
  "newest" selected the learned score orders the list first.
- Version 2.1.0, versionCode 2.
- Checked: all 88 JavaScript files parse (Babel parser), the three JSON files load. Not
  checked: no `npm install`, no Gradle build, not run on a phone. A Gradle build was not
  started because the two training processes hold 8.6 GB and 5 GB of RAM were free.
- Open: the application id is still `com.feedwell.app`, the same as the Play Store app. A
  separate id is needed to install both on one phone.

## 2026-10-05 17:37 — Laptop froze under the two-process batch; restarted 2026-10-06 10:55

- The GPU sampler shows 98 % utilisation, 83 to 86 °C and 87 to 97 W from 17:08 until its
  last sample at 17:36:37; two training processes held 8.6 GB of RAM. The desktop stopped
  responding at the lock screen and the machine was switched off.
- Lost: NRMS-GloVe seed 42 and teacher-linear seed 1, both in their last epoch. Nothing on
  disk was damaged: the four run records and the three stage files parse, the repository
  was clean, FeedWell-Edge was already pushed.
- Restart as ONE process at below-normal priority (`scripts/launch_p1.ps1`): seed 42 for
  all stages first, then seeds 1 and 2. Finished runs are skipped. Expected: about 2.5 h
  for seed 42, then about 5 h for the other two seeds. GPU sampler resumed.

## 2026-10-06 11:05 — Decisions (Sepehr) and FeedWell-Edge app identity

- Offline replay of click logs is acceptable as the evaluation of the on-device learner;
  the app may stay out of the first paper. Decide after the current experiments.
- FeedWell-Edge installs as `com.feedwelledge.app` ("FeedWell Edge", scheme
  `feedwelledge://`), next to the Play Store app. Done on branch
  `rebuild-on-feedwell-1.18` (FeedWell-Edge commit after 145b3b3), pushed; not built.
- The Lane-Change project has priority on the machine and on the WSL µNAS environment;
  this project keeps to one training process and no builds without asking.

## 2026-10-06 11:10 — Schedule (agreed order)

| # | Step | Needs | Effort | Status |
|---|---|---|---|---|
| 0 | Audit + repair (cost model, seeds, artifacts, paper sync) | – | done | done 2026-10-05 |
| 1 | P1 experiments: teacher bound, GloVe NRMS, scratch vs distilled, EN vs mixed-language clicks, 15 langs, 3 seeds | laptop GPU | ~7 h | running |
| 2 | Paper update with P1 numbers; notebook re-run; course sync | 1 | 1 day | next |
| 3 | Real µNAS search: byte-CNN in the lab µNAS fork (WSL), budgets for H7B3I-DK + F401RE, INT8 TFLite out; 120-grid exhaustive run as reference | 2; WSL free (Lane-Change first) | 2–3 days + search | waits |
| 4 | Real quantized models: full-int8 TFLite (weights + activations, AUC from the exported file); 1-bit via Larq → LCE or CBin-NN | 3 | 2–3 days | |
| 5 | Board numbers: ST Edge AI Developer Cloud (H7B3I-DK, F401RE) latency / flash / RAM; energy with the lab probe; 1-bit on a board | 4; ST login | 2 days | |
| 6 | FeedWell-Edge: npm + Gradle build, install com.feedwelledge.app; encoder inside the app (same .tflite), title embedding + user vector; phone latency / battery | 4; laptop slot + phone | 1 week | branch ready |
| 7 | Offline replay: MIND clicks in time order through the app learner | 2 | 2–3 days | optional for paper A |
| 8 | Paper A final: encoder + µNAS × precision + boards; 6 pages if ApplePies 2027 | 2–5; authors; venue | 1 week | |
| 9 | Houriyeh's thesis chapters | 2–6; her deadline | ? | |
| 10 | Paper B: app + on-device adaptation (+ pilot) | 6, 7 | PhD Y2–Y3 | later |

Critical path 1 → 2 → 3 → 4 → 5 → 8; the app track (6) runs beside it once step 4 yields a model.

## 2026-10-06 18:44 — P1 results, seeds 42 and 12 (seed 1 running)

| run | finished (local) | minutes | EN AUC | mean AUC, 14 languages | GPU util. | commit |
|---|---|--:|--:|--:|--:|---|
| p1/heuristics | 2026-10-05 16:55 | 1.1 | - | - | - | f5b7a0a* |
| p1/teacher/teacher_frozen/seed42 | 2026-10-05 17:03 | 7.2 | 0.5888 | 0.5748 | - | f5b7a0a* |
| p1/teacher/teacher_frozen/seed1 | 2026-10-05 17:08 | 5.1 | 0.589 | 0.5747 | 16% | c5957fa* |
| p1/teacher/teacher_linear/seed42 | 2026-10-05 17:08 | 5.1 | 0.6203 | 0.5923 | 16% | c5957fa* |
| p1/nrms/nrms_glove/seed42 | 2026-10-06 11:17 | 23.9 | 0.6603 | - | 95% | d42e82b |
| p1/nrms/nrms_reduced/seed42 | 2026-10-06 11:31 | 13.8 | 0.5943 | - | 89% | d42e82b |
| p1/student/scratch_en/seed42 | 2026-10-06 11:48 | 17.4 | 0.6203 | 0.5254 | 80% | d42e82b |
| p1/student/distill_frozen/seed42 | 2026-10-06 11:56 | 7.2 | 0.5675 | 0.5712 | 47% | d42e82b |
| p1/student/distill_ft_en/seed42 | 2026-10-06 12:12 | 16.7 | 0.6284 | 0.5565 | 88% | d42e82b |
| p1/student/distill_ft_mixed/seed42 | 2026-10-06 12:34 | 21.5 | 0.6139 | 0.5955 | 85% | d42e82b |
| p1/student/scratch_mixed/seed42 | 2026-10-06 12:54 | 20.6 | 0.5791 | 0.557 | 85% | d42e82b |
| p1/teacher/teacher_frozen/seed12 | 2026-10-06 12:58 | 3.9 | 0.589 | 0.5745 | 11% | d42e82b |
| p1/teacher/teacher_linear/seed12 | 2026-10-06 13:02 | 3.4 | 0.6172 | 0.5906 | 22% | d42e82b |
| p1/nrms/nrms_glove/seed12 | 2026-10-06 13:24 | 22.1 | 0.6662 | - | 93% | d42e82b |
| p1/nrms/nrms_reduced/seed12 | 2026-10-06 13:36 | 12.1 | 0.6186 | - | 91% | d42e82b |
| p1/student/scratch_en/seed12 | 2026-10-06 13:53 | 16.8 | 0.6154 | 0.5277 | 87% | d42e82b |
| p1/student/distill_frozen/seed12 | 2026-10-06 14:01 | 7.7 | 0.5768 | 0.571 | 39% | d42e82b |
| p1/student/distill_ft_en/seed12 | 2026-10-06 14:18 | 17.4 | 0.6252 | 0.5597 | 79% | d42e82b |
| p1/student/distill_ft_mixed/seed12 | 2026-10-06 14:41 | 23.4 | 0.5975 | 0.5872 | 82% | d42e82b |
| p1/student/scratch_mixed/seed12 | 2026-10-06 18:41 | 239.9 | 0.5805 | 0.5563 | 99% | d42e82b |

20 runs, 486 min of compute (8.1 h). A commit marked * had uncommitted code changes when the process started.

Reading of the table above (two seeds):
- NRMS with the reference GloVe configuration: 0.660 and 0.666. The June "0.607" was the
  reduced variant (0.594 and 0.619 under the new loop, 0.024 apart between seeds).
- Reference byte-CNN 64-5-384, English clicks: scratch 0.620 / 0.615; distilled start
  0.628 / 0.625 (+0.009 on average, smaller than the +0.033 of the June ablation, which used
  fixed negatives and 10 epochs).
- Cross-lingual (mean of 14 translations): scratch 0.525 / 0.528 (no transfer); distilled
  start + English clicks 0.557 / 0.560; distilled start + clicks shown in a random language
  0.596 / 0.587 at an English cost of 0.015 to 0.028. Frozen distilled student alone 0.571;
  frozen teacher 0.575; teacher + linear map 0.592 / 0.591.
- The last run (scratch_mixed, seed 12) took 240 min instead of 21: from about 16:30 another
  process held 5.7 GB of the 8 GB GPU memory (samples: 100 % utilisation at 30 W, 50 °C),
  so the training process ran from shared memory. Not a code problem; GPU sharing with the
  Lane-Change work is to be avoided.
- Launcher bug: `-Seeds 1,2` was bound to the single seed 12. Fixed (strings split on
  commas); seed 1 started at 18:44 to complete three seeds.

## 2026-10-06 20:50 — P1 complete (seeds 42, 12, 1); paper updated

- Seed 1 ran 18:43 to 20:49 (126 min, one process, GPU free). `scripts/summarize_p1.py` writes
  the seed aggregates to `paper/results/p1_summary.json`; `scripts/make_tables.py` writes
  Table 2 of the paper (`paper/tab_p1.tex`) from them.
- Three-seed means (EN AUC / mean over 14 translations): NRMS GloVe 0.664±0.003 (11.29 M
  params); NRMS reduced 0.604±0.010; byte-CNN 64-5-384 (68 k): scratch 0.618 / 0.528,
  distilled start 0.629 / 0.557, distilled start + mixed-language clicks 0.609 / 0.589,
  frozen distilled student 0.573 / 0.572; frozen teacher 0.589 / 0.575, + linear map
  0.618 / 0.590. Full table: `python -m scripts.log_report p1/`.
- Paper: abstract, setup, results (new Table 2, baselines, cross-lingual and history-bucket
  paragraphs), limitations and conclusion rewritten around these numbers; the June Table 1
  stays, marked as the earlier loop. `check_numbers.py` extended to the new values; PDF 9
  pages. Course notice updated.
- Notebook re-executed in QUICK mode after the batch (`scripts/make_notebook.py`).

## 2026-10-07 12:37 — MSc thesis of H. Emadoleslami rebuilt as a LaTeX project from the PDF draft

- Input: only the 67-page PDF draft (`H.E. Thesis/Thesis_UniGe_template.pdf`); text extracted with
  pdftotext and split into `H.E. Thesis/src/draft_parts/`. Output: `H.E. Thesis/src/` (main.tex, six
  chapters, four appendices, references.bib, build.ps1, README) compiling to 87 pages with
  pdflatex + bibtex (IEEEtran), no errors, no undefined references.
- Corrections against the draft: June cost columns replaced by the recomputed ones (energy
  166.3/17.91/12.54, 15.15/3.16/2.72, 14.98/4.35/3.96 µJ; 36.14/3.29/3.26 MMAC; RAM rule); the
  2 MMAC bound stated as not enforced; INT8 described as weight-only simulation; NRMS GloVe 0.664
  (three seeds) and reduced 0.604±0.010 as reference points instead of "recovers baseline 0.607";
  three-seed October conditions, per-language table (0.528/0.557/0.589), heuristics (0.525/0.598/0.617)
  and history buckets added; recovered binary 194 KB; conference paper (Appendix A, contribution 6)
  removed; Raspberry Pi 5 removed as a target (STM32H7B3I-DK, NUCLEO-F401RE, Android phone instead);
  Declaration and §1.7 per Sepehr's instruction; Appendix E placeholders filled with repositories,
  commits, environment and the 31-file SHA-256 manifest; bibliography fixed ([19] NAS-BNN authors,
  [20] LNEE 1553 pp. 85–90) and the GEMINI position paper verified (LNEE 1369, 2025).
- New assets in this repo: `paper/diagrams/*.tex` (five TikZ diagrams rendered to PNG),
  `scripts/make_thesis_assets.py` (seven charts in `paper/figures/thesis`, ten table fragments in
  `paper/tables/thesis`, all from paper/results and the data), `scripts/check_thesis_numbers.py`
  (every number quoted in the thesis chapters against the result files: 0 failures).
- Style sweep against DIMIR/paper/STYLE.md: no blocklist words left in the chapters; no em-dashes.
- Timing (file times): diagrams and charts 11:00–11:50; LaTeX writing 11:50–12:30; compile, number
  check and layout fixes 12:30–12:37.
- Open for Sepehr/Houriyeh: UniGe logo (drop `figures/logo_unige.png` into the thesis folder),
  final read-through, upload by the candidate.

## 2026-10-07 15:13 — Integer model exported and measured; board benchmarks started on the ST Edge AI Developer Cloud

- `scripts/deploy_int8.py` (run under the DIMIR venv, whose torch loads; the MIND venv's torch is blocked by
  Smart App Control today): the trained 64-5-384 encoder (October programme, distilled start, English clicks,
  seed 42) exported to ONNX with batch 1, then quantised with ONNX Runtime static quantisation (QDQ, per-channel
  symmetric int8 weights, int8 activations, Conv/Gemm/MatMul; 512 training titles for calibration). Dev AUC
  through the saved user encoder: PyTorch 0.6284, ONNX FP32 0.6283, ONNX INT8 QDQ 0.6223 (MRR 0.341 vs 0.344,
  nDCG@10 0.385 vs 0.387); cosine between FP32 and INT8 news vectors 0.990 mean, 0.937 min. Files in
  `artifacts/stedgeai/models` (FP32 275 KB, INT8 157 KB incl. the FP32 byte table); results in
  `paper/results/deploy_int8.json`. Whole script 1.1 min.
- `scripts/onnx_static.py`: batch-1 copies (and int32-input variants) of the June/quick ONNX exports of the
  other two architectures, verified bit-identical to the originals with ONNX Runtime.
- `scripts/stedgeai_cloud.py`: wrapper around ST's model-zoo client (vendored in DIMIR/Materials). Login with
  the student account's password failed in the client's SSO step (my.st.com timing out, then a 200 without a
  redirect); the token cached by the September DIMIR session on this laptop belongs to the same account and
  still refreshes, so the runs use it. No password stored anywhere. Cloud tool: ST Edge AI Core 4.0.1-20581.
- Local `stedgeai.exe` (C:/ST/STEdgeAI/4.0) is blocked by Smart App Control (DLL), so everything runs on the cloud.
- analyze, Micro-NAS INT8 QDQ (int64 input accepted): 3,389,828 MACC, weights 120,260 B, activations 33,796 B,
  flash total 169,638 B incl. 48,838 B kernel library, RAM 37,908 B.
- Benchmarks queued (background, log `artifacts/stedgeai/benchmark.log`, results
  `paper/results/stedgeai_cloud/`): the three architectures at FP32 and INT8 on STM32H7B3I-DK; the two small
  ones also on NUCLEO-F401RE.

## 2026-10-07 16:16 — Board measurements, integer export, app smoke test, thesis and paper updated

- Smart App Control was turned off by Sepehr this afternoon; torch loads again in the MIND venv
  (the integer export ran under the DIMIR venv before that).
- Integer export (`scripts/deploy_boards.py`, replaces deploy_int8.py and onnx_static.py, 0.8 min):
  64-5-384 encoder, October checkpoint (distilled start, English clicks, seed 42), exported without the
  byte-table lookup (caller gathers the 257 x 64 table, 65,792 B, and passes embedded sequence + mask),
  then ONNX Runtime static quantisation to QDQ int8 (per-channel weights, int8 activations, 512 training
  titles). Dev AUC: PyTorch 0.6284, ONNX FP32 0.6283, ONNX INT8 0.6223 (MRR 0.341, nDCG@10 0.385);
  FP32/INT8 news-vector cosine mean 0.990, min 0.937. Same files for 256-4-384 and 96-2-384 with random
  weights (latency and footprint do not depend on the weight values).
- Board runs, ST Edge AI Developer Cloud, Core 4.0.1-20581 (`scripts/stedgeai_cloud.py`), 15:38-16:03:

  | file | board | ms | weights KB | flash KB | RAM KB |
  |---|---|--:|--:|--:|--:|
  | 64-5-384 INT8 | H7B3I-DK | 31.19 | 53.2 | 77.7 | 49.4 |
  | 64-5-384 INT8 | F401RE | 449.64 | 53.2 | 78.0 | 44.5 |
  | 64-5-384 FP32 | H7B3I-DK | 76.99 | 200.0 | 210.0 | 70.9 |
  | 64-5-384 FP32 | F401RE | 444.08 | 200.0 | 210.2 | 65.7 |
  | 96-2-384 INT8 | H7B3I-DK | 58.57 | 63.9 | 84.7 | 67.6 |
  | 96-2-384 INT8 | F401RE | 431.30 | 63.9 | 84.8 | 59.3 |
  | 96-2-384 FP32 | H7B3I-DK | 84.14 | 245.6 | 254.6 | 84.9 |
  | 96-2-384 FP32 | F401RE | 397.22 | 245.6 | 254.8 | 65.7 |
  | 256-4-384 INT8 | H7B3I-DK | 649.41 | 381.5 | 411.3 | 188.5 |
  | 256-4-384 FP32 | H7B3I-DK | 837.37 | 1494.5 | 1504.2 | 180.7 |

  On-target check of the generated C code against the reference: relative error 0 (INT8), < 1e-5 (FP32).
  The 256-4-384 model needs more than the F401RE's 96 KB of RAM. INT8 is 2.5x faster than FP32 on the
  Cortex-M7 for 64-5-384, no faster on the Cortex-M4. No energy (the cloud reports time and memory only).
- Two earlier deployment variants were dropped: int64 byte ids (the on-target validation feeds random floats,
  the Gather got out-of-range indices) and float ids rescaled inside the graph (ran: 34.8 / 189 ms, but the
  on-target check failed, relative error 0.48 INT8 / 0.43 FP32). Their reports are kept in
  `paper/results/stedgeai_cloud/` (`*_fid_*`, `*__failed.json`).
- Cloud access: the password given today failed in the client's scripted SSO step (my.st.com timing out);
  the session token cached in the home directory from September (same account) refreshes and was used.
  The password is not stored anywhere; account ids were stripped from the saved reports.
- App smoke test, FeedWell Edge 2.1.0 on the test phone, 16:00-16:08: feed added through the URL field
  (BBC News), articles listed, preview and reader opened and scrolled, Settings shows local learning on;
  after a refresh the 7-day local summary counts the test events. The engine's event hooks are identical to
  the pre-rebuild code. Test data stays on the phone and is not used anywhere. Behavioural results need days of
  real use; for the thesis only the current state is described, the app study is for the paper.
- Thesis (`H.E. Thesis/src`): Figures 1.1, 2.1, 2.9, 4.1 and the overview diagram redrawn without overlaps;
  column widths of Tables 1.2, 1.1, 2.3, 3.1, 3.2 and others fixed; UniGe vertical colour logo on the title page
  (SVG converted to PDF with svglib); second style pass; new Section 5.8 "Measurements on the Boards" with two
  tables and a figure; abstract, contributions, threats, limitations, conclusion and future work updated.
  89 pages, no errors, `check_thesis_numbers` 0 failures. PDF: `H.E. Thesis/Thesis_Emadoleslami_2026-10-07.pdf`.
- Paper: TODOs for the integer export and the board numbers replaced by the measured values (energy and phone
  stay TODO); `check_numbers.py` extended and passes; 9 pages.

## 2026-10-07 16:58 — Schedule, updated

| # | Step | Needs | Effort | Status |
|---|---|---|---|---|
| 0 | Audit and repair (cost model, seeds, artifacts, paper sync) | – | done | done 10-05 |
| 1 | P1 experiments: teacher bound, GloVe NRMS, scratch vs distilled, EN vs mixed-language clicks, 15 languages x 3 seeds | laptop GPU | 10.2 h of runs | done 10-06 |
| 2 | Paper update with P1 numbers, notebook re-run, course sync | 1 | 1 day | done 10-06; course again 10-07 |
| 3 | Real µNAS search in the lab fork (WSL), budgets for H7B3I-DK and F401RE, MAC bound with the corrected counter; 120-grid exhaustive run as reference; matrix rerun over 3 seeds | WSL slot | 2–3 days + search | next |
| 4 | Integer and 1-bit models, accuracy from the exported file | 3 | 2–3 days | INT8 done for 64-5-384 (0.622 vs 0.628, 10-07); µNAS winners and 1-bit open |
| 5 | Board numbers: latency, flash, RAM; energy with the lab probe; 1-bit on a board | 4; ST session; probe | 2 days | latency/flash/RAM done 10-07 (H7 31.19 ms, F401 449.64 ms); energy and 1-bit open |
| 6 | FeedWell-Edge: build, install, encoder inside the app, phone latency and battery | 4; phone | 1 week | build, install, smoke test done 10-07; encoder in app and phone timing open |
| 7 | Offline replay: MIND clicks in time order through the app learner | 2 | 2–3 days | not started; optional for paper A |
| 8 | Paper A final | 3–5; authors; venue | 1 week | draft current (9 pp, P1 and board numbers); final after 3–5, 6 pp if ApplePies 2027 |
| 9 | Houriyeh's thesis | – | done | delivered 10-07 (89 pp); her read-through and upload, deadline 10-08 |
| 10 | Paper B: app and on-device adaptation, with pilot | 6, 7; days of real use | PhD Y2–Y3 | later; app behavioural results go here |

Critical path: 3, 4, 5, then 8. Step 6 can start now, since the integer encoder from step 4 exists.

## 2026-10-07 17:34 — Step 3 started: µNAS search of the encoder in the lab fork

- Boundaries checked at 17:00: GPU idle, WSL stopped (no Lane-Change job running), 936 GB free.
  The shared fork `~/uNAS` carries the Lane-Change-MCU patches and configs, so the search runs in a
  copy, `~/uNAS_mind`; `git status` of the shared fork is unchanged after setup. Venv `~/dmir_nas`
  (TF 2.21, Keras 3.12, GPU visible) used as it is.
- PyTorch was blocked again in the MIND venv at 17:04 (Windows application control, `shm.dll`), although
  Smart App Control is reported off. The byte table was extracted with the DIMIR venv's torch and the
  export made torch-free (`scripts/unas_export_data.py`).
- Adapter in `unas/` (README there): input = 128 byte ids through the distilled student's fixed byte
  table plus a padding-mask channel (128, 65); target = teacher anchor of the English title (384-d);
  rows in all 15 languages; fitness = 1 - validation mean cosine plus the fork's faithful costs; one
  search per board (H7: 256 KiB, 128 KiB, 2 M MACs; F401: 128 KiB, 48 KiB, 1 M MACs).
- Training recipe, three attempts on the June winners in the fork's space:

  | recipe | rows | epochs | 64-5-384 val cos, seeds 42/1/2 | note |
  |---|--:|---|---|---|
  | 1 | 32,000 | 10, patience 3 | 0.170 / 0.178 / 0.170 | best epoch = epoch 1 (BatchNorm lag); rejected |
  | 2 | 96,000 | 15, stop from 5 | 0.309 / 0.286 / 0.281 | val moves +-0.04 per epoch; rejected |
  | 3 | 96,000 | 15, lr 1e-3 then x0.7/epoch | 0.345 / 0.342 / 0.343 | search recipe, ~2.2 min per model |

  Recipe 3, other June winners: 256-4-384 0.478 (374 KiB, 36.2 M MACs, 6.3 min); 96-2-384 0.330.
  None of the three fits either MAC budget. Error bound = 1 - 0.343 = 0.657.
- Queue (`unas/run_queue.sh`): H7 search, F401 search (150 candidates each, population 40, sample 12,
  one worker), harvests, then the 30-architecture grid of the hand-designed family. Estimate 11-12 h.

## 2026-10-07 17:45 — First H7 search stopped (wrong metric in the Ray worker); trainer patch v2; queue relaunched

- 17:35 the queue started; by 17:41 rescoring saved candidates on the CPU showed the logged numbers were
  wrong: candidate 1 logged validation and test cosine 0.2659 / 0.2659, its saved model scores 0.2706 / 0.2736.
  Cause: in the Ray worker `tf.keras` is the legacy Keras 2 package (tfmot imported first); the Keras 2 metric
  object handed to the Keras 3 model was wrapped as a function whose running mean never resets, so it lagged
  (0.005 on validation, 0.03 on training) and the test pass repeated it. The loss in the same log equals the
  rescored cosine. In the single-process reference runs the metric equalled the loss, so they stay valid.
- Fix (`unas/patch_trainer.py` v2): Keras 3 loss and metric objects; val_error = 1 + min(val_loss), test_error
  = 1 + test loss; early stopping on val_loss; the patch is applied to a fresh copy of the shared fork's trainer.
  Smoke search (3 candidates, 6 epochs): metric = loss in the worker, test errors differ from validation ones.
- The stopped run is kept as `~/uNAS_mind/artifacts/mind_h7_flawed_metric` and not used.
- 17:45 queue relaunched (H7, F401, harvests, grid), selection watcher relaunched after it.

## 2026-10-08 09:42 — Overnight queue results (step 3); NaN bug in the H7 shortlist; H7 selection rerun

- Queue (`logs/nas/queue_10071745.log`): H7 search 17:45-19:29 (150 candidates, 1 h 44 min), F401 search
  19:29-21:03 (1 h 34 min), grid of the hand-designed family 21:03-22:20, selection 22:21-22:46.
  About one candidate per minute; GPU 40-85 %.
- H7 search: 110 of 150 candidates within 256 KiB / 128 KiB / 2 M MACs; best single run cos 0.331 at 2.00 M MACs.
  F401 search: 67 of 150 within 128 KiB / 48 KiB / 1 M MACs; best single run 0.298 at 0.65 M MACs.
- Grid (hand-designed 1x1 + depthwise-separable family, seed 42): within the H7 budget the best is 64-2-384,
  0.303 at 1.66 M MACs; within F401, 32-5-384, 0.296 at 1.00 M MACs. Unconstrained top: 256-5-384, 0.503 at 45 M.
  None of the in-budget models reaches the error bound (hand-designed 64-5-384, 0.343 at 3.31 M MACs).
- Bug: candidates stopped by TerminateOnNaN got val_error = NaN (1 in H7, inside the budget; 5 in F401, all
  outside it). NaN broke the sort in `select_by_seeds.py`: the H7 shortlist missed the true eight best
  (indices 108-146, all from late in the search). Fixed: harvest counts non-finite results as failed,
  selection keeps finite scores only, trainer patch v3 scores a NaN run as 1.0. The F401 shortlist was not
  affected. First H7 selection kept as `mind_h7_selection_nan_shortlist.json` (its winner, candidate 67:
  0.317 over three seeds at 1.74 M MACs).
- F401 selection (valid): candidate 134, mean val cos 0.2977 (SE 0.0006), test 0.298, 0.65 M MACs, 41.5 KiB,
  5.9 KiB activations: one block of two strided layers (up to 59 filters) and the GAP head.
- PyTorch loads again in the MIND venv this morning (2.12.1+cu130).
- 09:42 H7 selection relaunched on the corrected shortlist (24 runs).

## 2026-10-08 09:54 — Step 3 search phase done: corrected H7 selection

- H7 selection on the corrected shortlist, 09:41-09:53 (24 runs, 12 min): candidate 144 is both the best and
  the smallest within one standard error: mean val cos 0.3309 (SE 0.0007, seeds 42/1/2), test 0.331,
  2.00 M MACs, 51.9 KiB INT8 weights, 15.5 KiB activations; two convolution blocks, three layers, one
  pre-pooling, up to 98 filters, GAP head, no hidden dense layer. Next three: 0.322, 0.321, 0.321.
  F401 selection unchanged on rerun: candidate 134, 0.2977 (SE 0.0006), 0.65 M MACs.
- Against the hand-designed family under the same recipe and cost model (`grid.json`, one seed):
  H7 budget 64-2-384 0.303 at 1.66 M MACs, so the searched model is 0.028 higher at 2.00 M; F401 budget
  32-5-384 0.296 at 1.00 M, so the searched model is level at 0.65 M MACs.
- Neither choice reaches the hand-designed 64-5-384 (0.343 at 3.31 M MACs): within these budgets the encoder
  gives up 0.012 (H7) and 0.045 (F401) of teacher cosine. Ranking AUC of the chosen models follows in step 4.

## 2026-10-08 10:25 — Schedule, updated

| # | Step | Needs | Effort | Status |
|---|---|---|---|---|
| 0 | Audit and repair | – | done | done 10-05 |
| 1 | P1 experiments, 3 seeds | laptop GPU | 10.2 h of runs | done 10-06 |
| 2 | Paper update with P1, notebook, course | 1 | 1 day | done 10-06; course 10-07 |
| 3 | µNAS search in the lab fork, budgets for H7B3I-DK and F401RE; grid reference; seed-based choice; nine-variant matrix rerun over 3 seeds | WSL, GPU | searches 6.5 h; matrix ~9 h | searches, grid, choice done 10-08 (H7 cos 0.331 at 2.00 M MACs, F401 0.298 at 0.65 M); matrix rerun open |
| 4 | Full training of the two chosen models; integer export; ranking AUC from the exported files; 1-bit | 3 | 1–2 days | INT8 done for 64-5-384 (0.622 vs 0.628); chosen models next; 1-bit open |
| 5 | Board numbers: latency, flash, RAM; energy with the lab probe; 1-bit on a board | 4; ST session; probe | 1–2 days | June architectures measured 10-07; chosen models after 4; energy and 1-bit open |
| 6 | FeedWell-Edge: encoder inside the app; phone latency and battery | 4; phone | 1 week | build, install, smoke test done 10-07; integration open |
| 7 | Offline replay through the app learner | 2 | 2–3 days | not started; optional for paper A |
| 8 | Paper A final | 3–5; authors; venue | 1 week | draft 9 pp with P1 and board numbers; µNAS results to add |
| 9 | Houriyeh's thesis | – | done | delivered 10-07; her upload, deadline 10-08 |
| 10 | Paper B: app and on-device adaptation, pilot | 6, 7; days of real use | PhD Y2–Y3 | later |

Critical path: 4, then 5, then 8. The matrix rerun of step 3 can run overnight beside daytime work on 4.

## 2026-10-08 10:53 — Step 3 matrix rerun started; step 4 started (chosen µNAS encoders in the PyTorch pipeline)

Matrix rerun (`scripts/run_matrix.py`, commit 4a4ed5e)
- The nine cells of the June Table 1 (architectures 256-4-384, 64-5-384, 96-2-384 x FP32 / simulated INT8 /
  simulated binary) under the October loop, seeds 42, 1, 2: random initialisation, English clicks, 8 epochs; INT8 and
  binary = the trained FP32 model with simulated weight quantization of the inner layers plus 2 epochs of fine-tuning;
  full dev set. The architectures are the June winners and are not searched again. Every cell is recorded when it ends
  (`paper/results/matrix_runs.json`, `experiments.jsonl`), checkpoints are kept (`artifacts/runs/matrix`).
- Smoke run (one arm, 2,000 impressions) 0.9 min: sizes and MACs equal the recomputed June costs (266.75 / 203.94 /
  185.62 KB, 3.29 M MACs).
- Started 10:36 (one process, below-normal priority). First cell, 64-5-384 FP32 seed 42: AUC 0.6203 in 16.1 min, the
  same value as the P1 run `scratch_en/seed42` (same protocol and seed). GPU 95-97 %, 5.9 GB, 74-82 °C, 93-103 W.
- A file `artifacts/runs/matrix/PAUSE` stops the run before its next cell; it was set at 10:50 so that step 4 could use
  the GPU (one training process at a time). The run stopped at 10:52 and resumes after step 4 (`scripts/queue_step4.ps1`).
- The GPU sampler of the new launcher (`scripts/launch_job.ps1`) first used `nvidia-smi -l -f`, which buffers its file on
  Windows until it exits (empty CSV after 5 min); replaced by `scripts/gpu_sampler.ps1` (one appended line per 30 s
  while the job's process lives).

Step 4: chosen µNAS encoders rebuilt in PyTorch (commit 61723a1)
- The search's networks are Keras models of the fork's space; click training, evaluation and integer export here are
  PyTorch. `src/unas_encoder.py` rebuilds an architecture dictionary with the Keras semantics ("same" padding with the
  odd pad on the right for strided layers, BatchNorm epsilon 1e-3, MaxPool1D(2) pre-pooling, GAP over all 128
  positions, dropout before the output layer); the 257 x 64 byte table is a frozen parameter, as in the search.
- Check (`unas/dump_keras.py` in WSL on the CPU, `scripts/check_unas_port.py`): Keras builds with seeded random weights
  and non-trivial BatchNorm statistics, weights copied into the PyTorch builds, 64 validation rows. H7 candidate 144:
  largest output difference 2.3e-5 (outputs up to 72.5), smallest cosine 0.9999998; F401 candidate 134: 8.6e-6,
  0.9999999. Parameters: 53,404 and 42,664 trainable plus the 16,448-entry byte table. MACs counted here (convolution
  and dense layers): 1,986,304 and 648,736; the fork's resource model gives 1,995,648 and 650,624 because it also
  counts the pooling operations (9,344 and 1,888).
- `scripts/run_unas_full.py`: the reference encoder's programme (scripts/run_p1.py): distillation 15 epochs on all
  769k training rows in 15 languages (Matryoshka cosine loss), then click training from the distilled encoder,
  8 epochs, English clicks (`distill_ft_en`) or clicks in a random language (`distill_ft_mixed`); seeds 42, 12, 1 as in
  P1, so data order and negatives match the reference runs; full dev set, English by history bucket and 14 languages.
  CPU smoke run: 2.2 min; the byte table is unchanged after both stages (largest difference 0).
- 10:53 `distill_ft_en` started for both encoders and three seeds (estimate about 2.7 h); then the matrix resumes
  (about 8.4 h), then `distill_ft_mixed` (about 2.5 h).

Integer export with three seeds (`scripts/export_int8.py`, CPU)
- Same export as `scripts/deploy_boards.py` (embedded sequence and mask in, QDQ int8, 512 calibration titles), for the
  reference and the two µNAS encoders, three seeds each; accuracy of each file on the 73,152 dev impressions.
- Reference 64-5-384, `distill_ft_en`, 2.5 min for three seeds:

  | seed | PyTorch | ONNX FP32 | ONNX INT8 | FP32/INT8 cosine | INT8 file |
  |---|--:|--:|--:|--:|--:|
  | 42 | 0.6283 | 0.6283 | 0.6223 | 0.990 | 90,893 B |
  | 12 | 0.6252 | 0.6252 | 0.6210 | 0.990 | 90,902 B |
  | 1 | 0.6320 | 0.6320 | 0.6293 | 0.991 | 90,893 B |
  | mean | 0.6285 | 0.6285 | 0.6242 (sd 0.0045) | | |

  Seed 42 equals `paper/results/deploy_int8.json` to the last digit.
- The ONNX export of the µNAS builds matches PyTorch to 1.2e-7. A first test showed a large difference; the cause was
  the test itself: `torch.onnx.export` puts the wrapper module back into the mode it had, and a new wrapper is in
  training mode, so the encoder was left with BatchNorm on batch statistics and dropout on when the PyTorch reference
  was computed after the export. The export script sets the wrapper to evaluation mode before exporting; its earlier
  results are not affected (the model is put in evaluation mode before it is evaluated).
- 1-bit: not started. Route to a board as in `docs/binary_deployment.md` (Larq, LCE, TFLite Micro on the H7, as in
  `NAS-BNN/publish/nasbnn-wakevision-stm32`, or CBin-NN).

## 2026-10-08 11:24 — Step 4 first results (seed 42); board files of the µNAS encoders; Pad export fix

Full programme, seed 42 (`scripts/run_unas_full.py`; reference 64-5-384 from P1 for comparison):

| encoder | distillation | teacher cosine, EN dev | click training | EN AUC | mean of 14 languages |
|---|--:|--:|--:|--:|--:|
| µNAS H7, candidate 144 | 3.2 min | 0.390 | 11.0 min | 0.5984 | 0.5500 |
| µNAS F401, candidate 134 | 1.7 min | – | 5.8 min | 0.5949 | 0.5354 |
| reference 64-5-384 (P1, seed 42) | – | 0.468 | – | 0.6284 | 0.5565 |

- The search recipe put the H7 choice 0.012 below 64-5-384 in teacher cosine (0.331 against 0.343, 96k rows,
  15 epochs at most); after distillation on all 769k rows the gap is 0.078 (0.390 against 0.468) and the click-trained
  encoder is 0.030 AUC below the reference. The two differ in more than the architecture: the searched encoders read
  the byte table fixed (the reference trains its own) and average over all 128 positions, padding included (the
  reference averages over the title's bytes only). The hand-designed encoders in the same form (64-5-384, and the
  grid's best within each budget, 64-2-384 and 32-5-384) were added to the queue after the µNAS runs to separate the
  two effects.
- GPU during click training of the H7 encoder: 90-94 %, 2.8 GB, 70-73 °C, 93-95 W; distillation 75-77 %, 1.2 GB.

Integer files (`scripts/export_int8.py`), seed 42: µNAS H7 0.5984 (PyTorch) = 0.5984 (ONNX FP32), 0.5986 (ONNX INT8),
72,032 B; µNAS F401 0.5949 = 0.5949, 0.5931, 56,783 B.

Board runs, ST Edge AI Developer Cloud (Core 4.0.1), INT8 files of seed 42, 11:17-11:23:

| file | board | ms | MACC | weights B | flash B | RAM B | cycles/MACC |
|---|---|--:|--:|--:|--:|--:|--:|
| µNAS H7 | STM32H7B3I-DK | 18.68 | 2,047,954 | 55,716 | 74,059 | 32,772 | 2.55 |
| µNAS H7 | NUCLEO-F401RE | 252.26 | 2,047,954 | 55,716 | 74,935 | 26,020 | 10.35 |
| µNAS F401 | STM32H7B3I-DK | 15.08 | 661,128 | 44,532 | 63,393 | 19,192 | 6.39 |
| µNAS F401 | NUCLEO-F401RE | 32.15 | 661,128 | 44,532 | 64,297 | 17,956 | 4.08 |

On-target validation error 0 for all four. For comparison, 64-5-384 INT8 (10-07): 31.19 ms on the H7, 449.64 ms on
the F401.

Pad export fix
- The first board run of the H7 file (11:09-11:13) failed on both boards: the C code that ST Edge AI generates for the
  ONNX Pad node of the PyTorch export does not compile (the node's constant value is empty: `network.c: expected
  expression before ']'`, `_layers_1_Pad_output_0_value undeclared`). The board then ran the firmware of the previous
  job, so the validation and timing in the server's log belonged to another model (the 256-4-384 and 96-2-384 FP32
  tables of 10-07: 837.37 and 397.22 ms) and the check reported a cosine of -0.04. The two records are kept
  (`*__failed.json`, with a note; account id removed).
- `fold_pads` in `scripts/export_int8.py`: constant-fold the exported graph with ONNX Runtime (basic level) and move each
  zero Pad into the Conv that reads it (Conv `pads`: [1, 1] for the H7 encoder's kernel-3 layer, [0, 1] and [1, 2] for
  the F401 encoder's strided layers). Same function: largest difference to PyTorch 1.5e-7 on 200 titles. The H7 file was
  exported again; its AUC did not change. The reference files have no Pad node and are not affected.
- In these QDQ files only Conv and Gemm run in 8 bits; ReLU, max-pooling and the average run in float between them
  (`--op_types_to_quantize Conv Gemm MatMul`, as for the reference, so that the files compare). Quantising those
  operators too would cut RAM and time further; not done yet.

## 2026-10-08 12:08 — Step 4: µNAS encoders over three seeds; full-integer files; first comparator signal

µNAS encoders, English clicks, seeds 42, 12, 1 (10:53-11:53, one process; distillation 1.2-3.2 min, clicks 5.6-11.0 min):

| encoder | teacher cosine, EN dev | EN AUC | mean of 14 languages |
|---|--:|--:|--:|
| µNAS H7 (cand. 144) | 0.392 ± 0.001 | 0.595 ± 0.003 (0.598 / 0.596 / 0.590) | 0.543 ± 0.005 |
| µNAS F401 (cand. 134) | 0.349 ± 0.002 | 0.591 ± 0.003 (0.595 / 0.587 / 0.590) | 0.537 ± 0.004 |
| reference 64-5-384 (P1) | 0.472 | 0.629 ± 0.003 | 0.557 ± 0.002 |

Integer files, three seeds each (`scripts/export_int8.py`, now with two variants: `int8qdq` = Conv/Gemm/MatMul only, the
files measured so far; `int8full` = every operator ONNX Runtime quantises, ReLU folded into the convolution's range):

| encoder | PyTorch | int8qdq | int8full |
|---|--:|--:|--:|
| µNAS H7 | 0.5949 | 0.5959 ± 0.0027 | 0.5952 ± 0.0023 |
| µNAS F401 | 0.5905 | 0.5896 ± 0.0025 | 0.5899 ± 0.0028 |
| reference 64-5-384 | 0.6285 | 0.6242 ± 0.0036 | 0.6257 ± 0.0025 |

Board runs of the full-integer files (seed 42), 12:01-12:08:

| file | board | int8qdq ms | int8full ms | int8qdq RAM B | int8full RAM B |
|---|---|--:|--:|--:|--:|
| µNAS H7 | STM32H7B3I-DK | 18.68 | 31.58 | 32,772 | 33,200 |
| µNAS H7 | NUCLEO-F401RE | 252.26 | 264.32 | 26,020 | 26,448 |
| µNAS F401 | STM32H7B3I-DK | 15.08 | 15.12 | 19,192 | 19,192 |
| µNAS F401 | NUCLEO-F401RE | 32.15 | 31.91 | 17,956 | 17,956 |
| reference 64-5-384 | STM32H7B3I-DK | 31.19 | 39.55 | 49.4 KB | 41,212 |
| reference 64-5-384 | NUCLEO-F401RE | 449.64 | 449.48 | 44.5 KB | 35,072 |

- Quantising the remaining operators does not speed anything up. In the µNAS H7 file ST Edge AI then fuses the kernel-3
  convolution, its ReLU and the max-pooling into one integer "convolution with pooling" kernel, which takes 20.2 ms
  (5.8 cycles per MACC) on the Cortex-M7; the same convolution in the int8qdq file is part of an 18.7 ms total. RAM
  falls only for the reference (its masked average runs in 8 bits). The int8qdq files stay the deployment files.

First comparator signal: the hand-designed 64-5-384 in the fork's form (fixed byte table, mask channel, average over all
128 positions) reaches teacher cosine 0.460 after full distillation (seed 42), against 0.468 for the reference form. The
input handling costs about 0.01 of teacher cosine; the 0.07 gap of the H7 choice is its architecture (one kernel-3
convolution and one pooling: a receptive field of 4 bytes, against 11 for 64-5-384).

## 2026-10-08 12:54 — Step 4: hand-designed comparators, seed 42 (results and board runs)

Same programme as the µNAS encoders (`run_unas_full.py --models hand_64-5-384 hand_64-2-384 hand_32-5-384`, started
11:53 by the queue); integer files and boards as before (int8qdq, seed 42; ST Edge AI Cloud runs ending 12:25-12:26 and 12:51-12:54).

| encoder (budget) | MACs (search model) | teacher cos | EN AUC | 14 languages | INT8 AUC | H7 ms | F401 ms | flash B (H7) | RAM B (H7) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| µNAS cand. 144 (H7) | 2.00 M | 0.390 | 0.5984 | 0.5500 | 0.5986 | 18.68 | 252.26 | 74,059 | 32,772 |
| hand 64-2-384 (H7) | 1.66 M | 0.372 | 0.6021 | 0.5424 | 0.6054 | 16.57 | 191.68 | 61,198 | 40,880 |
| µNAS cand. 134 (F401) | 0.65 M | 0.347 | 0.5949 | 0.5354 | 0.5931 | 15.08 | 32.15 | 63,393 | 19,192 |
| hand 32-5-384 (F401) | 1.00 M | 0.333 | 0.5799 | 0.5419 | 0.5852 | 21.22 | 88.68 | 46,838 | 20,136 |
| hand 64-5-384 (none) | 3.31 M | 0.460 | 0.6223 | 0.5527 | 0.6199 | 31.47 | 426.85 | 80,182 | 41,672 |

- F401 budget: the searched encoder beats the best hand-designed shape on English AUC (+0.015) and teacher cosine
  (+0.014) with 35 % fewer MACs, and runs 2.8 times faster on the NUCLEO-F401RE (32.2 against 88.7 ms); the
  translations go the other way (-0.007).
- H7 budget: the hand-designed 64-2-384 is level or slightly ahead on English AUC (+0.004) with fewer MACs and runs
  faster on the H7 (16.6 against 18.7 ms); the searched encoder is ahead on teacher cosine (+0.018) and on the
  translations (+0.008). One seed; seeds 12 and 1 follow.
- The hand-designed 64-5-384 in the search's form loses 0.006 AUC to its original form (0.6223 against 0.6284,
  seed 42): the fixed byte table and the average over padded positions cost little.
- The on-target check of the 32-5-384 file reports a validation error of 0.008 (all other files 0).
- GPU: the 64-5-384 click training held 7.7-7.8 GB of the 8 GB (96-98 % utilisation, no slowdown seen); the smaller
  encoders 1.1-2.9 GB.

## 2026-10-08 14:37 — Step 4: comparators over three seeds; paper and course updated (µNAS section, Table 3)

Comparator runs 11:53-14:25 (152.5 min, one process); the queue then resumed the matrix at 14:25.

| encoder (budget) | MACs | teacher cos | EN AUC | 14 languages | INT8 AUC (int8qdq) |
|---|--:|--:|--:|--:|--:|
| µNAS cand. 144 (H7) | 2.00 M | 0.392 ± 0.001 | 0.595 ± 0.003 | 0.543 ± 0.005 | 0.596 ± 0.003 |
| hand 64-2-384 (H7) | 1.66 M | 0.371 ± 0.001 | 0.599 ± 0.005 (0.602 / 0.592 / 0.602) | 0.547 ± 0.003 | 0.598 ± 0.006 |
| µNAS cand. 134 (F401) | 0.65 M | 0.349 ± 0.002 | 0.591 ± 0.003 | 0.537 ± 0.004 | 0.590 ± 0.003 |
| hand 32-5-384 (F401) | 1.00 M | 0.333 ± 0.019 | 0.589 ± 0.008 (0.580 / 0.588 / 0.599) | 0.538 ± 0.005 | 0.589 ± 0.003 |
| hand 64-5-384 (none) | 3.31 M | 0.459 ± 0.000 | 0.618 ± 0.006 (0.622 / 0.609 / 0.623) | 0.551 ± 0.004 | 0.614 ± 0.010 |

Reading
- Within each budget the searched encoder and the best hand-designed shape rank the same within the seed spread
  (H7: -0.004, F401: +0.002); the searched ones lead on teacher cosine (+0.021, +0.016). The search's short fitness
  (96k rows, 15 epochs) placed the H7 choice 0.012 below 64-5-384, full distillation 0.067 below: the short fitness
  under-rates what a larger network gains from full training.
- The searched F401 encoder is the fastest of the five on both boards; on the NUCLEO-F401RE it takes 32.1 ms against
  88.7 ms for 32-5-384 (4.1 against 7.2 cycles per MACC; in 32-5-384 each depthwise layer runs at 22 cycles per MACC and
  the first pointwise layer over 65 channels at 8.8, from the per-node report of the board run).
- The teacher cosine of 32-5-384 varies by seed (0.333 / 0.310 / 0.356) with nearly identical training loss curves
  (final 0.658 / 0.657 / 0.660); the other encoders vary by 0.001-0.003.

Paper (`paper/paper.tex`, 11 pages, PDF rebuilt; `scripts/check_numbers.py` passes)
- Method: paragraph on the board searches (lab 1D µNAS, budgets, fitness, selection, PyTorch rebuild); reference
  `capello2025unas1d` added (Crossref: LNEE 1263, pp. 481-486, doi 10.1007/978-3-031-71518-1_59).
- Setup: the TODO on the June MAC bound replaced (the µNAS searches enforce it).
- Results: Table 3 (`paper/tab_unas.tex`, generated by `scripts/make_tab_unas.py`) and two paragraphs; abstract and
  conclusion carry the F401 result; limitations: the short fitness, and the 8-bit loss over three seeds.
- Wording: "integer-only file" replaced by "8-bit file" with the exact content of the QDQ files (8-bit weights and 8-bit
  inputs to every convolution and to the output layer; activation functions and averages in floating point).
- `check_numbers.py`: about 30 new checks recompute every value of the new text from `unas_summary.json`,
  `int8_export.json`, `p1_summary.json`, the board run log and the per-node board reports.
- Course, lesson 7: QDQ description corrected the same way; new part "روش ۵" with the five encoders on both boards.

Matrix (resumed 14:25): 64-5-384 INT8 seed 42, 0.6219 (7.3 min).

## 2026-10-08 16:37 — Matrix: 64-5-384 and 96-2-384 arms done; 256-4-384 arm stopped (GPU memory spilling into RAM); allocator cap

Matrix cells, resumed 14:25 (`matrix_runs.json`; AUC, seeds 42 / 1 / 2, June single run in brackets):

| arm | FP32 | INT8 (simulated) | binary (simulated) |
|---|---|---|---|
| 64-5-384 | 0.6203 / 0.6181 / 0.6187, mean 0.619 (0.605) | 0.6219 / 0.6073 / 0.6136, mean 0.614 (0.610) | 0.5319 / 0.5573 / 0.5734, mean 0.554 (0.521) |
| 96-2-384 | 0.6046 / 0.5977 / 0.6028, mean 0.602 (0.593) | 0.6085 / 0.5875 / 0.6061, mean 0.601 (0.601) | 0.5666 / 0.5564 / 0.5497, mean 0.558 (0.546) |

The 256-4-384 arm started at 16:13. Its first epoch took 17 min (P0 measured 9.2 min per epoch). GPU at 100 % but 30-62 W
(compute-bound runs draw 93-103 W), 7.8 of 8 GB; free RAM 1.5 of 31.5 GB, commit charge 55.8 GB. Per-process GPU counters:
7.63 GB dedicated plus 12.76 GB shared (system RAM) for the matrix process. Stopped at 16:32 (queue and matrix; the
18 finished cells are kept, about 19 min of the first 256-4-384 cell lost) to avoid a repeat of the freeze of 10-05.

Cause, measured with a 6,000-impression run of 256-4-384 (one epoch): 3.30 GiB of tensors at the peak, 19.24 GiB
reserved by PyTorch's caching allocator. On Windows the driver backs allocations beyond the card with system RAM instead
of failing them, the allocator frees its cache only after a failed allocation, and the batches vary in size (distinct
titles per step), so the cache grows without bound. With the allocator capped at 0.85 of the card
(`torch.cuda.set_per_process_memory_fraction`): 6.68 GiB reserved, same 3.30 GiB allocated, 34 s instead of 170 s.

- `src/gpu.py` (`cap_gpu_memory`, default 0.85, env MIND_GPU_FRACTION) is called by run_matrix, run_unas_full and run_p1.
  The computation is unchanged; earlier results stand, but those runs held more RAM and time than needed (the 7.7 GB of
  the fork-form 64-5-384 runs today, possibly the 240-min P1 run of 10-06).
- Queue relaunched (`scripts/queue_step4.ps1`; finished jobs skip): matrix from the 256-4-384 arm, then the µNAS
  encoders with mixed-language clicks.

## 2026-10-08 16:47 — Matrix relaunched with the allocator cap; ReActNet rerun queued; Table 1 and Fig. 2 generators

- Matrix relaunched 16:37. 256-4-384, first epoch in about 4.5 min (17 min before the cap; P0 had estimated 9.2 min):
  GPU 98-99 %, 95 W, 5.6 GB dedicated, no shared memory, 12 GB of RAM free; epoch-1 loss 1.4529, the same as in the
  stopped run. Expected end about 19:40, then the µNAS mixed-language runs (about 1 h).
- The matrix rerun changes a claim of the paper: the naive binary 64-5-384 scores 0.532 / 0.557 / 0.573 (mean 0.554)
  over three seeds, not "near chance" (June single run 0.521). The ReActNet encoder has only the June single run
  (0.572). `scripts/run_binary.py` now runs it with seeds 42, 1, 2 under the October loop (same programme: distillation
  15 epochs, English clicks 8 epochs; `paper/results/binary_reactnet.json`; CPU smoke run 2.1 min). Queued after the
  current queue.
- `scripts/make_tab_matrix.py` writes Table 1 (`paper/tab_matrix.tex`) from `matrix_summary.json`;
  `scripts/make_figures.py` draws Fig. 2 from the rerun (mean and SD over seeds) and the three-seed reduced NRMS when
  the summary exists. Paper, checker and course follow when the matrix is complete.
