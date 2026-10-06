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
