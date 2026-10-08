# µNAS search of the byte-level encoder (lab uNAS fork)

Step 3 of the schedule: search the encoder's network with the ELIOS lab fork of µNAS
(https://github.com/Elios-Lab/uNAS, TensorFlow), under the budgets of the two boards the
deployed files were measured on, with the same cost model as the Lane-Change-MCU searches.

The fork has no licence file, so it is not vendored. The shared checkout `~/uNAS` (WSL)
belongs to the Lane-Change-MCU project and carries its local patches; this project never
edits it. `run_mind_search.sh` copies it once to `~/uNAS_mind` and installs the files below
into the copy only. The venv `~/dmir_nas` (TF 2.21, Keras 3.12, GPU) is used as it is.

## What is searched

- **Input**: a title as 128 byte ids, embedded with the distilled student's fixed 257 × 64 byte
  table, plus a padding-mask channel: (128, 65). The caller applies the table on the board, as in
  `scripts/deploy_boards.py`, so the searched network is the deployed one.
- **Target**: the frozen teacher's L2-normalised 384-d embedding of the English title of the
  same article; titles are drawn from English and the 14 xMIND languages
  (`scripts/unas_export_data.py`: 96,000 training rows, 4,000 validation and 4,000 test rows from
  disjoint dev articles).
- **Fitness**: validation error = 1 − mean cosine to the target (the fork's trainer gets an
  "embedding" task, `patch_trainer.py`), plus the fork's three cost objectives from the faithful
  resource graph: peak activation memory, INT8 weight size, MACs.
- **Space**: the fork's 1D CNN space with a global-average-pooling head (`cnn1d_gap.py`, from the
  Lane-Change-MCU project): up to 10 convolution blocks of 1–3 layers (standard, 1×1 or depthwise,
  kernel 3–7, stride 1 or 2, optional pre-pooling, batch norm, ReLU), optional residual branches,
  pooling or GAP, 0–3 hidden dense layers, a 384-unit linear output.

## Budgets (`mind_config.py`)

| search | board | INT8 weights | activations | MACs per title |
|---|---|--:|--:|--:|
| `mind_h7` | STM32H7B3I-DK | 256 KiB | 128 KiB | 2.0 M |
| `mind_f401` | NUCLEO-F401RE | 128 KiB | 48 KiB | 1.0 M |

The hand-designed encoder (64-5-384) costs 51 KiB, 16 KiB and 3.31 M MACs under this cost model,
so it is outside both MAC budgets: the question is whether the search finds networks that keep
its quality at a fraction of its operations. The error bound is set from its validation error
under the search recipe (`eval_reference.py`).

## Files

| file | goes to (copy) | purpose |
|---|---|---|
| `mind_dataset.py` | `dataset/` | serves the exported rows, applies the byte table, adds the mask channel |
| `mind_config.py` | `configs/` | search configs, budgets, training recipe |
| `cnn1d_gap.py`, `safe_saver.py` | `configs/` | GAP-head space with faithful cost graph; saver with unique names and JSON sidecars (copied from the Lane-Change-MCU project) |
| `patch_trainer.py` | patches `uNAS/model_trainer.py` | cosine loss and error for the embedding task |
| `eval_reference.py` | run in the copy | June winners and the hand-designed family under the search recipe (`reference.json`, `grid.json`) |
| `harvest_search.py` | run in the copy | history and Pareto front of a search (`paper/results/unas/`) |
| `select_by_seeds.py` | run in the copy | final choice: eight best in-budget candidates re-trained with three seeds, validation only (`<name>_selection.json`) |
| `dump_keras.py` | run in the copy, CPU | Keras build of each chosen architecture with seeded random weights, for the check of the PyTorch build |
| `run_mind_search.sh` | WSL | `setup`, `reference`, `search <config>` (chunked, resumable, GPU sampler, logs in `logs/nas/`) |

## After the search (step 4)

The chosen architectures are rebuilt in PyTorch (`src/unas_encoder.py`, Keras semantics: "same"
padding with the odd pad on the right, BatchNorm epsilon 1e-3, pre-pooling, GAP over all
positions) so that they go through the same distillation, click training, evaluation and integer
export as the hand-designed encoder. `scripts/check_unas_port.py` copies the weights of the
`dump_keras.py` builds into the PyTorch ones and compares outputs on 64 validation rows
(`paper/results/unas/port_check.json`). Training: `scripts/run_unas_full.py`; integer files and
their accuracy: `scripts/export_int8.py`.

## Run

```bash
bash /mnt/c/Projects/PhD/MIND/unas/run_mind_search.sh reference
bash /mnt/c/Projects/PhD/MIND/unas/run_mind_search.sh search mind_h7
cd ~/uNAS_mind && ~/dmir_nas/bin/python /mnt/c/Projects/PhD/MIND/unas/harvest_search.py mind_h7
```

## Recipe history

1. 32,000 training rows, up to 10 epochs, early stopping with patience 3 (2026-10-07 17:08).
   The validation cosine of every reference model peaked in the first epoch (0.17–0.18) and dipped
   while the training cosine rose: BatchNorm's running statistics lag the weights over 125 steps
   per epoch, so early stopping kept epoch 1 and the fitness measured how fast a model starts.
   Results kept in `paper/results/unas/reference_recipe1.json`, not used.
2. 96,000 training rows, up to 15 epochs, early stopping from epoch 5 with patience 4
   (17:13). Training improved steadily, but the validation cosine of one model still moved by
   ±0.04 between epochs, and three seeds of the hand-designed encoder scored 0.309, 0.286 and 0.281.
   Kept in `reference_recipe2.json`, not used.
3. Recipe 2 with the learning rate held at 1e-3 for five epochs and multiplied by 0.7 per epoch
   afterwards (17:19). Three seeds of the hand-designed encoder: validation cosine 0.345, 0.342,
   0.343 (test 0.345, 0.345, 0.347), about 2.2 minutes each. This is the search recipe; the error
   bound is the mean validation error of these three runs, 0.657.

Because one run per candidate still carries some seed noise, the final choice from each search
is made afterwards by re-training the best candidates with several seeds, as in the
Lane-Change-MCU selection step.
