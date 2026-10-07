"""µNAS search configs for the MIND byte-level encoder (copied into the fork copy as
configs/mind_config.py by unas/run_mind_search.sh).

Search space: the fork's 1D CNN space with the global-average-pooling head and the faithful
resource graph (configs/cnn1d_gap.py, from the Lane-Change-MCU project). Fitness objectives,
as in the fork: validation error (1 - mean cosine to the teacher anchor, mixed languages),
peak activation memory, INT8 weight size and MACs, each normalised by its bound.

Budgets, one search per board (INT8 bytes and MACs per title):
  mind_h7    STM32H7B3I-DK  model 256 KiB, activations 128 KiB, 2.0 M MACs
  mind_f401  NUCLEO-F401RE  model 128 KiB, activations  48 KiB, 1.0 M MACs
The 2 M MAC bound is the one the June search was meant to enforce; at the measured 2.6
cycles per MAC on the H7 it is about 19 ms per title. On the F401 (11.2 cycles per MAC at
84 MHz) 1 M MACs is about 133 ms. The byte table (257 x 64) is held by the caller and is not
part of these budgets.

Error bound: MIND_ERROR_BOUND, set to the validation error of the hand-designed encoder
(64-5-384) under this recipe (unas/eval_reference.py), so the search asks whether it can
match that encoder within the board budgets.
"""
import os

import keras

from uNAS.config import AgingEvoConfig, BoundConfig, ModelSaverConfig, TrainingConfig
from uNAS.search_algorithms import AgingEvoSearch
from dataset.mind_dataset import MIND_Embedding_Dataset

ROUNDS = int(os.environ.get("MIND_ROUNDS", "150"))
POPULATION = int(os.environ.get("MIND_POPULATION", "40"))
SAMPLE = int(os.environ.get("MIND_SAMPLE", "12"))
EPOCHS = int(os.environ.get("MIND_EPOCHS", "15"))
PATIENCE = int(os.environ.get("MIND_PATIENCE", "4"))
START_STOPPING = int(os.environ.get("MIND_START_STOPPING", "5"))
LR, LR_DECAY = 1e-3, float(os.environ.get("MIND_LR_DECAY", "0.7"))   # constant for 5 epochs, then x0.7 per epoch


def lr_schedule(epoch, lr):
    return LR if epoch < 5 else LR * LR_DECAY ** (epoch - 4)
PARALLEL = int(os.environ.get("MIND_PARALLEL", "1"))
ERROR_BOUND = float(os.environ.get("MIND_ERROR_BOUND", "0.60"))
SUFFIX = os.environ.get("MIND_SUFFIX", "")

BUDGETS = {
    "h7": {"peak_mem": 128 * 1024, "model_size": 256 * 1024, "macs": 2_000_000},
    "f401": {"peak_mem": 48 * 1024, "model_size": 128 * 1024, "macs": 1_000_000},
}


def training_config(dataset, epochs=None):
    # keras.callbacks (Keras 3) as in DIMIR's configs: the fork's 1D models are Keras 3 models.
    # No stopping before START_STOPPING epochs: in the first epochs BatchNorm's running statistics
    # lag the weights and the validation cosine dips while training improves (reference runs of
    # 2026-10-07, first recipe), so an early stop would score how fast a model starts.
    # The learning rate is annealed after epoch 5: with a constant rate the validation cosine of one
    # model moved by +-0.04 between epochs and by 0.028 between seeds (second recipe), so the best
    # epoch's score was noisy.
    # Early stopping watches val_loss (= minus the validation mean cosine), the quantity the fitness
    # uses (unas/patch_trainer.py explains why the metric is not used).
    cbs = lambda: [
        keras.callbacks.LearningRateScheduler(lr_schedule),
        keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE, min_delta=0.002,
                                      restore_best_weights=True, start_from_epoch=START_STOPPING),
        keras.callbacks.TerminateOnNaN(),
    ]
    return TrainingConfig(dataset=dataset, optimizer="adam", callbacks=cbs,
                          epochs=epochs or EPOCHS, batch_size=256)


def _setup(board):
    from configs.cnn1d_gap import FaithfulGapCnn1DSearchSpace
    from configs.safe_saver import install
    install()                                   # unique file names and a JSON sidecar per saved model
    dataset = MIND_Embedding_Dataset()
    b = BUDGETS[board]
    name = f"mind_{board}{SUFFIX}"
    config = {
        "training_config": training_config(dataset),
        "bound_config": BoundConfig(error_bound=ERROR_BOUND, peak_mem_bound=b["peak_mem"],
                                    model_size_bound=b["model_size"], mac_bound=b["macs"]),
        "search_algorithm": AgingEvoSearch,
        "search_config": AgingEvoConfig(search_space=FaithfulGapCnn1DSearchSpace(),
                                        checkpoint_dir=f"artifacts/{name}", rounds=ROUNDS,
                                        population_size=POPULATION, sample_size=SAMPLE,
                                        max_parallel_evaluations=PARALLEL),
        # keep every candidate within the resource bounds; the final choice is made afterwards
        "model_saver_config": ModelSaverConfig(save_criteria="boundaries"),
        "serialized_dataset": False,
    }
    return {"config": config, "name": name, "load_from": None, "save_every": 5, "seed": 42}


def get_mind_h7_setup(**_):
    return _setup("h7")


def get_mind_f401_setup(**_):
    return _setup("f401")
