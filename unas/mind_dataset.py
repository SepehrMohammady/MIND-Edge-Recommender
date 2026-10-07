"""MIND byte-encoder dataset for the ELIOS uNAS fork (copied into the fork copy as
dataset/mind_dataset.py by unas/run_mind_search.sh).

Rows come from scripts/unas_export_data.py: title byte ids (128) in one of 15 languages and
the teacher's L2-normalised 384-d embedding of the English title of the same article. The
network input is the embedded sequence plus a padding mask, (128, 65): the 257 x 64 byte
table is fixed (the distilled student's) and applied here, as the caller applies it on the
board (scripts/deploy_boards.py).

task = "embedding": unas/patch_trainer.py makes the fork's trainer use a cosine loss for this
task and report val_error = 1 - max(val mean cosine).
"""
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from uNAS.dataset import Dataset

DATA_ROOT = Path(os.environ.get("MIND_UNAS_DATA", "/mnt/c/Projects/PhD/MIND/artifacts/unas/data"))


class MIND_Embedding_Dataset(Dataset):
    task = "embedding"

    def __init__(self, root=None):
        r = Path(root or DATA_ROOT)
        self.table = np.load(r / "byte_table.npy").astype(np.float32)          # (257, 64), row 0 = padding
        self._data = {}
        for split in ("train", "val", "test"):
            z = np.load(r / f"mind_unas_{split}.npz")
            self._data[split] = (z["ids"].astype(np.int32), z["y"].astype(np.float32))
        seq_len = self._data["train"][0].shape[1]
        self._input_shape = (seq_len, self.table.shape[1] + 1)
        self._num_classes = self._data["train"][1].shape[1]

    def _ds(self, split):
        ids, y = self._data[split]
        table = tf.constant(self.table)

        def embed(i, t):
            x = tf.gather(table, i)                                            # (L, 64)
            m = tf.cast(tf.not_equal(i, 0), tf.float32)[:, None]               # (L, 1)
            return tf.concat([x, m], axis=-1), t

        return tf.data.Dataset.from_tensor_slices((ids, y)).map(embed, num_parallel_calls=tf.data.AUTOTUNE)

    def train_dataset(self) -> tf.data.Dataset:
        return self._ds("train")

    def validation_dataset(self) -> tf.data.Dataset:
        return self._ds("val")

    def test_dataset(self) -> tf.data.Dataset:
        return self._ds("test")

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self._input_shape
