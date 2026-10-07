"""Idempotent patch of the MIND copy of the uNAS fork: an "embedding" task in the trainer.

The fork's ModelTrainer knows classification (val_error = 1 - accuracy) and scalar
regression (val_error = MAE). The MIND search distils a 384-d sentence embedding, so for a
dataset with task == "embedding" the patched trainer compiles with a cosine loss and the
Keras mean cosine metric, and reports val_error = 1 - max(val mean cosine) and test_error
= 1 - test mean cosine. Other datasets are unaffected.

Only the MIND copy (~/uNAS_mind) is patched; the shared fork (~/uNAS) used by DIMIR is
never touched.

Usage:  python3 patch_trainer.py ~/uNAS_mind/uNAS/model_trainer.py
"""
import sys
from pathlib import Path

MARK = "# MIND embedding task"

COMPILE_OLD = "        if dataset.num_classes < 2:#Regression\n"
COMPILE_NEW = (
    f"        {MARK}\n"
    "        _embedding = getattr(dataset, 'task', None) == 'embedding'\n"
    "        if _embedding:\n"
    "            loss = tf.keras.losses.CosineSimilarity(axis=-1)\n"
    "            metric = tf.keras.metrics.CosineSimilarity(name='cos', axis=-1)\n"
    "        elif dataset.num_classes < 2:#Regression\n"
)

ERROR_OLD = "        if(dataset.num_classes>=2):\n"
ERROR_NEW = (
    "        if _embedding:\n"
    "            val_error = 1.0 - max(log.history['val_cos'][check_logs_from_epoch:])\n"
    "            return {'val_error': val_error, 'test_error': 1.0 - test_acc,\n"
    "                    'pruned_weights': pruning_cb.weights if pruning_cb else None}\n"
    "        if(dataset.num_classes>=2):\n"
)


def main(path):
    p = Path(path)
    s = p.read_text()
    if MARK in s:
        print("already patched")
        return
    assert s.count(COMPILE_OLD) == 1 and s.count(ERROR_OLD) == 1, "trainer layout changed; patch by hand"
    s = s.replace(COMPILE_OLD, COMPILE_NEW).replace(ERROR_OLD, ERROR_NEW)
    p.write_text(s)
    print("patched", p)


if __name__ == "__main__":
    main(sys.argv[1])
