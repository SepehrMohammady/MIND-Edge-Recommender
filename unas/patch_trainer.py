"""Patch of the MIND copy of the uNAS fork: an "embedding" task in the trainer.

The fork's ModelTrainer knows classification (val_error = 1 - accuracy) and scalar
regression (val_error = MAE). The MIND search distils a 384-d sentence embedding, so for a
dataset with task == "embedding" the patched trainer compiles with the Keras 3 cosine loss and
metric, and reports

    val_error  = 1 + min(val_loss)  = 1 - best validation mean cosine
    test_error = 1 + test_loss      = 1 - test mean cosine

Errors come from the loss, not from the metric. In the search's Ray workers `tf.keras` resolves
to the legacy Keras 2 package (tensorflow_model_optimization is imported first), and a Keras 2
metric object handed to a Keras 3 model is wrapped as a plain function whose running mean is
never reset: it lagged the true cosine by 0.005 on validation and 0.03 on training, and the
"test" value repeated the validation one (first H7 search, 2026-10-07 17:35, stopped). The
first version of this patch read the metric; this version uses keras (Keras 3) objects and the loss.

Other datasets are unaffected. The patch is applied to a fresh copy of the shared fork's
trainer each time (unas/run_mind_search.sh setup); the shared fork (~/uNAS) is never touched.

Usage:  python3 patch_trainer.py ~/uNAS_mind/uNAS/model_trainer.py
"""
import sys
from pathlib import Path

MARK = "# MIND embedding task v3"
# v3 (2026-10-08): NaN-safe. A run stopped by TerminateOnNaN has NaN in its loss history; the
# best finite epoch counts, and a run with no finite epoch scores 1.0 (worst), instead of NaN.

COMPILE_OLD = "        if dataset.num_classes < 2:#Regression\n"
COMPILE_NEW = (
    f"        {MARK}\n"
    "        _embedding = getattr(dataset, 'task', None) == 'embedding'\n"
    "        if _embedding:\n"
    "            import keras as _k3\n"
    "            loss = _k3.losses.CosineSimilarity(axis=-1)\n"
    "            metric = _k3.metrics.CosineSimilarity(name='cos', axis=-1)\n"
    "        elif dataset.num_classes < 2:#Regression\n"
)

EVAL_OLD = "        _, test_acc = model.evaluate(test, verbose=0)\n"
EVAL_NEW = "        test_loss, test_acc = model.evaluate(test, verbose=0)\n"

ERROR_OLD = "        if(dataset.num_classes>=2):\n"
ERROR_NEW = (
    "        if _embedding:\n"
    "            import math as _m\n"
    "            _vl = [v for v in log.history['val_loss'][check_logs_from_epoch:] if _m.isfinite(v)]\n"
    "            val_error = 1.0 + min(_vl) if _vl else 1.0\n"
    "            test_error = 1.0 + test_loss if _m.isfinite(test_loss) else 1.0\n"
    "            return {'val_error': val_error, 'test_error': test_error,\n"
    "                    'pruned_weights': pruning_cb.weights if pruning_cb else None}\n"
    "        if(dataset.num_classes>=2):\n"
)


def main(path):
    p = Path(path)
    s = p.read_text()
    if MARK in s:
        print("already patched (v3)")
        return
    if "# MIND embedding task" in s:
        raise SystemExit("an older MIND patch is present; copy the shared fork's model_trainer.py first")
    for old in (COMPILE_OLD, EVAL_OLD, ERROR_OLD):
        assert s.count(old) == 1, f"trainer layout changed near {old.strip()!r}; patch by hand"
    s = s.replace(COMPILE_OLD, COMPILE_NEW).replace(EVAL_OLD, EVAL_NEW).replace(ERROR_OLD, ERROR_NEW)
    p.write_text(s)
    print("patched (v3)", p)


if __name__ == "__main__":
    main(sys.argv[1])
