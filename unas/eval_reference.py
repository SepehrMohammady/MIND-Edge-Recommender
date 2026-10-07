"""The June winners, written in the fork's 1D space, scored with the search recipe.

Each architecture is trained exactly as a search candidate (configs/mind_config.py: Adam,
batch 256, up to MIND_EPOCHS epochs with early stopping on the validation mean cosine), and
its cost is taken from the faithful resource graph that the search uses. The hand-designed
encoder 64-5-384 (the June constrained winner) is trained with three seeds to show how much
a single candidate's validation error moves between runs; 256-4-384 and 96-2-384 once.

The byte-level encoder in the fork's terms: a 1x1 convolution to C channels (the first
projection, no batch norm or ReLU), then D blocks of [depthwise convolution, kernel 3, and
1x1 convolution to C channels with batch norm and ReLU], global average pooling and a
384-unit linear output. Differences from the PyTorch encoder: the input has a 65th channel
(padding mask) and the pooling averages over all 128 positions, not only the non-padding ones.

Run inside the MIND fork copy (unas/run_mind_search.sh reference).
Writes /mnt/c/Projects/PhD/MIND/paper/results/unas/reference.json.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
import keras  # noqa: E402

from configs.cnn1d_gap import FaithfulGapCnn1DArchitecture  # noqa: E402
from configs.mind_config import EPOCHS, training_config  # noqa: E402
from dataset.mind_dataset import MIND_Embedding_Dataset  # noqa: E402
from uNAS.model_trainer import ModelTrainer  # noqa: E402
from uNAS.resource_models.models import inference_latency, model_size, peak_memory_usage  # noqa: E402

OUT = Path("/mnt/c/Projects/PhD/MIND/paper/results/unas/reference.json")
OUT.parent.mkdir(parents=True, exist_ok=True)


def byte_cnn(C, D):
    def pw(bn, relu):
        return {"type": "1x1Conv1D", "filters": C, "has_bn": bn, "has_relu": relu, "has_prepool": False}
    dw = {"type": "DWConv1D", "ker_size": 3, "1x_stride": False, "has_bn": False, "has_relu": False,
          "has_prepool": False}
    blocks = [{"is_branch": False, "layers": [pw(False, False)]}]
    blocks += [{"is_branch": False, "layers": [dict(dw), pw(True, True)]} for _ in range(D)]
    return {"conv_blocks": blocks, "pooling": {"type": "gap", "pool_size": 2}, "dense_blocks": [],
            "head_dropout": 0.0}


data = MIND_Embedding_Dataset()
trainer = ModelTrainer(training_config(data))
if len(sys.argv) > 1 and sys.argv[1] == "grid":
    # the hand-designed family under the search recipe: every channel width and depth of the June
    # space with the 384-d output all three June winners chose (6 x 5 = 30 architectures, seed 42)
    OUT = OUT.with_name("grid.json")
    runs = [(f"{C}-{D}-384", C, D, 42) for C in (32, 64, 96, 128, 192, 256) for D in (1, 2, 3, 4, 5)]
else:
    runs = [("64-5-384", 64, 5, s) for s in (42, 1, 2)] + [("256-4-384", 256, 4, 42), ("96-2-384", 96, 2, 42)]
results = json.loads(OUT.read_text()) if OUT.exists() else []
done = {(r["arch"], r["seed"]) for r in results}
for name, C, D, seed in runs:
    if (name, seed) in done:
        continue
    keras.utils.set_random_seed(seed)
    arch = FaithfulGapCnn1DArchitecture(byte_cnn(C, D))
    t0 = time.time()
    model = arch.to_keras_model(data.input_shape, data.num_classes)
    params = int(model.count_params())
    r = trainer.train_and_eval(model)
    rg = arch.to_resource_graph(data.input_shape, data.num_classes)
    row = {"arch": name, "seed": seed, "val_error": float(r["val_error"]), "test_error": float(r["test_error"]),
           "val_cos": 1 - float(r["val_error"]), "params": params, "peak_mem_bytes": int(peak_memory_usage(rg)),
           "model_size_bytes": int(model_size(rg)),
           "macs": int(inference_latency(rg, compute_weight=1, mem_access_weight=0)),
           "epochs_max": EPOCHS, "minutes": round((time.time() - t0) / 60, 2),
           "input_shape": list(data.input_shape)}
    results.append(row)
    OUT.write_text(json.dumps(results, indent=1))
    print(json.dumps(row), flush=True)
    keras.backend.clear_session()
