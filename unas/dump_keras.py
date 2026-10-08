"""Dump Keras builds of the chosen µNAS architectures, for the check of their PyTorch port.

For each search given, the architecture chosen by unas/select_by_seeds.py is built with the
fork's Keras builder (FaithfulGapCnn1DArchitecture.to_keras_model, input (128, 65), 384 outputs),
every weight is set to a seeded random value (BatchNorm variances and scales positive, so the
normalisation is not the identity), and the model is run in inference mode on 64 validation rows.
Written to artifacts/unas/port/<name>_keras.npz: the architecture (JSON), the weights of each
layer in model order, the byte ids, the Keras input and output. scripts/check_unas_port.py loads
the same weights into src/unas_encoder.py and compares the outputs.

CPU only (the GPU belongs to the training job). Run in WSL inside the MIND fork copy:
    cd ~/uNAS_mind && CUDA_VISIBLE_DEVICES= ~/dmir_nas/bin/python /mnt/c/Projects/PhD/MIND/unas/dump_keras.py mind_h7 mind_f401
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from configs.cnn1d_gap import FaithfulGapCnn1DArchitecture  # noqa: E402

REPO = Path("/mnt/c/Projects/PhD/MIND")
DATA = REPO / "artifacts/unas/data"
OUT = REPO / "artifacts/unas/port"
OUT.mkdir(parents=True, exist_ok=True)

table = np.load(DATA / "byte_table.npy").astype(np.float32)
ids = np.load(DATA / "mind_unas_val.npz")["ids"][:64].astype(np.int64)
x = np.concatenate([table[ids], (ids != 0).astype(np.float32)[..., None]], axis=-1)   # (64, 128, 65)

for name in sys.argv[1:]:
    sel = json.loads((REPO / f"paper/results/unas/{name}_selection.json").read_text())
    hist = json.loads((REPO / f"paper/results/unas/{name}_history.json").read_text())
    arch = next(c["arch"] for c in hist["candidates"] if c["index"] == sel["best"])
    model = FaithfulGapCnn1DArchitecture(json.loads(json.dumps(arch))).to_keras_model(x.shape[1:], 384)
    rng = np.random.default_rng(0)
    arrays, layers = {}, []
    for li, layer in enumerate(model.layers):
        for w in layer.weights:
            n = w.path.split("/")[-1]
            if n in ("gamma", "moving_variance"):
                v = rng.uniform(0.5, 1.5, w.shape)
            else:
                v = rng.normal(0.0, 0.3, w.shape)
            w.assign(v.astype(np.float32))
        if layer.weights:
            for w in layer.weights:
                arrays[f"{li}:{w.path.split('/')[-1]}"] = w.numpy()
        layers.append({"index": li, "class": layer.__class__.__name__, "name": layer.name,
                       "weights": [w.path.split("/")[-1] for w in layer.weights]})
    y = model(x, training=False).numpy()
    np.savez(OUT / f"{name}_keras.npz", arch=json.dumps(arch), layers=json.dumps(layers),
             ids=ids, x=x, y=y, **arrays)
    print(f"{name}: candidate {sel['best']}, {len(layers)} layers, output {y.shape}, "
          f"params {model.count_params()} -> {OUT / f'{name}_keras.npz'}")
