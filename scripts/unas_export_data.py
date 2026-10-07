"""Data for the µNAS search of the byte-level encoder in the lab's uNAS fork (WSL).

Each row is a news title in one of 15 languages (English or one of the 14 xMIND
translations), as 128 UTF-8 byte ids, paired with the frozen teacher's embedding of
the English title of the same article (L2-normalised, 384-d): the multilingual
distillation target of src/student.py.

  train  96,000 rows sampled from the training split (all languages); MIND_UNAS_N_TRAIN overrides
         (the first recipe used 32,000, too few steps per epoch for BatchNorm's running statistics)
  val     4,000 rows from the dev split, news ids of one half of the dev articles
  test    4,000 rows from the dev split, news ids of the other half (disjoint from val)

The searched networks take the embedded sequence, as in the deployed files of
scripts/deploy_boards.py: the byte table (257 x 64) is fixed to the distilled
student's table (seed 42, October programme) and written next to the rows.

    python -m scripts.unas_export_data
Writes artifacts/unas/data/{mind_unas_train,mind_unas_val,mind_unas_test}.npz and
manifest.json. The script does not import torch; the byte table is extracted from the
checkpoint beforehand (one line, any interpreter with torch):
    python -c "import torch,numpy as np; np.save('artifacts/unas/data/byte_table.npy',
               torch.load('artifacts/runs/p1/student_distilled_seed42.pt',map_location='cpu')['embed.weight'].float().numpy())"
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src import data_mind, data_xmind, teacher
from src.config import load_config


def text_to_bytes(text: str, max_len: int) -> list[int]:
    """Same rule as src.student.text_to_bytes: UTF-8 bytes -> ids 1..256, 0 = padding."""
    ids = [x + 1 for x in text.encode("utf-8")[:max_len]]
    return ids + [0] * (max_len - len(ids))


def build_distill_data(cfg: dict, split: str):
    """Same rows as src.student.build_distill_data, without importing torch."""
    nids, anchors = teacher.build_anchors(cfg, split)          # cached on disk
    L = cfg["data"]["max_title_bytes"]
    en = data_mind.read_news(cfg, split)
    rows = [np.asarray([text_to_bytes(en[n]["title"], L) for n in nids], dtype=np.int16)]
    tgt = [np.arange(len(nids))]
    for lang in data_xmind.available_langs(cfg):
        loc = data_xmind.localized_news(cfg, lang, split)
        rows.append(np.asarray([text_to_bytes(loc[n]["title"], L) for n in nids], dtype=np.int16))
        tgt.append(np.arange(len(nids)))
    return np.concatenate(rows), np.concatenate(tgt), anchors

cfg = load_config()
OUT = Path("artifacts/unas/data")
OUT.mkdir(parents=True, exist_ok=True)
CKPT = Path("artifacts/runs/p1/student_distilled_seed42.pt")
N_TRAIN = int(__import__("os").environ.get("MIND_UNAS_N_TRAIN", "96000"))
N_VAL, N_TEST, SEED = 4_000, 4_000, 0
t0 = time.time()
rng = np.random.default_rng(SEED)
langs = ["en"] + list(data_xmind.available_langs(cfg))


def normalise(a):
    a = a.astype(np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)


def save(split, ids, y, lang, nid_idx):
    path = OUT / f"mind_unas_{split}.npz"
    np.savez_compressed(path, ids=ids.astype(np.int16), y=y.astype(np.float32),
                        lang=lang.astype(np.int8), nid_idx=nid_idx.astype(np.int32))
    return path


files = {}
# ---------------------------------------------------------------- train
b, tgt, anchors = build_distill_data(cfg, "train")
n_news = len(anchors)
rows = rng.choice(len(b), size=N_TRAIN, replace=False)
files["train"] = save("train", b[rows], normalise(anchors)[tgt[rows]], rows // n_news, tgt[rows])

# ---------------------------------------------------------------- val / test, disjoint by article
b, tgt, anchors = build_distill_data(cfg, "dev")
n_news = len(anchors)
perm = rng.permutation(n_news)
half = np.zeros(n_news, dtype=bool)
half[perm[: n_news // 2]] = True
for split, mask, n in (("val", half, N_VAL), ("test", ~half, N_TEST)):
    pool = np.flatnonzero(mask[tgt])
    rows = rng.choice(pool, size=n, replace=False)
    files[split] = save(split, b[rows], normalise(anchors)[tgt[rows]], rows // n_news, tgt[rows])

# ---------------------------------------------------------------- fixed byte table (extracted beforehand)
table = np.load(OUT / "byte_table.npy")
assert table.shape == (257, 64), table.shape
files["byte_table"] = OUT / "byte_table.npy"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


manifest = {"rows": {"train": N_TRAIN, "val": N_VAL, "test": N_TEST}, "seed": SEED, "languages": langs,
            "byte_table_from": str(CKPT), "byte_table_row0_norm": float(np.linalg.norm(table[0])),
            "target": "teacher anchor of the English title, L2-normalised, 384-d",
            "files": {k: {"path": str(p), "sha256": sha(p), "bytes": Path(p).stat().st_size} for k, p in files.items()},
            "minutes": round((time.time() - t0) / 60, 2)}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
z = np.load(files["train"])
print(json.dumps({k: manifest[k] for k in ("rows", "languages", "byte_table_row0_norm", "minutes")}))
print("train lang counts", np.bincount(z["lang"], minlength=len(langs)).tolist())
