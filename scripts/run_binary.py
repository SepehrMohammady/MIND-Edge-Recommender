"""Improved-binary experiment: distil a ReActNet/Bi-Real byte-CNN to the teacher
anchors, fine-tune the recommender, evaluate on MINDsmall-dev. Compares against
the naive-binary numbers (Micro-NAS/Binary 0.521, Bin.uNAS/Binary 0.546).
Run: python -m scripts.run_binary
"""
import json
from pathlib import Path

from src.config import load_config, use_run_dir
from src.seed import seed_everything
from src import student, recommender
from src.binary import BinaryByteCNNEncoder

cfg = load_config()
use_run_dir(cfg, "binary")
seed_everything(cfg["seed"])
ARCH = dict(byte_embed_dim=64, channels=64, depth=5, out_dim=384)   # Micro-NAS shape
DISTILL_EP, TRAIN_EP = 15, 8

enc = student.distill_encoder(cfg, BinaryByteCNNEncoder(**ARCH), epochs=DISTILL_EP,
                              tag="bin-distill")

model = recommender.train_recommender(cfg, model=recommender.NewsRecommender(enc), epochs=TRAIN_EP)
res = recommender.evaluate(cfg, model, split="dev")
print("IMPROVED BINARY (distilled-init, ReActNet/Bi-Real):", res)
Path(cfg["paths"]["artifacts_dir"], "binary_improved.json").write_text(
    json.dumps({"arch": ARCH, "result": {k: round(v, 4) for k, v in res.items()}}, indent=2))
