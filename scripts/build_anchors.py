"""Precompute + cache frozen-teacher English anchor embeddings (train & dev).
Run once: python -m scripts.build_anchors
"""
from src.config import load_config
from src import teacher

cfg = load_config()
for split in ["train", "dev"]:
    nids, emb = teacher.build_anchors(cfg, split)
    print(f"[anchors] {split}: {len(nids)} nids, emb {emb.shape}")
print("ANCHORS OK")
