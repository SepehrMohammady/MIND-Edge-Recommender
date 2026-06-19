"""Frozen multilingual TEACHER (runs offline only; never shipped on-device).

We use a multilingual sentence-transformer to produce a 384-d *anchor*
embedding for each MIND article's ENGLISH title. Because xMIND titles are
parallel translations of the same articles, every language's translation is
distilled to map to this same English anchor (Reimers & Gurevych, EMNLP 2020)
-> one language-agnostic student usable across all 14 languages.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src import data_mind


def get_teacher(cfg: dict):
    """Load the frozen sentence-transformer onto the best device."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu"
    model = SentenceTransformer(cfg["teacher"]["model"], device=device)
    model.eval()
    return model


def embed_texts(model, texts: list[str], batch_size: int) -> np.ndarray:
    """Encode a list of texts to a (N, embed_dim) float32 array."""
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=True).astype("float32")


def build_anchors(cfg: dict, split: str) -> tuple[list[str], np.ndarray]:
    """Compute (and cache) English-title anchor embeddings for one split.

    Returns ``(nids, embeddings)`` aligned by index. Cached under artifacts/.
    """
    art = Path(cfg["paths"]["artifacts_dir"])
    npy = art / f"teacher_anchor_{cfg['data']['mind_size']}_{split}.npy"
    ids = art / f"teacher_anchor_{cfg['data']['mind_size']}_{split}.nids.txt"
    if npy.exists() and ids.exists():
        nids = ids.read_text(encoding="utf-8").split()
        return nids, np.load(npy)

    news = data_mind.read_news(cfg, split)
    nids = list(news.keys())
    titles = [news[n]["title"] for n in nids]
    model = get_teacher(cfg)
    emb = embed_texts(model, titles, cfg["teacher"]["batch_size"])

    np.save(npy, emb)
    ids.write_text(" ".join(nids), encoding="utf-8")
    return nids, emb
