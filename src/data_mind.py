"""MIND parsing: news.tsv / behaviors.tsv -> framework-agnostic structures.

We deliberately return plain Python/pandas structures (nid lists, titles,
labels) rather than tensors, so the SAME data feeds both the NRMS baseline
(word tokens) and the byte-level student (UTF-8 bytes). Tensorization happens
in the encoder modules.

MIND schema (tab-separated, header-less)
  news.tsv      : id, category, subcategory, title, abstract, url,
                  title_entities, abstract_entities
  behaviors.tsv : impression_id, user_id, time, history, impressions
                  history     = space-separated clicked nids (may be empty)
                  impressions = space-separated 'Nxxxx-1' / 'Nxxxx-0'
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

NEWS_COLS = ["nid", "category", "subcategory", "title", "abstract", "url",
             "title_entities", "abstract_entities"]
BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def _split_dir(cfg: dict, split: str) -> Path:
    return Path(cfg["paths"]["data_dir"]) / "mind" / cfg["data"]["mind_size"] / split


def read_news(cfg: dict, split: str) -> dict[str, dict]:
    """Return ``nid -> {category, subcategory, title, abstract}``."""
    path = _split_dir(cfg, split) / "news.tsv"
    df = pd.read_csv(path, sep="\t", names=NEWS_COLS, quoting=3, dtype=str).fillna("")
    df = df.set_index("nid")
    return {
        nid: {"category": r.category, "subcategory": r.subcategory,
              "title": r.title, "abstract": r.abstract}
        for nid, r in df.iterrows()
    }


def read_behaviors(cfg: dict, split: str) -> list[dict]:
    """Return a list of impressions with parsed history and candidates."""
    path = _split_dir(cfg, split) / "behaviors.tsv"
    df = pd.read_csv(path, sep="\t", names=BEHAVIOR_COLS, quoting=3, dtype=str)
    out = []
    for r in df.itertuples(index=False):
        history = r.history.split() if isinstance(r.history, str) and r.history else []
        cands, labels = [], []
        for tok in (r.impressions.split() if isinstance(r.impressions, str) else []):
            nid, _, lab = tok.rpartition("-")
            cands.append(nid)
            labels.append(int(lab) if lab in ("0", "1") else -1)  # -1 = hidden (test)
        out.append({"user": r.user_id, "history": history,
                    "cands": cands, "labels": labels})
    return out


def build_train_samples(behaviors: list[dict], neg_ratio: int, max_history: int,
                        seed: int) -> list[dict]:
    """NRMS-style instances: one positive + ``neg_ratio`` sampled negatives.

    Each instance = {history, cands=[pos, neg...], label=0 (pos is index 0)}.
    """
    rng = random.Random(seed)
    samples = []
    for imp in behaviors:
        pos = [c for c, l in zip(imp["cands"], imp["labels"]) if l == 1]
        neg = [c for c, l in zip(imp["cands"], imp["labels"]) if l == 0]
        if not pos or not neg:
            continue
        hist = imp["history"][-max_history:]
        for p in pos:
            chosen = [rng.choice(neg) for _ in range(neg_ratio)]
            samples.append({"history": hist, "cands": [p] + chosen, "label": 0})
    return samples


def build_eval_impressions(behaviors: list[dict], max_history: int) -> list[dict]:
    """Labeled impressions for ranking eval: {history, cands, labels}."""
    out = []
    for imp in behaviors:
        if not imp["cands"] or all(l < 0 for l in imp["labels"]):
            continue
        out.append({"history": imp["history"][-max_history:],
                    "cands": imp["cands"], "labels": imp["labels"]})
    return out


def category_vocab(news: dict[str, dict]) -> list[str]:
    """Sorted unique categories -> used for the cold-start topicWeights prior."""
    return sorted({v["category"] for v in news.values() if v["category"]})
