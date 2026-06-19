"""Impression-level ranking metrics (the MIND standard).

For each impression we score every candidate, then compute AUC / MRR /
nDCG@5 / nDCG@10 against the binary click labels, and average across
impressions. This matches Wu et al. (ACL 2020) and the MIND leaderboard.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _dcg(labels: np.ndarray, k: int) -> float:
    labels = labels[:k]
    gains = (2 ** labels - 1) / np.log2(np.arange(2, labels.size + 2))
    return float(gains.sum())


def _ndcg(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    order = np.argsort(y_score)[::-1]
    ideal = np.argsort(y_true)[::-1]
    idcg = _dcg(y_true[ideal], k)
    return _dcg(y_true[order], k) / idcg if idcg > 0 else 0.0


def _mrr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)[::-1]
    ranked = y_true[order]
    hits = np.where(ranked == 1)[0]
    return float(1.0 / (hits[0] + 1)) if hits.size else 0.0


def evaluate(impressions: list[dict]) -> dict[str, float]:
    """``impressions`` = list of {'labels': [...], 'scores': [...]}.

    Impressions with only one label class are skipped for AUC (undefined) but
    still count for MRR/nDCG.
    """
    aucs, mrrs, n5, n10 = [], [], [], []
    for imp in impressions:
        y = np.asarray(imp["labels"], dtype=float)
        s = np.asarray(imp["scores"], dtype=float)
        if y.size < 2 or len(set(y.tolist())) < 2:
            continue
        aucs.append(roc_auc_score(y, s))
        mrrs.append(_mrr(y, s))
        n5.append(_ndcg(y, s, 5))
        n10.append(_ndcg(y, s, 10))
    return {
        "auc": float(np.mean(aucs)) if aucs else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        "ndcg@5": float(np.mean(n5)) if n5 else 0.0,
        "ndcg@10": float(np.mean(n10)) if n10 else 0.0,
        "n_impressions": len(aucs),
    }
