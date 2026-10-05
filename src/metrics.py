"""Impression-level ranking metrics (the MIND standard).

For each impression we score every candidate, then compute AUC / MRR /
nDCG@5 / nDCG@10 against the binary click labels, and average across
impressions. This matches Wu et al. (ACL 2020) and the MIND leaderboard.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC from average ranks (Mann-Whitney U). Tied scores count one half,
    which is what ``sklearn.metrics.roc_auc_score`` returns; this form avoids
    its per-call input validation, the slow part of a 73k-impression loop."""
    n_pos = float(y_true.sum())
    n_neg = y_true.size - n_pos
    ranks = rankdata(y_score)                      # average rank for ties
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


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


HISTORY_BUCKETS = {"0": (0, 0), "1-5": (1, 5), "6-20": (6, 20), ">20": (21, None)}


def bucket_of(history_len: int) -> str:
    """Cold-start bucket of a user by number of clicked articles in history."""
    for name, (lo, hi) in HISTORY_BUCKETS.items():
        if history_len >= lo and (hi is None or history_len <= hi):
            return name
    raise ValueError(history_len)


def by_history(impressions: list[dict], scored: list[dict]) -> dict[str, dict]:
    """Ranking metrics per history-length bucket and over all impressions.
    ``impressions`` carry 'history'; ``scored`` is aligned with them."""
    groups: dict[str, list[dict]] = {name: [] for name in HISTORY_BUCKETS}
    for imp, sc in zip(impressions, scored):
        groups[bucket_of(len(imp["history"]))].append(sc)
    out = {name: evaluate(rows) for name, rows in groups.items()}
    out["all"] = evaluate(scored)
    return out


def evaluate(impressions: list[dict]) -> dict[str, float]:
    """``impressions`` = list of {'labels': [...], 'scores': [...]}.

    Impressions with a single label class are skipped (AUC is undefined), so
    all four metrics are averaged over the same impressions.
    """
    aucs, mrrs, n5, n10 = [], [], [], []
    for imp in impressions:
        y = np.asarray(imp["labels"], dtype=float)
        s = np.asarray(imp["scores"], dtype=float)
        if y.size < 2 or y.min() == y.max():
            continue
        aucs.append(_auc(y, s))
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
