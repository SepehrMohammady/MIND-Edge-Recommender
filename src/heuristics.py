"""Training-free scorers on MIND metadata: popularity and category histograms.

They need no network and no gradient step, so they bound from below what a
learned content encoder has to beat, and the category histogram is the offline
counterpart of the per-topic weights that FeedWell-Edge keeps on the device.

  popularity      distinct train users who clicked the article
  category        share of the user's clicked history in the candidate's
                  category; users without history fall back to the population's
                  category click distribution
  subcategory     the same on MIND's finer subcategory label, backing off to
                  the category share when the subcategory is unseen
"""
from __future__ import annotations

from collections import Counter

from src import data_mind, metrics, recommender


def train_statistics(cfg: dict):
    """Popularity per article and the population's category click distribution,
    both from the train log (history and clicked impressions)."""
    news = data_mind.read_news(cfg, "train")
    clicked_by: dict[str, set] = {}
    for imp in data_mind.read_behaviors(cfg, "train"):
        clicks = list(imp["history"]) + [c for c, l in zip(imp["cands"], imp["labels"]) if l == 1]
        for nid in clicks:
            clicked_by.setdefault(nid, set()).add(imp["user"])
    popularity = {nid: len(users) for nid, users in clicked_by.items()}
    cat_clicks: Counter = Counter()
    for nid, n in popularity.items():
        cat = news.get(nid, {}).get("category", "")
        if cat:
            cat_clicks[cat] += n
    total = sum(cat_clicks.values()) or 1
    return popularity, {c: n / total for c, n in cat_clicks.items()}


def score_heuristics(cfg: dict, split: str = "dev") -> tuple[list[dict], dict[str, list[dict]]]:
    """Scores of every heuristic for every labeled impression of ``split``."""
    news = data_mind.read_news(cfg, split)
    impressions = recommender.eval_impressions(cfg, split)
    popularity, global_cat = train_statistics(cfg)
    scored = {name: [] for name in ("popularity", "category", "subcategory")}
    for imp in impressions:
        cats = [news[n]["category"] for n in imp["history"] if n in news]
        subs = [news[n]["subcategory"] for n in imp["history"] if n in news]
        cat_share = ({c: k / len(cats) for c, k in Counter(cats).items()} if cats else global_cat)
        sub_share = {s: k / len(subs) for s, k in Counter(subs).items()} if subs else {}
        cand_cat = [news.get(c, {}).get("category", "") for c in imp["cands"]]
        cand_sub = [news.get(c, {}).get("subcategory", "") for c in imp["cands"]]
        cat_score = [cat_share.get(c, 0.0) for c in cand_cat]
        scored["popularity"].append({"labels": imp["labels"],
                                     "scores": [popularity.get(c, 0) for c in imp["cands"]]})
        scored["category"].append({"labels": imp["labels"], "scores": cat_score})
        scored["subcategory"].append({"labels": imp["labels"],
                                      "scores": [1.0 + sub_share[s] if s in sub_share else cs
                                                 for s, cs in zip(cand_sub, cat_score)]})
    return impressions, scored


def evaluate_heuristics(cfg: dict, split: str = "dev") -> dict[str, dict]:
    impressions, scored = score_heuristics(cfg, split)
    return {name: metrics.by_history(impressions, rows) for name, rows in scored.items()}
