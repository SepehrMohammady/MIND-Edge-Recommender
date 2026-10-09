"""Offline replay of MIND clicks through the FeedWell-Edge learner (schedule step 7).

Every impression of a MINDsmall split is scored in time order by the app's own ranking rule
(src/app_learner.py, checked against the app's JavaScript by scripts/check_app_learner.py) before the
learner sees that impression's clicks:

  topic        the MIND category (the app uses an RSS item's first category, else the feed title);
               a second run uses the MIND subcategory
  history      the impression's history column, replayed as "open" events, then the clicks of the user's
               earlier impressions in the same split (one learning step per impression, as the app runs one
               after reading)
  freshness    hours since the news first appears as a candidate in MINDsmall (train or dev): MIND has no
               publication time, so first appearance stands in for it
  read         the candidate is in the user's history or was clicked earlier
  content      the app encoder (8-bit ONNX file of artifacts/app/edge_encoder_v1, the one on the phone) gives
               every title a vector; the trained user encoder of the same checkpoint pools the last 50 clicked
               titles; content score = dot product. app + content: app score + lambda * z-score of the content
               scores within the impression (lambda chosen on the train split, reported on dev)

Clicks only: MIND has no reading time or scroll depth, so only "open" events are replayed. AUC counts ties
as one half; MRR and nDCG break ties in a seeded random order (the app's score ties within a topic).
Writes paper/results/app_replay.json and one record per split to experiments.jsonl.

    python -m scripts.replay_app          (CPU)
"""
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from src import metrics, recommender, runlog
from src.app_learner import AppLearner
from src.config import load_config
from src.student import ByteCNNEncoder, text_to_bytes

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "artifacts/app/edge_encoder_v1"
CKPT = ROOT / "artifacts/runs/p1/distill_ft_mixed_seed42.pt"
LAMBDAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
MAX_HISTORY = 50
cfg = load_config()
cfg["seed"] = 42
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f}m] {msg}", flush=True)


def read_split(split):
    d = ROOT / "data/mind/small" / split
    beh = pd.read_csv(d / "behaviors.tsv", sep="\t", header=None, quoting=3, dtype=str,
                      names=["iid", "user", "time", "history", "imps"])
    news = pd.read_csv(d / "news.tsv", sep="\t", header=None, quoting=3, dtype=str,
                       names=["nid", "cat", "subcat", "title", "abstract", "url", "te", "ae"])
    beh["t"] = pd.to_datetime(beh["time"], format="%m/%d/%Y %I:%M:%S %p")
    return beh, news


log("reading MINDsmall")
splits = {s: read_split(s) for s in ("train", "dev")}
news = pd.concat([n for _, n in splits.values()]).drop_duplicates("nid").set_index("nid")
first_seen = {}
for beh, _ in splits.values():
    for t, imps in zip(beh["t"], beh["imps"]):
        for tok in imps.split():
            nid = tok[:-2]
            if nid not in first_seen or t < first_seen[nid]:
                first_seen[nid] = t

log(f"encoding {len(news)} titles with the app's 8-bit file")
table = np.fromfile(APP / "byte_table.f32", dtype="<f4").reshape(257, 64)
sess = ort.InferenceSession(str(APP / "news_encoder_int8.onnx"), providers=["CPUExecutionProvider"])
L = cfg["data"]["max_title_bytes"]
nids = list(news.index)
row = {nid: i for i, nid in enumerate(nids)}
vecs = np.zeros((len(nids), 384), dtype=np.float32)
for i, title in enumerate(news["title"].fillna("")):
    ids = np.array([text_to_bytes(title, L)], dtype=np.int64)
    if ids.any():
        x = np.ascontiguousarray(table[ids].transpose(0, 2, 1)).astype(np.float32)
        vecs[i] = sess.run(None, {"embedded_bytes": x, "mask": (ids != 0).astype(np.float32)[:, None, :]})[0][0]

sd = torch.load(CKPT, map_location="cpu")
user_model = recommender.NewsRecommender(ByteCNNEncoder(64, 64, 5, 384))
user_model.load_state_dict(sd)
user_model.eval()
V = torch.from_numpy(vecs)


@torch.no_grad()
def user_vector(clicked_rows):
    h = V[clicked_rows[-MAX_HISTORY:]][None]
    return user_model.user_vector(h, torch.ones(1, h.shape[1]))[0].numpy()


TOPIC = {f: dict(zip(news.index, news[f].fillna("unknown").astype(str).str.lower())) for f in ("cat", "subcat")}


def replay(split):
    """One pass over a split in time order; app scores for both topic fields, content scores once."""
    beh, _ = splits[split]
    order = np.argsort(beh["t"].values, kind="stable")
    learners, clicked, uvec = {}, defaultdict(list), {}
    rng = np.random.default_rng(42)
    rows = []
    users, hists, times, imps = beh["user"].values, beh["history"].values, beh["t"].values, beh["imps"].values
    for k in order:
        user = users[k]
        if user not in learners:
            hist = [n for n in (hists[k].split() if isinstance(hists[k], str) else []) if n in row]
            learners[user] = {}
            for f in ("cat", "subcat"):
                learners[user][f] = AppLearner()
                learners[user][f].learn([{"type": "open", "topic": TOPIC[f][n]} for n in hist])
            clicked[user] = [row[n] for n in hist]
        if user not in uvec:
            uvec[user] = user_vector(clicked[user]) if clicked[user] else None
        toks = imps[k].split()
        cands = [t[:-2] for t in toks]
        labels = np.array([int(t[-1]) for t in toks])
        read = set(clicked[user])
        t = pd.Timestamp(times[k])
        hours = [(t - first_seen[n]).total_seconds() / 3600 for n in cands]
        app = {f: np.array([learners[user][f].score(TOPIC[f][n], h, row[n] in read) for n, h in zip(cands, hours)])
               for f in ("cat", "subcat")}
        u = uvec[user]
        content = vecs[[row[n] for n in cands]] @ u if u is not None else np.zeros(len(cands), dtype=np.float32)
        rows.append({"labels": labels, "app": app, "content": content.astype(np.float64),
                     "jitter": rng.random(len(cands)), "hist": len(clicked[user])})
        new = [n for n, y in zip(cands, labels) if y == 1]
        if new:
            for f in ("cat", "subcat"):
                learners[user][f].learn([{"type": "open", "topic": TOPIC[f][n]} for n in new])
            clicked[user].extend(row[n] for n in new)
            uvec.pop(user, None)
    return rows


def evaluate(rows, kind, field, lam=None, full=True):
    """AUC (ties count one half); with full=True also MRR and nDCG with ties broken in a seeded random order."""
    scored = []
    for r in rows:
        if kind == "app":
            s = r["app"][field]
        elif kind == "content":
            s = r["content"]
        else:
            c = r["content"]
            z = (c - c.mean()) / c.std() if c.std() > 0 else np.zeros_like(c)
            s = r["app"][field] + lam * z
        scored.append((r["labels"], s, r["jitter"]))
    aucs = [metrics._auc(y.astype(float), s) for y, s, _ in scored if 0 < y.sum() < len(y)]
    out = {"auc": round(float(np.mean(aucs)), 4), "n_impressions": len(aucs)}
    if full:
        tb = metrics.evaluate([{"labels": y, "scores": s + 1e-9 * j} for y, s, j in scored])
        out.update({k: round(tb[k], 4) for k in ("mrr", "ndcg@5", "ndcg@10")})
    return out


results = {"lambdas": LAMBDAS, "runs": {f: {} for f in ("cat", "subcat")}}
for split in ("train", "dev"):
    started = time.time()
    log(f"replay {split}")
    rows = replay(split)
    for f in ("cat", "subcat"):
        full = split == "dev"                      # train only chooses lambda: AUC is enough there
        out = {"app": evaluate(rows, "app", f, full=full), "content": evaluate(rows, "content", f, full=full),
               "app_plus_content": {str(l): evaluate(rows, "mix", f, l, full=full) for l in LAMBDAS},
               "impressions": len(rows), "with_history": int(sum(r["hist"] > 0 for r in rows))}
        results["runs"][f][split] = out
        log(f"{split}/{f}: app {out['app']['auc']}, content {out['content']['auc']}, "
            + ", ".join(f"l={l}: {out['app_plus_content'][str(l)]['auc']}" for l in LAMBDAS))
    minutes = round((time.time() - started) / 60, 2)
    for f in ("cat", "subcat"):
        results["runs"][f][split]["minutes_for_both_topic_fields"] = minutes
        runlog.append(cfg, f"replay/{f}/{split}", {"topic": f, "max_history": MAX_HISTORY,
                      "encoder": "artifacts/app/edge_encoder_v1/news_encoder_int8.onnx", "checkpoint": CKPT.name,
                      "lambdas": LAMBDAS}, results["runs"][f][split], started)
for f in ("cat", "subcat"):
    run = results["runs"][f]
    best = max(LAMBDAS, key=lambda l: run["train"]["app_plus_content"][str(l)]["auc"])
    run["lambda_chosen_on_train"] = best
    run["dev_with_chosen_lambda"] = run["dev"]["app_plus_content"][str(best)]
    log(f"{f}: lambda chosen on train {best}; dev app {run['dev']['app']['auc']} -> "
        f"app + content {run['dev_with_chosen_lambda']['auc']} (content alone {run['dev']['content']['auc']})")
results["minutes"] = round((time.time() - T0) / 60, 2)
(ROOT / "paper/results/app_replay.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
log("REPLAY DONE")
