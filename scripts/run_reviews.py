"""Reviewer-requested experiments (MINDsmall-dev):
  1. Multi-seed mean +/- std for the key configs (Micro-NAS INT8, NAS INT8,
     Micro-NAS Binary-improved).
  2. Matryoshka: cosine-to-anchor of the distilled student at truncated dims.
  3. Cold-start: AUC for cold users (no/short history) vs warm users.
Saves artifacts/reviews.json. Run: python -m scripts.run_reviews
"""
import json
import random
import statistics as st
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import load_config
from src import recommender, student, quantize
from src.nas.search import build_encoder
from src.binary import BinaryByteCNNEncoder

cfg = load_config()
device = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 1, 2]
MICRO = {"channels": 64, "depth": 5, "out_dim": 384}
NAS = {"channels": 256, "depth": 4, "out_dim": 384}
TRAIN_EP, QAT_EP, DISTILL_EP = 8, 2, 12


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    cfg["seed"] = s


def train_base(arch):
    return recommender.train_recommender(cfg, news_encoder=build_encoder(cfg, arch), epochs=TRAIN_EP)


def int8_auc(arch):
    m = train_base(arch)
    qm = quantize.convert_to_quant(m, "int8").to(device)
    qm = recommender.train_recommender(cfg, model=qm, epochs=QAT_EP)
    return recommender.evaluate(cfg, qm, split="dev")["auc"]


def binary_auc(arch):
    bn = student.build_distill_data  # noqa
    bytes_np, tgt_np, anchors = student.build_distill_data(cfg, "train")
    at = torch.tensor(anchors, device=device)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(bytes_np, dtype=torch.long),
                                       torch.tensor(tgt_np, dtype=torch.long)),
        batch_size=512, shuffle=True, drop_last=True)
    enc = BinaryByteCNNEncoder(cfg["student"]["byte_embed_dim"], **arch).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=cfg["train"]["distill_lr"])
    dims = (64, 128, 256, 384)
    enc.train()
    for _ in range(DISTILL_EP):
        for ids, ti in dl:
            ids = ids.to(device); tgt = at[ti.to(device)]
            pred = F.normalize(enc(ids), dim=-1)
            loss = sum(1 - F.cosine_similarity(pred[:, :d], tgt[:, :d], dim=-1).mean()
                       for d in dims) / len(dims)
            opt.zero_grad(); loss.backward(); opt.step()
    m = recommender.train_recommender(cfg, model=recommender.NewsRecommender(enc), epochs=TRAIN_EP)
    return recommender.evaluate(cfg, m, split="dev")["auc"]


def multiseed():
    out = {}
    for name, fn in [("micro_int8", lambda: int8_auc(MICRO)),
                     ("nas_int8", lambda: int8_auc(NAS)),
                     ("micro_binary_improved", lambda: binary_auc(MICRO))]:
        vals = []
        for s in SEEDS:
            seed_all(s)
            vals.append(round(fn(), 4))
            print(f"[{name}] seed {s}: {vals[-1]}", flush=True)
        out[name] = {"seeds": vals, "mean": round(st.mean(vals), 4),
                     "std": round(st.pstdev(vals), 4)}
    return out


def matryoshka():
    seed_all(42)
    enc = student.train_student(cfg).to(device).eval()
    nids, anc = student.teacher.build_anchors(cfg, "dev")
    import src.data_mind as dm
    news = dm.read_news(cfg, "dev")
    L = cfg["data"]["max_title_bytes"]
    from src.student import text_to_bytes
    X = torch.tensor([text_to_bytes(news[n]["title"], L) for n in nids[:8000]],
                     dtype=torch.long, device=device)
    A = torch.tensor(anc[:8000], device=device)
    with torch.no_grad():
        P = enc(X)
    return {str(d): round(F.cosine_similarity(P[:, :d], A[:, :d], dim=-1).mean().item(), 4)
            for d in (64, 128, 256, 384)}


def coldstart():
    seed_all(42)
    m = train_base(MICRO)
    warm = recommender.evaluate(cfg, m, split="dev", min_hist=5)["auc"]
    cold = recommender.evaluate(cfg, m, split="dev", max_hist=0)["auc"]
    short = recommender.evaluate(cfg, m, split="dev", min_hist=1, max_hist=4)["auc"]
    return {"cold_nohist": round(cold, 4), "short_1to4": round(short, 4), "warm_ge5": round(warm, 4)}


res = {"multiseed": multiseed(), "matryoshka": matryoshka(), "coldstart": coldstart()}
Path(cfg["paths"]["artifacts_dir"], "reviews.json").write_text(json.dumps(res, indent=2))
print("REVIEWS DONE:", json.dumps(res, indent=2))
