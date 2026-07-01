"""Remainder of the reviewer experiments (memory-safe): binary multi-seed +
Matryoshka + cold-start. micro_int8 / nas_int8 multi-seed already in
docs/run_log.md. Writes artifacts/reviews2.json + appends to docs/run_log.md.
Run: python -m scripts.run_reviews2
"""
import json
import random
import statistics as st
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import load_config
from src import recommender, student, metrics, export
import src.data_mind as dm
from src.nas.search import build_encoder
from src.binary import BinaryByteCNNEncoder
from src.student import ByteCNNEncoder, text_to_bytes

cfg = load_config()
device = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 1, 2]
MICRO = {"channels": 64, "depth": 5, "out_dim": 384}
TRAIN_EP, DISTILL_EP, TRAIN_IMPR, EVAL_IMPR = 6, 8, 40000, 8000
ART = Path(cfg["paths"]["artifacts_dir"])
LOG = Path("docs/run_log.md")
t0 = time.time()
results = {}


def log(msg):
    line = f"- [{(time.time()-t0)/60:.1f}m] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    cfg["seed"] = s


# --- 1. binary multi-seed (build distill data once) ---
bytes_np, tgt_np, anchors = student.build_distill_data(cfg, "train")
anchors_t = torch.tensor(anchors, device=device)
log("binary: distill data built")


def binary_auc():
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.permutation(len(bytes_np))[:120000]
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(bytes_np[idx], dtype=torch.long),
                                       torch.tensor(tgt_np[idx], dtype=torch.long)),
        batch_size=512, shuffle=True, drop_last=True)
    enc = BinaryByteCNNEncoder(cfg["student"]["byte_embed_dim"], **MICRO).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=cfg["train"]["distill_lr"])
    dims = (64, 128, 256, 384)
    enc.train()
    for _ in range(DISTILL_EP):
        for ids, ti in dl:
            ids = ids.to(device); tgt = anchors_t[ti.to(device)]
            loss = sum(1 - F.cosine_similarity(F.normalize(enc(ids), dim=-1)[:, :d], tgt[:, :d], dim=-1).mean()
                       for d in dims) / len(dims)
            opt.zero_grad(); loss.backward(); opt.step()
    m = recommender.train_recommender(cfg, model=recommender.NewsRecommender(enc),
                                      epochs=TRAIN_EP, max_train_impressions=TRAIN_IMPR)
    auc = recommender.evaluate(cfg, m, split="dev", max_impressions=EVAL_IMPR)["auc"]
    del enc, m, dl
    torch.cuda.empty_cache()
    return auc


vals = []
for s in SEEDS:
    ts = time.time(); seed_all(s); a = round(binary_auc(), 4); vals.append(a)
    log(f"binary micro seed {s}: AUC={a} ({(time.time()-ts)/60:.1f}m)")
results["binary_multiseed"] = {"seeds": vals, "mean": round(st.mean(vals), 4), "std": round(st.pstdev(vals), 4)}
(ART / "reviews2.json").write_text(json.dumps(results, indent=2))
log(f"binary micro: mean={results['binary_multiseed']['mean']} std={results['binary_multiseed']['std']}")

# --- 2. Matryoshka (cached distilled student) ---
seed_all(42)
s = cfg["student"]
enc = ByteCNNEncoder(s["byte_embed_dim"], s["channels"], s["depth"], s["out_dim"])
enc.load_state_dict(torch.load(ART / "student.pt", map_location="cpu"))
enc = enc.to(device).eval()
nids, anc = student.teacher.build_anchors(cfg, "dev")
news = dm.read_news(cfg, "dev"); L = cfg["data"]["max_title_bytes"]
X = torch.tensor([text_to_bytes(news[n]["title"], L) for n in nids[:8000]], dtype=torch.long, device=device)
A = torch.tensor(anc[:8000], device=device)
with torch.no_grad():
    P = enc(X)
results["matryoshka"] = {str(d): round(F.cosine_similarity(P[:, :d], A[:, :d], dim=-1).mean().item(), 4)
                         for d in (64, 128, 256, 384)}
(ART / "reviews2.json").write_text(json.dumps(results, indent=2))
log(f"matryoshka: {results['matryoshka']}")
del enc, P; torch.cuda.empty_cache()

# --- 3. cold-start ---
seed_all(42)
m = recommender.train_recommender(cfg, news_encoder=build_encoder(cfg, MICRO),
                                  epochs=TRAIN_EP, max_train_impressions=TRAIN_IMPR)
warm = recommender.evaluate(cfg, m, split="dev", max_impressions=EVAL_IMPR, mask_history=False)["auc"]
masked = recommender.evaluate(cfg, m, split="dev", max_impressions=EVAL_IMPR, mask_history=True)["auc"]
prior = export.build_topic_prior(cfg)
news_dev = dm.read_news(cfg, "dev")
imps = dm.build_eval_impressions(dm.read_behaviors(cfg, "dev"), cfg["data"]["max_history"])[:EVAL_IMPR]
scored = [{"labels": imp["labels"],
           "scores": [prior.get(news_dev.get(c, {}).get("category", ""), 0.0) for c in imp["cands"]]}
          for imp in imps]
prior_auc = metrics.evaluate(scored)["auc"]
results["coldstart"] = {"with_history": round(warm, 4), "masked_history": round(masked, 4),
                        "topic_prior": round(prior_auc, 4)}
(ART / "reviews2.json").write_text(json.dumps(results, indent=2))
log(f"coldstart: with_history={round(warm,4)} masked={round(masked,4)} topic_prior={round(prior_auc,4)}")
log("REVIEWS2 DONE")
