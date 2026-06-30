"""Reviewer experiments (FAST protocol; incrementally saved + timed):
  1. Multi-seed mean+/-std (Micro-NAS INT8, NAS INT8, Micro-NAS Binary-improved).
  2. Matryoshka: cosine-to-anchor of the distilled student at truncated dims.
  3. Cold-start: AUC with vs without user history (history masked).
Writes artifacts/reviews.json after each stage and appends timings to
docs/run_log.md. Run: python -m scripts.run_reviews
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
from src import recommender, student, quantize
from src.nas.search import build_encoder
from src.binary import BinaryByteCNNEncoder

cfg = load_config()
device = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 1, 2]
MICRO = {"channels": 64, "depth": 5, "out_dim": 384}
NAS = {"channels": 256, "depth": 4, "out_dim": 384}
TRAIN_EP, QAT_EP, DISTILL_EP = 6, 1, 8
TRAIN_IMPR, EVAL_IMPR = 40000, 8000
ART = Path(cfg["paths"]["artifacts_dir"])
LOG = Path("docs/run_log.md")
t0 = time.time()
results = {}


def log(msg):
    line = f"- [{(time.time()-t0)/60:.1f}m] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save():
    (ART / "reviews.json").write_text(json.dumps(results, indent=2))


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    cfg["seed"] = s


def train_base(arch):
    return recommender.train_recommender(cfg, news_encoder=build_encoder(cfg, arch),
                                         epochs=TRAIN_EP, max_train_impressions=TRAIN_IMPR)


def int8_auc(arch):
    m = train_base(arch)
    qm = quantize.convert_to_quant(m, "int8").to(device)
    qm = recommender.train_recommender(cfg, model=qm, epochs=QAT_EP, max_train_impressions=TRAIN_IMPR)
    return recommender.evaluate(cfg, qm, split="dev", max_impressions=EVAL_IMPR)["auc"]


def binary_auc(arch):
    bytes_np, tgt_np, anchors = student.build_distill_data(cfg, "train")
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.permutation(len(bytes_np))[:120000]
    bytes_np, tgt_np = bytes_np[idx], tgt_np[idx]
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
            loss = sum(1 - F.cosine_similarity(F.normalize(enc(ids), dim=-1)[:, :d], tgt[:, :d], dim=-1).mean()
                       for d in dims) / len(dims)
            opt.zero_grad(); loss.backward(); opt.step()
    m = recommender.train_recommender(cfg, model=recommender.NewsRecommender(enc),
                                      epochs=TRAIN_EP, max_train_impressions=TRAIN_IMPR)
    return recommender.evaluate(cfg, m, split="dev", max_impressions=EVAL_IMPR)["auc"]


# --- 1. multi-seed ---
ms = {}
for name, fn in [("micro_int8", lambda: int8_auc(MICRO)),
                 ("nas_int8", lambda: int8_auc(NAS)),
                 ("micro_binary_improved", lambda: binary_auc(MICRO))]:
    vals = []
    for s in SEEDS:
        ts = time.time(); seed_all(s); a = round(fn(), 4); vals.append(a)
        log(f"multiseed {name} seed {s}: AUC={a} ({(time.time()-ts)/60:.1f}m)")
    ms[name] = {"seeds": vals, "mean": round(st.mean(vals), 4), "std": round(st.pstdev(vals), 4)}
    results["multiseed"] = ms; save()
    log(f"multiseed {name}: mean={ms[name]['mean']} std={ms[name]['std']}")

# --- 2. Matryoshka (load cached distilled student; don't re-distill) ---
seed_all(42)
from src.student import ByteCNNEncoder
s = cfg["student"]
enc = ByteCNNEncoder(s["byte_embed_dim"], s["channels"], s["depth"], s["out_dim"])
sp = ART / "student.pt"
if sp.exists():
    enc.load_state_dict(torch.load(sp, map_location="cpu"))
else:
    enc = student.train_student(cfg)
enc = enc.to(device).eval()
nids, anc = student.teacher.build_anchors(cfg, "dev")
import src.data_mind as dm
from src.student import text_to_bytes
news = dm.read_news(cfg, "dev"); L = cfg["data"]["max_title_bytes"]
X = torch.tensor([text_to_bytes(news[n]["title"], L) for n in nids[:8000]], dtype=torch.long, device=device)
A = torch.tensor(anc[:8000], device=device)
with torch.no_grad():
    P = enc(X)
results["matryoshka"] = {str(d): round(F.cosine_similarity(P[:, :d], A[:, :d], dim=-1).mean().item(), 4)
                         for d in (64, 128, 256, 384)}
save(); log(f"matryoshka: {results['matryoshka']}")

# --- 3. cold-start: history-only model (warm vs masked) + topic-prior lift ---
seed_all(42)
m = train_base(MICRO)
warm = recommender.evaluate(cfg, m, split="dev", max_impressions=EVAL_IMPR, mask_history=False)["auc"]
masked = recommender.evaluate(cfg, m, split="dev", max_impressions=EVAL_IMPR, mask_history=True)["auc"]

# Cold-start prior: rank candidates by their category's population weight.
from src import export, metrics
import src.data_mind as dm2
prior = export.build_topic_prior(cfg)
news_dev = dm2.read_news(cfg, "dev")
imps = dm2.build_eval_impressions(dm2.read_behaviors(cfg, "dev"), cfg["data"]["max_history"])[:EVAL_IMPR]
scored = [{"labels": imp["labels"],
           "scores": [prior.get(news_dev.get(c, {}).get("category", ""), 0.0) for c in imp["cands"]]}
          for imp in imps]
prior_auc = metrics.evaluate(scored)["auc"]

results["coldstart"] = {"with_history": round(warm, 4),
                        "masked_history": round(masked, 4),
                        "topic_prior": round(prior_auc, 4)}
save(); log(f"coldstart: with_history={round(warm,4)} masked={round(masked,4)} topic_prior={round(prior_auc,4)}")
log("REVIEWS DONE")
