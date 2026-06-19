"""The news recommender: student news encoder + attention user encoder.

  news vec  = ByteCNNEncoder(title bytes)
  user vec  = additive-attention pool over the user's clicked-history news vecs
  score     = dot(user, candidate)

Training is NRMS-style: each instance is 1 positive + K negatives, optimised
with softmax cross-entropy (positive at index 0). Evaluation is impression-
level ranking (AUC/MRR/nDCG) via src.metrics, and can swap the news text to any
xMIND language for cross-lingual evaluation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import data_mind, data_xmind, metrics
from src.student import ByteCNNEncoder, text_to_bytes


# --------------------------------------------------------------- news vocab
class NewsVocab:
    """Maps nid -> row index and holds the (N, L) byte matrix. Index 0 = PAD."""

    def __init__(self, news: dict[str, dict], max_len: int):
        self.nids = ["<PAD>"] + list(news.keys())
        self.idx = {n: i for i, n in enumerate(self.nids)}
        rows = [[0] * max_len] + [text_to_bytes(news[n]["title"], max_len)
                                  for n in self.nids[1:]]
        self.byte_matrix = torch.tensor(rows, dtype=torch.long)
        self.max_len = max_len

    def to_indices(self, nid_list: list[str]) -> list[int]:
        return [self.idx.get(n, 0) for n in nid_list]


# ------------------------------------------------------------------- model
class NewsRecommender(nn.Module):
    def __init__(self, news_encoder: nn.Module, attn_dim: int = 128):
        super().__init__()
        self.news_encoder = news_encoder
        d = news_encoder.out_dim
        self.attn = nn.Linear(d, attn_dim)
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)

    def encode_news(self, ids):                       # (B, L) -> (B, D)
        return self.news_encoder(ids)

    def user_vector(self, hist_vecs, mask):           # (B,H,D),(B,H) -> (B,D)
        a = self.attn_v(torch.tanh(self.attn(hist_vecs))).squeeze(-1)   # (B,H)
        a = a.masked_fill(mask == 0, -1e4).softmax(-1).unsqueeze(-1)
        return (a * hist_vecs).sum(1)

    def forward(self, hist_ids, cand_ids):            # (B,H,L),(B,C,L) -> (B,C)
        B, H, L = hist_ids.shape
        C = cand_ids.shape[1]
        hist = self.encode_news(hist_ids.reshape(B * H, L)).reshape(B, H, -1)
        cand = self.encode_news(cand_ids.reshape(B * C, L)).reshape(B, C, -1)
        mask = (hist_ids.sum(-1) != 0).float()        # padded history rows = 0
        user = self.user_vector(hist, mask)
        return (user.unsqueeze(1) * cand).sum(-1)     # (B, C)


# -------------------------------------------------------------- train utils
def _collate(batch, max_history, byte_matrix):
    """Pad history indices, then gather byte rows (kept light: store indices,
    look up bytes here, not per-sample)."""
    B = len(batch)
    hist_idx = torch.zeros(B, max_history, dtype=torch.long)
    cand_idx = torch.stack([b["cand_idx"] for b in batch])
    for i, b in enumerate(batch):
        h = b["hist_idx"][-max_history:]
        if h.numel():
            hist_idx[i, -h.numel():] = h
    hist_ids = byte_matrix[hist_idx]      # (B, max_history, L)
    cand_ids = byte_matrix[cand_idx]      # (B, C, L)
    labels = torch.zeros(B, dtype=torch.long)
    return hist_ids, cand_ids, labels


def _make_dataset(cfg, vocab, behaviors):
    samples = data_mind.build_train_samples(
        behaviors, cfg["data"]["neg_ratio"], cfg["data"]["max_history"], cfg["seed"])
    return [{"hist_idx": torch.tensor(vocab.to_indices(s["history"]), dtype=torch.long),
             "cand_idx": torch.tensor(vocab.to_indices(s["cands"]), dtype=torch.long)}
            for s in samples]


def train_recommender(cfg: dict, news_encoder: nn.Module | None = None,
                      epochs: int | None = None,
                      max_train_impressions: int | None = None,
                      model: NewsRecommender | None = None) -> NewsRecommender:
    """Train end-to-end on MINDsmall train. Returns the fitted recommender.
    ``max_train_impressions`` slices the behaviour log for fast smoke runs.
    Pass ``model`` to continue/QAT-finetune an existing (e.g. quantized) model."""
    device = "cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu"
    news = data_mind.read_news(cfg, "train")
    vocab = NewsVocab(news, cfg["data"]["max_title_bytes"])
    behaviors = data_mind.read_behaviors(cfg, "train")
    if max_train_impressions:
        behaviors = behaviors[:max_train_impressions]

    if model is None:
        if news_encoder is None:
            s = cfg["student"]
            news_encoder = ByteCNNEncoder(s["byte_embed_dim"], s["channels"], s["depth"], s["out_dim"])
        model = NewsRecommender(news_encoder)
    model = model.to(device)

    ds = _make_dataset(cfg, vocab, behaviors)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True,
        collate_fn=lambda b: _collate(b, cfg["data"]["max_history"], vocab.byte_matrix))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

    model.train()
    for ep in range(epochs or cfg["train"]["epochs"]):
        total = 0.0
        for hist, cands, labels in dl:
            hist, cands, labels = hist.to(device), cands.to(device), labels.to(device)
            logits = model(hist, cands)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"[rec] epoch {ep+1}  loss={total/len(dl):.4f}")
    return model


# --------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(cfg: dict, model: NewsRecommender, split: str = "dev",
             lang: str | None = None, max_impressions: int | None = None) -> dict:
    """Impression-level ranking metrics. ``lang`` swaps title text to an xMIND
    language (cross-lingual transfer); ``None`` = English MIND."""
    device = next(model.parameters()).device
    news = (data_xmind.localized_news(cfg, lang, split) if lang
            else data_mind.read_news(cfg, split))
    vocab = NewsVocab(news, cfg["data"]["max_title_bytes"])
    model.eval()

    # Precompute all news embeddings once.
    bm = vocab.byte_matrix.to(device)
    embs = torch.cat([model.encode_news(bm[i:i + 1024]) for i in range(0, len(bm), 1024)])

    impressions = data_mind.build_eval_impressions(
        data_mind.read_behaviors(cfg, split), cfg["data"]["max_history"])
    if max_impressions:
        impressions = impressions[:max_impressions]

    scored = []
    for imp in impressions:
        h = torch.tensor(vocab.to_indices(imp["history"]), device=device)
        c = torch.tensor(vocab.to_indices(imp["cands"]), device=device)
        if len(h) == 0:
            user = torch.zeros(embs.shape[1], device=device)
        else:
            hv = embs[h].unsqueeze(0)
            mask = torch.ones(1, len(h), device=device)
            user = model.user_vector(hv, mask).squeeze(0)
        scores = (embs[c] * user).sum(-1).cpu().numpy()
        scored.append({"labels": imp["labels"], "scores": scores})
    return metrics.evaluate(scored)


def save(model: NewsRecommender, cfg: dict, name: str = "recommender.pt"):
    p = Path(cfg["paths"]["artifacts_dir"]) / name
    torch.save(model.state_dict(), p)
    return str(p)
