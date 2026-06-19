"""NRMS baseline (Wu et al., EMNLP 2019) -- the FP32 accuracy ceiling.

Word-embedding title encoder with multi-head self-attention + additive
attention, and an MHSA user encoder. English-only (its vocabulary is a word
table -- which is exactly the on-device flash problem the byte-CNN student
avoids). We report our reproduced MINDsmall-dev numbers and cite the published
MINDlarge-test ceiling (AUC 0.6776) for external reference.

Self-contained (own tokenizer/train/eval), reusing only data_mind + metrics.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import data_mind, metrics


# ----------------------------------------------------------------- vocab
class WordVocab:
    def __init__(self, news: dict, max_words: int, min_freq: int = 2):
        from collections import Counter
        cnt = Counter(w for v in news.values() for w in v["title"].lower().split())
        vocab = [w for w, c in cnt.items() if c >= min_freq]
        self.word2id = {w: i + 1 for i, w in enumerate(vocab)}   # 0 = PAD/OOV
        self.size = len(self.word2id) + 1
        self.max_words = max_words
        rows = [[0] * max_words] + [self._enc(v["title"]) for v in news.values()]
        self.nids = ["<PAD>"] + list(news.keys())
        self.idx = {n: i for i, n in enumerate(self.nids)}
        self.matrix = torch.tensor(rows, dtype=torch.long)

    def _enc(self, title: str) -> list[int]:
        ids = [self.word2id.get(w, 0) for w in title.lower().split()][:self.max_words]
        return ids + [0] * (self.max_words - len(ids))

    def to_indices(self, nids: list[str]) -> list[int]:
        return [self.idx.get(n, 0) for n in nids]


# --------------------------------------------------------------- modules
class _AdditiveAttn(nn.Module):
    def __init__(self, dim, attn=128):
        super().__init__()
        self.w = nn.Linear(dim, attn)
        self.q = nn.Linear(attn, 1, bias=False)

    def forward(self, x, mask):                       # x:(B,T,D) mask:(B,T) 1=keep
        a = self.q(torch.tanh(self.w(x))).squeeze(-1)
        a = a.masked_fill(mask == 0, -1e4).softmax(-1).unsqueeze(-1)
        return torch.nan_to_num((a * x).sum(1))


class NRMSModel(nn.Module):
    def __init__(self, vocab_size, embed=256, heads=8, attn=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed, padding_idx=0)
        self.news_mhsa = nn.MultiheadAttention(embed, heads, batch_first=True)
        self.news_pool = _AdditiveAttn(embed, attn)
        self.user_mhsa = nn.MultiheadAttention(embed, heads, batch_first=True)
        self.user_pool = _AdditiveAttn(embed, attn)
        self.out_dim = embed

    def encode_news(self, ids):                       # (B,L) -> (B,E)
        pad = ids == 0
        x = self.embed(ids)
        x, _ = self.news_mhsa(x, x, x, key_padding_mask=pad, need_weights=False)
        x = torch.nan_to_num(x)
        return self.news_pool(x, (~pad).float())

    def user_vector(self, hist_vecs, mask):           # (B,H,E),(B,H) -> (B,E)
        pad = mask == 0
        x, _ = self.user_mhsa(hist_vecs, hist_vecs, hist_vecs,
                              key_padding_mask=pad, need_weights=False)
        x = torch.nan_to_num(x)
        return self.user_pool(x, mask)

    def forward(self, hist_ids, cand_ids):
        B, H, L = hist_ids.shape
        C = cand_ids.shape[1]
        hist = self.encode_news(hist_ids.reshape(B * H, L)).reshape(B, H, -1)
        cand = self.encode_news(cand_ids.reshape(B * C, L)).reshape(B, C, -1)
        mask = (hist_ids.sum(-1) != 0).float()
        user = self.user_vector(hist, mask)
        return (user.unsqueeze(1) * cand).sum(-1)


# ------------------------------------------------------------- train/eval
def _collate(batch, max_history, matrix):
    B = len(batch)
    hist_idx = torch.zeros(B, max_history, dtype=torch.long)
    cand_idx = torch.stack([b["cand_idx"] for b in batch])
    for i, b in enumerate(batch):
        h = b["hist_idx"][-max_history:]
        if h.numel():
            hist_idx[i, -h.numel():] = h
    return matrix[hist_idx], matrix[cand_idx], torch.zeros(B, dtype=torch.long)


def train_nrms(cfg, epochs=None, max_train_impressions=None) -> NRMSModel:
    device = "cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu"
    n = cfg.get("nrms", {})
    news = data_mind.read_news(cfg, "train")
    vocab = WordVocab(news, n.get("max_words", 20))
    behaviors = data_mind.read_behaviors(cfg, "train")
    if max_train_impressions:
        behaviors = behaviors[:max_train_impressions]
    samples = data_mind.build_train_samples(behaviors, cfg["data"]["neg_ratio"],
                                            cfg["data"]["max_history"], cfg["seed"])
    ds = [{"hist_idx": torch.tensor(vocab.to_indices(s["history"]), dtype=torch.long),
           "cand_idx": torch.tensor(vocab.to_indices(s["cands"]), dtype=torch.long)}
          for s in samples]
    dl = torch.utils.data.DataLoader(
        ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True,
        collate_fn=lambda b: _collate(b, cfg["data"]["max_history"], vocab.matrix))

    model = NRMSModel(vocab.size, n.get("embed", 256), n.get("heads", 8)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    model.train()
    for ep in range(epochs or cfg["train"]["epochs"]):
        tot = 0.0
        for hist, cand, lab in dl:
            hist, cand, lab = hist.to(device), cand.to(device), lab.to(device)
            loss = F.cross_entropy(model(hist, cand), lab)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"[nrms] epoch {ep+1} loss={tot/len(dl):.4f}")
    model._vocab = vocab
    return model


@torch.no_grad()
def evaluate_nrms(cfg, model: NRMSModel, split="dev", max_impressions=None) -> dict:
    device = next(model.parameters()).device
    news = data_mind.read_news(cfg, split)
    vocab = WordVocab(news, model._vocab.max_words)
    vocab.word2id = model._vocab.word2id        # reuse trained vocabulary
    vocab.matrix = torch.tensor(
        [[0] * vocab.max_words] + [vocab._enc(v["title"]) for v in news.values()],
        dtype=torch.long)
    model.eval()
    bm = vocab.matrix.to(device)
    embs = torch.cat([model.encode_news(bm[i:i + 1024]) for i in range(0, len(bm), 1024)])
    imps = data_mind.build_eval_impressions(data_mind.read_behaviors(cfg, split),
                                            cfg["data"]["max_history"])
    if max_impressions:
        imps = imps[:max_impressions]
    scored = []
    for imp in imps:
        h = torch.tensor(vocab.to_indices(imp["history"]), device=device)
        c = torch.tensor(vocab.to_indices(imp["cands"]), device=device)
        if len(h) == 0:
            user = torch.zeros(embs.shape[1], device=device)
        else:
            user = model.user_vector(embs[h].unsqueeze(0),
                                     torch.ones(1, len(h), device=device)).squeeze(0)
        scored.append({"labels": imp["labels"], "scores": (embs[c] * user).sum(-1).cpu().numpy()})
    return metrics.evaluate(scored)
