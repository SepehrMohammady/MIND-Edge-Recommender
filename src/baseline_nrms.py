"""NRMS baseline (Wu et al., EMNLP 2019) -- the word-embedding reference point.

Word-embedding title encoder with multi-head self-attention + additive
attention, and a self-attentive user encoder. English-only: its vocabulary is a
word table, which is exactly the on-device flash problem the byte-CNN student
avoids.

Two variants share the training and scoring loops of ``src.recommender``:

  * ``NRMSGlove``  -- the reference configuration distributed with MIND
    (``MINDsmall_utils.zip``: GloVe-initialised 300-d word table, 20 heads x
    20 dims, additive attention 200, dropout 0.2, title length 30).
  * ``NRMSModel``  -- the reduced variant used in the June 2026 run (random
    256-d table, 8 heads via ``nn.MultiheadAttention``, title length 20).
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src import data_mind, metrics, recommender

_TOKEN = re.compile(r"[\w]+|[.,!?;|]")


def word_tokenize(sentence: str) -> list[str]:
    """Tokeniser of the reference MIND code: lower-cased words and . , ! ? ; |"""
    return _TOKEN.findall(sentence.lower()) if isinstance(sentence, str) else []


# ----------------------------------------------------------------- vocab
class WordVocab:
    """News index + (N, T) word-id matrix. Row 0 = PAD news; id 0 = PAD/OOV word."""

    def __init__(self, news: dict, max_words: int, min_freq: int = 2,
                 word2id: dict | None = None, tokenizer=None):
        from collections import Counter
        self.tokenize = tokenizer or (lambda s: s.lower().split())
        if word2id is None:
            cnt = Counter(w for v in news.values() for w in self.tokenize(v["title"]))
            vocab = [w for w, c in cnt.items() if c >= min_freq]
            word2id = {w: i + 1 for i, w in enumerate(vocab)}
        self.word2id = word2id
        self.size = max(word2id.values(), default=0) + 1
        self.max_words = max_words
        self.set_news(news)

    def set_news(self, news: dict) -> "WordVocab":
        self.nids = ["<PAD>"] + list(news.keys())
        self.idx = {n: i for i, n in enumerate(self.nids)}
        rows = [[0] * self.max_words] + [self._enc(v["title"]) for v in news.values()]
        self.matrix = torch.tensor(rows, dtype=torch.long)
        return self

    def for_news(self, news: dict) -> "WordVocab":
        """Same word table, another set of articles (e.g. the dev split)."""
        return WordVocab(news, self.max_words, word2id=self.word2id, tokenizer=self.tokenize)

    def _enc(self, title: str) -> list[int]:
        ids = [self.word2id.get(w, 0) for w in self.tokenize(title)][:self.max_words]
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


class _SelfAttention(nn.Module):
    """Multi-head self-attention that projects to heads x head_dim, as in the
    reference NRMS (three bias-free projections, no output projection)."""

    def __init__(self, in_dim: int, heads: int, head_dim: int):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        out = heads * head_dim
        self.q = nn.Linear(in_dim, out, bias=False)
        self.k = nn.Linear(in_dim, out, bias=False)
        self.v = nn.Linear(in_dim, out, bias=False)

    def forward(self, x, mask):                       # x:(B,T,D) mask:(B,T) 1=keep
        B, T, _ = x.shape

        def split(t):
            return t.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        a = (q @ k.transpose(-1, -2)) / self.head_dim ** 0.5          # (B,h,T,T)
        a = a.masked_fill(mask[:, None, None, :] == 0, -1e4).softmax(-1)
        return (a @ v).transpose(1, 2).reshape(B, T, self.heads * self.head_dim)


class NRMSGlove(nn.Module):
    """NRMS with the reference hyper-parameters and a pretrained word table."""

    def __init__(self, embedding: torch.Tensor, heads=20, head_dim=20, attn=200, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(embedding.float(), freeze=False)
        d = heads * head_dim
        self.drop = nn.Dropout(dropout)
        self.news_sa = _SelfAttention(embedding.shape[1], heads, head_dim)
        self.news_pool = _AdditiveAttn(d, attn)
        self.user_sa = _SelfAttention(d, heads, head_dim)
        self.user_pool = _AdditiveAttn(d, attn)
        self.out_dim = d

    def encode_news(self, ids):                       # (B,T) -> (B,D)
        mask = (ids != 0).float()
        x = self.drop(self.embed(ids))
        x = self.drop(self.news_sa(x, mask))
        return self.news_pool(x, mask)

    def user_vector(self, hist_vecs, mask):           # (B,H,D),(B,H) -> (B,D)
        return self.user_pool(self.user_sa(hist_vecs, mask), mask)


class NRMSModel(nn.Module):
    """Reduced NRMS of the June 2026 run (random-initialised 256-d table)."""

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


# ------------------------------------------------------------- train/eval
def _device(cfg):
    return "cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu"


def load_reference_assets(cfg: dict):
    """Word table and GloVe-initialised embedding shipped with MIND
    (``MINDsmall_utils.zip``, fetched by ``python -m src.download``)."""
    utils = Path(cfg["paths"]["data_dir"]) / "mind" / cfg["data"]["mind_size"] / "utils"
    with open(utils / "word_dict.pkl", "rb") as fh:
        word2id = pickle.load(fh)
    return word2id, torch.from_numpy(np.load(utils / "embedding.npy"))


def train_nrms(cfg, epochs=None, max_train_impressions=None, variant: str = "reduced"):
    """Train NRMS on the MIND train clicks. ``variant`` = 'glove' (reference
    configuration) or 'reduced' (June 2026 configuration)."""
    device = _device(cfg)
    news = data_mind.read_news(cfg, "train")
    if variant == "glove":
        word2id, embedding = load_reference_assets(cfg)
        vocab = WordVocab(news, 30, word2id=word2id, tokenizer=word_tokenize)
        model = NRMSGlove(embedding)
    else:
        n = cfg.get("nrms", {})
        vocab = WordVocab(news, n.get("max_words", 20))
        model = NRMSModel(vocab.size, n.get("embed", 256), n.get("heads", 8))
    behaviors = data_mind.read_behaviors(cfg, "train")
    if max_train_impressions:
        behaviors = behaviors[:max_train_impressions]
    idx = recommender.build_train_index(behaviors, vocab, cfg["data"]["max_history"], device)
    model = model.to(device).train()
    recommender.fit_ranker(cfg, model, vocab.matrix.to(device), idx, len(vocab.nids),
                           epochs or cfg["train"]["epochs"], tag=f"nrms-{variant}")
    model._vocab = vocab
    return model


@torch.no_grad()
def evaluate_nrms(cfg, model, split="dev", max_impressions=None, by_history=False) -> dict:
    """Ranking metrics on ``split``; ``by_history`` adds the cold-start buckets."""
    device = next(model.parameters()).device
    vocab = model._vocab.for_news(data_mind.read_news(cfg, split))
    model.eval()
    embs = recommender.encode_all_news(model, vocab.matrix.to(device))
    imps = recommender.filter_impressions(recommender.eval_impressions(cfg, split),
                                          max_impressions=max_impressions)
    scored = recommender.score_impressions(model, embs, vocab, imps)
    return metrics.by_history(imps, scored) if by_history else metrics.evaluate(scored)
