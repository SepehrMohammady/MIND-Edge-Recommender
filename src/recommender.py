"""The news recommender: student news encoder + attention user encoder.

  news vec  = ByteCNNEncoder(title bytes)
  user vec  = additive-attention pool over the user's clicked-history news vecs
  score     = dot(user, candidate)

Training is NRMS-style: each instance is 1 positive + K negatives, optimised
with softmax cross-entropy (positive at index 0). Negatives are redrawn from
the impression's non-clicked candidates at every step, as in the reference
NRMS iterator. Evaluation is impression-level ranking (AUC/MRR/nDCG) via
src.metrics, and can swap the news text to any xMIND language for cross-lingual
evaluation.

Both loops keep the index tensors on the training device and encode each
distinct title of a batch once, so padded history slots cost nothing. The same
loops train and score any model exposing ``encode_news`` and ``user_vector``
(the NRMS baseline uses them with a word-id table).
"""
from __future__ import annotations

from pathlib import Path

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


def load_news(cfg: dict, split: str, lang: str | None = None) -> dict[str, dict]:
    """English MIND news, or the same articles with xMIND ``lang`` text."""
    if lang in (None, "en"):
        return data_mind.read_news(cfg, split)
    return data_xmind.localized_news(cfg, lang, split)


# ------------------------------------------------------------------- model
class FixedVectors(nn.Module):
    """News 'encoder' for precomputed vectors (teacher or frozen student):
    passes them through unchanged, or through one trainable linear map."""

    def __init__(self, dim: int, project: bool = False):
        super().__init__()
        self.out_dim = dim
        self.proj = nn.Linear(dim, dim) if project else nn.Identity()

    def forward(self, vectors):
        return self.proj(vectors)


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
def build_train_index(behaviors: list[dict], vocab, max_history: int,
                      device) -> dict[str, torch.Tensor]:
    """Index tensors with one instance per clicked candidate.

    hist       (S, H)  right-aligned clicked history, 0 = PAD
    pos        (S,)    the clicked candidate
    neg_start, neg_count (S,)  slice of ``neg_flat`` holding the impression's
                       non-clicked candidates, from which negatives are drawn
    """
    hist_rows, pos, neg_start, neg_count, neg_flat = [], [], [], [], []
    for imp in behaviors:
        p = [c for c, l in zip(imp["cands"], imp["labels"]) if l == 1]
        n = [c for c, l in zip(imp["cands"], imp["labels"]) if l == 0]
        if not p or not n:
            continue
        h = vocab.to_indices(imp["history"][-max_history:])
        row = [0] * (max_history - len(h)) + h
        start = len(neg_flat)
        neg_flat.extend(vocab.to_indices(n))
        for c in vocab.to_indices(p):
            hist_rows.append(row)
            pos.append(c)
            neg_start.append(start)
            neg_count.append(len(n))

    def t(x):
        return torch.tensor(x, dtype=torch.long, device=device)

    return {"hist": t(hist_rows), "pos": t(pos), "neg_start": t(neg_start),
            "neg_count": t(neg_count), "neg_flat": t(neg_flat)}


_TABLE_CACHE: dict = {}


def stack_byte_matrices(cfg: dict, split: str, langs: list[str], device):
    """(n_langs * N, L) int16 byte table over one shared news index, so a title
    is addressed by ``lang_id * N + news_index``. Cached per process."""
    max_len = cfg["data"]["max_title_bytes"]
    key = (cfg["paths"]["data_dir"], cfg["data"]["mind_size"], split, tuple(langs), max_len,
           str(device))
    if key not in _TABLE_CACHE:
        vocab = NewsVocab(load_news(cfg, split, langs[0]), max_len)
        mats = [vocab.byte_matrix.to(torch.int16)]
        for lang in langs[1:]:
            v = NewsVocab(load_news(cfg, split, lang), max_len)
            assert v.nids == vocab.nids, f"news index of {lang} differs from {langs[0]}"
            mats.append(v.byte_matrix.to(torch.int16))
        _TABLE_CACHE[key] = (vocab, torch.cat(mats).to(device))
    return _TABLE_CACHE[key]


def batch_scores(model, hist_keys, cand_keys, token_table, vec_table=None):
    """Scores (B, C) for a batch of title keys into ``token_table``. Every
    distinct title is encoded once; all-zero rows are padding. With
    ``vec_table`` the news vectors are looked up instead of encoded."""
    B, H = hist_keys.shape
    keys = torch.cat([hist_keys.reshape(-1), cand_keys.reshape(-1)])
    uniq, inv = torch.unique(keys, return_inverse=True)
    ids = token_table[uniq]
    vecs = model.encode_news(vec_table[uniq] if vec_table is not None else ids.long())
    nonempty = (ids != 0).any(-1)
    hist = vecs[inv[:B * H]].reshape(B, H, -1)
    cand = vecs[inv[B * H:]].reshape(B, cand_keys.shape[1], -1)
    mask = nonempty[inv[:B * H]].reshape(B, H).float()
    user = model.user_vector(hist, mask)
    return (user.unsqueeze(1) * cand).sum(-1)


def fit_ranker(cfg: dict, model, token_table: torch.Tensor, idx: dict, n_news: int,
               epochs: int, n_langs: int = 1, vec_table: torch.Tensor | None = None,
               tag: str = "rec"):
    """Shared training loop (softmax over 1 positive + K resampled negatives)."""
    device = token_table.device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["train"]["lr"])
    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg["seed"]))
    S, B, K = idx["pos"].numel(), cfg["train"]["batch_size"], cfg["data"]["neg_ratio"]
    labels = torch.zeros(B, dtype=torch.long, device=device)
    for ep in range(epochs):
        perm = torch.randperm(S, device=device, generator=gen)
        total, steps = 0.0, 0
        for i in range(0, S - B + 1, B):                      # drop the ragged last batch
            b = perm[i:i + B]
            count = idx["neg_count"][b, None]
            pick = (torch.rand(B, K, device=device, generator=gen) * count).long()
            neg = idx["neg_flat"][idx["neg_start"][b, None] + torch.minimum(pick, count - 1)]
            cand = torch.cat([idx["pos"][b, None], neg], dim=1)
            lang = torch.randint(n_langs, (B, 1), device=device, generator=gen) * n_news
            logits = batch_scores(model, idx["hist"][b] + lang, cand + lang, token_table, vec_table)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        print(f"[{tag}] epoch {ep+1}  loss={total/max(steps, 1):.4f}", flush=True)
    return model


def train_recommender(cfg: dict, news_encoder: nn.Module | None = None,
                      epochs: int | None = None,
                      max_train_impressions: int | None = None,
                      model: NewsRecommender | None = None,
                      langs: list[str] | None = None,
                      fixed_vectors: torch.Tensor | None = None) -> NewsRecommender:
    """Train end-to-end on the MIND train clicks. Returns the fitted recommender.

    ``max_train_impressions`` slices the behaviour log for fast smoke runs.
    Pass ``model`` to continue or QAT-finetune an existing model.
    ``langs`` (e.g. ``["en", "zho", ...]``) draws one language per instance and
    shows that user's history and candidates in it; default is English only.
    ``fixed_vectors`` ((n_langs * N, D), row 0 of each language = PAD) replaces
    the encoder by a lookup, for the frozen-teacher and frozen-student bounds;
    ``model.news_encoder`` is then a ``FixedVectors``.
    """
    device = "cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu"
    langs = langs or ["en"]
    vocab, byte_table = stack_byte_matrices(cfg, "train", langs, device)
    behaviors = data_mind.read_behaviors(cfg, "train")
    if max_train_impressions:
        behaviors = behaviors[:max_train_impressions]
    idx = build_train_index(behaviors, vocab, cfg["data"]["max_history"], device)

    if model is None:
        if news_encoder is None:
            s = cfg["student"]
            news_encoder = ByteCNNEncoder(s["byte_embed_dim"], s["channels"], s["depth"], s["out_dim"])
        model = NewsRecommender(news_encoder)
    model = model.to(device).train()
    if fixed_vectors is not None:
        fixed_vectors = fixed_vectors.to(device)
    return fit_ranker(cfg, model, byte_table, idx, len(vocab.nids),
                      epochs or cfg["train"]["epochs"], n_langs=len(langs),
                      vec_table=fixed_vectors)


# --------------------------------------------------------------- evaluation
_VOCAB_CACHE: dict = {}
_IMPRESSION_CACHE: dict = {}


def eval_vocab(cfg: dict, split: str, lang: str | None = None) -> NewsVocab:
    """News index + byte matrix of one split/language (cached per process)."""
    key = (cfg["paths"]["data_dir"], cfg["data"]["mind_size"], split, lang or "en",
           cfg["data"]["max_title_bytes"])
    if key not in _VOCAB_CACHE:
        _VOCAB_CACHE[key] = NewsVocab(load_news(cfg, split, lang), cfg["data"]["max_title_bytes"])
    return _VOCAB_CACHE[key]


def eval_impressions(cfg: dict, split: str) -> list[dict]:
    """Labeled impressions of one split (cached per process)."""
    key = (cfg["paths"]["data_dir"], cfg["data"]["mind_size"], split, cfg["data"]["max_history"])
    if key not in _IMPRESSION_CACHE:
        _IMPRESSION_CACHE[key] = data_mind.build_eval_impressions(
            data_mind.read_behaviors(cfg, split), cfg["data"]["max_history"])
    return _IMPRESSION_CACHE[key]


@torch.no_grad()
def encode_all_news(model, token_matrix: torch.Tensor, batch: int = 1024) -> torch.Tensor:
    return torch.cat([model.encode_news(token_matrix[i:i + batch])
                      for i in range(0, len(token_matrix), batch)])


@torch.no_grad()
def encode_titles(encoder: nn.Module, byte_matrix: torch.Tensor, batch: int = 1024) -> torch.Tensor:
    """Vectors of a bare news encoder for a (N, L) byte matrix; all-zero (PAD)
    rows map to the zero vector, the layout ``fixed_vectors`` expects."""
    encoder.eval()
    device = next(encoder.parameters()).device
    out = torch.cat([encoder(byte_matrix[i:i + batch].to(device).long())
                     for i in range(0, len(byte_matrix), batch)])
    return out * (byte_matrix.to(device) != 0).any(-1, keepdim=True).float()


@torch.no_grad()
def score_impressions(model, embs: torch.Tensor, vocab, impressions: list[dict],
                      mask_history: bool = False, batch: int = 2048) -> list[dict]:
    """Candidate scores for every impression: [{'labels', 'scores'}, ...].

    A user with no history gets the zero vector, so all candidates tie."""
    device = embs.device
    H = max((len(i["history"]) for i in impressions), default=1) or 1
    scored = []
    for s in range(0, len(impressions), batch):
        chunk = impressions[s:s + batch]
        hist = torch.zeros(len(chunk), H, dtype=torch.long, device=device)
        lens = torch.zeros(len(chunk), dtype=torch.long, device=device)
        for r, imp in enumerate(chunk):
            h = [] if mask_history else vocab.to_indices(imp["history"])
            if h:
                hist[r, :len(h)] = torch.tensor(h, device=device)
                lens[r] = len(h)
        mask = (torch.arange(H, device=device)[None, :] < lens[:, None]).float()
        user = model.user_vector(embs[hist], mask)
        user = user * (lens > 0).float().unsqueeze(-1)
        n_cands = [len(imp["cands"]) for imp in chunk]
        cand = torch.tensor([c for imp in chunk for c in vocab.to_indices(imp["cands"])],
                            device=device)
        owner = torch.repeat_interleave(torch.arange(len(chunk), device=device),
                                        torch.tensor(n_cands, device=device))
        flat = (embs[cand] * user[owner]).sum(-1).float().cpu().numpy()
        off = 0
        for imp, n in zip(chunk, n_cands):
            scored.append({"labels": imp["labels"], "scores": flat[off:off + n]})
            off += n
    return scored


def filter_impressions(impressions: list[dict], min_hist=None, max_hist=None,
                       max_impressions=None) -> list[dict]:
    if min_hist is not None:
        impressions = [i for i in impressions if len(i["history"]) >= min_hist]
    if max_hist is not None:
        impressions = [i for i in impressions if len(i["history"]) <= max_hist]
    if max_impressions:
        impressions = impressions[:max_impressions]
    return impressions


@torch.no_grad()
def evaluate(cfg: dict, model: NewsRecommender, split: str = "dev",
             lang: str | None = None, max_impressions: int | None = None,
             min_hist: int | None = None, max_hist: int | None = None,
             mask_history: bool = False,
             news_vectors: torch.Tensor | None = None) -> dict:
    """Impression-level ranking metrics. ``lang`` swaps title text to an xMIND
    language (cross-lingual transfer); ``None`` = English MIND.
    ``min_hist``/``max_hist`` filter impressions by clicked-history length (for
    cold-start analysis: e.g. max_hist=0 = users with no history).
    ``news_vectors`` ((N, D), row 0 = PAD) supplies precomputed vectors for the
    split instead of encoding the titles."""
    device = next(model.parameters()).device
    vocab = eval_vocab(cfg, split, lang)
    model.eval()
    if news_vectors is not None:
        embs = encode_all_news(model, news_vectors.to(device))
    else:
        embs = encode_all_news(model, vocab.byte_matrix.to(device))
    impressions = filter_impressions(eval_impressions(cfg, split), min_hist, max_hist,
                                     max_impressions)
    return metrics.evaluate(score_impressions(model, embs, vocab, impressions, mask_history))


@torch.no_grad()
def evaluate_by_history(cfg: dict, model, split: str = "dev", lang: str | None = None,
                        news_vectors: torch.Tensor | None = None,
                        max_impressions: int | None = None) -> dict[str, dict]:
    """``evaluate`` on the full split, broken down by history-length bucket
    (0, 1-5, 6-20, >20 clicks) plus 'all', from one scoring pass."""
    device = next(model.parameters()).device
    vocab = eval_vocab(cfg, split, lang)
    model.eval()
    source = news_vectors if news_vectors is not None else vocab.byte_matrix
    embs = encode_all_news(model, source.to(device))
    impressions = filter_impressions(eval_impressions(cfg, split), max_impressions=max_impressions)
    return metrics.by_history(impressions, score_impressions(model, embs, vocab, impressions))


def save(model: NewsRecommender, cfg: dict, name: str = "recommender.pt"):
    p = Path(cfg["paths"]["artifacts_dir"]) / name
    torch.save(model.state_dict(), p)
    return str(p)
