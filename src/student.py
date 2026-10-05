"""Byte-level char-CNN STUDENT news encoder + multilingual distillation.

The student's whole "vocabulary" is the 256 UTF-8 byte values (id 0 = pad,
1..256 = byte+1), shared identically by all 14 languages -> no per-language
embedding table, so it fits an MCU and serves every language with one model.

It is distilled to reproduce the frozen teacher's English anchor embedding
(see teacher.py). Architecture (channels / depth / out_dim) is the NAS search
space, so the encoder is fully parameterized.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import data_mind, data_xmind, teacher


# ---------------------------------------------------------------- tokenizer
def text_to_bytes(text: str, max_len: int) -> list[int]:
    """UTF-8 bytes -> ids in 1..256 (0 reserved for padding), truncated/padded."""
    b = text.encode("utf-8")[:max_len]
    ids = [x + 1 for x in b]
    return ids + [0] * (max_len - len(ids))


# ------------------------------------------------------------------- model
class _DSConv(nn.Module):
    """Depthwise-separable 1D conv block (cheap, MCU-friendly)."""

    def __init__(self, ch: int):
        super().__init__()
        self.dw = nn.Conv1d(ch, ch, kernel_size=3, padding=1, groups=ch)
        self.pw = nn.Conv1d(ch, ch, kernel_size=1)
        self.bn = nn.BatchNorm1d(ch)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))


class ByteCNNEncoder(nn.Module):
    """Bytes (B, L) -> news embedding (B, out_dim)."""

    def __init__(self, byte_embed_dim=64, channels=128, depth=3, out_dim=384):
        super().__init__()
        self.embed = nn.Embedding(257, byte_embed_dim, padding_idx=0)
        self.proj_in = nn.Conv1d(byte_embed_dim, channels, kernel_size=1)
        self.blocks = nn.ModuleList(_DSConv(channels) for _ in range(depth))
        self.head = nn.Linear(channels, out_dim)
        self.out_dim = out_dim

    def forward(self, ids):                      # ids: (B, L) long
        mask = (ids != 0).float().unsqueeze(1)   # (B, 1, L)
        x = self.embed(ids).transpose(1, 2)      # (B, E, L)
        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x)
        x = (x * mask).sum(-1) / mask.sum(-1).clamp(min=1)   # masked mean pool
        return self.head(x)                      # (B, out_dim)


# ------------------------------------------------------------ distillation
def _byte_matrix(titles: list[str], max_len: int) -> np.ndarray:
    return np.asarray([text_to_bytes(t, max_len) for t in titles], dtype=np.int16)


def build_distill_data(cfg: dict, split: str = "train"):
    """Stack byte rows for English + every available language against shared
    English anchors. Returns (bytes[int16, M, L], target_idx[M], anchors[N, D])."""
    nids, anchors = teacher.build_anchors(cfg, split)
    idx = {n: i for i, n in enumerate(nids)}
    max_len = cfg["data"]["max_title_bytes"]

    en = data_mind.read_news(cfg, split)
    byte_rows, tgt = [_byte_matrix([en[n]["title"] for n in nids], max_len)], [np.arange(len(nids))]

    for lang in data_xmind.available_langs(cfg):
        loc = data_xmind.localized_news(cfg, lang, split)
        byte_rows.append(_byte_matrix([loc[n]["title"] for n in nids], max_len))
        tgt.append(np.arange(len(nids)))

    return np.concatenate(byte_rows), np.concatenate(tgt), anchors


def _mrl_loss(pred, target, dims):
    """Matryoshka: cosine loss over nested prefixes so early dims stay usable."""
    loss = 0.0
    for d in dims:
        loss = loss + (1 - F.cosine_similarity(pred[:, :d], target[:, :d], dim=-1)).mean()
    return loss / len(dims)


_DISTILL_CACHE: dict = {}


def distill_tensors(cfg: dict, split: str = "train"):
    """``build_distill_data`` as device tensors, built once per process:
    (bytes int16 (M, L), target index (M,), anchors (N, D))."""
    device = "cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu"
    key = (cfg["paths"]["data_dir"], cfg["data"]["mind_size"], split,
           cfg["data"]["max_title_bytes"], tuple(data_xmind.available_langs(cfg)), device)
    if key not in _DISTILL_CACHE:
        bytes_np, tgt_np, anchors = build_distill_data(cfg, split)
        _DISTILL_CACHE[key] = (torch.tensor(bytes_np, device=device),
                               torch.tensor(tgt_np, dtype=torch.long, device=device),
                               torch.tensor(anchors, device=device))
    return _DISTILL_CACHE[key]


def distill_encoder(cfg: dict, encoder: nn.Module, epochs: int | None = None,
                    max_titles: int | None = None, dims=(64, 128, 256, 384),
                    batch: int = 512, tag: str = "distill") -> nn.Module:
    """Distil ``encoder`` (any module mapping byte ids to ``out_dim`` vectors)
    to the teacher anchors with the Matryoshka cosine loss. English and every
    available xMIND translation of a title share the English anchor.
    ``max_titles`` draws a seeded random subset for short runs."""
    X, T, anchors = distill_tensors(cfg, "train")
    device = X.device
    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg["seed"]))
    if max_titles and max_titles < len(X):
        keep = torch.randperm(len(X), device=device, generator=gen)[:max_titles]
        X, T = X[keep], T[keep]
    encoder = encoder.to(device).train()
    opt = torch.optim.AdamW(encoder.parameters(), lr=cfg["train"]["distill_lr"])
    dims = tuple(d for d in dims if d <= encoder.out_dim)
    for ep in range(epochs or cfg["train"]["distill_epochs"]):
        perm = torch.randperm(len(X), device=device, generator=gen)
        total, steps = 0.0, 0
        for i in range(0, len(X) - batch + 1, batch):
            b = perm[i:i + batch]
            pred = F.normalize(encoder(X[b].long()), dim=-1)
            loss = _mrl_loss(pred, anchors[T[b]], dims)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        print(f"[{tag}] epoch {ep+1}  loss={total/max(steps, 1):.4f}", flush=True)
    return encoder


@torch.no_grad()
def teacher_similarity(cfg: dict, encoder: nn.Module, split: str = "dev",
                       dims=(64, 128, 256, 384), max_titles: int = 8000) -> dict:
    """Mean cosine between the student's and the teacher's embedding of the
    English ``split`` titles, on nested prefixes of the vector."""
    nids, anchors = teacher.build_anchors(cfg, split)
    news = data_mind.read_news(cfg, split)
    device = next(encoder.parameters()).device
    L = cfg["data"]["max_title_bytes"]
    X = torch.tensor([text_to_bytes(news[n]["title"], L) for n in nids[:max_titles]],
                     dtype=torch.long, device=device)
    A = torch.tensor(anchors[:max_titles], device=device)
    encoder.eval()
    P = torch.cat([encoder(X[i:i + 1024]) for i in range(0, len(X), 1024)])
    return {str(d): round(F.cosine_similarity(P[:, :d], A[:, :d], dim=-1).mean().item(), 4)
            for d in dims if d <= P.shape[1]}


def train_student(cfg: dict, dims=(64, 128, 256, 384)) -> ByteCNNEncoder:
    """Distill a ByteCNNEncoder to the teacher anchors; save to artifacts/."""
    s = cfg["student"]
    model = ByteCNNEncoder(s["byte_embed_dim"], s["channels"], s["depth"], s["out_dim"])
    model = distill_encoder(cfg, model, dims=dims)
    out = Path(cfg["paths"]["artifacts_dir"]) / "student.pt"
    torch.save(model.state_dict(), out)
    print(f"[distill] saved -> {out}")
    return model
