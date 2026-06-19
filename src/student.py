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


def train_student(cfg: dict, dims=(64, 128, 256, 384)) -> ByteCNNEncoder:
    """Distill a ByteCNNEncoder to the teacher anchors; save to artifacts/."""
    device = "cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu"
    bytes_np, tgt_np, anchors = build_distill_data(cfg, "train")
    anchors_t = torch.tensor(anchors, device=device)

    ds = torch.utils.data.TensorDataset(torch.tensor(bytes_np, dtype=torch.long),
                                        torch.tensor(tgt_np, dtype=torch.long))
    dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True, drop_last=True)

    s = cfg["student"]
    model = ByteCNNEncoder(s["byte_embed_dim"], s["channels"], s["depth"], s["out_dim"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["distill_lr"])
    dims = tuple(d for d in dims if d <= s["out_dim"])

    model.train()
    for ep in range(cfg["train"]["distill_epochs"]):
        total = 0.0
        for ids, ti in dl:
            ids, target = ids.to(device), anchors_t[ti.to(device)]
            pred = F.normalize(model(ids), dim=-1)
            loss = _mrl_loss(pred, target, dims)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"[distill] epoch {ep+1}/{cfg['train']['distill_epochs']}  loss={total/len(dl):.4f}")

    out = Path(cfg["paths"]["artifacts_dir"]) / "student.pt"
    torch.save(model.state_dict(), out)
    print(f"[distill] saved -> {out}")
    return model
