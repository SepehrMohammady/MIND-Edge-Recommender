"""PyTorch build of the encoders found by the µNAS search in the lab fork.

The search (unas/, results in paper/results/unas/) trains Keras models of the fork's 1D-CNN
space (uNAS/cnn1d/cnn1d_architecture.py, global-average-pooling head from unas/cnn1d_gap.py).
The click training, evaluation and export code of this repository is PyTorch, so the chosen
architectures are rebuilt here from their architecture dictionaries with the Keras semantics:

  * Conv1D / DepthwiseConv1D with bias and padding "same" (TensorFlow rule: total padding
    max((out - 1) * stride + kernel - length, 0), the odd pad on the right), stride 2 when the
    layer's "1x_stride" flag is set (the fork's naming);
  * BatchNormalization (epsilon 1e-3) and ReLU after the convolution when the layer has them;
  * MaxPool1D(2) without padding before a layer with "has_prepool";
  * global average pooling over all positions (padding included, as in Keras: the mask channel
    tells the network which positions are padding);
  * dropout ("head_dropout") before each Dense layer; the last Dense has no activation.

Input, as in the search (unas/mind_dataset.py): byte ids through the 257 x 64 byte table of the
distilled reference student (seed 42), plus a padding-mask channel, 65 channels in all. The table
is a frozen parameter: it stays fixed in every training stage, as during the search, and on a
board the caller holds it (scripts/deploy_boards.py). Branch blocks and the strided pooling head
of the fork's space are not supported; neither chosen architecture has them.

scripts/check_unas_port.py loads the weights of a Keras build into this one and compares outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BYTE_TABLE = ROOT / "artifacts/unas/data/byte_table.npy"
BN_EPS = 1e-3                                   # Keras BatchNormalization default


def same_padding(length: int, kernel: int, stride: int) -> tuple[int, int]:
    out = -(-length // stride)
    total = max((out - 1) * stride + kernel - length, 0)
    return total // 2, total - total // 2


class KerasConv1d(nn.Module):
    """Conv1D or DepthwiseConv1D with padding "same", then optional BatchNorm and ReLU."""

    def __init__(self, spec: dict, in_ch: int, length: int):
        super().__init__()
        self.prepool = bool(spec["has_prepool"]) and length > 1
        if self.prepool:
            length //= 2
        k = 1 if spec["type"] == "1x1Conv1D" else min(spec["ker_size"], length)
        self.stride = 1 if spec["type"] == "1x1Conv1D" or not spec["1x_stride"] else 2
        self.depthwise = spec["type"] == "DWConv1D"
        out_ch = in_ch if self.depthwise else spec["filters"]
        self.pad = same_padding(length, k, self.stride)
        self.conv = nn.Conv1d(in_ch, out_ch, k, stride=self.stride, groups=in_ch if self.depthwise else 1)
        self.bn = nn.BatchNorm1d(out_ch, eps=BN_EPS) if spec["has_bn"] else None
        self.relu = bool(spec["has_relu"])
        self.out_ch, self.out_len = out_ch, -(-length // self.stride)

    def forward(self, x):
        if self.prepool:
            x = F.max_pool1d(x, 2)
        x = self.conv(F.pad(x, self.pad))
        if self.bn is not None:
            x = self.bn(x)
        return F.relu(x) if self.relu else x


class UnasEncoder(nn.Module):
    """Bytes (B, L) -> news embedding (B, out_dim), architecture from a µNAS dictionary."""

    def __init__(self, arch: dict, out_dim: int = 384, max_len: int = 128, table: np.ndarray | None = None):
        super().__init__()
        assert (arch.get("pooling") or {}).get("type") == "gap", "only the GAP head is supported"
        assert not any(b["is_branch"] for b in arch["conv_blocks"]), "branch blocks are not supported"
        table = np.load(BYTE_TABLE) if table is None else table
        self.embed = nn.Embedding(table.shape[0], table.shape[1], padding_idx=0)
        self.embed.weight.data.copy_(torch.from_numpy(np.asarray(table, dtype=np.float32)))
        self.embed.weight.requires_grad_(False)
        ch, length = table.shape[1] + 1, max_len
        layers = []
        for block in arch["conv_blocks"]:
            for spec in block["layers"]:
                layers.append(KerasConv1d(spec, ch, length))
                ch, length = layers[-1].out_ch, layers[-1].out_len
        self.layers = nn.ModuleList(layers)
        self.dropout = float(arch.get("head_dropout", 0.0))
        dense = []
        for d in arch.get("dense_blocks", []):
            assert d.get("activation") in (None, "relu"), d
            dense.append(nn.Linear(ch, d["units"]))
            ch = d["units"]
        self.dense = nn.ModuleList(dense)
        self.dense_act = [d.get("activation") for d in arch.get("dense_blocks", [])]
        self.head = nn.Linear(ch, out_dim)
        self.out_dim, self.arch = out_dim, arch

    def body(self, embedded, mask):
        """Network after the byte-table lookup: embedded (B, 64, L) and mask (B, 1, L)."""
        x = torch.cat([embedded, mask], dim=1)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(-1)
        for lin, act in zip(self.dense, self.dense_act):
            x = lin(F.dropout(x, self.dropout, self.training))
            x = F.relu(x) if act == "relu" else x
        return self.head(F.dropout(x, self.dropout, self.training))

    def forward(self, ids):                      # ids: (B, L) long
        mask = (ids != 0).float().unsqueeze(1)
        return self.body(self.embed(ids).transpose(1, 2), mask)


def load_keras_weights(model: UnasEncoder, npz) -> None:
    """Copy the weights of a Keras build (unas/dump_keras.py) layer by layer."""
    layers = [l for l in json.loads(str(npz["layers"])) if l["weights"]]
    targets = []
    for layer in model.layers:
        targets.append(("conv", layer))
        if layer.bn is not None:
            targets.append(("bn", layer.bn))
    targets += [("dense", lin) for lin in model.dense] + [("dense", model.head)]
    assert len(targets) == len(layers), (len(targets), [l["class"] for l in layers])
    for (kind, mod), l in zip(targets, layers):
        w = {n: torch.from_numpy(np.asarray(npz[f"{l['index']}:{n}"])) for n in l["weights"]}
        if kind == "conv":
            k = w["kernel"]                       # Conv1D (k, in, out); DepthwiseConv1D (k, in, 1)
            mod.conv.weight.data.copy_(k.permute(1, 2, 0) if mod.depthwise else k.permute(2, 1, 0))
            mod.conv.bias.data.copy_(w["bias"])
        elif kind == "bn":
            mod.weight.data.copy_(w["gamma"])
            mod.bias.data.copy_(w["beta"])
            mod.running_mean.copy_(w["moving_mean"])
            mod.running_var.copy_(w["moving_variance"])
        else:
            mod.weight.data.copy_(w["kernel"].T)
            mod.bias.data.copy_(w["bias"])


def hand_designed(C: int, D: int) -> dict:
    """The hand-designed byte-CNN C-D-384 in the fork's terms, as in unas/eval_reference.py (grid of
    step 3): 1x1 convolution to C channels, then D blocks of depthwise convolution (kernel 3) and 1x1
    convolution with batch norm and ReLU, global average pooling, 384-unit output."""
    def pw(bn):
        return {"type": "1x1Conv1D", "filters": C, "has_bn": bn, "has_relu": bn, "has_prepool": False}
    dw = {"type": "DWConv1D", "ker_size": 3, "1x_stride": False, "has_bn": False, "has_relu": False,
          "has_prepool": False}
    blocks = [{"is_branch": False, "layers": [pw(False)]}]
    blocks += [{"is_branch": False, "layers": [dict(dw), pw(True)]} for _ in range(D)]
    return {"conv_blocks": blocks, "pooling": {"type": "gap", "pool_size": 2}, "dense_blocks": [],
            "head_dropout": 0.0}


def model_arch(name: str) -> tuple:
    """(label, architecture) of a step-4 model: "mind_h7" / "mind_f401" = the search's choice
    (label = candidate index), "hand_<C>-<D>-384" = the hand-designed family in the fork's terms."""
    if name.startswith("hand_"):
        C, D, _ = (int(v) for v in name[len("hand_"):].split("-"))
        return name[len("hand_"):], hand_designed(C, D)
    return chosen_arch(name)


def chosen_arch(name: str) -> tuple[int, dict]:
    """Index and architecture chosen by unas/select_by_seeds.py for search ``name``."""
    res = ROOT / "paper/results/unas"
    sel = json.loads((res / f"{name}_selection.json").read_text(encoding="utf-8"))
    hist = json.loads((res / f"{name}_history.json").read_text(encoding="utf-8"))
    return sel["best"], next(c["arch"] for c in hist["candidates"] if c["index"] == sel["best"])
