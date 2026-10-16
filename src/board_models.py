"""Checkpoint and byte table of an exported 8-bit encoder file (artifacts/stedgeai/models/<stem>_seed<s>_body_*.onnx),
for the board scripts (firmware/build.py, scripts/board_validate.py). Same naming as scripts/export_int8.py."""
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_for(stem: str, seed: int) -> Path:
    """stem: unas_f401_c142, unas_h7, hand_32-5-384, ref_64-5-384, ..."""
    if stem.startswith("unas_"):
        return ROOT / f"artifacts/runs/unas_full/mind_{stem[len('unas_'):]}_distill_ft_en_seed{seed}.pt"
    if stem.startswith("hand_"):
        return ROOT / f"artifacts/runs/unas_full/{stem}_distill_ft_en_seed{seed}.pt"
    if stem == "ref_64-5-384":
        return ROOT / f"artifacts/runs/p1/distill_ft_en_seed{seed}.pt"
    if stem == "ref_mixed_64-5-384":
        return ROOT / f"artifacts/runs/p1/distill_ft_mixed_seed{seed}.pt"
    raise ValueError(f"no checkpoint rule for {stem}")


def byte_table(stem: str, seed: int) -> np.ndarray:
    """(257, 64) float32: the model's own byte embedding (the lookup a device does before the network)."""
    sd = torch.load(checkpoint_for(stem, seed), map_location="cpu")
    return sd["news_encoder.embed.weight"].numpy().astype(np.float32)


def split_name(onnx: Path):
    """unas_f401_c142_seed42_body_int8qdq.onnx -> ("unas_f401_c142", 42, "int8qdq")"""
    stem, rest = onnx.stem.split("_seed")
    seed, kind = rest.split("_body_")
    return stem, int(seed), kind
