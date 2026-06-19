"""Search-space primitives: sample, mutate, and key architectures.

An architecture is a small dict over the byte-CNN's tunable dims:
``{channels, depth, out_dim}`` (byte_embed_dim is fixed for a shared embedding).
"""
from __future__ import annotations

import random


def random_arch(space: dict, rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in space.items()}


def mutate(arch: dict, space: dict, rng: random.Random, p: float = 0.5) -> dict:
    a = dict(arch)
    for k, choices in space.items():
        if rng.random() < p:
            a[k] = rng.choice(choices)
    return a


def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}


def arch_key(arch: dict) -> tuple:
    return tuple(sorted(arch.items()))
