"""Evolutionary architecture search with a distillation-quality fitness.

Candidates are scored by how well they reproduce the teacher anchors (cosine,
Matryoshka-sliced to the candidate's out_dim) after a short QAT distillation in
the arm's precision, subject to STM32H7 footprint feasibility (Micro-NAS arms).
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from src import footprint, quantize, student
from src.student import ByteCNNEncoder
from .search_space import arch_key, crossover, mutate, random_arch

# arm -> (precision, constrained?)
ARMS = {
    "nas": ("fp32", False),
    "micro_nas": ("int8", True),
    "binarized_micro_nas": ("binary", True),
}


def build_encoder(cfg: dict, arch: dict) -> ByteCNNEncoder:
    return ByteCNNEncoder(cfg["student"]["byte_embed_dim"],
                          arch["channels"], arch["depth"], arch["out_dim"])


def estimate_ram_kb(arch: dict, cfg: dict, precision: str) -> float:
    """Peak activation buffer proxy = channels x seq_len x bytes/elem."""
    bytes_el = {"fp32": 4, "int8": 1, "binary": 1}[precision]
    return arch["channels"] * cfg["data"]["max_title_bytes"] * bytes_el / 1024


def arch_cost(cfg: dict, arch: dict, precision: str) -> dict:
    enc = quantize.convert_to_quant(build_encoder(cfg, arch), precision)
    ex = torch.zeros(1, cfg["data"]["max_title_bytes"], dtype=torch.long)
    fp = quantize.quant_fp_fraction(enc)
    foot = footprint.summarize(enc, ex, precision, cfg, fp_fraction=fp)
    foot["ram_kb"] = round(estimate_ram_kb(arch, cfg, precision), 2)
    return foot


def feasible(foot: dict, c: dict | None) -> bool:
    if not c:
        return True
    return (foot["size_kb"] <= c["flash_kb"] and foot["ram_kb"] <= c["ram_kb"]
            and foot["macs"] <= c["max_macs"])


def make_distill_fitness(cfg: dict, precision: str, n_train=20000, n_val=4000,
                         epochs=2, batch=512, device=None):
    """Prepare a subsampled distillation set once; return ``fitness(arch)``."""
    device = device or ("cuda" if torch.cuda.is_available()
                        and cfg["train"]["device"] == "cuda" else "cpu")
    bytes_np, tgt_np, anchors = student.build_distill_data(cfg, "train")
    anchors_t = torch.tensor(anchors, device=device)
    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(len(bytes_np))
    tr, va = perm[:n_train], perm[n_train:n_train + n_val]

    Xtr = torch.tensor(bytes_np[tr], dtype=torch.long)
    Ttr = torch.tensor(tgt_np[tr], dtype=torch.long)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, Ttr),
                                     batch_size=batch, shuffle=True, drop_last=True)
    Xva = torch.tensor(bytes_np[va], dtype=torch.long, device=device)
    Tva = anchors_t[torch.tensor(tgt_np[va], dtype=torch.long, device=device)]

    def fitness(arch: dict) -> float:
        d = arch["out_dim"]
        enc = quantize.convert_to_quant(build_encoder(cfg, arch), precision).to(device)
        opt = torch.optim.AdamW(enc.parameters(), lr=cfg["train"]["distill_lr"])
        enc.train()
        for _ in range(epochs):
            for ids, ti in dl:
                ids = ids.to(device)
                tgt = anchors_t[ti.to(device)][:, :d]
                loss = (1 - F.cosine_similarity(enc(ids), tgt, dim=-1)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        enc.eval()
        with torch.no_grad():
            q = F.cosine_similarity(enc(Xva), Tva[:, :d], dim=-1).mean().item()
        return q

    return fitness


def search(cfg: dict, arm: str, generations: int | None = None,
           population: int | None = None, fitness_fn=None, verbose=True) -> list[dict]:
    """Run one arm. Returns all assessed archs, best-first."""
    precision, constrained = ARMS[arm]
    constraints = cfg["nas"]["constraints"] if constrained else None
    space = cfg["nas"]["search_space"]
    rng = random.Random(cfg["seed"])
    pop_n = population or cfg["nas"]["population"]
    gens = generations or cfg["nas"]["generations"]
    if fitness_fn is None:
        fitness_fn = make_distill_fitness(cfg, precision)

    seen: dict = {}

    def assess(arch: dict) -> dict:
        k = arch_key(arch)
        if k in seen:
            return seen[k]
        foot = arch_cost(cfg, arch, precision)
        ok = feasible(foot, constraints)
        quality = fitness_fn(arch) if ok else -1.0
        rec = {"arch": arch, "arm": arm, "precision": precision,
               "feasible": ok, "quality": round(quality, 4), **foot}
        seen[k] = rec
        return rec

    results = [assess(random_arch(space, rng)) for _ in range(pop_n)]
    for g in range(gens):
        feas = [r for r in results if r["feasible"]] or results
        parents = sorted(feas, key=lambda r: r["quality"], reverse=True)[:max(2, pop_n // 4)]
        children = []
        while len(children) < pop_n - len(parents):
            pa, pb = rng.choice(parents), rng.choice(parents)
            children.append(assess(mutate(crossover(pa["arch"], pb["arch"], rng), space, rng)))
        results = parents + children
        if verbose:
            best = max(results, key=lambda r: r["quality"])
            nfeas = sum(r["feasible"] for r in seen.values())
            print(f"[nas:{arm}] gen {g+1}/{gens} best_q={best['quality']:.4f} "
                  f"size_kb={best['size_kb']} feasible={nfeas}/{len(seen)}")
    return sorted(seen.values(), key=lambda r: (r["feasible"], r["quality"]), reverse=True)
