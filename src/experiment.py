"""High-level orchestration so the notebook stays short.

Pipeline: anchors -> distill student -> NAS (3 arms) -> train a recommender per
arm's best architecture -> evaluate across the precision axis (FP32/INT8/Binary)
-> evaluate across all 14 languages -> assemble the results matrix. NRMS is the
FP32 ceiling. Each function is one notebook cell.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from src import (baseline_nrms, data_xmind, footprint, metrics, quantize,
                 recommender, student, teacher)
from src.nas import search
from src.nas.search import build_encoder

PRECISIONS = ["fp32", "int8", "binary"]


# ---------------------------------------------------------------- stages
def ensure_anchors(cfg: dict) -> None:
    for split in ("train", "dev"):
        teacher.build_anchors(cfg, split)


def distill(cfg: dict):
    return student.train_student(cfg)


def search_arms(cfg: dict, generations=None, population=None,
                n_train=20000, n_val=4000, distill_epochs=2) -> dict:
    """Run all three NAS arms; cache the per-arch records to artifacts."""
    out = {}
    fitness_cache = {}
    for arm, (precision, _) in search.ARMS.items():
        if precision not in fitness_cache:
            fitness_cache[precision] = search.make_distill_fitness(
                cfg, precision, n_train, n_val, distill_epochs)
        out[arm] = search.search(cfg, arm, generations, population,
                                 fitness_fn=fitness_cache[precision])
    Path(cfg["paths"]["artifacts_dir"], "nas_results.json").write_text(
        json.dumps({k: v[:10] for k, v in out.items()}, indent=2), encoding="utf-8")
    return out


def best_arch(arm_results: list[dict]) -> dict:
    feas = [r for r in arm_results if r["feasible"]] or arm_results
    return max(feas, key=lambda r: r["quality"])["arch"]


# -------------------------------------------------- train + precision sweep
def train_for_arch(cfg: dict, arch: dict, epochs=None, max_train_impressions=None):
    enc = build_encoder(cfg, arch)
    return recommender.train_recommender(cfg, news_encoder=enc, epochs=epochs,
                                         max_train_impressions=max_train_impressions)


def precision_sweep(cfg: dict, model, arch: dict, arm: str,
                    qat_epochs=2, max_eval=None) -> list[dict]:
    """Evaluate one trained model at FP32/INT8/Binary (PTQ + short QAT)."""
    ex = torch.zeros(1, cfg["data"]["max_title_bytes"], dtype=torch.long)
    rows = []
    for prec in PRECISIONS:
        qm = quantize.convert_to_quant(model, prec).to(next(model.parameters()).device)
        if prec != "fp32" and qat_epochs:
            qm = recommender.train_recommender(cfg, model=qm, epochs=qat_epochs,
                                               max_train_impressions=max_train_impressions_for(cfg))
        res = recommender.evaluate(cfg, qm, split="dev", max_impressions=max_eval)
        fp = quantize.quant_fp_fraction(qm.news_encoder)
        foot = footprint.summarize(qm.news_encoder, ex, prec, cfg, fp_fraction=fp)
        rows.append({"arm": arm, "precision": prec, **arch, "auc": round(res["auc"], 4),
                     "mrr": round(res["mrr"], 4), "ndcg@10": round(res["ndcg@10"], 4),
                     "size_kb": foot["size_kb"], "ram_kb": search.estimate_ram_kb(arch, cfg, prec),
                     "macs": foot["macs"], "energy_uj": foot["energy_uj_per_inf"]})
    return rows


def max_train_impressions_for(cfg: dict):
    return cfg["train"].get("qat_max_impressions")


# ------------------------------------------------------------- multilingual
@torch.no_grad()
def eval_languages(cfg: dict, model, langs=None, max_impressions=2000) -> pd.DataFrame:
    langs = langs or (["en"] + data_xmind.available_langs(cfg))
    rows = []
    for lang in langs:
        res = recommender.evaluate(cfg, model, split="dev",
                                   lang=None if lang == "en" else lang,
                                   max_impressions=max_impressions)
        rows.append({"lang": lang, **{k: round(v, 4) for k, v in res.items()
                                      if k != "n_impressions"}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ baseline
def baseline_row(cfg: dict, epochs=None, max_eval=None) -> dict:
    m = baseline_nrms.train_nrms(cfg, epochs=epochs)
    res = baseline_nrms.evaluate_nrms(cfg, m, split="dev", max_impressions=max_eval)
    return {"arm": "NRMS (baseline)", "precision": "fp32", "auc": round(res["auc"], 4),
            "mrr": round(res["mrr"], 4), "ndcg@10": round(res["ndcg@10"], 4)}


def save_matrix(cfg: dict, rows: list[dict], name="results_matrix.csv") -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(Path(cfg["paths"]["artifacts_dir"]) / name, index=False)
    return df


# ------------------------------------------------------------------ ablation
def run_ablation(cfg: dict, arch: dict | None = None,
                 distill_epochs: int = 12, train_epochs: int | None = None,
                 max_train_impressions: int | None = None) -> dict:
    """Does distillation help? Compare a news encoder trained on clicks from
    scratch vs initialised from the distilled student, at a fixed architecture.
    ``max_train_impressions`` caps the click log for fast (QUICK) runs."""
    import torch.nn.functional as F

    arch = arch or {"channels": 64, "depth": 5, "out_dim": 384}
    be = cfg["student"]["byte_embed_dim"]
    train_epochs = train_epochs or cfg["train"]["epochs"]
    device = ("cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu")

    def mk():
        return student.ByteCNNEncoder(be, arch["channels"], arch["depth"], arch["out_dim"])

    scratch = recommender.train_recommender(cfg, news_encoder=mk(), epochs=train_epochs,
                                            max_train_impressions=max_train_impressions)
    r_scratch = recommender.evaluate(cfg, scratch, split="dev")

    bytes_np, tgt_np, anchors = student.build_distill_data(cfg, "train")
    anchors_t = torch.tensor(anchors, device=device)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(bytes_np, dtype=torch.long),
                                       torch.tensor(tgt_np, dtype=torch.long)),
        batch_size=512, shuffle=True, drop_last=True)
    enc = mk().to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=cfg["train"]["distill_lr"])
    dims = tuple(d for d in (64, 128, 256, 384) if d <= arch["out_dim"])
    enc.train()
    for _ in range(distill_epochs):
        for ids, ti in dl:
            ids = ids.to(device)
            tgt = anchors_t[ti.to(device)]
            pred = F.normalize(enc(ids), dim=-1)
            loss = sum(1 - F.cosine_similarity(pred[:, :d], tgt[:, :d], dim=-1).mean()
                       for d in dims) / len(dims)
            opt.zero_grad(); loss.backward(); opt.step()
    distilled = recommender.train_recommender(
        cfg, model=recommender.NewsRecommender(enc), epochs=train_epochs,
        max_train_impressions=max_train_impressions)
    r_distilled = recommender.evaluate(cfg, distilled, split="dev")

    return {"arch": arch, "train_epochs": train_epochs, "distill_epochs": distill_epochs,
            "scratch": {k: round(v, 4) for k, v in r_scratch.items()},
            "distilled_init": {k: round(v, 4) for k, v in r_distilled.items()}}
