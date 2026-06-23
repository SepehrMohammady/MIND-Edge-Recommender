"""Export artifacts for the app: ONNX content encoder + cold-start topic prior.

Two core outputs, mapped to the ``edgeml_model_state_v1`` schema:
  1. content encoder  -> ONNX (byte ids -> embedding); deployable to ORT / X-CUBE-AI.
  2. cold-start prior -> per-category affinity weights to bootstrap ``topicWeights``.

Multi-model export (``export_all``) exports every (arm, precision) combination
and writes a ``models_manifest.json`` comparison table so callers can pick
the right trade-off between AUC, flash footprint, RAM and energy at runtime.

Naming convention:
  content_encoder_{arm}_{precision}.onnx   e.g. content_encoder_micro_nas_int8.onnx
  edgeml_{arm}_{precision}.json            e.g. edgeml_micro_nas_int8.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src import data_mind

# Canonical arm and precision identifiers (used in filenames and JSON).
ARM_LABELS     = ("nas", "micro_nas", "bin_unas")
PRECISIONS     = ("fp32", "int8", "binary")
PRECISION_BYTES = {"fp32": 4, "int8": 1, "binary": 0.125}   # bytes per weight


def export_encoder_onnx(encoder, cfg: dict, name: str = "content_encoder.onnx") -> str:
    """Export the byte-CNN news encoder to ONNX (dynamic batch)."""
    encoder = encoder.eval().cpu()
    path = Path(cfg["paths"]["artifacts_dir"]) / name
    dummy = torch.zeros(1, cfg["data"]["max_title_bytes"], dtype=torch.long)
    torch.onnx.export(
        encoder, (dummy,), str(path),
        input_names=["title_bytes"], output_names=["news_embedding"],
        dynamic_axes={"title_bytes": {0: "batch"}, "news_embedding": {0: "batch"}},
        opset_version=18)
    return str(path)


def build_topic_prior(cfg: dict) -> dict[str, float]:
    """Population cold-start prior = normalised click frequency per category."""
    news = data_mind.read_news(cfg, "train")
    behaviors = data_mind.read_behaviors(cfg, "train")
    counts: dict[str, int] = {}
    for imp in behaviors:
        clicked = list(imp["history"])
        clicked += [c for c, l in zip(imp["cands"], imp["labels"]) if l == 1]
        for nid in clicked:
            cat = news.get(nid, {}).get("category", "")
            if cat:
                counts[cat] = counts.get(cat, 0) + 1
    total = sum(counts.values()) or 1
    return {c: round(n / total, 6) for c, n in sorted(counts.items(), key=lambda x: -x[1])}


def export_one(
    cfg: dict,
    encoder,
    langs: list[str],
    arm: str = "micro_nas",
    precision: str = "int8",
    prior: dict[str, float] | None = None,
    metrics: dict[str, Any] | None = None,
) -> str:
    """Export a single (arm, precision) encoder with its state JSON.

    Args:
        cfg:       Loaded config dict.
        encoder:   Trained ByteLevelEncoder (already quantized to ``precision``).
        langs:     List of supported language codes.
        arm:       Search arm label — one of 'nas', 'micro_nas', 'bin_unas'.
        precision: Precision label — one of 'fp32', 'int8', 'binary'.
        prior:     Pre-computed topic prior (built once to avoid re-reading data).
        metrics:   Optional evaluation metrics dict {auc, mrr, ndcg5, ndcg10,
                   size_kb, ram_kb, macs, energy_uj} to embed in the JSON.

    Returns:
        Path to the written state JSON file.
    """
    onnx_name = f"content_encoder_{arm}_{precision}.onnx"
    onnx_path = export_encoder_onnx(encoder, cfg, name=onnx_name)

    if prior is None:
        prior = build_topic_prior(cfg)

    state: dict[str, Any] = {
        "schema": "edgeml_model_state_v1",
        "arm": arm,
        "precision": precision,
        "topicWeights": prior,
        "contentEncoder": {
            "format": "onnx",
            "path": Path(onnx_path).name,
            "inputBytes": cfg["data"]["max_title_bytes"],
            "embedDim": encoder.out_dim,
            "matryoshkaDim": cfg["student"]["matryoshka_dim"],
            "bytesPerWeight": PRECISION_BYTES.get(precision, 4),
        },
        "languages": langs,
        "notes": (
            f"Byte-level encoder ({arm}/{precision}); one model serves all languages. "
            f"See models_manifest.json for the full arm × precision comparison."
        ),
    }
    if metrics:
        state["metrics"] = metrics

    json_name = f"edgeml_{arm}_{precision}.json"
    out = Path(cfg["paths"]["artifacts_dir"]) / json_name
    out.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out)


def export_all(
    cfg: dict,
    models: dict[tuple[str, str], Any],
    langs: list[str],
    metrics_table: dict[tuple[str, str], dict] | None = None,
) -> str:
    """Export every (arm, precision) model and write a comparison manifest.

    Args:
        cfg:           Loaded config dict.
        models:        Dict mapping (arm, precision) → trained encoder.
                       e.g. {("nas","fp32"): enc1, ("micro_nas","int8"): enc2, ...}
        langs:         List of supported language codes.
        metrics_table: Optional dict mapping (arm, precision) → metrics dict.

    Returns:
        Path to the written ``models_manifest.json``.

    Example::

        export_all(cfg, {
            ("nas",       "fp32"):   model_nas_fp32,
            ("nas",       "int8"):   model_nas_int8,
            ("nas",       "binary"): model_nas_bin,
            ("micro_nas", "fp32"):   model_mnas_fp32,
            ("micro_nas", "int8"):   model_mnas_int8,   # ← on-device sweet spot
            ("micro_nas", "binary"): model_mnas_bin,
            ("bin_unas",  "fp32"):   model_bunas_fp32,
            ("bin_unas",  "int8"):   model_bunas_int8,
            ("bin_unas",  "binary"): model_bunas_bin,
        }, langs=cfg["languages"])
    """
    prior = build_topic_prior(cfg)   # build once, reuse for all models
    manifest_entries = []

    for (arm, precision), encoder in models.items():
        m = (metrics_table or {}).get((arm, precision))
        export_one(cfg, encoder, langs, arm=arm, precision=precision,
                   prior=prior, metrics=m)
        entry: dict[str, Any] = {
            "arm": arm,
            "precision": precision,
            "onnx_file": f"content_encoder_{arm}_{precision}.onnx",
            "state_file": f"edgeml_{arm}_{precision}.json",
            "recommended": (arm == "micro_nas" and precision == "int8"),
        }
        if m:
            entry["metrics"] = m
        manifest_entries.append(entry)

    # Sort: nas → micro_nas → bin_unas, fp32 → int8 → binary
    _arm_ord  = {a: i for i, a in enumerate(ARM_LABELS)}
    _prec_ord = {p: i for i, p in enumerate(PRECISIONS)}
    manifest_entries.sort(
        key=lambda e: (_arm_ord.get(e["arm"], 9), _prec_ord.get(e["precision"], 9))
    )

    manifest = {
        "schema": "edgeml_models_manifest_v1",
        "description": (
            "All exported arm × precision combinations. "
            "recommended=true marks the on-device sweet spot (Micro-NAS / INT8)."
        ),
        "models": manifest_entries,
    }
    out = Path(cfg["paths"]["artifacts_dir"]) / "models_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out)


# ── Backward-compatible alias ─────────────────────────────────────────────────
def export_artifacts(cfg: dict, encoder, langs: list[str]) -> str:
    """Legacy single-model export (Micro-NAS / INT8 sweet spot).

    Kept for notebook backward compatibility. New code should call
    ``export_one()`` or ``export_all()`` instead.
    """
    return export_one(cfg, encoder, langs, arm="micro_nas", precision="int8")
