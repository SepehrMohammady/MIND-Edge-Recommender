"""Export artifacts for the app: ONNX content encoder + cold-start topic prior.

Two outputs (handoff deliverables 6.6), mapped to the app's
``edgeml_model_state_v1`` schema:
  1. content encoder  -> ONNX (byte ids -> embedding); deployable to ORT/TFLite.
  2. cold-start prior -> per-category affinity weights to initialise the app's
     ``topicWeights`` instead of zeros (fixes cold start).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from src import data_mind


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


def export_artifacts(cfg: dict, encoder, langs: list[str]) -> str:
    """Write encoder ONNX + topic prior + schema-mapped state JSON."""
    onnx_path = export_encoder_onnx(encoder, cfg)
    prior = build_topic_prior(cfg)
    state = {
        "schema": "edgeml_model_state_v1",
        "topicWeights": prior,
        "contentEncoder": {
            "format": "onnx",
            "path": Path(onnx_path).name,
            "inputBytes": cfg["data"]["max_title_bytes"],
            "embedDim": encoder.out_dim,
            "matryoshkaDim": cfg["student"]["matryoshka_dim"],
        },
        "languages": langs,
        "notes": "byte-level encoder; one model serves all languages.",
    }
    out = Path(cfg["paths"]["artifacts_dir"]) / "edgeml_model_state_v1.json"
    out.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out)
