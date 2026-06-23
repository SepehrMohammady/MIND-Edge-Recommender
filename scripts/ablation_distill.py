"""Ablation: does teacher distillation help the deployed recommender?

Controlled comparison at the Micro-NAS architecture (64-5-384), MINDsmall-dev:
scratch (clicks only) vs distilled-init (pre-distilled to teacher anchors, then
fine-tuned). Thin wrapper over experiment.run_ablation. Saves artifacts/ablation.json.

    python -m scripts.ablation_distill
"""
import json
from pathlib import Path

from src.config import load_config
from src import experiment

cfg = load_config()
out = experiment.run_ablation(cfg, distill_epochs=12, train_epochs=cfg["train"]["epochs"])
Path(cfg["paths"]["artifacts_dir"], "ablation.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print("ABLATION DONE:", json.dumps(out, indent=2))
