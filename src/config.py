"""Central configuration loader.

All experiment knobs live in ``config.yaml`` at the repo root. The notebook
loads a config dict, optionally overrides a few keys in-memory, and passes it
to every other module -- so the notebook stays the single control surface.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    """Load ``config.yaml`` and resolve/create the declared directories.

    Paths under ``cfg['paths']`` are made absolute (relative to the repo root)
    and created if missing, so downstream code never worries about cwd.
    """
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["paths"] = {k: str((ROOT / v).resolve()) for k, v in cfg["paths"].items()}
    for directory in cfg["paths"].values():
        Path(directory).mkdir(parents=True, exist_ok=True)

    return cfg


def resolve_langs(cfg: dict) -> list[str]:
    """Return the concrete list of xMIND language codes to evaluate."""
    return list(cfg["data"]["xmind_langs"])
