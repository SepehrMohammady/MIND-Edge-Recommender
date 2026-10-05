"""Machine-readable experiment log.

Every run appends one JSON line to ``<results_dir>/experiments.jsonl`` (kept
under version control): what was run, with which seed and settings, on which
commit and hardware, how long it took and what it measured. Numbers quoted in
the paper are meant to be traceable to these records.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


_ENV: dict = {}


def environment() -> dict:
    """Software, hardware and code version of this process. Read once, at the
    first record, so a commit made while a long batch is running does not change
    what its later records report. ``dirty`` looks at the code only (src/,
    scripts/, config.yaml), not at the paper or the result files."""
    if not _ENV:
        _ENV.update({
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "commit": _git("rev-parse", "--short", "HEAD"),
            "dirty": bool(_git("status", "--porcelain", "--untracked-files=no", "--",
                               "src", "scripts", "config.yaml")),
        })
    return dict(_ENV)


def append(cfg: dict, name: str, settings: dict, results: dict, started: float) -> dict:
    """Append one run record and return it. ``started`` = ``time.time()`` at the
    beginning of the run."""
    record = {
        "run": name,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "minutes": round((time.time() - started) / 60, 2),
        "seed": cfg.get("seed"),
        "settings": settings,
        "results": results,
        "env": environment(),
    }
    path = Path(cfg["paths"]["results_dir"]) / "experiments.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read(cfg: dict) -> list[dict]:
    path = Path(cfg["paths"]["results_dir"]) / "experiments.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
