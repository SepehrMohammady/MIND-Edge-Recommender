"""Offline download of MIND (English) + xMIND (multilingual).

Run this BEFORE any training so a later loss of internet cannot interrupt
experiments::

    python -m src.download

Everything is driven by ``config.yaml`` (dataset size, language list, paths).

Sources (verified 2026-06-18)
-----------------------------
* MIND  : HF mirror ``Recommenders/MIND`` -- the canonical Azure blob
          (``mind201910small.blob.core.windows.net``) now returns HTTP 409.
* xMIND : ``aiana94/xMINDsmall`` / ``aiana94/xMINDlarge`` (CC-BY-NC-SA-4.0).
          Ships ONLY translated ``title``/``abstract`` keyed by MIND ``nid``;
          MIND ``behaviors.tsv`` is reused unchanged and joined on ``nid``.

A SHA256 manifest is written to ``data/manifest.json`` for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

from src.config import load_config

MIND_REPO = "Recommenders/MIND"
XMIND_REPO = {"small": "aiana94/xMINDsmall", "large": "aiana94/xMINDlarge"}
MIND_SPLITS = {"small": ["train", "dev"], "large": ["train", "dev", "test"]}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_mind(cfg: dict) -> dict:
    """Download + unzip the MIND zips for the configured size."""
    size = cfg["data"]["mind_size"]
    data_dir = Path(cfg["paths"]["data_dir"])
    raw_dir = data_dir / "mind_raw"
    out_dir = data_dir / "mind" / size
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = [f"MIND{size}_{s}.zip" for s in MIND_SPLITS[size]]
    print(f"[MIND] downloading {patterns} from {MIND_REPO} ...")
    snapshot_download(
        repo_id=MIND_REPO,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(raw_dir),
    )

    manifest = {}
    for split in MIND_SPLITS[size]:
        zip_path = raw_dir / f"MIND{size}_{split}.zip"
        dest = out_dir / split
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[MIND] extracting {zip_path.name} -> {dest}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        manifest[f"mind_{size}_{split}.zip"] = _sha256(zip_path)
    return manifest


def download_xmind(cfg: dict) -> dict:
    """Snapshot the full xMIND repo (all languages) for the configured size."""
    size = cfg["data"]["mind_size"]
    data_dir = Path(cfg["paths"]["data_dir"])
    out_dir = data_dir / "xmind" / size
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = XMIND_REPO[size]
    print(f"[xMIND] snapshotting {repo} (all languages) ...")
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(out_dir))

    # Manifest = sha256 of every parquet/tsv file pulled.
    manifest = {}
    for f in sorted(out_dir.rglob("*")):
        if f.is_file() and ".cache" not in f.parts and (
                f.name.endswith((".parquet", ".parquet.gzip", ".tsv", ".csv"))):
            manifest[f"xmind/{f.relative_to(out_dir).as_posix()}"] = _sha256(f)
    return manifest


def main() -> None:
    cfg = load_config()
    manifest = {}
    manifest.update(download_mind(cfg))
    manifest.update(download_xmind(cfg))

    manifest_path = Path(cfg["paths"]["data_dir"]) / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"\n[OK] datasets ready. SHA256 manifest -> {manifest_path}")
    print(f"     {len(manifest)} files pinned.")


if __name__ == "__main__":
    main()
