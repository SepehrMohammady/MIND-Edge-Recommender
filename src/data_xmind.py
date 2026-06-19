"""xMIND loading + cross-lingual fusion with MIND.

xMIND ships ONLY translated text (columns: nid, title, abstract) per language,
at::

    data/xmind/<size>/data/<lang>/<split>.parquet.gzip

It carries no behaviours, categories, or entities. For multilingual evaluation
we therefore reuse MIND's English impressions and category labels and merely
swap the title/abstract text to the target language, joined on ``nid``.

=> "multilingual eval" = cross-lingual transfer over identical English
   impressions, NOT native-language user behaviour. (Stated in the paper.)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import data_mind


def _lang_path(cfg: dict, lang: str, split: str) -> Path:
    return (Path(cfg["paths"]["data_dir"]) / "xmind" / cfg["data"]["mind_size"]
            / "data" / lang / f"{split}.parquet.gzip")


def available_langs(cfg: dict) -> list[str]:
    """Languages actually present on disk (intersection with config)."""
    base = Path(cfg["paths"]["data_dir"]) / "xmind" / cfg["data"]["mind_size"] / "data"
    on_disk = {p.name for p in base.iterdir() if p.is_dir()} if base.exists() else set()
    return [l for l in cfg["data"]["xmind_langs"] if l in on_disk]


def read_xmind_text(cfg: dict, lang: str, split: str) -> dict[str, dict]:
    """Return ``nid -> {title, abstract}`` for one language/split."""
    df = pd.read_parquet(_lang_path(cfg, lang, split)).fillna("")
    return {r.nid: {"title": r.title, "abstract": r.abstract}
            for r in df.itertuples(index=False)}


def localized_news(cfg: dict, lang: str, split: str) -> dict[str, dict]:
    """MIND news with title/abstract replaced by the ``lang`` translation.

    Category/subcategory come from MIND (English); text comes from xMIND.
    News ids missing in xMIND keep their English text (rare; logged by caller).
    """
    news = data_mind.read_news(cfg, split)          # English base (has categories)
    trans = read_xmind_text(cfg, lang, split)       # translated text
    for nid, item in news.items():
        if nid in trans:
            item["title"] = trans[nid]["title"] or item["title"]
            item["abstract"] = trans[nid]["abstract"] or item["abstract"]
    return news
