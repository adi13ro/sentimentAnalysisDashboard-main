"""Acquire raw review data.

Primary source: the UCI "Sentiment Labelled Sentences" dataset (3,000 reviews from
Amazon, IMDb and Yelp, labelled 1 = positive, 0 = negative). If the download fails
(offline machine, CI without network) a small synthetic set is generated instead so
the rest of the pipeline and the dashboard still run end to end.

The UCI data has no timestamps, so a deterministic review_date spread over the last
twelve months is attached to support trend charts. Bring your own CSV with a real
date column to replace it (see README).
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

UCI_FILES = {
    "amazon": "sentiment labelled sentences/amazon_cells_labelled.txt",
    "imdb": "sentiment labelled sentences/imdb_labelled.txt",
    "yelp": "sentiment labelled sentences/yelp_labelled.txt",
}

REQUIRED_COLUMNS = ["review_id", "text", "label", "source", "review_date"]


def _attach_dates(df: pd.DataFrame, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(months=12)
    span_days = (end - start).days
    # Slightly skew towards recent months, as real review volumes tend to grow.
    offsets = (rng.beta(2.0, 1.3, size=len(df)) * span_days).astype(int)
    df = df.copy()
    df["review_date"] = (start + pd.to_timedelta(offsets, unit="D")).strftime("%Y-%m-%d")
    return df


def download_uci(url: str = config.UCI_URL, timeout: int = 90) -> pd.DataFrame:
    log.info("Downloading %s", url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    frames = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for source, member in UCI_FILES.items():
            with zf.open(member) as fh:
                part = pd.read_csv(
                    fh, sep="\t", header=None, names=["text", "label"],
                    quoting=3, encoding="utf-8", on_bad_lines="skip",
                )
            part["source"] = source
            frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    return df


def synthetic_reviews(n: int = 600, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Template-based fallback data. Clearly weaker than real reviews, but keeps the demo alive."""
    rng = np.random.default_rng(seed)
    pos = [
        "Absolutely love this {thing}, works perfectly and arrived early.",
        "Great {thing}. Battery life is excellent and the build quality is solid.",
        "The {thing} exceeded my expectations, highly recommend it.",
        "Fantastic service and the {thing} is exactly as described.",
        "Five stars. Best {thing} I have bought this year.",
    ]
    neg = [
        "Terrible {thing}, stopped working after two days.",
        "Waste of money. The {thing} feels cheap and the support was useless.",
        "Very disappointed, the {thing} does not match the description at all.",
        "The {thing} arrived broken and the refund took weeks.",
        "Do not buy this {thing}. Poor quality and awful customer service.",
    ]
    things = ["phone case", "headset", "charger", "blender", "keyboard", "jacket", "lamp"]
    sources = ["amazon", "imdb", "yelp"]
    rows = []
    for i in range(n):
        label = int(rng.integers(0, 2))
        template = (pos if label else neg)[rng.integers(0, 5)]
        rows.append({
            "text": template.format(thing=things[rng.integers(0, len(things))]),
            "label": label,
            "source": sources[rng.integers(0, 3)],
        })
    return pd.DataFrame(rows)


def load_raw(force_download: bool = False, allow_synthetic: bool = True) -> pd.DataFrame:
    """Return the raw review frame, downloading and caching it on first use."""
    path: Path = config.RAW_CSV
    if path.exists() and not force_download:
        return pd.read_csv(path)

    try:
        df = download_uci()
        origin = "uci"
    except Exception as exc:
        if not allow_synthetic:
            raise
        log.warning("Download failed (%s); using synthetic data", exc)
        df = synthetic_reviews()
        origin = "synthetic"

    df = _attach_dates(df)
    df.insert(0, "review_id", range(1, len(df) + 1))
    df["origin"] = origin
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved %d raw reviews to %s", len(df), path)
    return df


def load_user_csv(path: str | Path) -> pd.DataFrame:
    """Load a user-supplied CSV with at least text and label columns."""
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    if "source" not in df:
        df["source"] = "custom"
    if "review_date" not in df:
        df = _attach_dates(df)
    if "review_id" not in df:
        df.insert(0, "review_id", range(1, len(df) + 1))
    df["origin"] = "custom"
    return df
