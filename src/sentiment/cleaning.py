"""Text normalisation for review data.

Cleaning is deliberately conservative. Negation words ("not", "never") and
contractions carry sentiment, so they are preserved rather than stripped as stopwords.
"""

from __future__ import annotations

import html
import re
import unicodedata

import pandas as pd

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_TEXT_RE = re.compile(r"[^a-z0-9'!?.,\s]")
_MULTI_PUNCT_RE = re.compile(r"([!?.,])\1+")
_WS_RE = re.compile(r"\s+")

CONTRACTIONS = {
    "won't": "will not", "can't": "can not", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'ve": " have", "'m": " am",
}


def expand_contractions(text: str) -> str:
    text = text.replace("won't", "will not").replace("can't", "can not")
    for short, long in CONTRACTIONS.items():
        if short in ("won't", "can't"):
            continue
        text = text.replace(short, long)
    return text


def clean_text(text: object) -> str:
    """Normalise a single review. Returns an empty string for null or non-string input."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.lower()
    text = expand_contractions(text)
    text = _NON_TEXT_RE.sub(" ", text)
    text = _MULTI_PUNCT_RE.sub(r"\1", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add a text_clean column and derived length features."""
    out = df.copy()
    out["text_clean"] = out["text"].map(clean_text)
    out["n_words"] = out["text_clean"].str.split().str.len().fillna(0).astype(int)
    out["n_chars"] = out["text_clean"].str.len()
    return out
