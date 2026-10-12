"""Data-quality checks run before any model sees the data.

Each check records what it dropped and why, so the quality report in the dashboard
explains the gap between raw row count and training row count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

MIN_WORDS = 2
MAX_WORDS = 400
VALID_LABELS = {0, 1}


@dataclass
class QualityReport:
    rows_in: int = 0
    rows_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    label_balance: dict[str, float] = field(default_factory=dict)
    sources: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _drop(df: pd.DataFrame, mask: pd.Series, reason: str, report: QualityReport) -> pd.DataFrame:
    n = int(mask.sum())
    if n:
        report.dropped[reason] = report.dropped.get(reason, 0) + n
    return df.loc[~mask]


def validate_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]:
    """Return the validated frame and a report of everything that was removed."""
    report = QualityReport(rows_in=len(df))
    required = {"review_id", "text", "text_clean", "label", "source", "review_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Frame is missing columns: {sorted(missing)}")

    out = df.copy()
    out = _drop(out, out["text"].isna(), "null_text", report)
    out = _drop(out, out["text_clean"].str.len() == 0, "empty_after_cleaning", report)

    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = _drop(out, out["label"].isna(), "null_label", report)
    out = _drop(out, ~out["label"].isin(VALID_LABELS), "label_out_of_range", report)
    out["label"] = out["label"].astype(int)

    out = _drop(out, out["n_words"] < MIN_WORDS, "too_short", report)
    out = _drop(out, out["n_words"] > MAX_WORDS, "too_long", report)

    dates = pd.to_datetime(out["review_date"], errors="coerce")
    out = _drop(out, dates.isna(), "invalid_date", report)
    out = _drop(out, pd.to_datetime(out["review_date"]) > pd.Timestamp.today(), "future_date", report)

    # Same text with conflicting labels is unusable as ground truth: drop every copy.
    conflicts = out.groupby("text_clean")["label"].nunique()
    conflicting = out["text_clean"].isin(conflicts[conflicts > 1].index)
    out = _drop(out, conflicting, "conflicting_labels", report)

    # Remaining exact duplicates are re-posts or scraping artefacts: keep one.
    dup = out.duplicated(subset=["text_clean"], keep="first")
    out = _drop(out, dup, "duplicate_text", report)

    report.rows_out = len(out)
    if len(out):
        balance = out["label"].value_counts(normalize=True)
        report.label_balance = {str(k): round(float(v), 4) for k, v in balance.items()}
        minority = balance.min()
        if minority < 0.3:
            report.warnings.append(
                f"Label imbalance: minority class is {minority:.1%} of rows; consider class weights"
            )
        report.sources = {str(k): int(v) for k, v in out["source"].value_counts().items()}
    else:
        report.warnings.append("No rows survived validation")

    drop_ratio = 1 - report.rows_out / max(report.rows_in, 1)
    if drop_ratio > 0.2:
        report.warnings.append(f"{drop_ratio:.1%} of rows were dropped; inspect the source data")

    return out.reset_index(drop=True), report
