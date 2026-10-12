import pandas as pd
import pytest

from sentiment.cleaning import clean_frame
from sentiment.validation import validate_frame


def make(rows):
    df = pd.DataFrame(rows)
    if "review_id" not in df:
        df.insert(0, "review_id", range(len(df)))
    if "source" not in df:
        df["source"] = "amazon"
    if "review_date" not in df:
        df["review_date"] = "2025-01-15"
    df["review_date"] = df["review_date"].fillna("2025-01-15")
    return clean_frame(df)


def test_drops_bad_rows_and_reports_reasons():
    df = make([
        {"text": "really good product", "label": 1},
        {"text": "really good product", "label": 1},           # duplicate
        {"text": None, "label": 0},                            # null text
        {"text": "ok", "label": 1},                            # too short
        {"text": "awful awful awful", "label": 7},             # label out of range
        {"text": "bad bad experience", "label": None},         # null label
        {"text": "fine but slow shipping", "label": 0, "review_date": "not-a-date"},
        {"text": "works as described", "label": 1, "review_date": "2999-01-01"},
    ])
    _, report = validate_frame(df)
    assert report.rows_in == 8
    assert report.rows_out == 1
    assert report.dropped == {
        "null_text": 1, "too_short": 1, "label_out_of_range": 1, "null_label": 1,
        "invalid_date": 1, "future_date": 1, "duplicate_text": 1,
    }
    assert any("dropped" in w for w in report.warnings)


def test_conflicting_labels_are_removed():
    df = make([
        {"text": "Mixed feelings here", "label": 1},
        {"text": "mixed feelings HERE", "label": 0},   # same text after cleaning, other label
        {"text": "clearly great stuff", "label": 1},
        {"text": "clearly great stuff", "label": 1},   # plain duplicate
    ])
    out, report = validate_frame(df)
    assert report.dropped == {"conflicting_labels": 2, "duplicate_text": 1}
    assert list(out["text_clean"]) == ["clearly great stuff"]


def test_imbalance_warning():
    rows = [{"text": f"good thing number {i}", "label": 1} for i in range(9)]
    rows.append({"text": "terrible thing here", "label": 0})
    _, report = validate_frame(make(rows))
    assert any("imbalance" in w.lower() for w in report.warnings)
    assert report.label_balance["1"] == 0.9


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        validate_frame(pd.DataFrame({"text": ["x"]}))
