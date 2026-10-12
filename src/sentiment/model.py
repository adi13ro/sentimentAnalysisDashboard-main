"""Sentiment classifier: TF-IDF (word + character n-grams) feeding logistic regression.

A linear model on sparse n-grams is a strong, fast baseline for short reviews and its
coefficients are directly interpretable, which the dashboard uses to show which terms
drive each prediction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from . import config

LABELS = {0: "negative", 1: "positive"}


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    cv_f1_mean: float
    cv_f1_std: float
    confusion: list[list[int]]
    n_train: int
    n_test: int

    def to_dict(self) -> dict:
        return asdict(self)


def build_pipeline(seed: int = config.RANDOM_SEED) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced", random_state=seed)
    return Pipeline([("features", features), ("clf", clf)])


def train_and_evaluate(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = config.RANDOM_SEED,
) -> tuple[Pipeline, Metrics]:
    X, y = df["text_clean"], df["label"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed,
                                              stratify=y)
    pipe = build_pipeline(seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1")

    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, 1]
    metrics = Metrics(
        accuracy=round(float(accuracy_score(y_te, pred)), 4),
        precision=round(float(precision_score(y_te, pred)), 4),
        recall=round(float(recall_score(y_te, pred)), 4),
        f1=round(float(f1_score(y_te, pred)), 4),
        roc_auc=round(float(roc_auc_score(y_te, proba)), 4),
        cv_f1_mean=round(float(cv_scores.mean()), 4),
        cv_f1_std=round(float(cv_scores.std()), 4),
        confusion=confusion_matrix(y_te, pred).tolist(),
        n_train=len(X_tr),
        n_test=len(X_te),
    )
    return pipe, metrics


def score_frame(pipe: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    proba = pipe.predict_proba(out["text_clean"])[:, 1]
    out["p_positive"] = np.round(proba, 4)
    out["predicted"] = (proba >= 0.5).astype(int)
    out["predicted_label"] = out["predicted"].map(LABELS)
    out["true_label"] = out["label"].map(LABELS)
    out["correct"] = out["predicted"] == out["label"]
    return out


def top_terms(pipe: Pipeline, n: int = 20) -> pd.DataFrame:
    """Most positive and most negative word n-grams by model coefficient."""
    word_vec = pipe.named_steps["features"].transformer_list[0][1]
    coef = pipe.named_steps["clf"].coef_[0][: len(word_vec.vocabulary_)]
    terms = np.array(word_vec.get_feature_names_out())
    order = np.argsort(coef)
    neg = pd.DataFrame({"term": terms[order[:n]], "weight": coef[order[:n]], "polarity": "negative"})
    pos = pd.DataFrame({"term": terms[order[-n:]][::-1], "weight": coef[order[-n:]][::-1],
                        "polarity": "positive"})
    return pd.concat([pos, neg], ignore_index=True)


def explain(pipe: Pipeline, text_clean: str, n: int = 8) -> pd.DataFrame:
    """Per-term contribution to the log-odds for one cleaned review (word features only)."""
    word_vec = pipe.named_steps["features"].transformer_list[0][1]
    vec = word_vec.transform([text_clean])
    coef = pipe.named_steps["clf"].coef_[0][: len(word_vec.vocabulary_)]
    terms = word_vec.get_feature_names_out()
    contrib = vec.multiply(coef).toarray()[0]
    idx = np.nonzero(contrib)[0]
    df = pd.DataFrame({"term": terms[idx], "contribution": contrib[idx]})
    df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index)
    return df.head(n).reset_index(drop=True)


def save(pipe: Pipeline, metrics: Metrics, model_path: Path = config.MODEL_PATH,
         metrics_path: Path = config.METRICS_JSON) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    metrics_path.write_text(json.dumps(metrics.to_dict(), indent=2))


def load(model_path: Path = config.MODEL_PATH) -> Pipeline:
    return joblib.load(model_path)
