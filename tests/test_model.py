import pandas as pd

from sentiment import model
from sentiment.cleaning import clean_text


def sample_frame(n_each: int = 40) -> pd.DataFrame:
    pos = ["great product love it", "excellent quality highly recommend", "works perfectly fast delivery",
           "fantastic value very happy", "amazing sound best purchase"]
    neg = ["terrible waste of money", "broke after two days awful", "poor quality very disappointed",
           "horrible support never again", "cheap and useless do not buy"]
    rows = []
    for i in range(n_each):
        rows.append({"text_clean": clean_text(pos[i % 5] + f" item{i}"), "label": 1})
        rows.append({"text_clean": clean_text(neg[i % 5] + f" item{i}"), "label": 0})
    return pd.DataFrame(rows)


def test_train_scores_and_explains(tmp_path):
    df = sample_frame()
    pipe, metrics = model.train_and_evaluate(df, test_size=0.25)
    assert metrics.f1 > 0.9
    assert metrics.n_train + metrics.n_test == len(df)
    assert sum(sum(r) for r in metrics.confusion) == metrics.n_test

    scored = model.score_frame(pipe, df)
    assert {"p_positive", "predicted", "predicted_label", "correct"} <= set(scored.columns)
    assert scored["p_positive"].between(0, 1).all()

    terms = model.top_terms(pipe, n=5)
    assert len(terms) == 10 and set(terms["polarity"]) == {"positive", "negative"}

    contrib = model.explain(pipe, "terrible product")
    assert not contrib.empty and (contrib["contribution"] < 0).any()

    model.save(pipe, metrics, tmp_path / "m.joblib", tmp_path / "m.json")
    loaded = model.load(tmp_path / "m.joblib")
    assert loaded.predict(["great product"])[0] == 1
