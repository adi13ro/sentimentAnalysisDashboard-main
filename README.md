# Sentiment Analysis Dashboard

Classifies customer reviews as positive or negative and visualises sentiment trends in an
interactive browser dashboard. The pipeline downloads a public review dataset, cleans and
validates it with an auditable quality report, trains a TF-IDF + logistic regression model,
scores every review and hands the results to a Streamlit app.

```
UCI reviews (Amazon, IMDb, Yelp) ──▶ clean ──▶ validate ──▶ train / evaluate ──▶ score
                                                   │                                │
                                        quality_report.json              reviews_scored.csv
                                                   └──────────▶ Streamlit dashboard ◀┘
```

## Results

Hold-out split (20 %) on 2,967 validated reviews:

| Metric | Value |
|---|---|
| Accuracy | 0.852 |
| F1 | 0.854 |
| ROC-AUC | 0.928 |
| 5-fold CV F1 (train) | 0.827 ± 0.007 |

## Stack

Python 3.12, pandas, scikit-learn, Streamlit, Plotly, pytest, ruff, uv. Docker optional.

## Run it

```bash
make install     # uv sync --extra dev
make pipeline    # download -> clean -> validate -> train -> score (about 10 s)
make dashboard   # http://localhost:8501
make test        # 10 tests incl. a headless render of the dashboard
```

Or in Docker (trains at build time, so the container starts ready):

```bash
docker build -t sentiment-dashboard . && docker run -p 8501:8501 sentiment-dashboard
```

### Use your own reviews

```bash
uv run sentiment-pipeline --csv path/to/reviews.csv
```

The CSV needs `text` and `label` (1 positive, 0 negative) columns. `source` and
`review_date` (ISO date) are optional. The UCI data has no timestamps, so the pipeline
attaches a deterministic date spread over the last twelve months purely so trend charts
have an x-axis; with your own data pass real dates.

If the machine is offline the pipeline falls back to a small synthetic review set so the
dashboard still runs. Pass `--no-synthetic` to fail instead.

## Dashboard

| Tab | What it shows |
|---|---|
| Trend | Review volume and positive share per month, average P(positive) by source |
| Distribution | Probability histogram by true label, sentiment by source, least confident predictions |
| Top terms | Strongest positive and negative n-grams from the model coefficients |
| Model | Hold-out metrics, confusion matrix, per-source accuracy |
| Data quality | Rows in/out, drop reasons, label balance, cleaning checklist |
| Try it | Type a review, get a probability and the terms that drove the prediction |

Sidebar filters: source, date range, predicted vs. labelled sentiment.

## Data cleaning and validation

Cleaning (`src/sentiment/cleaning.py`): HTML unescape and tag removal, NFKC Unicode
normalisation, URL stripping, lower-casing, contraction expansion that keeps negations,
punctuation and whitespace collapsing.

Validation (`src/sentiment/validation.py`) drops and counts rows that are null or empty after
cleaning, have a missing or out-of-range label, are shorter than 2 or longer than 400 words,
carry an invalid or future date, share text with a row of the opposite label, or duplicate an
earlier row. It warns on label imbalance (< 30 % minority) and when more than 20 % of rows
are dropped. The report is written to `data/processed/quality_report.json` and shown in the
dashboard.

## Layout

```
src/sentiment/
  data.py         download UCI zip, synthetic fallback, custom CSV loader
  cleaning.py     text normalisation
  validation.py   quality checks + QualityReport
  model.py        pipeline, metrics, top terms, per-review explanation
  pipeline.py     CLI entry point
app/dashboard.py  Streamlit app
tests/            unit tests + dashboard smoke test
```

## Dataset

Kotzias et al., "From Group to Individual Labels using Deep Features", KDD 2015.
UCI Machine Learning Repository: Sentiment Labelled Sentences (3,000 sentences, balanced).
