"""Streamlit dashboard for customer review sentiment.

Run: streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment import cleaning, config, model

st.set_page_config(page_title="Review Sentiment", page_icon=":speech_balloon:", layout="wide")

POS, NEG = "#2a9d8f", "#e76f51"
COLORS = {"positive": POS, "negative": NEG}


@st.cache_data(show_spinner=False)
def load_scored() -> pd.DataFrame:
    df = pd.read_csv(config.SCORED_CSV, parse_dates=["review_date"])
    df["month"] = df["review_date"].dt.to_period("M").dt.to_timestamp()
    return df


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


@st.cache_resource(show_spinner=False)
def load_model():
    return model.load()


def require_artifacts() -> None:
    missing = [p.name for p in (config.SCORED_CSV, config.MODEL_PATH, config.METRICS_JSON,
                                config.QUALITY_JSON) if not p.exists()]
    if missing:
        st.error("Pipeline artefacts not found: " + ", ".join(missing))
        st.code("uv run sentiment-pipeline", language="bash")
        st.stop()


require_artifacts()
df = load_scored()
metrics = load_json(config.METRICS_JSON)
quality = load_json(config.QUALITY_JSON)
pipe = load_model()

# ---------- sidebar filters ----------
st.sidebar.header("Filters")
sources = sorted(df["source"].unique())
chosen_sources = st.sidebar.multiselect("Source", sources, default=sources)
min_d, max_d = df["review_date"].min().date(), df["review_date"].max().date()
date_range = st.sidebar.slider("Review date", min_d, max_d, (min_d, max_d))
use_pred = st.sidebar.radio("Sentiment from", ["Model prediction", "Ground-truth label"]) == \
    "Model prediction"
label_col = "predicted_label" if use_pred else "true_label"

mask = df["source"].isin(chosen_sources) & df["review_date"].dt.date.between(*date_range)
view = df.loc[mask]
if view.empty:
    st.warning("No reviews match the current filters.")
    st.stop()

# ---------- headline ----------
st.title("Customer Review Sentiment")
st.caption("TF-IDF + logistic regression over Amazon, IMDb and Yelp reviews. "
           "Filter by source and date; switch between predicted and labelled sentiment.")

share_pos = (view[label_col] == "positive").mean()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Reviews", f"{len(view):,}")
c2.metric("Positive share", f"{share_pos:.1%}")
c3.metric("Avg. P(positive)", f"{view['p_positive'].mean():.2f}")
c4.metric("Test F1", f"{metrics['f1']:.3f}")
c5.metric("Test ROC-AUC", f"{metrics['roc_auc']:.3f}")

tab_trend, tab_dist, tab_terms, tab_model, tab_quality, tab_live = st.tabs(
    ["Trend", "Distribution", "Top terms", "Model", "Data quality", "Try it"]
)

# ---------- trend ----------
with tab_trend:
    monthly = (view.groupby(["month", label_col]).size().rename("count").reset_index())
    totals = monthly.groupby("month")["count"].transform("sum")
    monthly["share"] = monthly["count"] / totals

    left, right = st.columns(2)
    fig = px.bar(monthly, x="month", y="count", color=label_col, color_discrete_map=COLORS,
                 title="Review volume per month", labels={"month": "", "count": "reviews"})
    fig.update_layout(legend_title_text="", bargap=0.2)
    left.plotly_chart(fig, width="stretch")

    pos_share = monthly[monthly[label_col] == "positive"][["month", "share"]]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=pos_share["month"], y=pos_share["share"], mode="lines+markers",
                              line=dict(color=POS, width=3), name="positive share"))
    fig2.add_hline(y=share_pos, line_dash="dot", line_color="grey",
                   annotation_text=f"overall {share_pos:.0%}")
    fig2.update_layout(title="Positive share per month", yaxis_tickformat=".0%",
                       yaxis_range=[0, 1], xaxis_title="", yaxis_title="")
    right.plotly_chart(fig2, width="stretch")

    by_source = (view.groupby(["month", "source"])["p_positive"].mean().reset_index())
    fig3 = px.line(by_source, x="month", y="p_positive", color="source", markers=True,
                   title="Average P(positive) by source", labels={"month": "", "p_positive": ""})
    fig3.update_layout(yaxis_range=[0, 1], legend_title_text="")
    st.plotly_chart(fig3, width="stretch")

# ---------- distribution ----------
with tab_dist:
    left, right = st.columns(2)
    fig = px.histogram(view, x="p_positive", color="true_label", nbins=40, barmode="overlay",
                       color_discrete_map=COLORS, opacity=0.75,
                       title="Predicted probability by true label",
                       labels={"p_positive": "P(positive)", "true_label": ""})
    fig.add_vline(x=0.5, line_dash="dash", line_color="grey")
    left.plotly_chart(fig, width="stretch")

    per_source = (view.groupby(["source", label_col]).size().rename("count").reset_index())
    fig2 = px.bar(per_source, x="source", y="count", color=label_col, barmode="group",
                  color_discrete_map=COLORS, title="Sentiment by source",
                  labels={"source": "", "count": "reviews"})
    fig2.update_layout(legend_title_text="")
    right.plotly_chart(fig2, width="stretch")

    st.subheader("Least confident predictions")
    uncertain = view.assign(margin=(view["p_positive"] - 0.5).abs()).nsmallest(15, "margin")
    st.dataframe(uncertain[["review_date", "source", "text", "true_label", "p_positive"]],
                 width="stretch", hide_index=True)

# ---------- top terms ----------
with tab_terms:
    terms = model.top_terms(pipe, n=20)
    fig = px.bar(terms.sort_values("weight"), x="weight", y="term", orientation="h",
                 color="polarity", color_discrete_map=COLORS,
                 title="Strongest word features (logistic regression weights)",
                 labels={"weight": "coefficient", "term": ""})
    fig.update_layout(height=700, legend_title_text="", yaxis=dict(dtick=1))
    st.plotly_chart(fig, width="stretch")

# ---------- model ----------
with tab_model:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Hold-out metrics")
        mdf = pd.DataFrame({
            "metric": ["accuracy", "precision", "recall", "f1", "roc_auc"],
            "value": [metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]],
        })
        st.dataframe(mdf, hide_index=True, width="stretch")
        st.write(f"5-fold CV F1 on the training split: **{metrics['cv_f1_mean']:.3f}** "
                 f"(± {metrics['cv_f1_std']:.3f}). Train/test rows: "
                 f"{metrics['n_train']:,} / {metrics['n_test']:,}.")
    with right:
        cm = metrics["confusion"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=["pred negative", "pred positive"], y=["true negative", "true positive"],
            text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
        fig.update_layout(title="Confusion matrix (test split)", height=350)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Accuracy on the filtered view")
    acc_by_source = view.groupby("source")["correct"].mean().reset_index()
    fig = px.bar(acc_by_source, x="source", y="correct", title="", text_auto=".1%",
                 labels={"source": "", "correct": "accuracy"}, color_discrete_sequence=["#264653"])
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], height=300)
    st.plotly_chart(fig, width="stretch")
    st.caption("Note: this view includes training rows, so it overstates generalisation. "
               "Use the hold-out metrics above for model quality.")

# ---------- data quality ----------
with tab_quality:
    st.subheader("Validation report")
    a, b, c = st.columns(3)
    a.metric("Rows in", f"{quality['rows_in']:,}")
    b.metric("Rows kept", f"{quality['rows_out']:,}")
    dropped_total = sum(quality["dropped"].values())
    c.metric("Rows dropped", f"{dropped_total:,}")
    if quality["dropped"]:
        dd = pd.DataFrame(sorted(quality["dropped"].items(), key=lambda kv: -kv[1]),
                          columns=["reason", "rows"])
        fig = px.bar(dd, x="rows", y="reason", orientation="h", title="Drop reasons",
                     color_discrete_sequence=["#8d99ae"])
        fig.update_layout(height=300, yaxis=dict(dtick=1))
        st.plotly_chart(fig, width="stretch")
    else:
        st.success("No rows were dropped.")
    for w in quality["warnings"]:
        st.warning(w)
    st.write("Label balance:", quality["label_balance"])
    st.write("Rows per source:", quality["sources"])
    st.subheader("Cleaning checklist")
    st.markdown(
        "- HTML entities and tags removed, Unicode normalised (NFKC)\n"
        "- URLs stripped, text lower-cased, contractions expanded (negations kept)\n"
        "- Non-alphanumeric noise and repeated punctuation collapsed\n"
        "- Rows dropped when text is null/empty, label missing or outside {0, 1}, "
        "length outside 2–400 words, date invalid or in the future\n"
        "- Exact duplicate texts and texts with conflicting labels removed"
    )

# ---------- live classifier ----------
with tab_live:
    st.subheader("Classify a review")
    text = st.text_area("Review text", "The battery died after a week and support never replied.",
                        height=120)
    if text.strip():
        clean = cleaning.clean_text(text)
        p = float(pipe.predict_proba([clean])[0, 1])
        label = "positive" if p >= 0.5 else "negative"
        st.markdown(f"### :{'green' if label == 'positive' else 'red'}[{label}]  "
                    f"P(positive) = {p:.2f}")
        st.progress(p)
        contrib = model.explain(pipe, clean)
        if not contrib.empty:
            contrib["polarity"] = contrib["contribution"].map(
                lambda c: "positive" if c > 0 else "negative")
            fig = px.bar(contrib.iloc[::-1], x="contribution", y="term", orientation="h",
                         color="polarity", color_discrete_map=COLORS,
                         title="Terms driving this prediction", labels={"term": ""})
            fig.update_layout(height=320, legend_title_text="", yaxis=dict(dtick=1))
            st.plotly_chart(fig, width="stretch")
        st.caption(f"Cleaned input: `{clean}`")
