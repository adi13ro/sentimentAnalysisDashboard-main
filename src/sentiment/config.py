from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

UCI_URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
RAW_CSV = DATA_RAW / "reviews_raw.csv"
CLEAN_CSV = DATA_PROCESSED / "reviews_clean.csv"
SCORED_CSV = DATA_PROCESSED / "reviews_scored.csv"
QUALITY_JSON = DATA_PROCESSED / "quality_report.json"
METRICS_JSON = MODELS / "metrics.json"
MODEL_PATH = MODELS / "sentiment_model.joblib"

RANDOM_SEED = 42
