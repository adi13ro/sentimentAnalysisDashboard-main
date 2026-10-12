"""Smoke test: the Streamlit app must render without exceptions once artefacts exist."""

import pytest
from streamlit.testing.v1 import AppTest

from sentiment import config

pytestmark = pytest.mark.skipif(
    not config.SCORED_CSV.exists() or not config.MODEL_PATH.exists(),
    reason="run `sentiment-pipeline` first to produce dashboard artefacts",
)


def test_dashboard_renders():
    at = AppTest.from_file(str(config.ROOT / "app" / "dashboard.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "Customer Review Sentiment"
    assert len(at.metric) >= 5
    assert len(at.tabs) == 6
