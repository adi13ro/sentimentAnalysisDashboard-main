import pandas as pd

from sentiment.cleaning import clean_frame, clean_text


def test_lowercases_and_strips_html_and_urls():
    raw = "<b>GREAT</b> product!!! see https://example.com &amp; enjoy"
    assert clean_text(raw) == "great product! see enjoy"


def test_preserves_negation_when_expanding_contractions():
    assert clean_text("I don't like it") == "i do not like it"
    assert clean_text("Won't buy again") == "will not buy again"


def test_non_string_input_becomes_empty():
    assert clean_text(None) == ""
    assert clean_text(12.5) == ""


def test_clean_frame_adds_length_features():
    df = pd.DataFrame({"text": ["Nice one", None]})
    out = clean_frame(df)
    assert list(out["text_clean"]) == ["nice one", ""]
    assert list(out["n_words"]) == [2, 0]
    assert list(out["n_chars"]) == [8, 0]
