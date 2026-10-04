"""
tests/test_limitations.py

The standing statements about what a report does NOT say.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vechealth.interpretation.limitations import (
    LIMITATIONS,
    SHORT_READING_NOTE,
    limitations_text,
)


def test_limitations_state_the_negative_prediction_result():
    text = " ".join(LIMITATIONS)
    assert "did not predict retrieval quality" in text
    assert "p = 0.0511" in text
    assert "no predictive claim is made" in text


def test_limitations_state_the_scope_and_the_calibration_caveats():
    text = " ".join(LIMITATIONS)
    assert "one corpus" in text and "Qwen3-Embedding-0.6B" in text
    assert "only five were specific" in text
    assert "recalibrating" in text and "(0.05)" in text and "(1.2)" in text
    assert "calibrated and evaluated on the same benchmark" in text
    assert "not implemented in this package" in text


def test_every_limitation_is_a_nonempty_sentence():
    assert len(LIMITATIONS) >= 8
    for item in LIMITATIONS:
        assert item.strip().endswith(".")
        assert len(item) > 60


def test_text_rendering_numbers_the_items():
    rendered = limitations_text()
    assert rendered.splitlines()[0].startswith("1. ")
    assert f"{len(LIMITATIONS)}. " in rendered


def test_short_note_is_what_the_cli_prints():
    assert "did not predict retrieval quality" in SHORT_READING_NOTE
    assert "not validated as good or bad" in SHORT_READING_NOTE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
