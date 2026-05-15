"""Static-contract tests for the delimiter atlas detector."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from src.atlas.thresholds import DELIMITER
from src.detectors import Detector
from src.detectors.delimiter import (
    DEFAULT_CORPUS_PATH,
    DELIMITER_CHARS,
    DelimiterDetector,
)


def test_class_attrs():
    assert DelimiterDetector.motif == "delimiter"
    assert DelimiterDetector.name == "delimiter_attention"
    inst = DelimiterDetector()
    assert inst.threshold is DELIMITER


def test_satisfies_protocol():
    inst = DelimiterDetector()
    assert isinstance(inst, Detector)


def test_score_signature():
    sig = inspect.signature(DelimiterDetector.score)
    params = list(sig.parameters.values())
    assert params[0].name == "self"
    assert params[1].name == "model"
    for p in params[2:]:
        assert p.default is not inspect.Parameter.empty


def test_delimiter_chars_contains_canonical_set():
    assert "," in DELIMITER_CHARS
    assert "." in DELIMITER_CHARS
    assert "\n" in DELIMITER_CHARS


def test_corpus_file_exists_and_nonempty():
    assert DEFAULT_CORPUS_PATH.exists(), f"corpus missing at {DEFAULT_CORPUS_PATH}"
    corpus = json.loads(DEFAULT_CORPUS_PATH.read_text())
    assert isinstance(corpus, list)
    assert len(corpus) > 0
    assert all(isinstance(s, str) for s in corpus)
    assert all(len(s) > 0 for s in corpus)


def test_threshold_value_matches_atlas_calibrated():
    """Calibrated post-smoke at 410M step143000: tightened to 0.40."""
    inst = DelimiterDetector()
    assert inst.threshold.value == pytest.approx(0.40)
    assert inst.threshold.motif == "delimiter"
