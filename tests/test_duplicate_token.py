"""Unit tests for the duplicate-token attention-head detector.

Behavioral tests against a real model are deferred to the atlas-v1 smoke
notebook; these tests pin the static contract (protocol conformance,
metadata, signature, validation).
"""

from __future__ import annotations

import inspect

import pytest

from src.atlas.thresholds import DUPLICATE_TOKEN
from src.detectors.duplicate_token import DuplicateTokenDetector
from src.detectors.protocol import Detector
from src.locked_thresholds import LockedThreshold


# ---- class-attribute contract --------------------------------------------


def test_class_attrs():
    """Detector declares correct motif, name, and atlas-v1 threshold."""
    assert DuplicateTokenDetector.motif == "duplicate_token"
    assert DuplicateTokenDetector.name == "duplicate_token_attention"
    inst = DuplicateTokenDetector()
    assert inst.threshold is DUPLICATE_TOKEN
    assert isinstance(inst.threshold, LockedThreshold)


# ---- protocol conformance -------------------------------------------------


def test_satisfies_protocol():
    """Runtime-checkable Protocol: instance passes isinstance(Detector)."""
    inst = DuplicateTokenDetector()
    assert isinstance(inst, Detector)


# ---- score() signature ----------------------------------------------------


def test_score_method_takes_only_model():
    """score() takes exactly one positional argument (the model)."""
    sig = inspect.signature(DuplicateTokenDetector.score)
    params = list(sig.parameters.values())
    assert params[0].name == "self"
    assert params[1].name == "model"
    for p in params[2:]:
        assert p.default is not inspect.Parameter.empty, (
            f"DuplicateTokenDetector.score has required arg {p.name} besides model — "
            f"detector-specific config should live on __init__"
        )


# ---- seq_len validation ---------------------------------------------------


def test_seq_len_must_be_even():
    """Odd seq_len raises ValueError citing duplicate-by-construction."""
    from src.detectors.duplicate_token import duplicate_token_score

    # No real model needed: validation runs before any model access.
    with pytest.raises(ValueError, match="duplicate by construction"):
        duplicate_token_score(model=None, seq_len=99)


def test_detector_seq_len_validation_propagates():
    """Detector with odd seq_len fails at score() with ValueError."""
    inst = DuplicateTokenDetector(seq_len=101)
    with pytest.raises(ValueError, match="duplicate by construction"):
        inst.score(model=None)
