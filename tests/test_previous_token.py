"""Tests for the PreviousTokenDetector adapter (atlas-v1).

Mirrors ``tests/test_detector_protocol.py``: static contract only — class
attributes, protocol conformance, and ``score()`` signature. Behavioral
tests against a real model fixture are deferred.
"""

from __future__ import annotations

import inspect

from src.atlas.thresholds import PREVIOUS_TOKEN
from src.detectors.previous_token import PreviousTokenDetector
from src.detectors.protocol import Detector
from src.locked_thresholds import LockedThreshold


# ---- class-attribute contract --------------------------------------------


def test_previous_token_class_attrs():
    """Adapter declares correct motif, name, and atlas-v1 threshold."""
    assert PreviousTokenDetector.motif == "previous_token"
    assert PreviousTokenDetector.name == "previous_token_attention"
    inst = PreviousTokenDetector()
    assert inst.threshold is PREVIOUS_TOKEN
    assert isinstance(inst.threshold, LockedThreshold)


# ---- protocol conformance -------------------------------------------------


def test_previous_token_satisfies_protocol():
    """Runtime-checkable Protocol: adapter passes isinstance(Detector)."""
    inst = PreviousTokenDetector()
    assert isinstance(inst, Detector)


# ---- score() signature ----------------------------------------------------


def test_previous_token_score_signature():
    """score() takes exactly one positional argument (the model)."""
    sig = inspect.signature(PreviousTokenDetector.score)
    params = list(sig.parameters.values())
    assert params[0].name == "self"
    assert params[1].name == "model"
    for p in params[2:]:
        assert p.default is not inspect.Parameter.empty, (
            f"PreviousTokenDetector.score has required arg {p.name} besides "
            f"model — detector-specific config should live on __init__"
        )
