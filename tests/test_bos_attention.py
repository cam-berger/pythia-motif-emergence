"""Tests for the BosAttentionDetector adapter (atlas-v1).

Mirrors ``tests/test_detector_protocol.py`` and
``tests/test_previous_token.py``: static contract only — class attributes,
protocol conformance, and ``score()`` signature. Behavioral tests against
a real model fixture are deferred.
"""

from __future__ import annotations

import inspect

from src.atlas.thresholds import BOS_ATTENTION
from src.detectors.bos_attention import BosAttentionDetector
from src.detectors.protocol import Detector
from src.locked_thresholds import LockedThreshold


# ---- class-attribute contract --------------------------------------------


def test_bos_attention_class_attrs():
    """Adapter declares correct motif, name, and atlas-v1 threshold."""
    assert BosAttentionDetector.motif == "bos_attention"
    assert BosAttentionDetector.name == "bos_attention"
    inst = BosAttentionDetector()
    assert inst.threshold is BOS_ATTENTION
    assert isinstance(inst.threshold, LockedThreshold)


# ---- protocol conformance -------------------------------------------------


def test_bos_attention_satisfies_protocol():
    """Runtime-checkable Protocol: adapter passes isinstance(Detector)."""
    inst = BosAttentionDetector()
    assert isinstance(inst, Detector)


# ---- score() signature ----------------------------------------------------


def test_bos_attention_score_signature():
    """score() takes exactly one positional argument (the model)."""
    sig = inspect.signature(BosAttentionDetector.score)
    params = list(sig.parameters.values())
    assert params[0].name == "self"
    assert params[1].name == "model"
    for p in params[2:]:
        assert p.default is not inspect.Parameter.empty, (
            f"BosAttentionDetector.score has required arg {p.name} besides "
            f"model — detector-specific config should live on __init__"
        )
