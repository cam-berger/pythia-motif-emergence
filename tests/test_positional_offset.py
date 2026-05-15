"""Tests for the PositionalOffsetDetector adapter (atlas-v1).

Mirrors ``tests/test_previous_token.py``: static contract only — class
attributes, protocol conformance, ``score()`` signature, and the
``K_SET`` invariants (k=-1 is deliberately excluded; the family has five
nonzero offsets). Behavioral tests against a real model are deferred to
the atlas-v1 smoke notebook.
"""

from __future__ import annotations

import inspect

from src.atlas.thresholds import POSITIONAL_OFFSET
from src.detectors.positional_offset import K_SET, PositionalOffsetDetector
from src.detectors.protocol import Detector
from src.locked_thresholds import LockedThreshold


# ---- class-attribute contract --------------------------------------------


def test_positional_offset_class_attrs():
    """Adapter declares correct motif, name, and atlas-v1 threshold."""
    assert PositionalOffsetDetector.motif == "positional_offset"
    assert PositionalOffsetDetector.name == "positional_offset_attention"
    inst = PositionalOffsetDetector()
    assert inst.threshold is POSITIONAL_OFFSET
    assert isinstance(inst.threshold, LockedThreshold)


# ---- protocol conformance -------------------------------------------------


def test_positional_offset_satisfies_protocol():
    """Runtime-checkable Protocol: adapter passes isinstance(Detector)."""
    inst = PositionalOffsetDetector()
    assert isinstance(inst, Detector)


# ---- score() signature ----------------------------------------------------


def test_positional_offset_score_signature():
    """score() takes exactly one positional argument (the model)."""
    sig = inspect.signature(PositionalOffsetDetector.score)
    params = list(sig.parameters.values())
    assert params[0].name == "self"
    assert params[1].name == "model"
    for p in params[2:]:
        assert p.default is not inspect.Parameter.empty, (
            f"PositionalOffsetDetector.score has required arg {p.name} besides "
            f"model — detector-specific config should live on __init__"
        )


# ---- K_SET invariants -----------------------------------------------------


def test_k_set_excludes_minus_one():
    """k=-1 is the previous-token detector's territory; must be excluded."""
    assert -1 not in K_SET


def test_k_set_length():
    """The positional-offset family has exactly five nonzero offsets."""
    assert len(K_SET) == 5
