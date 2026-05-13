"""Tests for the Detector protocol and adapter class metadata.

These tests assert the static contract: that every adapter declares the
expected motif / name / threshold and structurally satisfies the
``Detector`` protocol. Behavioral tests (golden-head scores against
synthetic transformers) belong in Candidate #5 — they require a real
model fixture and are deferred to that commit.
"""

from __future__ import annotations

import inspect

import pytest

from src.detectors import (
    Detector,
    InductionDetector,
    InductionOvDetector,
    PerHeadScores,
    SInhibitionCompdlaDetector,
    SInhibitionDetector,
    SuccessorArgmaxKDetector,
    SuccessorDetector,
)
from src.locked_thresholds import (
    ALT_K_MIN,
    ALT_TAU_IND_OV,
    ALT_TAU_SI_DLA,
    INDUCTION_QK,
    S_INHIBITION_DELTA,
    SUCCESSOR_LIFT,
    LockedThreshold,
)


# Adapters that need no constructor args (self-contained).
_SELF_CONTAINED = [
    (InductionDetector,        "induction",    "induction_qk",          INDUCTION_QK),
    (InductionOvDetector,      "induction",    "induction_ov",          ALT_TAU_IND_OV),
    (SuccessorArgmaxKDetector, "successor",    "successor_argmax_k_of_7", ALT_K_MIN),
]

# Adapters that take prompts at construction; instantiated with empty/sentinel
# lists for metadata-only tests (no .score() call).
_PROMPT_DRIVEN = [
    (SuccessorDetector,         "successor",    "successor_lift",       SUCCESSOR_LIFT, {"prompts": []}),
    (SInhibitionDetector,       "s_inhibition", "s_inhibition_delta_h", S_INHIBITION_DELTA, {"clean_prompts": [], "corrupt_prompts": []}),
    (SInhibitionCompdlaDetector, "s_inhibition", "s_inhibition_compdla", ALT_TAU_SI_DLA, {"prompts": []}),
]


# ---- class-attribute contract --------------------------------------------


@pytest.mark.parametrize("cls,motif,name,threshold", _SELF_CONTAINED)
def test_self_contained_adapter_class_attrs(cls, motif, name, threshold):
    """Adapter declares correct motif, name, and locked threshold."""
    assert cls.motif == motif
    assert cls.name == name
    inst = cls()
    assert inst.threshold is threshold
    assert isinstance(inst.threshold, LockedThreshold)


@pytest.mark.parametrize("cls,motif,name,threshold,kwargs", _PROMPT_DRIVEN)
def test_prompt_driven_adapter_class_attrs(cls, motif, name, threshold, kwargs):
    """Prompt-driven adapter declares correct motif/name/threshold; needs prompts."""
    assert cls.motif == motif
    assert cls.name == name
    inst = cls(**kwargs)
    assert inst.threshold is threshold
    assert isinstance(inst.threshold, LockedThreshold)


# ---- protocol conformance -------------------------------------------------


@pytest.mark.parametrize("cls", [cls for cls, *_ in _SELF_CONTAINED])
def test_self_contained_adapter_satisfies_protocol(cls):
    """Runtime-checkable Protocol: every adapter passes isinstance(Detector)."""
    inst = cls()
    assert isinstance(inst, Detector)


@pytest.mark.parametrize("cls,_m,_n,_t,kwargs", _PROMPT_DRIVEN)
def test_prompt_driven_adapter_satisfies_protocol(cls, _m, _n, _t, kwargs):
    inst = cls(**kwargs)
    assert isinstance(inst, Detector)


# ---- score() signature ----------------------------------------------------


_ALL_ADAPTERS = [
    InductionDetector,
    InductionOvDetector,
    SuccessorArgmaxKDetector,
    SuccessorDetector,
    SInhibitionDetector,
    SInhibitionCompdlaDetector,
]


@pytest.mark.parametrize("cls", _ALL_ADAPTERS)
def test_score_method_takes_only_model(cls):
    """score() takes exactly one positional argument (the model)."""
    sig = inspect.signature(cls.score)
    params = list(sig.parameters.values())
    assert params[0].name == "self"
    assert params[1].name == "model"
    # No other required positional params: detector-specific config lives on __init__.
    for p in params[2:]:
        assert p.default is not inspect.Parameter.empty, (
            f"{cls.__name__}.score has required arg {p.name} besides model — "
            f"detector-specific config should live on __init__"
        )


# ---- detector names are unique -------------------------------------------


def test_detector_names_are_unique():
    """Names are how the Sweep Driver routes; collisions would corrupt parquet."""
    names = [cls.name for cls in _ALL_ADAPTERS]
    assert len(names) == len(set(names))


# ---- PerHeadScores dataclass shape ---------------------------------------


def test_per_head_scores_is_frozen():
    import torch
    p = PerHeadScores(scores=torch.zeros(2, 3), motif="induction", detector_name="x")
    with pytest.raises(Exception):  # FrozenInstanceError
        p.scores = torch.zeros(2, 3)  # type: ignore[misc]


def test_per_head_scores_aux_defaults_to_none():
    import torch
    p = PerHeadScores(scores=torch.zeros(2, 3), motif="induction", detector_name="x")
    assert p.aux is None
