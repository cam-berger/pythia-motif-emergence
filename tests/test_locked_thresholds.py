"""Audit test for src/locked_thresholds.

EXPECTED is the hand-maintained source of truth. If a value drifts in
``src/locked_thresholds.py`` without a matching edit to ``EXPECTED`` below,
this test fails — that failure is the deliberate moment of pause:
"am I allowed to change this pre-registered value?"

EXPECTED reflects HYPOTHESIS.md as of the most recent amendment date that
introduced each threshold. Update EXPECTED *only* alongside a corresponding
HYPOTHESIS.md amendment (and reference the amendment ID in the commit
message).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.locked_thresholds import (
    ALL_LOCKED,
    LockedThreshold,
    SUCCESSOR_LIFT,
)


EXPECTED: dict[str, tuple[float, str, str]] = {
    # name                  value      comparator   motif
    "INDUCTION_QK":       (0.30,       "gt",        "induction"),
    "SUCCESSOR_LIFT":     (0.13496,    "ge",        "successor"),
    "S_INHIBITION_DELTA": (0.0372,     "ge",        "s_inhibition"),
    "NULL_LOW":           (0.8,        "ge",        None),
    "NULL_HIGH":          (1.2,        "ge",        None),
    "DEP_THRESHOLD":      (0.5,        "ge",        None),
    "GENERIC_THRESHOLD":  (0.7,        "ge",        None),
    "HARD_K_MIN":         (4,          "ge",        "induction"),
    "N_MIN_SUPERSEDE":    (5,          "ge",        None),
    "ALT_TAU_IND_OV":     (13.592629,  "ge",        "induction"),
    "ALT_K_MIN":          (2,          "ge",        "successor"),
    "ALT_TAU_SI_DLA":     (0.247095,   "ge",        "s_inhibition"),
}


def test_registry_matches_expected_values():
    """Every registered threshold must match the hand-maintained EXPECTED table."""
    for name, (value, comparator, motif) in EXPECTED.items():
        assert name in ALL_LOCKED, f"missing from registry: {name}"
        t = ALL_LOCKED[name]
        assert t.value == value, f"{name}.value drift: {t.value} != {value}"
        assert t.comparator == comparator, f"{name}.comparator drift: {t.comparator} != {comparator}"
        assert t.motif == motif, f"{name}.motif drift: {t.motif} != {motif}"


def test_registry_has_no_unexpected_entries():
    """Every registered threshold must appear in EXPECTED (catch silent additions)."""
    extras = set(ALL_LOCKED) - set(EXPECTED)
    assert not extras, (
        f"registry has entries not in EXPECTED: {sorted(extras)}. "
        f"Add them to EXPECTED (and HYPOTHESIS.md) before merging."
    )


def test_all_records_have_amendment_and_description():
    """Audit-trail invariants: every threshold carries provenance."""
    for name, t in ALL_LOCKED.items():
        assert isinstance(t, LockedThreshold), f"{name} is not a LockedThreshold"
        assert t.amendment.strip(), f"{name} has empty amendment"
        assert t.lock_date.strip(), f"{name} has empty lock_date"
        assert t.description.strip(), f"{name} has empty description"


def test_passes_scalar_gt():
    """Olsson gate uses strict `>`."""
    t = ALL_LOCKED["INDUCTION_QK"]
    assert t.passes(0.31) is True
    assert t.passes(0.30) is False  # strict >
    assert t.passes(0.29) is False


def test_passes_scalar_ge():
    """Successor gate uses `≥`."""
    t = SUCCESSOR_LIFT
    assert t.passes(0.13496) is True   # inclusive
    assert t.passes(0.14) is True
    assert t.passes(0.13) is False


def test_passes_vectorizes_numpy():
    t = SUCCESSOR_LIFT
    scores = np.array([0.10, 0.13496, 0.20])
    np.testing.assert_array_equal(t.passes(scores), np.array([False, True, True]))


def test_passes_vectorizes_pandas():
    t = SUCCESSOR_LIFT
    s = pd.Series([0.10, 0.13496, 0.20])
    pd.testing.assert_series_equal(t.passes(s), pd.Series([False, True, True]))


def test_locked_thresholds_are_frozen():
    """Mutation guard: dataclass(frozen=True) is part of the contract."""
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        SUCCESSOR_LIFT.value = 999.0  # type: ignore[misc]
