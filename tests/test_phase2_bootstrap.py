"""Regression tests for §H2-9-R bootstrap reversal-rate primitive.

Synthetic paired bootstrap arrays with known holds_rate; no real-parquet
dependencies. Targets the per-pair reversal-rate function and its censored-tie
edge case (the bug that made `descriptive_p` constants by construction; the
§H2-9-R reframe).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.phase2_bootstrap import (
    ReversalRateResult,
    bootstrap_pair_reversal_rate,
)
from src.analysis.phase2_logistic import RIGHT_CENSOR_STEP


def test_reversal_rate_zero_when_ordering_always_holds():
    """Every replicate has μ_early < μ_late → holds_rate = 1, reversal = 0."""
    rng = np.random.default_rng(0)
    early = rng.uniform(1_000, 5_000, size=1_000)
    late = rng.uniform(50_000, 60_000, size=1_000)

    r = bootstrap_pair_reversal_rate(
        early, late, size="70m",
        motif_early="induction", motif_late="successor",
    )

    assert isinstance(r, ReversalRateResult)
    assert r.n_holds == 1_000
    assert r.n_total == 1_000
    assert r.n_undetermined == 0
    assert r.holds_rate == 1.0
    assert r.reversal_rate == 0.0


def test_reversal_rate_one_when_ordering_always_reverses():
    """Every replicate has μ_early > μ_late → holds_rate = 0, reversal = 1."""
    rng = np.random.default_rng(1)
    early = rng.uniform(50_000, 60_000, size=500)
    late = rng.uniform(1_000, 5_000, size=500)

    r = bootstrap_pair_reversal_rate(
        early, late, "160m", "induction", "s_inhibition",
    )
    assert r.n_holds == 0
    assert r.holds_rate == 0.0
    assert r.reversal_rate == 1.0


def test_reversal_rate_recovers_known_mixed_fraction():
    """Construct a deterministic 70/30 mix and verify the recovered rate."""
    n = 1_000
    early = np.empty(n, dtype=np.float64)
    late = np.empty(n, dtype=np.float64)
    # First 700 hold; last 300 reverse.
    early[:700] = 5_000.0
    late[:700] = 50_000.0
    early[700:] = 50_000.0
    late[700:] = 5_000.0

    r = bootstrap_pair_reversal_rate(early, late, "410m", "induction", "successor")
    assert r.n_holds == 700
    assert r.holds_rate == pytest.approx(0.700)
    assert r.reversal_rate == pytest.approx(0.300)


def test_reversal_rate_censored_ties_count_as_undetermined_not_holding():
    """§H2-4 ties-fail: both μ at RIGHT_CENSOR_STEP → undetermined, not-held."""
    n_total = 100
    early = np.full(n_total, float(RIGHT_CENSOR_STEP))
    late = np.full(n_total, float(RIGHT_CENSOR_STEP))

    r = bootstrap_pair_reversal_rate(early, late, "70m", "successor", "s_inhibition")

    assert r.n_undetermined == n_total
    assert r.n_holds == 0
    assert r.holds_rate == 0.0
    assert r.reversal_rate == 1.0


def test_reversal_rate_partial_censoring_distinguishes_undetermined_from_real_reversal():
    """Mixed census: 40 censored ties (undetermined) + 60 real holds.

    The contract: undetermined replicates contribute to n_undetermined and to
    `not held` (so reversal_rate counts them), but they are categorically
    distinguishable from a genuine reversal.
    """
    n_total = 100
    early = np.empty(n_total)
    late = np.empty(n_total)
    # 40 censored ties.
    early[:40] = float(RIGHT_CENSOR_STEP)
    late[:40] = float(RIGHT_CENSOR_STEP)
    # 60 genuine holds.
    early[40:] = 5_000.0
    late[40:] = 50_000.0

    r = bootstrap_pair_reversal_rate(early, late, "70m", "induction", "successor")

    assert r.n_total == 100
    assert r.n_undetermined == 40
    assert r.n_holds == 60
    assert r.holds_rate == pytest.approx(0.60)
    # reversal_rate folds undetermined in, but n_undetermined is queryable.
    assert r.reversal_rate == pytest.approx(0.40)


def test_reversal_rate_skips_non_finite_replicates():
    """NaNs in one or both arrays are dropped before computing the rate."""
    early = np.array([5_000.0, np.nan, 5_000.0, 5_000.0], dtype=np.float64)
    late = np.array([50_000.0, 50_000.0, np.nan, 50_000.0], dtype=np.float64)

    r = bootstrap_pair_reversal_rate(early, late, "70m", "induction", "successor")

    # Only 2 paired-finite replicates remain (indices 0 and 3); both hold.
    assert r.n_total == 2
    assert r.n_holds == 2
    assert r.holds_rate == 1.0
    assert r.reversal_rate == 0.0


def test_reversal_rate_requires_paired_arrays():
    """Mismatched shapes are a programmer error; should raise (assert)."""
    early = np.zeros(10)
    late = np.zeros(11)
    with pytest.raises(AssertionError):
        bootstrap_pair_reversal_rate(early, late, "70m", "a", "b")
