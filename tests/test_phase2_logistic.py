"""Regression tests for the §H2-3 / §H2-4 / §H2-5 logistic-fit primitives.

Synthetic toy curves with known μ; no dependence on `data/exploration/` parquets.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.phase2_logistic import (
    H1C_GATE_P,
    JOINT_NULL_P,
    N_CENS,
    N_EMERGED,
    RIGHT_CENSOR_STEP,
    evaluate_ordering,
    fit_logistic_to_counts,
    joint_h1c_test,
    logistic_count,
    tiered_fit,
)


# §H2-1 grid (kept here so the tests do not import the runner module).
H2_1_GRID = np.array(
    [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
     1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
     10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
     20000, 24000, 29000, 35000, 41000, 49000, 59000, 70000,
     84000, 100000, 120000, 143000],
    dtype=np.int64,
)


def _synthetic_counts(mu_target: float, *, L: float = 8.0, k: float = 2.0,
                      steps: np.ndarray = H2_1_GRID) -> np.ndarray:
    """Logistic counts on the §H2-1 grid with known μ (raw step units)."""
    log_steps = np.log(np.maximum(steps.astype(float), 0.5))
    raw = logistic_count(log_steps, L=L, k=k, mu=np.log(mu_target))
    return np.round(raw).astype(np.int32)


# ---------------------------------------------------------------------------
# tiered_fit — emerged regime
# ---------------------------------------------------------------------------


def test_tiered_fit_emerged_recovers_mu_within_tolerance():
    mu_true = 8000.0
    counts = _synthetic_counts(mu_true, L=8.0, k=3.0)
    assert counts.max() >= N_EMERGED, "fixture must hit emerged tier"

    result = tiered_fit("70m", "induction", H2_1_GRID, counts)

    assert result.regime == "emerged"
    assert result.max_count == int(counts.max())
    # log-step recovery is tight; raw-step μ is within ±20 % of the planted value.
    assert mu_true * 0.8 <= result.mu <= mu_true * 1.2
    assert np.isfinite(result.L) and np.isfinite(result.k)


def test_tiered_fit_emerged_at_threshold_boundary():
    """max_count == N_EMERGED is the inclusive lower bound of `emerged`."""
    counts = np.zeros_like(H2_1_GRID, dtype=np.int32)
    counts[-5:] = N_EMERGED  # saturates at exactly N_EMERGED

    result = tiered_fit("70m", "induction", H2_1_GRID, counts)
    assert result.regime == "emerged"
    assert result.max_count == N_EMERGED


# ---------------------------------------------------------------------------
# tiered_fit — marginal regime
# ---------------------------------------------------------------------------


def test_tiered_fit_marginal_uses_bootstrap_median_when_supplied():
    counts = np.zeros_like(H2_1_GRID, dtype=np.int32)
    counts[-3:] = 3  # 2 ≤ max < N_EMERGED → marginal
    bootstrap_median = 50_000.0

    result = tiered_fit(
        "70m", "successor", H2_1_GRID, counts,
        bootstrap_median_mu=bootstrap_median,
    )

    assert result.regime == "marginal"
    assert result.mu == bootstrap_median
    assert result.max_count == 3


def test_tiered_fit_marginal_falls_back_to_fit_when_no_bootstrap():
    counts = np.zeros_like(H2_1_GRID, dtype=np.int32)
    counts[-4:] = 3
    result = tiered_fit("70m", "successor", H2_1_GRID, counts)
    assert result.regime == "marginal"
    # No bootstrap supplied → either a finite fit μ or NaN if curve_fit failed.
    assert np.isnan(result.mu) or result.mu > 0


# ---------------------------------------------------------------------------
# tiered_fit — censored regime
# ---------------------------------------------------------------------------


def test_tiered_fit_censored_when_max_below_n_cens():
    """max < N_CENS (i.e. max ∈ {0, 1}) → right-censor at step143000."""
    for max_c in (0, 1):
        counts = np.zeros_like(H2_1_GRID, dtype=np.int32)
        if max_c == 1:
            counts[-1] = 1
        result = tiered_fit("70m", "s_inhibition", H2_1_GRID, counts)
        assert result.regime == "censored", f"max={max_c} should censor"
        assert result.mu == float(RIGHT_CENSOR_STEP)
        assert np.isnan(result.L) and np.isnan(result.k)
        assert result.max_count == max_c


def test_tiered_fit_n_cens_boundary_is_marginal_not_censored():
    """max == N_CENS (== 2) is *marginal*, not censored — boundary check."""
    counts = np.zeros_like(H2_1_GRID, dtype=np.int32)
    counts[-2:] = N_CENS
    result = tiered_fit("70m", "successor", H2_1_GRID, counts)
    assert result.regime == "marginal"
    assert result.mu != float(RIGHT_CENSOR_STEP)


# ---------------------------------------------------------------------------
# evaluate_ordering — ties-fail policy (§H2-4)
# ---------------------------------------------------------------------------


def test_evaluate_ordering_strict_ascending_holds():
    o = evaluate_ordering("160m", mu_ind=2_000.0, mu_suc=10_000.0, mu_si=80_000.0)
    assert o.holds_strict
    assert o.pair_ind_suc_holds and o.pair_suc_si_holds
    assert not o.pair_ind_suc_undetermined
    assert not o.pair_suc_si_undetermined


def test_evaluate_ordering_both_censored_pair_is_undetermined():
    """Two μ both right-censored → ties-fail per §H2-4."""
    o = evaluate_ordering(
        "70m",
        mu_ind=2_000.0,
        mu_suc=float(RIGHT_CENSOR_STEP),
        mu_si=float(RIGHT_CENSOR_STEP),
    )
    assert o.pair_suc_si_undetermined
    assert not o.pair_suc_si_holds  # undetermined → does NOT hold
    assert not o.holds_strict


def test_evaluate_ordering_one_censored_resolves_via_strict_inequality():
    """Only one μ at the sentinel — pair is determined; strict inequality applies."""
    o = evaluate_ordering(
        "410m",
        mu_ind=5_000.0,
        mu_suc=20_000.0,
        mu_si=float(RIGHT_CENSOR_STEP),
    )
    assert not o.pair_suc_si_undetermined
    assert o.pair_suc_si_holds  # 20_000 < RIGHT_CENSOR_STEP
    assert o.holds_strict


def test_evaluate_ordering_equal_non_censored_fails_strict():
    """Equal but not at the sentinel → not undetermined, fails strict (<)."""
    o = evaluate_ordering("70m", mu_ind=5_000.0, mu_suc=5_000.0, mu_si=10_000.0)
    assert not o.pair_ind_suc_undetermined
    assert not o.pair_ind_suc_holds
    assert not o.holds_strict


# ---------------------------------------------------------------------------
# joint_h1c_test — gate composition (§H2-5)
# ---------------------------------------------------------------------------


def test_joint_h1c_passes_when_all_three_sizes_hold():
    orderings = [
        evaluate_ordering(s, mu_ind=2_000.0, mu_suc=10_000.0, mu_si=80_000.0)
        for s in ("70m", "160m", "410m")
    ]
    verdict = joint_h1c_test(orderings)
    assert verdict.n_sizes_holding == 3
    assert verdict.passes
    assert verdict.joint_p_under_h0 == pytest.approx(JOINT_NULL_P)
    assert verdict.joint_p_under_h0 < H1C_GATE_P


def test_joint_h1c_fails_when_any_size_misses():
    orderings = [
        evaluate_ordering("70m", 2_000.0, 10_000.0, 80_000.0),       # holds
        evaluate_ordering("160m", 10_000.0, 2_000.0, 80_000.0),      # ind > suc, fails
        evaluate_ordering("410m", 2_000.0, 10_000.0, 80_000.0),      # holds
    ]
    verdict = joint_h1c_test(orderings)
    assert verdict.n_sizes_holding == 2
    assert not verdict.passes
    assert np.isnan(verdict.joint_p_under_h0)


# ---------------------------------------------------------------------------
# fit_logistic_to_counts — graceful failure
# ---------------------------------------------------------------------------


def test_fit_logistic_returns_nans_on_unfittable_input():
    """All-zero counts cannot constrain (L, k, μ); the fitter must not raise."""
    counts = np.zeros_like(H2_1_GRID, dtype=np.int32)
    L, k, mu = fit_logistic_to_counts(H2_1_GRID, counts)
    # Either NaNs (fit raised internally) or a degenerate fit at the boundary;
    # the contract is "no exception, returns a finite-or-NaN tuple".
    for v in (L, k, mu):
        assert isinstance(v, float)
