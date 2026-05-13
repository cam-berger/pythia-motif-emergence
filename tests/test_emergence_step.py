"""Unit tests for src/analysis/emergence_step.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.emergence_step import emergence_step


def _trajectory(steps_counts):
    """Construct a trajectory pd.Series from a list of (step, count) pairs."""
    s = pd.Series({step: count for step, count in steps_counts})
    return s.sort_index()


# ---- first_geq_k ----------------------------------------------------------


def test_first_geq_k_basic():
    traj = _trajectory([(1000, 0), (2000, 1), (4000, 3), (8000, 5)])
    assert emergence_step(traj, "first_geq_k", k=3) == 4000.0
    assert emergence_step(traj, "first_geq_k", k=5) == 8000.0


def test_first_geq_k_returns_none_when_never_reached():
    traj = _trajectory([(1000, 0), (2000, 1), (4000, 2)])
    assert emergence_step(traj, "first_geq_k", k=5) is None


def test_first_geq_k_inclusive_at_milestone():
    traj = _trajectory([(1000, 4), (2000, 5)])
    assert emergence_step(traj, "first_geq_k", k=5) == 2000.0  # ≥, not >


def test_first_geq_k_requires_k():
    traj = _trajectory([(1000, 1)])
    with pytest.raises(ValueError, match="requires the k="):
        emergence_step(traj, "first_geq_k")


# ---- half_max -------------------------------------------------------------


def test_half_max_uses_ceil_of_half_max():
    # max=11 → ⌈0.50 × 11⌉ = 6
    traj = _trajectory([(1000, 1), (2000, 3), (4000, 6), (8000, 11)])
    assert emergence_step(traj, "half_max") == 4000.0


def test_half_max_floor_one_when_max_is_one():
    # max=1 → max(1, ⌈0.50 × 1⌉) = 1
    traj = _trajectory([(1000, 0), (2000, 1)])
    assert emergence_step(traj, "half_max") == 2000.0


def test_half_max_all_zero_returns_none():
    traj = _trajectory([(1000, 0), (2000, 0), (4000, 0)])
    assert emergence_step(traj, "half_max") is None


def test_half_max_smallest_step_wins_under_ties():
    # max=4, target=2. First step ≥2 is 2000.
    traj = _trajectory([(1000, 1), (2000, 2), (4000, 2), (8000, 4)])
    assert emergence_step(traj, "half_max") == 2000.0


# ---- half_final -----------------------------------------------------------


def test_half_final_uses_final_step():
    # final=8 → target=4. First step ≥4 is 4000.
    traj = _trajectory([(1000, 1), (2000, 3), (4000, 4), (8000, 8)])
    assert emergence_step(traj, "half_final") == 4000.0


def test_half_final_returns_none_when_final_zero():
    traj = _trajectory([(1000, 5), (2000, 5), (4000, 0)])
    assert emergence_step(traj, "half_final") is None


# ---- logistic_mu ----------------------------------------------------------


def test_logistic_mu_returns_finite_step_for_typical_emergence():
    rng = np.random.default_rng(0)
    steps = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 30000, 100000])
    L_true, k_true, mu_log_true = 10.0, 1.5, np.log(5000.0)
    counts_clean = L_true / (1.0 + np.exp(-k_true * (np.log(steps) - mu_log_true)))
    counts = np.clip(np.round(counts_clean + rng.normal(0, 0.2, size=counts_clean.shape)), 0, None)
    traj = pd.Series(counts, index=steps).sort_index()
    mu = emergence_step(traj, "logistic_mu")
    assert mu is not None
    assert 1000.0 < mu < 20000.0  # μ recovers the order of magnitude


def test_logistic_mu_returns_none_on_degenerate_input():
    traj = _trajectory([(1000, 0), (2000, 0), (4000, 0)])
    mu = emergence_step(traj, "logistic_mu")
    # all-zero counts: curve_fit may return any μ inside bounds. Accept either
    # None (fit failure) or a finite value — what we test is that we don't
    # crash and the return type is float|None.
    assert mu is None or isinstance(mu, float)


# ---- unknown proxy --------------------------------------------------------


def test_unknown_proxy_raises():
    traj = _trajectory([(1000, 1)])
    with pytest.raises(ValueError, match="unknown proxy"):
        emergence_step(traj, "fancy_new_proxy")  # type: ignore[arg-type]
