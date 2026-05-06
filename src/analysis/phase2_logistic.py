"""Logistic-fit emergence analysis for Phase 2 (HYPOTHESIS.md §H2-3, §H2-4, §H2-5).

Implements the tiered logistic-fit handling from §H2-3 (full fit / marginal /
right-censored), the ties-fail permutation-test mechanics from §H2-4, and the
joint H1-C verdict from §H2-5.

Logistic form (brief §3):

    count(step) ≈ L / (1 + exp(-k · (log(step) − μ)))

We fit (L, k, μ) to per-cell head counts via scipy.optimize.curve_fit;
μ (the inflection in log-step space) is the emergence step. When max(count)
across all cells is below the §H2-3 N_cens threshold (2), μ is right-
censored at step143000 — "did not emerge during training" — and treated as
+∞ for ordering-test purposes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

# §H2-3 lock
N_EMERGED = 5
N_CENS = 2
RIGHT_CENSOR_STEP = 143000

# §H2-5 lock
H1C_GATE_P = 0.005
JOINT_NULL_P = (1 / 6) ** 3  # ≈ 0.00463


@dataclass(frozen=True)
class FitResult:
    """One (size, motif) fit result.

    `regime` is one of "emerged" (full fit, max count ≥ N_EMERGED), "marginal"
    (2 ≤ max < N_EMERGED, bootstrap-median μ), or "censored" (max < N_CENS,
    μ = right-censored at step143000).

    `mu` is in raw step units (not log-step). For the ordering test, censored
    μ values are treated as +∞.
    """

    size: str
    motif: str
    regime: str
    mu: float
    L: float
    k: float
    max_count: int


def logistic_count(log_step: np.ndarray, L: float, k: float, mu: float) -> np.ndarray:
    """Forward model: count as a function of log(step).

    `mu` is in *log-step* space here; the FitResult exposes it in raw-step
    units (exp(mu)).
    """
    return L / (1.0 + np.exp(-k * (log_step - mu)))


def fit_logistic_to_counts(
    steps: np.ndarray, counts: np.ndarray
) -> tuple[float, float, float]:
    """Fit (L, k, μ) to (steps, counts) via scipy curve_fit.

    Returns μ in raw step units (not log).
    """
    nonzero_steps = np.maximum(steps.astype(float), 0.5)  # avoid log(0)
    log_steps = np.log(nonzero_steps)
    counts_f = counts.astype(float)
    L0 = max(counts_f.max(), 1.0)
    mu0 = np.log(8000.0)  # a-priori reasonable emergence step
    k0 = 1.0
    p0 = (L0, k0, mu0)
    bounds = ([0.0, 0.01, np.log(0.5)], [3 * L0 + 1, 50.0, np.log(2_000_000.0)])
    try:
        popt, _ = curve_fit(
            logistic_count, log_steps, counts_f, p0=p0, bounds=bounds, maxfev=5000
        )
        L_fit, k_fit, mu_log = popt
    except (RuntimeError, ValueError):
        return float("nan"), float("nan"), float("nan")
    return float(L_fit), float(k_fit), float(np.exp(mu_log))


def tiered_fit(
    size: str,
    motif: str,
    steps: np.ndarray,
    counts: np.ndarray,
    *,
    bootstrap_median_mu: float | None = None,
) -> FitResult:
    """Apply §H2-3 tiered handling and return the FitResult.

    `bootstrap_median_mu`, if supplied, is used as the point estimate in the
    marginal regime (where the direct logistic fit is unstable).
    """
    max_count = int(counts.max())
    if max_count < N_CENS:
        return FitResult(
            size=size, motif=motif, regime="censored",
            mu=float(RIGHT_CENSOR_STEP), L=float("nan"), k=float("nan"),
            max_count=max_count,
        )
    L_fit, k_fit, mu_fit = fit_logistic_to_counts(steps, counts)
    if max_count < N_EMERGED:
        if bootstrap_median_mu is not None:
            return FitResult(
                size=size, motif=motif, regime="marginal",
                mu=float(bootstrap_median_mu), L=L_fit, k=k_fit,
                max_count=max_count,
            )
        return FitResult(
            size=size, motif=motif, regime="marginal",
            mu=mu_fit, L=L_fit, k=k_fit, max_count=max_count,
        )
    return FitResult(
        size=size, motif=motif, regime="emerged",
        mu=mu_fit, L=L_fit, k=k_fit, max_count=max_count,
    )


@dataclass(frozen=True)
class OrderingResult:
    """Per-(size) H1-C ordering verdict.

    `holds_strict` is True iff μ_induction < μ_successor < μ_S_inhibition
    holds with strict inequality (no ties between right-censored values).
    """

    size: str
    mu_induction: float
    mu_successor: float
    mu_s_inhibition: float
    pair_ind_suc_holds: bool
    pair_suc_si_holds: bool
    holds_strict: bool
    pair_ind_suc_undetermined: bool
    pair_suc_si_undetermined: bool


def evaluate_ordering(
    size: str,
    mu_ind: float,
    mu_suc: float,
    mu_si: float,
    *,
    censor_step: float = RIGHT_CENSOR_STEP,
) -> OrderingResult:
    """Evaluate H1-C ordering for one size with §H2-4 ties-fail policy.

    Two μ values both at `censor_step` (right-censored ties) → pair ordering
    is undetermined; that pair fails the strict ordering. Otherwise strict
    inequality is the test.
    """
    is_cens = lambda v: bool(np.isclose(v, censor_step))  # noqa: E731

    ind_suc_undet = is_cens(mu_ind) and is_cens(mu_suc)
    suc_si_undet = is_cens(mu_suc) and is_cens(mu_si)

    pair_ind_suc = (not ind_suc_undet) and (mu_ind < mu_suc)
    pair_suc_si = (not suc_si_undet) and (mu_suc < mu_si)
    holds = pair_ind_suc and pair_suc_si

    return OrderingResult(
        size=size,
        mu_induction=float(mu_ind),
        mu_successor=float(mu_suc),
        mu_s_inhibition=float(mu_si),
        pair_ind_suc_holds=pair_ind_suc,
        pair_suc_si_holds=pair_suc_si,
        holds_strict=holds,
        pair_ind_suc_undetermined=ind_suc_undet,
        pair_suc_si_undetermined=suc_si_undet,
    )


@dataclass(frozen=True)
class JointVerdict:
    """Phase 2 H1-C joint verdict per §H2-5.

    `passes` is True iff `holds_strict` for all 3 sizes. The joint p-value
    under H_0 (exchangeable order in all 3 sizes) is (1/6)³ ≈ 0.00463 if
    all three hold, otherwise effectively 1 (the gate fails).
    """

    sizes: tuple[str, ...]
    per_size: tuple[OrderingResult, ...]
    n_sizes_holding: int
    joint_p_under_h0: float
    passes: bool


def joint_h1c_test(orderings: list[OrderingResult]) -> JointVerdict:
    """Combine per-size orderings into the joint H1-C verdict per §H2-5."""
    n_holding = sum(1 for o in orderings if o.holds_strict)
    if n_holding == len(orderings):
        joint_p = JOINT_NULL_P
    else:
        joint_p = float("nan")
    passes = (joint_p < H1C_GATE_P) if not np.isnan(joint_p) else False
    return JointVerdict(
        sizes=tuple(o.size for o in orderings),
        per_size=tuple(orderings),
        n_sizes_holding=n_holding,
        joint_p_under_h0=joint_p,
        passes=passes,
    )
