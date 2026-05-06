"""Bootstrap CI + threshold-sensitivity post-processing (HYPOTHESIS.md §H2-2).

Per-prompt bootstrap with B = 1000 resamples (95% percentile CI on μ) and
± 25% threshold-sensitivity in 5 increments. Operates on the per-(head,
prompt) score matrices cached by the Phase 2 sweep runners (§H2-6b).

Per-motif aggregation differs because score units differ:
- Induction: per-sequence per-(layer, head) score; aggregate = mean across
  sequences. Resampled unit = sequence index.
- Successor: per-prompt real and null DLA per (layer, head); aggregate =
  mean cross-category lift = mean over categories of mean over prompts in
  category of (real - null). Resampled unit = prompt index.
- S-inhibition: per-prompt (sender, NM) delta; aggregate = mean over prompts
  per (sender, NM), then mean over downstream NMs per sender. Resampled
  unit = prompt index.

Each function returns a 1000-resample array of μ values per (size, motif),
ready for percentile-based 95% CI extraction.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.analysis.phase2_logistic import (
    N_CENS,
    N_EMERGED,
    RIGHT_CENSOR_STEP,
    fit_logistic_to_counts,
)

B_BOOTSTRAP = 1000
CI_LEVEL = 0.95
THRESHOLD_SENSITIVITY_FRACTIONS = (-0.25, -0.125, 0.0, 0.125, 0.25)


@dataclass(frozen=True)
class BootstrapResult:
    size: str
    motif: str
    threshold: float
    mu_point_estimate: float
    mu_bootstrap_mean: float
    mu_bootstrap_median: float
    mu_ci_low: float
    mu_ci_high: float
    n_bootstrap: int


def _mu_from_counts_per_step(
    steps: np.ndarray, counts: np.ndarray
) -> float:
    if counts.max() < N_CENS:
        return float(RIGHT_CENSOR_STEP)
    _, _, mu = fit_logistic_to_counts(steps, counts)
    if not np.isfinite(mu) or mu <= 0:
        return float(RIGHT_CENSOR_STEP)
    return float(mu)


def _percentile_ci(values: np.ndarray, level: float = CI_LEVEL) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - level
    low = float(np.percentile(finite, 100 * alpha / 2))
    high = float(np.percentile(finite, 100 * (1 - alpha / 2)))
    return low, high


def bootstrap_induction(
    per_seq_by_step: dict[int, np.ndarray],
    threshold: float,
    rng: np.random.Generator,
    *,
    B: int = B_BOOTSTRAP,
) -> tuple[np.ndarray, float]:
    """Bootstrap μ from per-sequence prefix-matching scores.

    `per_seq_by_step[step]` has shape (n_sequences, n_layers, n_heads).

    Returns (mu_bootstrap_array of shape (B,), mu_point_estimate).
    """
    steps = np.array(sorted(per_seq_by_step.keys()), dtype=np.int64)
    arrays = [per_seq_by_step[s] for s in steps]
    n_seq = arrays[0].shape[0]
    assert all(a.shape[0] == n_seq for a in arrays), "n_sequences must match across cells"

    point_counts = np.array(
        [(arr.mean(axis=0) > threshold).sum() for arr in arrays], dtype=np.int32
    )
    mu_point = _mu_from_counts_per_step(steps, point_counts)

    mus = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n_seq, size=n_seq)
        counts = np.array(
            [(arr[idx].mean(axis=0) > threshold).sum() for arr in arrays], dtype=np.int32
        )
        mus[b] = _mu_from_counts_per_step(steps, counts)
    return mus, mu_point


def bootstrap_successor(
    per_prompt_by_step: dict[int, dict[str, np.ndarray]],
    threshold: float,
    rng: np.random.Generator,
    *,
    B: int = B_BOOTSTRAP,
) -> tuple[np.ndarray, float]:
    """Bootstrap μ from per-prompt successor real/null DLA arrays.

    `per_prompt_by_step[step]` is a dict with keys 'real', 'null', 'cats'.
    'real' and 'null' shapes: (n_layers, n_heads, n_prompts). 'cats' is a
    (n_prompts,) array of category strings.

    Lift = mean over 4 categories of (mean over prompts in category of (real
    - null)).
    """
    steps = np.array(sorted(per_prompt_by_step.keys()), dtype=np.int64)
    arrays = [per_prompt_by_step[s] for s in steps]

    cats = arrays[0]["cats"]
    n_prompts = cats.size
    assert all(a["real"].shape[-1] == n_prompts for a in arrays)
    unique_cats = sorted(set(cats.tolist()))
    cat_to_idx = {c: np.flatnonzero(cats == c) for c in unique_cats}

    def _lift(real: np.ndarray, null: np.ndarray, idx: np.ndarray) -> np.ndarray:
        # real, null: (n_layers, n_heads, n_prompts)
        per_cat = []
        for c in unique_cats:
            cat_pos_in_idx = np.intersect1d(cat_to_idx[c], idx, assume_unique=False)
            # Actually we want prompts at idx[k] where cats[idx[k]] == c.
            mask = cats[idx] == c
            if not mask.any():
                # category absent from this resample; use point lift
                per_cat.append(np.zeros(real.shape[:2], dtype=np.float64))
                continue
            sel_real = real[..., idx[mask]].mean(axis=-1)
            sel_null = null[..., idx[mask]].mean(axis=-1)
            per_cat.append((sel_real - sel_null).astype(np.float64))
        return np.stack(per_cat, axis=0).mean(axis=0)

    point_counts = []
    for a in arrays:
        lift_full = _lift(a["real"], a["null"], np.arange(n_prompts))
        point_counts.append(int((lift_full >= threshold).sum()))
    point_counts = np.array(point_counts, dtype=np.int32)
    mu_point = _mu_from_counts_per_step(steps, point_counts)

    mus = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n_prompts, size=n_prompts)
        counts = []
        for a in arrays:
            lift = _lift(a["real"], a["null"], idx)
            counts.append(int((lift >= threshold).sum()))
        mus[b] = _mu_from_counts_per_step(steps, np.array(counts, dtype=np.int32))
    return mus, mu_point


def bootstrap_s_inhibition(
    per_prompt_by_step: dict[int, np.ndarray],
    threshold: float,
    nm_layers_by_step: dict[int, list[int]],
    sender_layers: np.ndarray,
    rng: np.random.Generator,
    *,
    B: int = B_BOOTSTRAP,
) -> tuple[np.ndarray, float]:
    """Bootstrap μ from per-prompt S-inhibition (sender, NM, prompt) deltas.

    `per_prompt_by_step[step]` has shape (n_senders, k_nm, n_prompts).
    `nm_layers_by_step[step]` is a list of length k_nm with NM layer indices.
    `sender_layers` is a flat (n_senders,) array of sender layer indices.

    Per-(sender, NM) score = mean over prompts of delta. Per-sender score =
    mean over downstream NMs of per-(sender, NM) score (NMs at layer ≤ sender
    are "not downstream" and excluded). Senders with no downstream NMs get 0.
    """
    steps = np.array(sorted(per_prompt_by_step.keys()), dtype=np.int64)

    def _delta_h_per_sender(
        deltas: np.ndarray, nm_layers: list[int], idx: np.ndarray
    ) -> np.ndarray:
        # deltas: (n_senders, k_nm, n_prompts) → mean over resampled prompts
        per_sender_nm = deltas[..., idx].mean(axis=-1)  # (n_senders, k_nm)
        n_senders = per_sender_nm.shape[0]
        out = np.zeros(n_senders, dtype=np.float64)
        nm_layer_arr = np.asarray(nm_layers)
        for s_idx in range(n_senders):
            ds_mask = nm_layer_arr > sender_layers[s_idx]
            if ds_mask.any():
                out[s_idx] = per_sender_nm[s_idx, ds_mask].mean()
        return out

    n_prompts = next(iter(per_prompt_by_step.values())).shape[-1]
    point_counts = []
    for s in steps:
        d = _delta_h_per_sender(
            per_prompt_by_step[s], nm_layers_by_step[s], np.arange(n_prompts)
        )
        point_counts.append(int((d >= threshold).sum()))
    point_counts = np.array(point_counts, dtype=np.int32)
    mu_point = _mu_from_counts_per_step(steps, point_counts)

    mus = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n_prompts, size=n_prompts)
        counts = []
        for s in steps:
            d = _delta_h_per_sender(per_prompt_by_step[s], nm_layers_by_step[s], idx)
            counts.append(int((d >= threshold).sum()))
        mus[b] = _mu_from_counts_per_step(steps, np.array(counts, dtype=np.int32))
    return mus, mu_point


def threshold_sensitivity_curve(
    bootstrap_fn,
    *args,
    base_threshold: float,
    fractions: tuple[float, ...] = THRESHOLD_SENSITIVITY_FRACTIONS,
    **kwargs,
) -> dict[float, float]:
    """Compute μ point estimates at each ±-threshold variant per §H2-2.

    Returns {threshold_value: mu_point_estimate} for each of the 5 fractions.
    Uses 0 bootstraps internally; only the point estimate matters.
    """
    out: dict[float, float] = {}
    rng = np.random.default_rng(0)
    for frac in fractions:
        thr = base_threshold * (1.0 + frac)
        # Run a single point-estimate (B=0 not supported; use B=1, ignore mus)
        _, mu = bootstrap_fn(*args, threshold=thr, rng=rng, B=1, **kwargs)
        out[thr] = mu
    return out


@dataclass(frozen=True)
class ReversalRateResult:
    """Per-(size, pair) bootstrap reversal-rate result.

    Replaces the hardcoded prior-probability `descriptive_p` column. The
    reversal rate measures the *empirical* fraction of bootstrap replicates
    in which the pair's predicted ordering does NOT hold — i.e., a real
    summary of how robust the per-size point ordering is to per-prompt
    resampling, rather than a prior under exchangeability.

    Censored ties (both μ at the right-censor sentinel) are counted as
    undetermined per §H2-4 ties-fail; undetermined replicates are treated as
    not-holding when computing `reversal_rate`.
    """

    size: str
    motif_early: str
    motif_late: str
    n_holds: int
    n_undetermined: int
    n_total: int
    holds_rate: float
    reversal_rate: float


def bootstrap_pair_reversal_rate(
    mus_early: np.ndarray,
    mus_late: np.ndarray,
    size: str,
    motif_early: str,
    motif_late: str,
    *,
    censor_step: float = RIGHT_CENSOR_STEP,
) -> ReversalRateResult:
    """Empirical reversal rate for one (size, pair) under per-prompt bootstrap.

    For each bootstrap replicate b, the predicted ordering μ_early < μ_late
    holds iff strict inequality holds. Censored ties (both at censor_step)
    are undetermined per §H2-4 and counted as not-holding.

    `reversal_rate = 1 - holds_rate` is the more reviewer-legible direction:
    "fraction of bootstrap replicates where the registered ordering fails."
    """
    assert mus_early.shape == mus_late.shape, "bootstrap arrays must be paired"
    finite_mask = np.isfinite(mus_early) & np.isfinite(mus_late)
    early = mus_early[finite_mask]
    late = mus_late[finite_mask]
    both_cens = np.isclose(early, censor_step) & np.isclose(late, censor_step)
    holds = (early < late) & ~both_cens
    n_total = int(early.size)
    n_holds = int(holds.sum())
    n_undet = int(both_cens.sum())
    holds_rate = float(n_holds) / max(n_total, 1)
    reversal_rate = 1.0 - holds_rate
    return ReversalRateResult(
        size=size,
        motif_early=motif_early,
        motif_late=motif_late,
        n_holds=n_holds,
        n_undetermined=n_undet,
        n_total=n_total,
        holds_rate=holds_rate,
        reversal_rate=reversal_rate,
    )


def summarize_bootstrap(
    size: str, motif: str, threshold: float, mus: np.ndarray, mu_point: float,
) -> BootstrapResult:
    finite = mus[np.isfinite(mus)]
    if finite.size == 0:
        mean_v = float("nan")
        median_v = float("nan")
        low, high = float("nan"), float("nan")
    else:
        mean_v = float(finite.mean())
        median_v = float(np.median(finite))
        low, high = _percentile_ci(finite, level=CI_LEVEL)
    return BootstrapResult(
        size=size, motif=motif, threshold=float(threshold),
        mu_point_estimate=float(mu_point),
        mu_bootstrap_mean=mean_v,
        mu_bootstrap_median=median_v,
        mu_ci_low=low, mu_ci_high=high,
        n_bootstrap=int(mus.size),
    )
