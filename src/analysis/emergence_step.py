"""Single seam for emergence-step extraction across all proxies.

Four proxies coexist in the project. Before this module, each was reimplemented
inline in 3+ runner scripts and one notebook cell with subtle edge-case drift
(``None`` vs ``inf`` vs ``step_max`` returns; ``>`` vs ``≥`` at the milestone).
The proxies are now unified here:

- ``'logistic_mu'`` — μ from the logistic fit (PROJECT_BRIEF.md §3, §H2-3).
  Used by the §H1-C 3-size joint sign-test at p ≈ 0.00463.
- ``'half_max'`` — first step where pass-count ≥ ⌈0.50 × max⌉ over the
  trajectory's max. Used by §H1-C-2.8b-extension (within-motif normalization,
  locked 2026-05-12).
- ``'half_final'`` — first step where pass-count ≥ ⌈0.50 × final-step count⌉.
  Used by §H1-C-altdetectors-2-rr-4.
- ``'first_geq_k'`` — first step where pass-count ≥ k (k passed explicitly).

The return is ``float | None``. ``None`` means the milestone is never reached
(or the logistic fit failed). Callers that need ``inf`` sentinel semantics
for ordering tests should do the conversion at the call site:

    step = emergence_step(traj, 'half_max')
    step_for_ordering = step if step is not None else math.inf
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

from src.analysis.phase2_logistic import fit_logistic_to_counts

Proxy = Literal["logistic_mu", "half_max", "half_final", "first_geq_k"]


def emergence_step(
    trajectory: pd.Series,
    proxy: Proxy,
    *,
    k: int | None = None,
) -> float | None:
    """Step at which the trajectory reaches the proxy milestone, or ``None``.

    Args:
        trajectory: pass-count series indexed by training step. Must be sorted
            by step ascending. Index values are training steps; series values
            are non-negative integer pass-counts.
        proxy: one of ``'logistic_mu'``, ``'half_max'``, ``'half_final'``,
            ``'first_geq_k'``.
        k: target pass-count for ``proxy='first_geq_k'``. Required iff that
            proxy is selected; ignored otherwise.

    Returns:
        Step value as ``float`` (raw step units, not log-step), or ``None``
        if the milestone is never reached. For ``'logistic_mu'``, ``None``
        is also returned when the scipy fit fails.
    """
    if proxy == "first_geq_k":
        if k is None:
            raise ValueError("'first_geq_k' proxy requires the k= keyword")
        return _first_step_geq(trajectory, k)

    if proxy == "half_max":
        max_count = int(trajectory.max())
        if max_count <= 0:
            return None
        target = max(1, math.ceil(0.50 * max_count))
        return _first_step_geq(trajectory, target)

    if proxy == "half_final":
        final_count = int(trajectory.iloc[-1])
        if final_count <= 0:
            return None
        target = max(1, math.ceil(0.50 * final_count))
        return _first_step_geq(trajectory, target)

    if proxy == "logistic_mu":
        steps = np.asarray(trajectory.index, dtype=float)
        counts = np.asarray(trajectory.values, dtype=float)
        _, _, mu = fit_logistic_to_counts(steps, counts)
        if math.isnan(mu):
            return None
        return float(mu)

    raise ValueError(f"unknown proxy: {proxy!r}")


def _first_step_geq(trajectory: pd.Series, target: int) -> float | None:
    """Smallest index value where the series reaches ``target``, or ``None``."""
    hits = trajectory[trajectory >= target]
    if len(hits) == 0:
        return None
    return float(hits.index[0])
