"""Locked numeric pass/fail thresholds — programmatic mirror of HYPOTHESIS.md.

Every value in this module is a number that (i) is named in a HYPOTHESIS.md
amendment and (ii) controls a downstream pass/fail decision (detector gate,
classification gate, sample-size minimum, etc.). Methodology constants
(bootstrap parameters, batch sizes, bracket widths) live with their callers.

The audit test in ``tests/test_locked_thresholds.py`` holds a hand-maintained
``EXPECTED`` dict of {python_name: value} that must match this file. Editing a
threshold here without also editing the test triggers a failure — that
failure is the moment of pause: *"am I allowed to change this locked value?"*

The amendment-ID and lock-date on each ``LockedThreshold`` record the
pre-registration commitment so the registry is legible without bouncing back
to HYPOTHESIS.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LockedThreshold:
    """A pre-registered numeric gate.

    Use ``passes(score)`` to test a score (scalar, np.ndarray, or pd.Series —
    `>` and `>=` are already vectorized). Use ``.value`` and ``.comparator``
    when you need the raw number/operator (e.g. for parquet groupby filters).
    """

    value: float
    amendment: str
    lock_date: str
    comparator: Literal["gt", "ge"]
    motif: str | None
    description: str

    def passes(self, score):
        if self.comparator == "gt":
            return score > self.value
        return score >= self.value


# ----- Primary §H1-C detector gates ---------------------------------------

INDUCTION_QK = LockedThreshold(
    value=0.30,
    amendment="§IND-1 / Olsson 2022",
    lock_date="pre-registration",
    comparator="gt",
    motif="induction",
    description=(
        "Olsson 2022 QK-circuit prefix-matching score on n=50, len=100 "
        "repetition sequences (PROJECT_BRIEF.md §4)."
    ),
)

SUCCESSOR_LIFT = LockedThreshold(
    value=0.13496,
    amendment="§SU-1b / §SU-tau",
    lock_date="2026-05-06",
    comparator="ge",
    motif="successor",
    description=(
        "Successor-head lift_dla minimum, locked at the 95th percentile of "
        "the pooled per-head null distribution."
    ),
)

S_INHIBITION_DELTA = LockedThreshold(
    value=0.0372,
    amendment="§S-1 / §S-tau",
    lock_date="pre-Phase-2",
    comparator="ge",
    motif="s_inhibition",
    description=(
        "S-inhibition path-patching Δ_h minimum (Wang 2023 L7H3 = 0.0372, "
        "min over the 4 Wang heads — strict gate)."
    ),
)


# ----- §H5/§H6 causal-readout classification gates ------------------------

NULL_LOW = LockedThreshold(
    value=0.8,
    amendment="§H5-causal-2-6 / §SU-1b (Readouts A, C)",
    lock_date="2026-05-11",
    comparator="ge",
    motif=None,
    description="Lower edge of the NULL acceptance band (ratio_lift ≥ 0.8).",
)

NULL_HIGH = LockedThreshold(
    value=1.2,
    amendment="§H5-causal-2-6 / §SU-1b (Readouts A, C)",
    lock_date="2026-05-11",
    comparator="ge",  # band-upper used via `score <= NULL_HIGH.value`; ge is conventional
    motif=None,
    description="Upper edge of the NULL acceptance band (ratio_lift ≤ 1.2).",
)

DEP_THRESHOLD = LockedThreshold(
    value=0.5,
    amendment="§H5-causal-2-6 (Readouts A, C)",
    lock_date="2026-05-11",
    comparator="ge",  # "DEP" fires on `ratio_lift < DEP_THRESHOLD`; ge is conventional
    motif=None,
    description="DEP classification threshold (ratio_lift < 0.5 → DEP).",
)

GENERIC_THRESHOLD = LockedThreshold(
    value=0.7,
    amendment="§H5-causal-2-6 (Readouts A, C)",
    lock_date="2026-05-11",
    comparator="ge",
    motif=None,
    description="GENERIC classification threshold (ratio_ctrl_lift < 0.7).",
)

HARD_K_MIN = LockedThreshold(
    value=4,
    amendment="§H6-causal-2",
    lock_date="2026-05-11",
    comparator="ge",
    motif="induction",
    description=(
        "Minimum number of top-K induction heads after 3-way exclusion "
        "(NM / SI-sender / suc-receiver) before §H6-causal-2 halts with "
        "HARD_K_MIN_UNMET."
    ),
)


# ----- §H4-supersede emergence-count gate ---------------------------------

N_MIN_SUPERSEDE = LockedThreshold(
    value=5,
    amendment="§H4-supersede",
    lock_date="2026-05-10",
    comparator="ge",
    motif=None,
    description=(
        "Minimum pass-count for an (size, motif) cell to clear the "
        "§H4-supersede emergence gate."
    ),
)


# ----- §H1-C-altdetectors-2-r-supersede cross-family thresholds ----------
# These are the SINGLE cross-family thresholds derived from GPT-2 small's
# 95th percentile of each alt-score axis. They are SUPERSEDED for the
# headline §H1-C-altdetectors robustness analysis by the per-size
# runtime-derived thresholds of §H1-C-altdetectors-2-rr-supersede (which
# are not in this registry because they are computed from each model's
# final-checkpoint pass-count). The cross-family values below remain
# operationally locked for the §H1-C-altdetectors-2-r-supersede readout.

ALT_TAU_IND_OV = LockedThreshold(
    value=13.592629,
    amendment="§H1-C-altdetectors-2-r-supersede",
    lock_date="2026-05-12",
    comparator="ge",
    motif="induction",
    description=(
        "Alt induction OV-circuit DLA threshold (95th-pct across 144 GPT-2 "
        "small heads, c2-percentile rule)."
    ),
)

ALT_K_MIN = LockedThreshold(
    value=2,
    amendment="§H1-C-altdetectors-2-r-supersede",
    lock_date="2026-05-12",
    comparator="ge",
    motif="successor",
    description=(
        "Alt successor argmax-K-of-7 minimum (ceil-95th-pct of K-score "
        "across 144 GPT-2 small heads)."
    ),
)

ALT_TAU_SI_DLA = LockedThreshold(
    value=0.247095,
    amendment="§H1-C-altdetectors-2-r-supersede",
    lock_date="2026-05-12",
    comparator="ge",
    motif="s_inhibition",
    description=(
        "Alt S-inhibition CompDLA-at-S2 threshold (95th-pct across 144 "
        "GPT-2 small heads)."
    ),
)


# ----- All locked thresholds for programmatic iteration -------------------

ALL_LOCKED: dict[str, LockedThreshold] = {
    "INDUCTION_QK":       INDUCTION_QK,
    "SUCCESSOR_LIFT":     SUCCESSOR_LIFT,
    "S_INHIBITION_DELTA": S_INHIBITION_DELTA,
    "NULL_LOW":           NULL_LOW,
    "NULL_HIGH":          NULL_HIGH,
    "DEP_THRESHOLD":      DEP_THRESHOLD,
    "GENERIC_THRESHOLD":  GENERIC_THRESHOLD,
    "HARD_K_MIN":         HARD_K_MIN,
    "N_MIN_SUPERSEDE":    N_MIN_SUPERSEDE,
    "ALT_TAU_IND_OV":     ALT_TAU_IND_OV,
    "ALT_K_MIN":          ALT_K_MIN,
    "ALT_TAU_SI_DLA":     ALT_TAU_SI_DLA,
}
