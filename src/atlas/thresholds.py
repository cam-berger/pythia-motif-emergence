"""Exploratory thresholds for the atlas-v1 head-family detectors.

These values are EXPLORATORY — not pre-registered in HYPOTHESIS.md, not
audited by ``tests/test_locked_thresholds.py``. They share the
``LockedThreshold`` dataclass with ``src.locked_thresholds`` so the
Detector protocol is uniformly satisfied, but the ``amendment`` field
explicitly flags them as ``atlas-v1 (exploratory)`` to keep pre-registered
and exploratory thresholds visually distinct at the call site.

The 0.20 value uniform across the 5 new families is a starting-point gate:
"this head puts 20%+ of its attention mass on the family's target." Tune
in v2 if signal is too noisy or too sparse.
"""

from __future__ import annotations

from src.locked_thresholds import LockedThreshold

_AMENDMENT = "atlas-v1 (exploratory)"
_LOCK_DATE = "2026-05-13"
# Per-family thresholds calibrated at Pythia-410M step143000 (commit 6...).
# Loose families (bos_attention, delimiter) tightened to be selective; sparse
# families (positional_offset) slightly relaxed. The previous_token /
# duplicate_token defaults at 0.20 were already in the sensible range.

PREVIOUS_TOKEN = LockedThreshold(
    value=0.20,
    amendment=_AMENDMENT,
    lock_date=_LOCK_DATE,
    comparator="ge",
    motif="previous_token",
    description=(
        "Mean attention from p to p-1 on Olsson repetition sequences. "
        "Calibration: 34/384 pass at 410M step143000."
    ),
)

DUPLICATE_TOKEN = LockedThreshold(
    value=0.20,
    amendment=_AMENDMENT,
    lock_date=_LOCK_DATE,
    comparator="ge",
    motif="duplicate_token",
    description=(
        "Mean attention from p∈[50,99] to position p-50 on Olsson "
        "repetition sequences (the duplicate by construction). "
        "Calibration: 6/384 pass at 410M step143000."
    ),
)

POSITIONAL_OFFSET = LockedThreshold(
    value=0.15,
    amendment=_AMENDMENT,
    lock_date=_LOCK_DATE,
    comparator="ge",
    motif="positional_offset",
    description=(
        "max over k in {-3,-2,+1,+2,+3} of mean attention from p to p+k. "
        "Excludes k=-1 (which is the previous_token detector). "
        "Relaxed from 0.20 → 0.15 at calibration; max=0.22 at 410M "
        "step143000, so 0.20 is at the ceiling and only 4 heads pass."
    ),
)

BOS_ATTENTION = LockedThreshold(
    value=0.50,
    amendment=_AMENDMENT,
    lock_date=_LOCK_DATE,
    comparator="ge",
    motif="bos_attention",
    description=(
        "Mean attention from positions p∈[1,99] to position 0 on Olsson "
        "repetition sequences. Tightened from 0.20 → 0.50 at calibration; "
        "median attention on BOS across all 384 heads is 0.47 ('BOS sink' "
        "phenomenon), so 0.50 selects the strong-BOS heads only."
    ),
)

DELIMITER = LockedThreshold(
    value=0.40,
    amendment=_AMENDMENT,
    lock_date=_LOCK_DATE,
    comparator="ge",
    motif="delimiter",
    description=(
        "Fraction of attention mass on {',', '.', '\\n'} delimiter tokens "
        "on 50-snippet natural-prose corpus × 256 tokens. Tightened from "
        "0.20 → 0.40 at calibration; top heads have 70-80% delimiter mass."
    ),
)

ALL_ATLAS_THRESHOLDS: dict[str, LockedThreshold] = {
    "PREVIOUS_TOKEN":    PREVIOUS_TOKEN,
    "DUPLICATE_TOKEN":   DUPLICATE_TOKEN,
    "POSITIONAL_OFFSET": POSITIONAL_OFFSET,
    "BOS_ATTENTION":     BOS_ATTENTION,
    "DELIMITER":         DELIMITER,
}
