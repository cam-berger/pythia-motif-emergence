"""Atlas-v1 exploratory head-family detectors.

This subpackage hosts the exploratory attention-head families being mapped
in the atlas-v1 branch (previous-token, duplicate-token, positional-offset,
BOS-attention, delimiter). Detectors here are NOT pre-registered — see
``src.atlas.thresholds`` for the exploratory-threshold registry, distinct
from the locked pre-registration registry in ``src.locked_thresholds``.

Each family's adapter satisfies the Detector protocol in
``src.detectors.protocol`` and uses the threshold from
``src.atlas.thresholds`` so the Sweep Driver (Candidate #1) can run atlas
detectors and locked detectors through the same code path.
"""
