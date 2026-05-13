"""Uniform Detector protocol across the seven motif detectors.

Before this module each detector had a unique signature and a unique result
dataclass; callers had to special-case each motif. The protocol unifies the
runtime seam: every adapter exposes ``score(model) -> PerHeadScores`` with
a uniform ``(n_layers, n_heads)`` score tensor. Detector-specific quirks
(prompt sets, seed-driven sequence construction, batch handling) move to
construction time on each adapter.

The existing rich result types (``PrefixMatchingResult``, ``SuccessorResult``,
``SInhibitionResult``) are preserved as the ``aux`` payload, so the §H6-causal
pipeline and debug notebooks reach into ``aux`` without changes.

Copy-suppression is intentionally NOT adapted to this protocol (Path A is
not pursued — PROJECT_BRIEF.md §3). It returns two tensors used as a
boolean mask, which doesn't fit a single ``scores`` surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from src.locked_thresholds import LockedThreshold


@dataclass(frozen=True)
class PerHeadScores:
    """A uniform per-(layer, head) score grid plus motif identity.

    The ``scores`` tensor has shape ``(n_layers, n_heads)`` and is the
    polymorphic surface — Sweep Driver, per-step pass-counting, and rank-
    overlap analyses operate on it without knowing which detector produced
    it. ``aux`` carries detector-specific rich diagnostics (the existing
    bespoke Result dataclass) and is opaque from the protocol's perspective.
    """

    scores: torch.Tensor
    motif: str
    detector_name: str
    aux: object = None


@runtime_checkable
class Detector(Protocol):
    """Adapter protocol for all motif detectors.

    Class-attribute contract:
        motif: str
            One of {'induction', 'successor', 's_inhibition'}.
        name: str
            Unique detector identifier (e.g., 'induction_qk',
            'successor_lift', 'successor_argmax_k_of_7').
        threshold: LockedThreshold
            The canonical pre-registered gate from
            ``src.locked_thresholds``. For runtime-derived thresholds (the
            §H1-C-altdetectors-2-rr-supersede per-size scheme), the Sweep
            Driver overrides at call time; the detector's own threshold
            reports the canonical scheme.

    Runtime contract:
        score(model) -> PerHeadScores
            Returns a uniform ``(n_layers, n_heads)`` per-head score grid.
            All detector-specific config (prompts, seeds, batch_size) is
            captured at construction time on the adapter.
    """

    motif: str
    name: str
    threshold: LockedThreshold

    def score(self, model) -> PerHeadScores: ...
