"""Motif detectors and the uniform Detector protocol.

The six adapter classes below all conform to the ``Detector`` protocol in
``src.detectors.protocol``. Each exposes a uniform ``score(model) ->
PerHeadScores`` runtime call; detector-specific config (prompts, seeds,
batch sizes) lives at construction time.

Copy-suppression is intentionally NOT adapted (Path A is not pursued per
PROJECT_BRIEF.md §3); use ``copy_suppression_score`` directly if needed.
"""

from src.detectors.altdetectors import (
    InductionOvDetector,
    SInhibitionCompdlaDetector,
    SuccessorArgmaxKDetector,
)
from src.detectors.induction import InductionDetector
from src.detectors.protocol import Detector, PerHeadScores
from src.detectors.s_inhibition import SInhibitionDetector
from src.detectors.successor import SuccessorDetector

__all__ = [
    "Detector",
    "PerHeadScores",
    "InductionDetector",
    "SuccessorDetector",
    "SInhibitionDetector",
    "InductionOvDetector",
    "SuccessorArgmaxKDetector",
    "SInhibitionCompdlaDetector",
]
