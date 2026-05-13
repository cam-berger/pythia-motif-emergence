"""Smoke-test all 5 atlas detectors at Pythia-410M step143000.

Loads the model ONCE and runs each of the 5 atlas detectors against it,
printing the top-10 (layer, head, score) per family. This is the
sanity-check before kicking off the full 40-checkpoint sweep — we want
to see that:

  - Each detector returns a (n_layers, n_heads) score tensor with no NaN.
  - Top-scoring heads per family look plausible (e.g., previous-token
    heads tend to cluster in early layers).
  - The 0.20 threshold passes a small-but-nonzero count per family at
    convergence (sanity for the atlas-v1 threshold choice).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time

import torch
from transformer_lens import HookedTransformer

from src.detectors.bos_attention import BosAttentionDetector
from src.detectors.delimiter import DelimiterDetector
from src.detectors.duplicate_token import DuplicateTokenDetector
from src.detectors.positional_offset import PositionalOffsetDetector
from src.detectors.previous_token import PreviousTokenDetector


def _top10(scores: torch.Tensor, n_heads: int, *, label: str, extra=None):
    flat = scores.flatten()
    topk = torch.topk(flat, k=10)
    print(f"\n{label}:")
    header = f"{'rank':>4} {'L':>3} {'H':>3} {'score':>8}"
    if extra is not None:
        header += f" {'extra':>8}"
    print(header)
    for r, (s, idx) in enumerate(zip(topk.values, topk.indices), start=1):
        L = int(idx) // n_heads
        H = int(idx) % n_heads
        line = f"{r:>4} {L:>3} {H:>3} {float(s):>8.4f}"
        if extra is not None:
            line += f" {extra[L, H].item():>8}"
        print(line)


def _pass_count(scores: torch.Tensor, threshold: float, comparator: str) -> int:
    if comparator == "gt":
        return int((scores > threshold).sum())
    return int((scores >= threshold).sum())


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    t0 = time.time()
    print("loading Pythia-410M-deduped @ step143000 ...")
    model = HookedTransformer.from_pretrained(
        "pythia-410m-deduped",
        revision="step143000",
        device=device,
    )
    n_layers, n_heads = int(model.cfg.n_layers), int(model.cfg.n_heads)
    print(f"  n_layers={n_layers}  n_heads={n_heads}  load={time.time()-t0:.1f}s")

    detectors = [
        ("previous_token",    PreviousTokenDetector()),
        ("duplicate_token",   DuplicateTokenDetector()),
        ("positional_offset", PositionalOffsetDetector()),
        ("bos_attention",     BosAttentionDetector()),
        ("delimiter",         DelimiterDetector()),
    ]

    summary = []
    for name, det in detectors:
        t1 = time.time()
        print(f"\n{'='*60}\nRunning {det.__class__.__name__} ...")
        result = det.score(model)
        elapsed = time.time() - t1
        scores = result.scores
        n_nan = int(torch.isnan(scores).sum())
        n_pass = _pass_count(scores, det.threshold.value, det.threshold.comparator)
        n_total = scores.numel()
        max_score = float(scores.max())
        median_score = float(scores.median())
        print(f"  shape={tuple(scores.shape)}  n_nan={n_nan}  "
              f"max={max_score:.4f}  median={median_score:.4f}  "
              f"n_pass(≥{det.threshold.value})={n_pass}/{n_total}  "
              f"elapsed={elapsed:.1f}s")

        extra = None
        if det.name == "positional_offset_attention" and isinstance(result.aux, dict):
            extra = result.aux.get("dominant_k")
        _top10(scores, n_heads, label=f"top-10 {name}", extra=extra)

        summary.append(dict(
            name=name, n_pass=n_pass, n_total=n_total,
            max=max_score, median=median_score, n_nan=n_nan, elapsed=elapsed,
        ))

    print(f"\n{'='*60}\nSUMMARY (atlas-v1 smoke @ 410M step143000):")
    print(f"{'family':>20} {'n_pass/total':>14} {'max':>8} {'median':>8} {'elapsed':>8}")
    for s in summary:
        pass_str = f"{s['n_pass']}/{s['n_total']}"
        print(f"{s['name']:>20} {pass_str:>14} "
              f"{s['max']:>8.4f} {s['median']:>8.4f} {s['elapsed']:>7.1f}s")


if __name__ == "__main__":
    main()
