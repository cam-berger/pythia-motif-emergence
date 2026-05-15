"""Smoke test for the atlas-v1 PositionalOffsetDetector.

Loads Pythia-410M-deduped @ step143000, runs the positional-offset
detector with defaults, and prints the top-10 (layer, head, score,
dominant_k) tuples by score. Intended as a one-shot manual sanity check
— at the final checkpoint we expect to surface a small number of heads
that lock onto one of the five nonzero offsets in
``K_SET = (-3, -2, +1, +2, +3)``.

Run:
    uv run python notebooks/_run_atlas_positional_offset_smoke.py
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detectors.positional_offset import PositionalOffsetDetector


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    torch.set_grad_enabled(False)
    device = _select_device()
    print(f"Loading Pythia-410M-deduped @ step143000 on {device} ...", flush=True)

    model = HookedTransformer.from_pretrained(
        "pythia-410m-deduped",
        revision="step143000",
        device=device,
    )
    model.eval()

    detector = PositionalOffsetDetector()
    print(
        f"Scoring {model.cfg.n_layers} layers x {model.cfg.n_heads} heads "
        f"(threshold={detector.threshold.value}) ...",
        flush=True,
    )
    per_head = detector.score(model)
    scores = per_head.scores  # (n_layers, n_heads), CPU float32
    dominant_k = per_head.aux["dominant_k"]  # (n_layers, n_heads), CPU int64

    n_layers, n_heads = scores.shape
    flat = scores.flatten()
    top_vals, top_idx = torch.topk(flat, k=min(10, flat.numel()))

    print()
    print(
        f"Top-10 (layer, head, score, dominant_k) for motif={per_head.motif} "
        f"detector={per_head.detector_name}:"
    )
    for val, idx in zip(top_vals, top_idx, strict=True):
        i = int(idx.item())
        layer = i // n_heads
        head = i % n_heads
        k = int(dominant_k[layer, head].item())
        print(
            f"  L{layer:>2}  H{head:>2}  score={float(val.item()):.4f}  "
            f"dominant_k={k:+d}"
        )


if __name__ == "__main__":
    main()
