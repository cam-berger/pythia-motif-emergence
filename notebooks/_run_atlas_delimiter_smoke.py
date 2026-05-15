"""Smoke-test: run the delimiter detector at Pythia-410M step 143000.

Loads the model, runs DelimiterDetector with defaults (50 sequences x 256
tokens from the seeded corpus), prints top-10 (layer, head, score) heads
by score. Intended for sanity-checking the adapter and the corpus
substrate before adding the detector to the atlas sweep.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformer_lens import HookedTransformer

from src.detectors.delimiter import DelimiterDetector


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")
    print("loading Pythia-410M-deduped @ step143000 ...")
    model = HookedTransformer.from_pretrained(
        "pythia-410m-deduped",
        revision="step143000",
        device=device,
    )
    print(f"  n_layers={model.cfg.n_layers}  n_heads={model.cfg.n_heads}")

    detector = DelimiterDetector()
    print("running detector ...")
    result = detector.score(model)
    scores = result.scores  # (n_layers, n_heads), CPU float32
    aux = result.aux or {}
    n_delim = aux.get("n_delimiter_positions")
    if n_delim is not None:
        print(f"delimiter positions per sample: min={int(n_delim.min())}, "
              f"max={int(n_delim.max())}, mean={float(n_delim.float().mean()):.1f}")

    flat = scores.flatten()
    topk = torch.topk(flat, k=10)
    n_heads = scores.shape[1]
    print("\ntop-10 delimiter-attention heads at step143000:")
    print(f"{'rank':>4} {'L':>3} {'H':>3} {'score':>8}")
    for r, (s, idx) in enumerate(zip(topk.values, topk.indices), start=1):
        L = int(idx) // n_heads
        H = int(idx) % n_heads
        print(f"{r:>4} {L:>3} {H:>3} {float(s):>8.4f}")


if __name__ == "__main__":
    main()
