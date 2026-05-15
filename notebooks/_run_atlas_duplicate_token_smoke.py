"""Smoke test for the duplicate-token detector on Pythia-410M @ step143000.

Loads Pythia-410M-deduped at the canonical final-checkpoint revision,
instantiates ``DuplicateTokenDetector`` with its default Olsson-repetition
config, scores every (layer, head), and prints the top-10 candidates.

This script is NOT a test; it is an exploratory smoke probe for the
atlas-v1 head-family inventory. Run it by hand, not under pytest:

    .venv/bin/python notebooks/_run_atlas_duplicate_token_smoke.py
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from src.detectors.duplicate_token import DuplicateTokenDetector


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    device = _pick_device()
    print(f"[smoke] device = {device}")

    model = HookedTransformer.from_pretrained(
        "pythia-410m-deduped",
        revision="step143000",
        device=device,
    )
    print(
        f"[smoke] loaded pythia-410m-deduped @ step143000 "
        f"(n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads})"
    )

    detector = DuplicateTokenDetector()
    result = detector.score(model)
    scores = result.scores  # (n_layers, n_heads), CPU
    n_layers, n_heads = scores.shape

    flat = scores.flatten()
    top_vals, top_idx = torch.topk(flat, k=min(10, flat.numel()))
    print(f"[smoke] motif={result.motif} detector={result.detector_name}")
    print(f"[smoke] threshold = {detector.threshold.value} ({detector.threshold.comparator})")
    print("[smoke] top-10 (layer, head, score):")
    for val, idx in zip(top_vals, top_idx, strict=True):
        layer = int(idx.item() // n_heads)
        head = int(idx.item() % n_heads)
        print(f"  L{layer:02d}.H{head:02d}  {float(val.item()):.4f}")


if __name__ == "__main__":
    main()
