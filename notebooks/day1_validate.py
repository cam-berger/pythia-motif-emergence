"""Day 1 pilot validation: induction-head detection on Pythia-410M.

Gate (PROJECT_BRIEF.md §3, Day 1): ≥1 head on Pythia-410M @ step143000 has
prefix-matching score > 0.5.

Run:
    uv run python notebooks/day1_validate.py
"""

from __future__ import annotations

import sys
import time

from src.detectors.induction import prefix_matching_score
from src.utils.mps_compat import assert_mps_fallback_enabled, get_device
from src.utils.pythia_loader import load_pythia

GATE_THRESHOLD = 0.5
SIZE = "410m"
STEP = 143000
N_SEQUENCES = 50
SEQ_LEN = 100


def main() -> int:
    assert_mps_fallback_enabled()
    device = get_device()
    print(f"device: {device}")

    print(f"loading Pythia-{SIZE} @ step{STEP}...")
    t0 = time.time()
    model = load_pythia(SIZE, step=STEP)
    print(f"  loaded in {time.time() - t0:.1f}s")
    print(f"  n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads}, "
          f"d_model={model.cfg.d_model}")

    print(f"running prefix-matching detector ({N_SEQUENCES} sequences, len {SEQ_LEN})...")
    t0 = time.time()
    result = prefix_matching_score(
        model,
        n_sequences=N_SEQUENCES,
        seq_len=SEQ_LEN,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    top10 = result.top_k(10)
    print("\ntop-10 candidate induction heads (layer, head, score):")
    for layer, head, score in top10:
        marker = "  <-- gate" if score > GATE_THRESHOLD else ""
        print(f"  L{layer:2d}H{head:2d}: {score:.4f}{marker}")

    n_above_gate = int((result.scores > GATE_THRESHOLD).sum().item())
    n_above_03 = int((result.scores > 0.3).sum().item())
    print(f"\nheads with score > 0.3: {n_above_03}")
    print(f"heads with score > {GATE_THRESHOLD}: {n_above_gate}")

    if n_above_gate >= 1:
        print("\nDay 1 gate: PASS")
        return 0
    print(f"\nDay 1 gate: FAIL (no head exceeded {GATE_THRESHOLD})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
