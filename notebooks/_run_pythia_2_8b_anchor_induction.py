"""Pythia-2.8B @ step143000 induction anchor inspection (§H4-4).

Runs the Olsson prefix-matching detector on Pythia-2.8B-deduped @ step143000.
Single anchor gate per HYPOTHESIS.md §H4-4 (induction anchor):

  Numerical gate:  ≥1 head clears prefix-match score > 0.3
                   (PROJECT_BRIEF.md §4 threshold).

Per §H4-4: regardless of anchor gate outcome, the 40-cell sweep proceeds
unconditionally. Anchor result is documentation. This script also serves as
the pre-flight memory probe: a single forward-pass screen at d_model=2560
that surfaces any fp32 OOM before committing to the full anchor sequence.

Outputs:
  - data/exploration/induction_pythia_2_8b_anchor.parquet (long format)
  - data/exploration/induction_pythia_2_8b_anchor_per_seq.npz (raw)
"""

from __future__ import annotations

import os

# Must be set BEFORE any huggingface_hub import. The pythia_loader.py
# runtime workaround sets HF_HUB_OFFLINE=1 *after* huggingface_hub has
# been imported, which is too late for Pythia-2.8B's late-training
# checkpoints (which have only pytorch_model.bin, not safetensors —
# transformers' loader exhausts its preferred-format probe and hits
# the empty-Bearer auth bug on a network fallback).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.detectors.induction import prefix_matching_score  # noqa: E402
from src.utils.mps_compat import assert_mps_fallback_enabled  # noqa: E402
from src.utils.pythia_loader import load_pythia  # noqa: E402

OUT_AGG = (
    REPO_ROOT
    / "data"
    / "exploration"
    / "induction_pythia_2_8b_anchor.parquet"
)
OUT_RAW = (
    REPO_ROOT
    / "data"
    / "exploration"
    / "induction_pythia_2_8b_anchor_per_seq.npz"
)

SIZE = "2.8b"
STEP = 143000
THRESHOLD = 0.3  # PROJECT_BRIEF.md §4
N_SEQUENCES = 50
SEQ_LEN = 100
SEED = 0
BATCH_SIZE = 2  # halved from 1B's 4 for d_model=2560 MPS safety per §H4-7


def main() -> None:
    assert_mps_fallback_enabled()
    print(f"Loading Pythia-{SIZE}-deduped @ step{STEP}...", flush=True)
    t0 = time.time()
    model = load_pythia(SIZE, step=STEP)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"  loaded in {time.time() - t0:.1f}s; n_layers={n_layers}, n_heads={n_heads}")

    print(
        f"\nRunning prefix-matching screen "
        f"(N={N_SEQUENCES} sequences, len={SEQ_LEN}, batch_size={BATCH_SIZE})...",
        flush=True,
    )
    t0 = time.time()
    result = prefix_matching_score(
        model,
        n_sequences=N_SEQUENCES,
        seq_len=SEQ_LEN,
        seed=SEED,
        batch_size=BATCH_SIZE,
        return_per_sequence=True,
    )
    elapsed = time.time() - t0
    print(f"  Screen complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    rows: list[dict] = []
    for L in range(n_layers):
        for H in range(n_heads):
            rows.append(
                dict(
                    model=f"pythia-{SIZE}-deduped",
                    step=STEP,
                    layer=L,
                    head=H,
                    metric="prefix_matching_score",
                    value=float(result.scores[L, H].item()),
                )
            )

    OUT_AGG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT_AGG, index=False)
    assert result.per_sequence_scores is not None
    np.savez_compressed(
        OUT_RAW,
        scores=result.scores.numpy(),
        per_sequence_scores=result.per_sequence_scores.numpy(),
        n_sequences=result.n_sequences,
        seq_len=result.seq_len,
        seed=SEED,
    )
    print(f"\nWrote {OUT_AGG.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_RAW.relative_to(REPO_ROOT)}")

    # §H4-4 gate
    print("\n=== HYPOTHESIS.md §H4-4 induction anchor gate ===")
    flat = result.scores.flatten()
    sorted_idx = flat.argsort(descending=True)

    print("Top 10 Pythia-2.8B candidates by prefix-matching score:")
    for rank in range(min(10, flat.numel())):
        idx = int(sorted_idx[rank].item())
        L, H = idx // n_heads, idx % n_heads
        v = float(flat[idx].item())
        clears = v > THRESHOLD
        flag = " [clears 0.3]" if clears else ""
        print(f"  #{rank + 1:2d}: L{L:>2d}H{H:<2d}  score={v:.4f}{flag}")

    n_clear = int((flat > THRESHOLD).sum().item())
    print(f"\nHeads clearing prefix-match > {THRESHOLD}: {n_clear}")
    numerical_pass = n_clear >= 1
    print(
        f"Numerical gate (≥1 head clears > 0.3): "
        f"{'PASS' if numerical_pass else 'FAIL'}"
    )

    print("\n=== Anchor verdict ===")
    print(f"Numerical:    {'PASS' if numerical_pass else 'FAIL'}")
    print(
        f"Per §H4-4: regardless of anchor outcome, the 40-cell sweep proceeds "
        f"unconditionally. Anchor result is documentation."
    )


if __name__ == "__main__":
    main()
