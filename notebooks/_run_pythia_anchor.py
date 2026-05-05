"""Run the pre-registered pilot anchor: Pythia-410M-deduped @ step143000.

Apply the McDougall two-criterion copy-suppression detector to all 384 heads of
Pythia-410M-deduped at the final checkpoint, on the canonical corpus. Output
both the per-(layer, head) scores parquet and the per-position OV side-cache
required for the proof notebook's data-driven worked-example selection.

This is the gating evidence for the pre-reg pilot decision rule. The
calibrated supplementary scheme failed validation on GPT-2 (HYPOTHESIS.md
amendment 2026-05-05 §1: L10H7 OV rank=15, mean QK=0.019), so the supplementary
analysis is dropped and this anchor reports strict-criterion results only.

Output:
    data/pilot/copy_suppression_pythia_410m_step143000.parquet
        Long-format scores: (size, step, layer, head, motif, score) for
        motif in {"copy_suppression_qk", "copy_suppression_ov"}.

    data/pilot/copy_suppression_pythia_410m_step143000_per_position.npz
        per_position_ov: (n_eligible, n_layers, n_heads) float32
        per_position_meta: (n_eligible, 2) int64 — (passage_idx, position)

Run:
    uv run python notebooks/_run_pythia_anchor.py
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from pathlib import Path
from time import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notebooks._lib.sweep_io import SweepRow, write_long
from src.detectors.copy_suppression import copy_suppression_score
from src.utils.corpus_io import load_corpus
from src.utils.pythia_loader import load_pythia

OUT_PARQUET = ROOT / "data" / "pilot" / "copy_suppression_pythia_410m_step143000.parquet"
OUT_NPZ = ROOT / "data" / "pilot" / "copy_suppression_pythia_410m_step143000_per_position.npz"

QK_STRICT = 0.3
OV_THRESHOLD = 0.0
SIZE = "410m"
STEP = 143000


def main() -> None:
    torch.set_grad_enabled(False)

    print(f"Loading Pythia-{SIZE}-deduped @ step{STEP} ...", flush=True)
    t0 = time()
    model = load_pythia(SIZE, step=STEP)
    n_layers, n_heads = int(model.cfg.n_layers), int(model.cfg.n_heads)
    print(f"  loaded in {time() - t0:.1f}s: n_layers={n_layers}, n_heads={n_heads}")

    print("Loading canonical corpus ...", flush=True)
    passages = load_corpus()
    sequences = [model.to_tokens(p.text)[0].cpu() for p in passages]
    n_tokens = sum(len(s) for s in sequences)
    print(
        f"  {len(sequences)} passages, {n_tokens} tokens "
        f"(min={min(len(s) for s in sequences)}, "
        f"max={max(len(s) for s in sequences)})"
    )

    print("Running detector with per-position OV collection ...", flush=True)
    t0 = time()
    result = copy_suppression_score(model, sequences, collect_per_position=True)
    print(f"  detector finished in {time() - t0:.1f}s")
    print(f"  eligible positions: {result.n_positions}")

    # Long-format scores parquet
    rows: list[SweepRow] = []
    for layer in range(n_layers):
        for head in range(n_heads):
            qk = float(result.qk_scores[layer, head].item())
            ov = float(result.ov_scores[layer, head].item())
            rows.append(
                SweepRow(SIZE, STEP, layer, head, "copy_suppression_qk", qk)
            )
            rows.append(
                SweepRow(SIZE, STEP, layer, head, "copy_suppression_ov", ov)
            )
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    write_long(rows, OUT_PARQUET)
    print(f"  wrote {OUT_PARQUET.relative_to(ROOT)}: {len(rows)} rows")

    # Per-position OV side cache
    assert result.per_position_ov is not None and result.per_position_meta is not None
    np.savez_compressed(
        OUT_NPZ,
        per_position_ov=result.per_position_ov.numpy(),
        per_position_meta=np.array(result.per_position_meta, dtype=np.int64),
    )
    print(f"  wrote {OUT_NPZ.relative_to(ROOT)}: "
          f"shape={tuple(result.per_position_ov.shape)}")

    # Pre-reg gating summary — STRICT criterion only
    strict_candidates = result.candidates(QK_STRICT, OV_THRESHOLD)
    print()
    print("=" * 72)
    print("Pre-reg pilot anchor — STRICT criterion (QK > 0.3 AND OV < 0)")
    print("=" * 72)
    print(f"\n  Heads passing both strict criteria: {len(strict_candidates)} of {n_layers * n_heads}")
    if strict_candidates:
        print("  Strict candidates (sorted by most-negative OV):")
        sorted_cands = sorted(
            strict_candidates,
            key=lambda lh: float(result.ov_scores[lh].item()),
        )
        for layer, head in sorted_cands[:10]:
            qk = float(result.qk_scores[layer, head].item())
            ov = float(result.ov_scores[layer, head].item())
            print(f"    L{layer:2d}H{head:2d}: QK={qk:.3f} OV={ov:+.3f}")

    # Reference: top-10 by most-negative OV regardless of QK (for §Supplementary)
    print(f"\n  Top-10 heads by most-negative OV (descriptive, non-gating):")
    for layer, head, score in result.top_k_ov_negative(10):
        qk = float(result.qk_scores[layer, head].item())
        ov = float(result.ov_scores[layer, head].item())
        passes_strict = (qk > QK_STRICT) and (ov < OV_THRESHOLD)
        marker = "  <-- passes strict" if passes_strict else ""
        print(f"    L{layer:2d}H{head:2d}: QK={qk:.3f} OV={ov:+.3f}{marker}")

    print(f"\n  Top-10 heads by QK (descriptive):")
    for layer, head, score in result.top_k_qk(10):
        qk = float(result.qk_scores[layer, head].item())
        ov = float(result.ov_scores[layer, head].item())
        passes_strict = (qk > QK_STRICT) and (ov < OV_THRESHOLD)
        marker = "  <-- passes strict" if passes_strict else ""
        print(f"    L{layer:2d}H{head:2d}: QK={qk:.3f} OV={ov:+.3f}{marker}")

    print()
    print("Pilot decision (apply rule from HYPOTHESIS.md §Pilot decision rule):")
    n_strict = len(strict_candidates)
    if n_strict >= 3:
        print(f"  -> Strong positive (Path A): {n_strict} heads pass strict (>=3 required)")
    elif 1 <= n_strict <= 2:
        print(f"  -> Weak positive (Path A with caveat): {n_strict} heads pass strict")
    else:
        print(f"  -> Negative (Path C): {n_strict} heads pass strict")
    print()
    print("Note: pilot decision is locked in PILOT_RESULTS.md only after qualitative")
    print("inspection of any numerically-passing heads (Day 4 step). Sweep findings")
    print("are non-gating and reported in PILOT_RESULTS.md §Supplementary.")


if __name__ == "__main__":
    main()
