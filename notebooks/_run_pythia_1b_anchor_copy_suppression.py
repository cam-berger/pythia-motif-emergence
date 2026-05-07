"""Pythia-1B @ step143000 copy-suppression anchor (curiosity, post-§H3-scale).

Mirrors the existing `_run_pythia_anchor.py` (Path-A pilot for 410m) but
applied to Pythia-1B-deduped. Runs the McDougall two-criterion copy-
suppression detector across all 128 heads of Pythia-1B at step143000 on
the canonical corpus.

**Pre-reg note.** This is exploratory and NOT pre-registered as part of
§H3-scale (which locked to Path C / S-inhibition). The original Phase 1
pilot at 410m established Path A's negative result (0 heads passed McDougall
strict at 410m, motivating the pivot to Path C). This anchor checks whether
scale unlocks any copy-suppression-like behavior at 1B that wasn't visible
at 410m. Either outcome (still none, or some at 1B) is a side observation
on scaling.

Output:
    data/exploration/copy_suppression_pythia_1b_step143000.parquet
        Long-format scores: (size, step, layer, head, motif, score) for
        motif in {"copy_suppression_qk", "copy_suppression_ov"}.

    data/exploration/copy_suppression_pythia_1b_step143000_per_position.npz
        per_position_ov: (n_eligible, n_layers, n_heads) float32
        per_position_meta: (n_eligible, 2) int64 — (passage_idx, position)
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

OUT_PARQUET = ROOT / "data" / "exploration" / "copy_suppression_pythia_1b_step143000.parquet"
OUT_NPZ = ROOT / "data" / "exploration" / "copy_suppression_pythia_1b_step143000_per_position.npz"

QK_STRICT = 0.3
OV_THRESHOLD = 0.0
SIZE = "1b"
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
    print(f"  detector finished in {time() - t0:.1f}s ({(time()-t0)/60:.1f} min)")
    print(f"  eligible positions: {result.n_positions}")

    rows: list[SweepRow] = []
    for layer in range(n_layers):
        for head in range(n_heads):
            qk = float(result.qk_scores[layer, head].item())
            ov = float(result.ov_scores[layer, head].item())
            rows.append(SweepRow(SIZE, STEP, layer, head, "copy_suppression_qk", qk))
            rows.append(SweepRow(SIZE, STEP, layer, head, "copy_suppression_ov", ov))
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    write_long(rows, OUT_PARQUET)
    print(f"  wrote {OUT_PARQUET.relative_to(ROOT)}: {len(rows)} rows")

    assert result.per_position_ov is not None and result.per_position_meta is not None
    np.savez_compressed(
        OUT_NPZ,
        per_position_ov=result.per_position_ov.numpy(),
        per_position_meta=np.array(result.per_position_meta, dtype=np.int64),
    )
    print(
        f"  wrote {OUT_NPZ.relative_to(ROOT)}: "
        f"shape={tuple(result.per_position_ov.shape)}"
    )

    # McDougall strict
    strict_candidates = result.candidates(QK_STRICT, OV_THRESHOLD)
    print()
    print("=" * 72)
    print("Pythia-1B copy-suppression — STRICT criterion (QK > 0.3 AND OV < 0)")
    print("=" * 72)
    print(
        f"\n  Heads passing both strict criteria: "
        f"{len(strict_candidates)} of {n_layers * n_heads}"
    )
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

    print("\n  Top-10 heads by most-negative OV (descriptive):")
    for layer, head, score in result.top_k_ov_negative(10):
        qk = float(result.qk_scores[layer, head].item())
        ov = float(result.ov_scores[layer, head].item())
        passes_strict = (qk > QK_STRICT) and (ov < OV_THRESHOLD)
        marker = "  <-- passes strict" if passes_strict else ""
        print(f"    L{layer:2d}H{head:2d}: QK={qk:.3f} OV={ov:+.3f}{marker}")

    print("\n  Top-10 heads by QK (descriptive):")
    for layer, head, score in result.top_k_qk(10):
        qk = float(result.qk_scores[layer, head].item())
        ov = float(result.ov_scores[layer, head].item())
        passes_strict = (qk > QK_STRICT) and (ov < OV_THRESHOLD)
        marker = "  <-- passes strict" if passes_strict else ""
        print(f"    L{layer:2d}H{head:2d}: QK={qk:.3f} OV={ov:+.3f}{marker}")

    print()
    print("=" * 72)
    print("Comparison reference: Phase 1 pilot at 410m found 0 heads passing")
    print("strict McDougall (motivated the pivot to Path C / S-inhibition).")
    print("=" * 72)
    n_strict = len(strict_candidates)
    if n_strict >= 3:
        print(f"  -> Strong positive: {n_strict} heads pass strict at 1B")
        print("     (would have been Path A had this been the pilot.)")
    elif 1 <= n_strict <= 2:
        print(f"  -> Weak positive: {n_strict} heads pass strict at 1B")
    else:
        print(f"  -> Negative: {n_strict} heads pass strict at 1B (matches 410m).")
    print()
    print("Note: side observation only; not a §H3-scale gate.")


if __name__ == "__main__":
    main()
