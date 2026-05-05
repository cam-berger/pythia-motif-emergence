"""Emergence sweep: copy-suppression detector across 3 sizes x 6 checkpoints.

Mirrors the (size, step) grid of the induction emergence preview for direct
cell-for-cell comparability. Each cell produces per-(layer, head) QK and OV
scores; the long-format parquet aggregates everything.

Compute: per-cell ~10-40s (proportional to model size and corpus length).
Total wall-clock ~5-12 min on M5 Pro.

Run:
    uv run python notebooks/_run_copy_suppression_sweep.py
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from pathlib import Path
from time import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notebooks._lib.sweep_io import SweepRow, write_long
from src.detectors.copy_suppression import copy_suppression_score
from src.utils.corpus_io import load_corpus
from src.utils.pythia_loader import load_pythia, prefetch_pythia

OUT_PARQUET = ROOT / "data" / "exploration" / "copy_suppression_emergence_preview.parquet"

SIZES = ["70m", "160m", "410m"]
STEPS = [0, 1000, 3000, 8000, 25000, 143000]


def main() -> None:
    torch.set_grad_enabled(False)

    print("Loading canonical corpus once (re-tokenized per model) ...")
    passages = load_corpus()
    n_text_passages = len(passages)
    text_chunks = [p.text for p in passages]

    rows: list[SweepRow] = []
    t_total = time()

    for size in SIZES:
        for step in STEPS:
            print(f"\n[{size} @ step{step}] prefetching if needed ...", flush=True)
            try:
                prefetch_pythia(size, step=step)
            except Exception as e:
                print(f"  prefetch warning ({e!r}); attempting load anyway")

            print(f"[{size} @ step{step}] loading model ...", flush=True)
            t0 = time()
            model = load_pythia(size, step=step)
            n_layers, n_heads = int(model.cfg.n_layers), int(model.cfg.n_heads)
            print(
                f"  loaded in {time() - t0:.1f}s: n_layers={n_layers}, n_heads={n_heads}"
            )

            sequences = [model.to_tokens(t)[0].cpu() for t in text_chunks]

            t0 = time()
            result = copy_suppression_score(model, sequences)
            print(
                f"  detector: {time() - t0:.1f}s, n_eligible={result.n_positions}, "
                f"strict_pass={len(result.candidates(0.3, 0.0))}, "
                f"max_qk={result.qk_scores.max().item():.3f}, "
                f"min_ov={result.ov_scores.min().item():+.3f}"
            )

            for layer in range(n_layers):
                for head in range(n_heads):
                    qk = float(result.qk_scores[layer, head].item())
                    ov = float(result.ov_scores[layer, head].item())
                    rows.append(SweepRow(size, step, layer, head, "copy_suppression_qk", qk))
                    rows.append(SweepRow(size, step, layer, head, "copy_suppression_ov", ov))

            # Free model from device before loading the next checkpoint.
            del model, result
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    write_long(rows, OUT_PARQUET)
    print(f"\nTotal sweep time: {time() - t_total:.1f}s")
    print(f"Wrote {OUT_PARQUET.relative_to(ROOT)}: {len(rows)} rows "
          f"({n_text_passages} passages × {len(SIZES) * len(STEPS)} cells × 2 motifs × variable n_heads)")


if __name__ == "__main__":
    main()
