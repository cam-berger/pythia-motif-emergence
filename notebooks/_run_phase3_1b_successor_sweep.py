"""Phase 3 1B successor sweep — 40 cells × Pythia-1B-deduped (HYPOTHESIS.md §H3-scale).

Per cell: run successor_screen with `return_per_prompt=True` so the
bootstrap post-processing (§H2-2 inherited per §H3-scale-10) can resample
prompts. Save per-cell row in long-format parquet plus per-cell `.npz` of
per-prompt scores.

Outputs:
  - data/exploration/phase3_1b_successor_sweep.parquet — long format
    (size, step, layer, head, motif, score) where score = lift_dla
  - data/exploration/phase3_1b_successor_per_prompt/1b_step{step}.npz —
    per-prompt real and null DLA arrays for bootstrap input
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.detectors.successor import (  # noqa: E402
    build_successor_prompts,
    successor_screen,
)
from src.utils.mps_compat import assert_mps_fallback_enabled  # noqa: E402
from src.utils.pythia_loader import load_pythia, prefetch_pythia  # noqa: E402

OUT_PARQUET = (
    REPO_ROOT / "data" / "exploration" / "phase3_1b_successor_sweep.parquet"
)
OUT_NPZ_DIR = REPO_ROOT / "data" / "exploration" / "phase3_1b_successor_per_prompt"

SIZE = "1b"
CHECKPOINTS = (
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
    10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
    20000, 24000, 29000, 35000, 41000, 49000, 59000, 70000,
    84000, 100000, 120000, 143000,
)

BATCH_SIZE = 4  # smaller for d_model=2048 MPS safety (matches anchor)


def main() -> None:
    assert_mps_fallback_enabled()
    OUT_NPZ_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    overall_t0 = time.time()
    cell_n = 0
    total = len(CHECKPOINTS)
    for step in CHECKPOINTS:
        cell_n += 1
        label = f"{SIZE}_step{step}"
        print(f"\n=== [{cell_n}/{total}] {label} ===", flush=True)
        t0 = time.time()
        prefetch_pythia(SIZE, step=step)
        model = load_pythia(SIZE, step=step)
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        print(
            f"  loaded in {time.time() - t0:.1f}s; "
            f"n_layers={n_layers}, n_heads={n_heads}",
            flush=True,
        )

        prompts = build_successor_prompts(model.tokenizer, seed=0)

        t0 = time.time()
        result = successor_screen(
            model, prompts, batch_size=BATCH_SIZE, return_per_prompt=True
        )
        elapsed = time.time() - t0
        top_idx = int(result.lift_dla.flatten().argmax().item())
        top_L, top_H = top_idx // n_heads, top_idx % n_heads
        print(
            f"  screen complete in {elapsed:.1f}s; "
            f"top L{top_L}H{top_H} lift={float(result.lift_dla[top_L, top_H]):+.4f}",
            flush=True,
        )

        for L in range(n_layers):
            for H in range(n_heads):
                rows.append(
                    dict(
                        size=SIZE,
                        step=step,
                        layer=L,
                        head=H,
                        motif="successor_lift_dla",
                        score=float(result.lift_dla[L, H].item()),
                    )
                )

        np.savez_compressed(
            OUT_NPZ_DIR / f"{label}.npz",
            per_prompt_real=result.per_prompt_real.numpy(),
            per_prompt_null=result.per_prompt_null.numpy(),
            prompt_categories=np.array(result.prompt_categories, dtype="U16"),
            n_prompts=result.n_prompts,
        )

        del model, result
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nWrote {OUT_PARQUET.relative_to(REPO_ROOT)}  ({len(df):,} rows)")
    print(f"\nTotal sweep wall time: {(time.time() - overall_t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
