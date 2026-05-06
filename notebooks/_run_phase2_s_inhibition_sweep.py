"""Phase 2 S-inhibition sweep — 40 cells × 3 sizes (HYPOTHESIS.md §H2-1, §H2-6).

Per cell: run s_inhibition_screen with `return_per_prompt=True` so the
bootstrap post-processing (§H2-2) can resample IOI prompts. Save per-cell
row in long-format parquet plus per-cell `.npz` of per-prompt path-
patching deltas.

This is the longest sweep (path-patching is per-(sender, NM) per cell);
estimated ~2 hours wall time.

Outputs:
  - data/exploration/phase2_s_inhibition_sweep.parquet — long format
    (size, step, layer, head, motif, score) where score = delta_h
  - data/exploration/phase2_s_inhibition_per_prompt/{size}_step{step}.npz —
    per-prompt (sender, NM) deltas for bootstrap input
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

from src.detectors.s_inhibition import (  # noqa: E402
    build_abc_corrupted_prompts,
    s_inhibition_screen,
)
from src.replication.tigges_ioi import load_ioi_prompts  # noqa: E402
from src.utils.mps_compat import assert_mps_fallback_enabled  # noqa: E402
from src.utils.pythia_loader import load_pythia, prefetch_pythia  # noqa: E402

OUT_PARQUET = (
    REPO_ROOT / "data" / "exploration" / "phase2_s_inhibition_sweep.parquet"
)
OUT_NPZ_DIR = REPO_ROOT / "data" / "exploration" / "phase2_s_inhibition_per_prompt"
PROMPTS_PATH = REPO_ROOT / "data" / "prompts" / "ioi_prompts.tsv"

SIZES = ("70m", "160m", "410m")
CHECKPOINTS = (
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
    10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
    20000, 24000, 29000, 35000, 41000, 49000, 59000, 70000,
    84000, 100000, 120000, 143000,
)

BATCH_SIZE = 50


def main() -> None:
    assert_mps_fallback_enabled()
    OUT_NPZ_DIR.mkdir(parents=True, exist_ok=True)

    clean_prompts = load_ioi_prompts(PROMPTS_PATH)
    print(f"Loaded {len(clean_prompts)} IOI prompts")

    rows: list[dict] = []
    overall_t0 = time.time()
    cell_n = 0
    total = len(SIZES) * len(CHECKPOINTS)
    for size in SIZES:
        for step in CHECKPOINTS:
            cell_n += 1
            label = f"{size}_step{step}"
            print(f"\n=== [{cell_n}/{total}] {label} ===", flush=True)
            t0 = time.time()
            prefetch_pythia(size, step=step)
            model = load_pythia(size, step=step)
            n_layers = model.cfg.n_layers
            n_heads = model.cfg.n_heads
            print(
                f"  loaded in {time.time() - t0:.1f}s; "
                f"n_layers={n_layers}, n_heads={n_heads}",
                flush=True,
            )

            corrupt_prompts = build_abc_corrupted_prompts(clean_prompts, model.tokenizer, seed=0)

            t0 = time.time()
            result = s_inhibition_screen(
                model,
                clean_prompts,
                corrupt_prompts,
                batch_size=BATCH_SIZE,
                return_per_prompt=True,
            )
            elapsed = time.time() - t0
            top_idx = int(result.delta_h.flatten().argmax().item())
            top_L, top_H = top_idx // n_heads, top_idx % n_heads
            print(
                f"  screen complete in {elapsed:.1f}s; "
                f"top L{top_L}H{top_H} Δ_h={float(result.delta_h[top_L, top_H]):+.4f}; "
                f"NMs={result.nm_heads}",
                flush=True,
            )

            for L in range(n_layers):
                for H in range(n_heads):
                    rows.append(
                        dict(
                            size=size,
                            step=step,
                            layer=L,
                            head=H,
                            motif="s_inhibition_attn_shift",
                            score=float(result.delta_h[L, H].item()),
                        )
                    )

            np.savez_compressed(
                OUT_NPZ_DIR / f"{label}.npz",
                per_prompt_delta=result.per_prompt_delta.numpy(),
                nm_heads=np.array(result.nm_heads, dtype=np.int32),
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
