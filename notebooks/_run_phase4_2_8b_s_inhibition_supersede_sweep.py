"""Phase 4 2.8B S-inhibition reduced-grid sweep — 10 cells × Pythia-2.8B-deduped (HYPOTHESIS.md §H4-supersede).

Reduced-grid re-attempt registered by §H4-supersede after the §H4-7-supersede
DEFERRED verdict (8/40 cells halted at ~57 min/cell). 10 cells in the
emergence-likely range (steps 5000–70000); ~10 h overnight budget per §H4-supersede-4.

Outputs:
  - data/exploration/phase4_2_8b_s_inhibition_supersede_sweep.parquet
  - data/exploration/phase4_2_8b_s_inhibition_supersede_per_prompt/2.8b_step{step}.npz
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

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
    REPO_ROOT / "data" / "exploration" / "phase4_2_8b_s_inhibition_supersede_sweep.parquet"
)
OUT_NPZ_DIR = (
    REPO_ROOT / "data" / "exploration" / "phase4_2_8b_s_inhibition_supersede_per_prompt"
)
PROMPTS_PATH = REPO_ROOT / "data" / "prompts" / "ioi_prompts.tsv"

SIZE = "2.8b"
# §H4-supersede-1 locked grid
CHECKPOINTS = (5000, 7000, 10000, 14000, 20000, 29000, 41000, 49000, 59000, 70000)

BATCH_SIZE = 10  # inherited from §H4-7 / §H4-7-supersede


def main() -> None:
    assert_mps_fallback_enabled()
    OUT_NPZ_DIR.mkdir(parents=True, exist_ok=True)

    clean_prompts = load_ioi_prompts(PROMPTS_PATH)
    print(f"Loaded {len(clean_prompts)} IOI prompts")

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

        corrupt_prompts = build_abc_corrupted_prompts(
            clean_prompts, model.tokenizer, seed=0
        )

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
                        size=SIZE,
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

        # Write parquet incrementally after every cell, so a mid-sweep halt
        # preserves partial results (lesson from §H4-7-supersede where parquet
        # was written end-of-loop only and the 8 completed cells were not in
        # the parquet).
        OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(OUT_PARQUET, index=False)

        del model, result
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nWrote {OUT_PARQUET.relative_to(REPO_ROOT)}  ({len(df):,} rows)")
    print(f"\nTotal sweep wall time: {(time.time() - overall_t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
