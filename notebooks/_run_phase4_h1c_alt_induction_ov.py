"""Phase 4 §H1-C-altdetectors alt-induction sweep — 3 sizes × 40 cells.

Runs the Olsson OV-circuit verification detector (`induction_ov_score`) at
Pythia-{70m, 160m, 410m} × all 40 §H2-1 checkpoints. Locked threshold per
§H1-C-altdetectors-2-r-supersede: τ_ind_OV = +13.592629.

Outputs (long-format):
  - data/exploration/phase4_h1c_alt_induction_ov.parquet
      columns: size, step, layer, head, score
  - data/exploration/phase4_h1c_alt_induction_ov_per_cell.parquet
      columns: size, step, n_pass (heads with score >= τ_ind_OV)
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

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.detectors.altdetectors import (  # noqa: E402
    TAU_IND_OV,
    induction_ov_score,
)
from src.utils.mps_compat import assert_mps_fallback_enabled  # noqa: E402
from src.utils.pythia_loader import load_pythia, prefetch_pythia  # noqa: E402

OUT_PARQUET = REPO_ROOT / "data" / "exploration" / "phase4_h1c_alt_induction_ov.parquet"
OUT_CELL = REPO_ROOT / "data" / "exploration" / "phase4_h1c_alt_induction_ov_per_cell.parquet"

SIZES = ("70m", "160m", "410m")
CHECKPOINTS = (
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
    10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
    20000, 24000, 29000, 35000, 41000, 49000, 59000, 70000,
    84000, 100000, 120000, 143000,
)

N_SEQUENCES = 50
SEQ_LEN = 100
SEED = 0
BATCH_SIZE = 8


def main() -> None:
    assert_mps_fallback_enabled()

    rows: list[dict] = []
    cell_rows: list[dict] = []
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
            print(f"  loaded in {time.time()-t0:.1f}s; n_layers={n_layers}, n_heads={n_heads}", flush=True)

            t0 = time.time()
            ov_scores = induction_ov_score(
                model,
                n_sequences=N_SEQUENCES,
                seq_len=SEQ_LEN,
                seed=SEED,
                batch_size=BATCH_SIZE,
            )
            elapsed = time.time() - t0
            n_pass = int((ov_scores >= TAU_IND_OV).sum().item())
            top_idx = int(ov_scores.flatten().argmax().item())
            top_L, top_H = top_idx // n_heads, top_idx % n_heads
            top_v = float(ov_scores[top_L, top_H].item())
            print(
                f"  OV-screen in {elapsed:.1f}s; "
                f"top L{top_L}H{top_H} OV={top_v:+.4f}; "
                f"n_pass(τ={TAU_IND_OV:.3f})={n_pass}/{n_layers * n_heads}",
                flush=True,
            )

            for L in range(n_layers):
                for H in range(n_heads):
                    rows.append(dict(
                        size=size, step=step, layer=L, head=H,
                        motif="alt_induction_ov",
                        score=float(ov_scores[L, H].item()),
                    ))
            cell_rows.append(dict(
                size=size, step=step,
                n_pass=n_pass,
                top_layer=top_L, top_head=top_H, top_score=top_v,
                tau=TAU_IND_OV,
            ))

            OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(OUT_PARQUET, index=False)
            pd.DataFrame(cell_rows).to_parquet(OUT_CELL, index=False)

            del model, ov_scores
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    print(f"\n=== alt-induction-OV sweep complete ===")
    print(f"Wrote {OUT_PARQUET.relative_to(REPO_ROOT)}  ({len(rows):,} rows)")
    print(f"Wrote {OUT_CELL.relative_to(REPO_ROOT)}  ({len(cell_rows):,} rows)")
    print(f"Total wall: {(time.time() - overall_t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
