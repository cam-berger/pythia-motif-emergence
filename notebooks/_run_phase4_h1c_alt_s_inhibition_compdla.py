"""Phase 4 §H1-C-altdetectors alt-S-inhibition sweep — 3 sizes × 40 cells.

Runs the Wang 2023 §3 Component-DLA-at-S2 detector (`s_inhibition_compdla_at_s2`)
at Pythia-{70m, 160m, 410m} × all 40 §H2-1 checkpoints. Locked threshold per
§H1-C-altdetectors-2-r-supersede: τ_si_DLA = +0.247095.

Uses the same 200 IOI prompts (BABA + ABBA, seed=0) as the locked §S-1 detector.

Outputs:
  - data/exploration/phase4_h1c_alt_s_inhibition_compdla.parquet
      columns: size, step, layer, head, score
  - data/exploration/phase4_h1c_alt_s_inhibition_compdla_per_cell.parquet
      columns: size, step, n_pass, top_layer, top_head, top_score
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
    TAU_SI_DLA,
    s_inhibition_compdla_at_s2,
)
from src.replication.tigges_ioi import load_ioi_prompts  # noqa: E402
from src.utils.mps_compat import assert_mps_fallback_enabled  # noqa: E402
from src.utils.pythia_loader import load_pythia, prefetch_pythia  # noqa: E402

OUT_PARQUET = REPO_ROOT / "data" / "exploration" / "phase4_h1c_alt_s_inhibition_compdla.parquet"
OUT_CELL = REPO_ROOT / "data" / "exploration" / "phase4_h1c_alt_s_inhibition_compdla_per_cell.parquet"
PROMPTS_PATH = REPO_ROOT / "data" / "prompts" / "ioi_prompts.tsv"

SIZES = ("70m", "160m", "410m")
CHECKPOINTS = (
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
    10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
    20000, 24000, 29000, 35000, 41000, 49000, 59000, 70000,
    84000, 100000, 120000, 143000,
)

BATCH_SIZE = 16


def main() -> None:
    assert_mps_fallback_enabled()

    clean_prompts = load_ioi_prompts(PROMPTS_PATH)
    print(f"Loaded {len(clean_prompts)} IOI prompts")

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
            compdla = s_inhibition_compdla_at_s2(
                model, clean_prompts, batch_size=BATCH_SIZE,
            )
            elapsed = time.time() - t0
            n_pass = int((compdla >= TAU_SI_DLA).sum().item())
            top_idx = int(compdla.flatten().argmax().item())
            top_L, top_H = top_idx // n_heads, top_idx % n_heads
            top_v = float(compdla[top_L, top_H].item())
            print(
                f"  CompDLA-S2 in {elapsed:.1f}s; "
                f"top L{top_L}H{top_H} CompDLA={top_v:+.4f}; "
                f"n_pass(τ={TAU_SI_DLA:.3f})={n_pass}/{n_layers * n_heads}",
                flush=True,
            )

            for L in range(n_layers):
                for H in range(n_heads):
                    rows.append(dict(
                        size=size, step=step, layer=L, head=H,
                        motif="alt_s_inhibition_compdla",
                        score=float(compdla[L, H].item()),
                    ))
            cell_rows.append(dict(
                size=size, step=step,
                n_pass=n_pass,
                top_layer=top_L, top_head=top_H, top_score=top_v,
                tau=TAU_SI_DLA,
            ))

            OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(OUT_PARQUET, index=False)
            pd.DataFrame(cell_rows).to_parquet(OUT_CELL, index=False)

            del model, compdla
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    print(f"\n=== alt-S-inhibition-CompDLA sweep complete ===")
    print(f"Wrote {OUT_PARQUET.relative_to(REPO_ROOT)}  ({len(rows):,} rows)")
    print(f"Wrote {OUT_CELL.relative_to(REPO_ROOT)}  ({len(cell_rows):,} rows)")
    print(f"Total wall: {(time.time() - overall_t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
