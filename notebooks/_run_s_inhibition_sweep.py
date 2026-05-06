"""18-cell S-inhibition exploration sweep (Phase 1.3 deliverable iii).

Sweep grid: Pythia {70m, 160m, 410m} × {0, 1000, 3000, 8000, 25000, 143000}.
Per cell: re-derive NMs via component-DLA top-4 (NMs change across training),
run the full path-patching screen, save per-head Δ_h and per-(sender, NM)
matrix.

Output:
  - data/exploration/s_inhibition_emergence_preview.parquet (long format,
    canonical schema (size, step, layer, head, motif, score))
  - data/exploration/s_inhibition_emergence_per_cell.npz (raw per-cell
    delta_h, per_nm_matrix, nm_heads — used by the emergence notebook for
    per-cell visualization)

Wall-time estimate: ~60-90 min on M5 Pro (measure-and-adjust after first
cell). Per Q9-(b), this runs unconditionally regardless of anchor outcome.
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
    REPO_ROOT / "data" / "exploration" / "s_inhibition_emergence_preview.parquet"
)
OUT_NPZ = (
    REPO_ROOT / "data" / "exploration" / "s_inhibition_emergence_per_cell.npz"
)
PROMPTS_PATH = REPO_ROOT / "data" / "prompts" / "ioi_prompts.tsv"

SIZES = ("70m", "160m", "410m")
STEPS = (0, 1000, 3000, 8000, 25000, 143000)
BATCH_SIZE = 50


def main() -> None:
    assert_mps_fallback_enabled()
    clean = load_ioi_prompts(PROMPTS_PATH)
    print(f"Loaded {len(clean)} IOI prompts (GPT-NeoX BPE, seed=0)")

    rows: list[dict] = []
    raw_per_cell: dict[str, np.ndarray] = {}
    nm_records: list[dict] = []

    overall_t0 = time.time()
    for size in SIZES:
        for step in STEPS:
            cell_label = f"{size}_step{step}"
            print(f"\n=== {cell_label} ===", flush=True)
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

            corrupt = build_abc_corrupted_prompts(clean, model.tokenizer, seed=0)

            t0 = time.time()
            result = s_inhibition_screen(
                model, clean, corrupt, batch_size=BATCH_SIZE
            )
            elapsed = time.time() - t0
            print(
                f"  screen complete in {elapsed:.1f}s; "
                f"NMs (component-DLA top-4): {result.nm_heads}",
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
            raw_per_cell[f"delta_h__{cell_label}"] = result.delta_h.numpy()
            raw_per_cell[f"per_nm__{cell_label}"] = result.per_nm_matrix.numpy()
            raw_per_cell[f"nm_heads__{cell_label}"] = np.array(
                result.nm_heads, dtype=np.int32
            )
            for nm_idx, nm_lh in enumerate(result.nm_heads):
                nm_records.append(
                    dict(
                        size=size,
                        step=step,
                        nm_rank=nm_idx + 1,
                        nm_layer=int(nm_lh[0]),
                        nm_head=int(nm_lh[1]),
                    )
                )

            del model, result
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    # Save aggregate parquet (canonical schema).
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nWrote {OUT_PARQUET.relative_to(REPO_ROOT)}  ({len(df):,} rows)")

    # Side artifact for per-cell raw data.
    np.savez_compressed(OUT_NPZ, **raw_per_cell)
    print(f"Wrote {OUT_NPZ.relative_to(REPO_ROOT)}")

    nm_df = pd.DataFrame(nm_records)
    print("\nComponent-DLA NM evolution across cells:")
    print(nm_df.to_string(index=False))

    print(
        f"\nTotal sweep wall time: {(time.time() - overall_t0)/60:.1f} min"
    )


if __name__ == "__main__":
    main()
