"""18-cell successor-head exploration sweep (Phase 1.4 deliverable iii).

Sweep grid: Pythia {70m, 160m, 410m} × {0, 1000, 3000, 8000, 25000, 143000}.
Per cell: re-build prompts under that model's tokenizer (first-token
mappings will differ across sizes since 70m/160m/410m share GPT-NeoX BPE
but tokenization is context-sensitive), run the §SU-1b lift-form successor
screen, save per-head lift_dla and per-category breakdowns.

Output:
  - data/exploration/successor_emergence_preview.parquet (long format,
    canonical schema (size, step, layer, head, motif, score) — motif =
    "successor_lift_dla")
  - data/exploration/successor_emergence_per_cell.npz (raw per-cell
    real/null/lift tensors and per-category breakdowns; used by the
    emergence notebook)

Wall-time estimate: ~5-10 min on M5 Pro (DLA is much cheaper than path-
patching). Per Q9-(b), runs unconditionally regardless of anchor outcome.
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
    CATEGORIES,
    build_successor_prompts,
    successor_screen,
)
from src.utils.mps_compat import assert_mps_fallback_enabled  # noqa: E402
from src.utils.pythia_loader import load_pythia, prefetch_pythia  # noqa: E402

OUT_PARQUET = (
    REPO_ROOT / "data" / "exploration" / "successor_emergence_preview.parquet"
)
OUT_NPZ = (
    REPO_ROOT / "data" / "exploration" / "successor_emergence_per_cell.npz"
)

SIZES = ("70m", "160m", "410m")
STEPS = (0, 1000, 3000, 8000, 25000, 143000)
BATCH_SIZE = 8
TAU_LIFT = 0.13496  # locked §SU-tau


def main() -> None:
    assert_mps_fallback_enabled()
    rows: list[dict] = []
    raw_per_cell: dict[str, np.ndarray] = {}

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

            prompts = build_successor_prompts(model.tokenizer, seed=0)
            t0 = time.time()
            result = successor_screen(model, prompts, batch_size=BATCH_SIZE)
            elapsed = time.time() - t0
            top_idx = int(result.lift_dla.flatten().argmax().item())
            top_lift = float(result.lift_dla.flatten()[top_idx].item())
            print(
                f"  screen complete in {elapsed:.1f}s; "
                f"top head L{top_idx//n_heads}H{top_idx%n_heads} "
                f"lift={top_lift:+.4f}",
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
                            motif="successor_lift_dla",
                            score=float(result.lift_dla[L, H].item()),
                        )
                    )

            raw_per_cell[f"lift__{cell_label}"] = result.lift_dla.numpy()
            raw_per_cell[f"real__{cell_label}"] = result.real_dla.numpy()
            raw_per_cell[f"null__{cell_label}"] = result.null_dla.numpy()
            for c_idx, cat in enumerate(
                [c for c in CATEGORIES if c in result.per_category_real]
            ):
                lift_cat = (
                    result.per_category_real[cat] - result.per_category_null[cat]
                ).numpy()
                raw_per_cell[f"lift_{cat}__{cell_label}"] = lift_cat

            del model, result
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nWrote {OUT_PARQUET.relative_to(REPO_ROOT)}  ({len(df):,} rows)")

    np.savez_compressed(OUT_NPZ, **raw_per_cell)
    print(f"Wrote {OUT_NPZ.relative_to(REPO_ROOT)}")

    print(f"\nTotal sweep wall time: {(time.time() - overall_t0)/60:.1f} min")
    print(f"\nτ_lift = {TAU_LIFT} (locked §SU-tau, applied uniformly to all cells)")
    print("\nPer-cell summary (heads ≥ τ_lift):")
    for size in SIZES:
        for step in STEPS:
            sub = df[(df["size"] == size) & (df["step"] == step)]
            n_strict = int((sub["score"] >= TAU_LIFT).sum())
            max_lift = float(sub["score"].max())
            print(f"  {size:>4s} step{step:>6d}: n_strict={n_strict:>2d}  max_lift={max_lift:+.4f}")


if __name__ == "__main__":
    main()
