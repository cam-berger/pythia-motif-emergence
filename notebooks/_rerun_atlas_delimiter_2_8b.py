"""One-shot rerun of the delimiter detector at Pythia-2.8b across all 40 checkpoints.

Background: the original 2.8b atlas sweep (data/atlas/atlas_v1_2_8b_sweep.parquet)
returned all-zero delimiter scores because of a tokenizer-config drift between
Pythia size variants — HookedTransformer's ``to_tokens(",")`` returns an empty
sequence at 2.8b. The delimiter detector now uses ``tokenizer.encode(s,
add_special_tokens=False)`` directly, sidestepping the brittle wrapper.

This script reruns only the delimiter detector at 2.8b for all 40 checkpoints
and replaces the delimiter rows in the existing sweep parquet. The other four
detectors (previous_token, duplicate_token, positional_offset, bos_attention)
ran cleanly and are preserved.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pandas as pd
import torch

from src.detectors.delimiter import DelimiterDetector
from src.utils.pythia_loader import load_pythia, prefetch_pythia


CHECKPOINTS = (
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
    10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
    20000, 24000, 29000, 35000, 41000, 49000, 59000, 70000,
    84000, 100000, 120000, 143000,
)

OUT_PARQUET = REPO_ROOT / "data" / "atlas" / "atlas_v1_2_8b_sweep.parquet"
SIZE = "2.8b"
BATCH = 1


def main() -> None:
    # Load existing sweep; drop delimiter rows (which are all-zero from the broken run).
    existing = pd.read_parquet(OUT_PARQUET)
    kept = existing[existing["motif"] != "delimiter"].copy()
    print(f"existing rows: {len(existing):,}  non-delimiter rows kept: {len(kept):,}")

    new_rows: list[dict] = []
    t_total = time.time()
    for i, step in enumerate(CHECKPOINTS, start=1):
        t_step = time.time()
        print(f"\n[{i}/{len(CHECKPOINTS)}] loading 2.8b @ step{step} ...", flush=True)
        prefetch_pythia(SIZE, step=step)
        model = load_pythia(SIZE, step=step)
        n_layers = int(model.cfg.n_layers)
        n_heads = int(model.cfg.n_heads)

        det = DelimiterDetector(batch_size=BATCH)
        result = det.score(model)
        scores = result.scores
        n_pass = int((scores >= det.threshold.value).sum())
        print(
            f"  delimiter: max={float(scores.max()):.4f}  "
            f"n_pass(≥{det.threshold.value})={n_pass}/{scores.numel()}  "
            f"elapsed={time.time() - t_step:.1f}s",
            flush=True,
        )

        for L in range(n_layers):
            for H in range(n_heads):
                new_rows.append(dict(
                    size=SIZE,
                    step=int(step),
                    layer=int(L),
                    head=int(H),
                    motif="delimiter",
                    detector_name=result.detector_name,
                    score=float(scores[L, H]),
                    dominant_k=None,
                ))

        del model, det, result, scores
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Incremental write — merge kept + new_rows so partial progress is durable.
        merged = pd.concat([kept, pd.DataFrame(new_rows)], ignore_index=True)
        merged.to_parquet(OUT_PARQUET, index=False)

    print(f"\n=== DONE === wall time: {(time.time() - t_total) / 60:.1f}m  "
          f"new rows: {len(new_rows):,}  out: {OUT_PARQUET.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
