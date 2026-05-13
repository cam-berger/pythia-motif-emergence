"""Atlas-v1 sweep: 5 head-family detectors × 40 checkpoints, parameterized by size.

Generalized version of _run_atlas_410m_sweep.py — pass ``--size {70m,160m,410m,2.8b}``
to run any single Pythia size. Output parquet lands at
``data/atlas/atlas_v1_<size>_sweep.parquet``.

Reuses the project's ``src.utils.pythia_loader.load_pythia`` for size-aware
device + dtype selection (CPU/MPS, fp32). Batch sizes scale down at 2.8b
per existing project precedent (§H4-7).

Output schema is canonical long-format
(size, step, layer, head, motif, detector_name, score) plus a side
``dominant_k`` column populated only on positional_offset rows.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

# Set HF_HUB_OFFLINE only if HF cache already has the model; otherwise the
# first run needs to fetch. Default-on for offline robustness.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.detectors.bos_attention import BosAttentionDetector
from src.detectors.delimiter import DelimiterDetector
from src.detectors.duplicate_token import DuplicateTokenDetector
from src.detectors.positional_offset import PositionalOffsetDetector
from src.detectors.previous_token import PreviousTokenDetector
from src.utils.pythia_loader import load_pythia, prefetch_pythia


CHECKPOINTS = (
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
    10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
    20000, 24000, 29000, 35000, 41000, 49000, 59000, 70000,
    84000, 100000, 120000, 143000,
)

SIZES = ("70m", "160m", "410m", "1b", "2.8b")

# Per-size batch tuning. Smaller for 2.8b per §H4-7 MPS-safety precedent.
# Olsson-rep detectors run at seq_len=100; delimiter runs at seq_len=256
# (4× memory pressure), so its batch is roughly halved.
BATCH_OLSSON = {
    "70m":  8,
    "160m": 8,
    "410m": 8,
    "1b":   4,
    "2.8b": 2,
}
BATCH_DELIMITER = {
    "70m":  4,
    "160m": 4,
    "410m": 4,
    "1b":   2,
    "2.8b": 1,
}


def _make_detectors(size: str):
    bo = BATCH_OLSSON[size]
    bd = BATCH_DELIMITER[size]
    return [
        PreviousTokenDetector(batch_size=bo),
        DuplicateTokenDetector(batch_size=bo),
        PositionalOffsetDetector(batch_size=bo),
        BosAttentionDetector(batch_size=bo),
        DelimiterDetector(batch_size=bd),
    ]


def _result_to_rows(result, *, size: str, step: int) -> list[dict]:
    scores = result.scores
    n_layers, n_heads = scores.shape
    dominant_k = None
    if result.detector_name == "positional_offset_attention" and isinstance(result.aux, dict):
        dominant_k = result.aux.get("dominant_k")
    rows = []
    for L in range(n_layers):
        for H in range(n_heads):
            row = dict(
                size=size,
                step=int(step),
                layer=int(L),
                head=int(H),
                motif=result.motif,
                detector_name=result.detector_name,
                score=float(scores[L, H]),
            )
            row["dominant_k"] = int(dominant_k[L, H]) if dominant_k is not None else None
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True, choices=SIZES,
                        help="Pythia size to sweep.")
    parser.add_argument("--prefetch", action="store_true",
                        help="Prefetch all 40 checkpoints first (one HF call per ckpt) before sweeping.")
    args = parser.parse_args()
    size = args.size

    out_parquet = REPO_ROOT / "data" / "atlas" / f"atlas_v1_{size.replace('.', '_')}_sweep.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    print(f"size: {size}  out: {out_parquet.relative_to(REPO_ROOT)}")
    print(f"checkpoints: {len(CHECKPOINTS)}  detectors: 5  "
          f"batch(olsson)={BATCH_OLSSON[size]}  batch(delim)={BATCH_DELIMITER[size]}")

    if args.prefetch:
        print("\nprefetching all checkpoints ...")
        for step in CHECKPOINTS:
            prefetch_pythia(size, step=step)
        print("prefetch complete\n")

    all_rows: list[dict] = []
    t_total = time.time()
    for i, step in enumerate(CHECKPOINTS, start=1):
        t_step = time.time()
        print(f"\n[{i}/{len(CHECKPOINTS)}] loading {size} @ step{step} ...", flush=True)
        try:
            prefetch_pythia(size, step=step)
            model = load_pythia(size, step=step)
        except Exception as e:
            print(f"  LOAD ERROR for {size} step{step}: {e!r}", flush=True)
            continue
        load_elapsed = time.time() - t_step
        n_layers = int(model.cfg.n_layers)
        n_heads = int(model.cfg.n_heads)

        detectors = _make_detectors(size)
        per_step_rows: list[dict] = []
        for det in detectors:
            t_det = time.time()
            result = det.score(model)
            elapsed = time.time() - t_det
            new_rows = _result_to_rows(result, size=size, step=step)
            n_pass = int((result.scores >= det.threshold.value).sum())
            n_total = result.scores.numel()
            print(
                f"  {det.name:>32} : {elapsed:6.1f}s  "
                f"n_pass(≥{det.threshold.value})={n_pass}/{n_total}  "
                f"max={float(result.scores.max()):.3f}",
                flush=True,
            )
            per_step_rows.extend(new_rows)

        all_rows.extend(per_step_rows)
        del model, detectors
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Incremental durable write
        pd.DataFrame(all_rows).to_parquet(out_parquet, index=False)
        print(
            f"  step total: {time.time() - t_step:.1f}s "
            f"(load={load_elapsed:.1f}s, layers={n_layers}, heads={n_heads})  "
            f"rows so far: {len(all_rows):,}",
            flush=True,
        )

    print(f"\n=== DONE {size} === wall time: {(time.time() - t_total) / 60:.1f}m  "
          f"rows: {len(all_rows):,}  out: {out_parquet.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
