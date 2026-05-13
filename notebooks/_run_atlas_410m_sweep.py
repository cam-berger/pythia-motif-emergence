"""Atlas-v1 sweep: 5 head-family detectors × 40 checkpoints @ Pythia-410M.

Loads each of the 40 §H2-1 checkpoints in turn, runs all 5 atlas
detectors against it, and emits canonical long-format parquet rows
(size, step, layer, head, motif, score) to
``data/atlas/atlas_v1_410m_sweep.parquet``. For positional_offset, the
dominant_k aux is stored in a side column ``dominant_k`` (int8; NaN for
the other 4 families).

Schedule: the locked 40-cell grid from ``checkpoints.yaml`` /
HYPOTHESIS.md §H2-1 — same x-axis as the existing 3-motif §H1-C work,
so the integration notebook can overlay atlas families with the
locked motif trajectories.

Compute envelope at 410M-deduped on MPS: ~30s model load per
checkpoint + ~8s of detector time per checkpoint → ~25-30 minutes
wall time total.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch
from transformer_lens import HookedTransformer

from src.detectors.bos_attention import BosAttentionDetector
from src.detectors.delimiter import DelimiterDetector
from src.detectors.duplicate_token import DuplicateTokenDetector
from src.detectors.positional_offset import PositionalOffsetDetector
from src.detectors.previous_token import PreviousTokenDetector


CHECKPOINTS = (
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
    10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
    20000, 24000, 29000, 35000, 41000, 49000, 59000, 70000,
    84000, 100000, 120000, 143000,
)

OUT_PARQUET = REPO_ROOT / "data" / "atlas" / "atlas_v1_410m_sweep.parquet"
MODEL_NAME = "pythia-410m-deduped"
SIZE_TAG = "410m"


def _make_detectors():
    return [
        PreviousTokenDetector(),
        DuplicateTokenDetector(),
        PositionalOffsetDetector(),
        BosAttentionDetector(),
        DelimiterDetector(),
    ]


def _result_to_rows(result, *, size: str, step: int) -> list[dict]:
    """Flatten a PerHeadScores result into long-format rows."""
    scores = result.scores  # (n_layers, n_heads), CPU float32
    n_layers, n_heads = scores.shape
    dominant_k = None
    if result.detector_name == "positional_offset_attention" and isinstance(result.aux, dict):
        dominant_k = result.aux.get("dominant_k")  # (n_layers, n_heads) int

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
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}  out: {OUT_PARQUET.relative_to(REPO_ROOT)}")
    print(f"checkpoints: {len(CHECKPOINTS)}  detectors: 5  total cells: {len(CHECKPOINTS) * 5}")

    all_rows: list[dict] = []
    t_total = time.time()
    for i, step in enumerate(CHECKPOINTS, start=1):
        t_step = time.time()
        revision = f"step{step}"
        print(f"\n[{i}/{len(CHECKPOINTS)}] loading {MODEL_NAME} @ {revision} ...", flush=True)
        try:
            model = HookedTransformer.from_pretrained(
                MODEL_NAME, revision=revision, device=device
            )
        except Exception as e:
            print(f"  LOAD ERROR for {revision}: {e!r}", flush=True)
            continue
        load_elapsed = time.time() - t_step

        detectors = _make_detectors()
        per_step_rows: list[dict] = []
        for det in detectors:
            t_det = time.time()
            result = det.score(model)
            elapsed = time.time() - t_det
            new_rows = _result_to_rows(result, size=SIZE_TAG, step=step)
            n_pass = int((result.scores >= det.threshold.value).sum())
            n_total = result.scores.numel()
            print(
                f"  {det.name:>32} : {elapsed:5.1f}s  "
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

        # Incremental parquet write so partial progress is durable.
        df = pd.DataFrame(all_rows)
        df.to_parquet(OUT_PARQUET, index=False)
        print(
            f"  step total: {time.time() - t_step:.1f}s "
            f"(load={load_elapsed:.1f}s)  rows so far: {len(all_rows):,}",
            flush=True,
        )

    print(f"\n=== DONE === wall time: {(time.time() - t_total) / 60:.1f}m  "
          f"rows: {len(all_rows):,}  out: {OUT_PARQUET.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
