"""Prefetch all 40-cell × 3-size Pythia checkpoints needed for Phase 2.

Idempotent: prefetch_pythia checks the HF cache first and skips already-
cached snapshots. Up to 4 concurrent downloads.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.pythia_loader import prefetch_pythia  # noqa: E402

SIZES = ("70m", "160m", "410m")
CHECKPOINTS = (
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
    10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
    20000, 24000, 29000, 35000, 41000, 49000, 59000, 70000,
    84000, 100000, 120000, 143000,
)


def fetch_one(size: str, step: int) -> tuple[str, int, str]:
    try:
        path = prefetch_pythia(size, step=step)
        return (size, step, f"OK -> {path}")
    except Exception as e:
        return (size, step, f"ERROR: {e}")


def main() -> None:
    jobs = [(size, step) for size in SIZES for step in CHECKPOINTS]
    print(f"Prefetching {len(jobs)} (size, step) snapshots with 4 concurrent workers...", flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_one, s, c): (s, c) for s, c in jobs}
        for fut in as_completed(futures):
            size, step, status = fut.result()
            done += 1
            print(f"  [{done:>3d}/{len(jobs)}] pythia-{size} step{step}: {status}", flush=True)
    print(f"\nTotal prefetch time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
