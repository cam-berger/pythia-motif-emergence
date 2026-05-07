"""Prefetch all 40 Pythia-2.8B-deduped checkpoints for §H4-scaling.

Per HYPOTHESIS.md §H4-1, §H4-7, §H4-8: 2.8B-deduped only, 40-cell §H2-1 grid,
safetensors-only (`*.bin` explicitly excluded; ~700 GB total).

Idempotent: snapshot_download checks the HF cache and skips already-cached
revisions. Up to 4 concurrent downloads.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import snapshot_download  # noqa: E402

from src.utils.pythia_loader import pythia_repo, revision_for_step  # noqa: E402

SIZE = "2.8b"
CHECKPOINTS = (
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
    10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
    20000, 24000, 29000, 35000, 41000, 49000, 59000, 70000,
    84000, 100000, 120000, 143000,
)

# Per §H4-8: safetensors-only patterns. `*.bin` excluded explicitly.
_SNAPSHOT_PATTERNS_2_8B = [
    "*.json",
    "*.safetensors",
    "tokenizer*",
    "special_tokens*",
    "merges*",
    "vocab*",
]


def fetch_one(step: int) -> tuple[str, int, str]:
    try:
        path = snapshot_download(
            repo_id=pythia_repo(SIZE),
            revision=revision_for_step(step),
            allow_patterns=_SNAPSHOT_PATTERNS_2_8B,
        )
        return (SIZE, step, f"OK -> {path}")
    except Exception as e:
        return (SIZE, step, f"ERROR: {e}")


def main() -> None:
    jobs = list(CHECKPOINTS)
    print(
        f"Prefetching {len(jobs)} pythia-{SIZE}-deduped snapshots "
        f"(safetensors-only) with 4 concurrent workers...",
        flush=True,
    )
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_one, step): step for step in jobs}
        for fut in as_completed(futures):
            size, step, status = fut.result()
            done += 1
            print(
                f"  [{done:>3d}/{len(jobs)}] pythia-{size} step{step}: {status}",
                flush=True,
            )
    print(f"\nTotal prefetch time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
