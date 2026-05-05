"""Run the Tigges IOI replication across 8 Pythia-410M checkpoints.

Locked design (from Phase 1.2 grilling, see NOTES.md "2026-05-05" entry):
  - Model: Pythia-410M-deduped (note: Tigges used `-no-dropout`)
  - Prompts: data/prompts/ioi_prompts.tsv (N=200, seed=0; Tigges used N=70)
  - Checkpoints: 0, 1000, 3000, 8000, 25000, 50000, 100000, 143000
  - Metrics: accuracy (gate) + mean_logit_diff (supplementary)

Outputs:
  - data/exploration/tigges_ioi_replication.parquet — aggregate metrics, one
    row per (step, metric).
  - data/exploration/tigges_ioi_per_prompt.parquet — per-prompt logit_diff
    for the notebook to plot distributions and debug if the gate fails.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gc

import pandas as pd
import torch

from src.replication.tigges_ioi import load_ioi_prompts, score_prompts  # noqa: E402
from src.utils.mps_compat import assert_mps_fallback_enabled  # noqa: E402
from src.utils.pythia_loader import load_pythia, prefetch_pythia  # noqa: E402

PROMPTS_PATH = REPO_ROOT / "data" / "prompts" / "ioi_prompts.tsv"
AGG_OUT = REPO_ROOT / "data" / "exploration" / "tigges_ioi_replication.parquet"
PER_PROMPT_OUT = (
    REPO_ROOT / "data" / "exploration" / "tigges_ioi_per_prompt.parquet"
)
SIZE = "410m"
STEPS: tuple[int, ...] = (0, 1000, 3000, 8000, 25000, 50000, 100000, 143000)
BATCH_SIZE = 16


def main() -> None:
    assert_mps_fallback_enabled()
    prompts = load_ioi_prompts(PROMPTS_PATH)
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_PATH.relative_to(REPO_ROOT)}")

    agg_rows: list[dict] = []
    per_prompt_rows: list[dict] = []

    for step in STEPS:
        print(f"\n=== step{step} ===", flush=True)
        t0 = time.time()
        prefetch_pythia(SIZE, step=step)
        model = load_pythia(SIZE, step=step)
        print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

        t0 = time.time()
        result = score_prompts(model, prompts, batch_size=BATCH_SIZE)
        print(
            f"  scored in {time.time() - t0:.1f}s  "
            f"acc={result.accuracy:.4f}  mld={result.mean_logit_diff:+.4f}",
            flush=True,
        )

        agg_rows.append(
            dict(
                size=SIZE,
                step=step,
                metric="accuracy",
                value=result.accuracy,
                n_prompts=len(prompts),
            )
        )
        agg_rows.append(
            dict(
                size=SIZE,
                step=step,
                metric="mean_logit_diff",
                value=result.mean_logit_diff,
                n_prompts=len(prompts),
            )
        )
        for i, p in enumerate(prompts):
            per_prompt_rows.append(
                dict(
                    size=SIZE,
                    step=step,
                    prompt_idx=i,
                    template_kind=p.template_kind,
                    io_name=p.io_name,
                    s_name=p.s_name,
                    logit_io=float(result.logit_io[i].item()),
                    logit_s=float(result.logit_s[i].item()),
                    logit_diff=float(result.logit_diff[i].item()),
                )
            )

        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    AGG_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(agg_rows).to_parquet(AGG_OUT, index=False)
    pd.DataFrame(per_prompt_rows).to_parquet(PER_PROMPT_OUT, index=False)
    print(f"\nWrote {AGG_OUT.relative_to(REPO_ROOT)}")
    print(f"Wrote {PER_PROMPT_OUT.relative_to(REPO_ROOT)}")

    print("\n=== Aggregate accuracy curve ===")
    agg_df = pd.DataFrame(agg_rows)
    acc = (
        agg_df[agg_df["metric"] == "accuracy"]
        .sort_values("step")[["step", "value"]]
        .reset_index(drop=True)
    )
    print(acc.to_string(index=False))


if __name__ == "__main__":
    main()
