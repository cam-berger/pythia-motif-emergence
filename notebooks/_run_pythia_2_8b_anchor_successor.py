"""Pythia-2.8B @ step143000 successor anchor inspection (§H4-4).

Inherits §SU-5 anchor gates verbatim:
  Numerical:   ≥1 head clears τ_lift = 0.13496 (locked §SU-tau).
  Qualitative: top candidate has positive lift in ≥3/4 categories
               (days, months, numerals, letters).

Per §H4-4: regardless of anchor outcome, the 40-cell sweep proceeds
unconditionally. Anchor result is documentation.

Outputs:
  - data/exploration/successor_pythia_2_8b_anchor.parquet (long format)
  - data/exploration/successor_pythia_2_8b_anchor_per_head.npz (raw)
"""

from __future__ import annotations

import os

# Must be set before any huggingface_hub import (see _run_pythia_2_8b_anchor_induction.py).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

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
    save_successor_prompts,
    successor_screen,
)
from src.utils.mps_compat import assert_mps_fallback_enabled  # noqa: E402
from src.utils.pythia_loader import load_pythia  # noqa: E402

OUT_AGG = (
    REPO_ROOT
    / "data"
    / "exploration"
    / "successor_pythia_2_8b_anchor.parquet"
)
OUT_RAW = (
    REPO_ROOT
    / "data"
    / "exploration"
    / "successor_pythia_2_8b_anchor_per_head.npz"
)
OUT_PROMPTS = REPO_ROOT / "data" / "prompts" / "successor_prompts_pythia_2_8b.tsv"

SIZE = "2.8b"
STEP = 143000
TAU_LIFT = 0.13496  # locked §SU-tau
BATCH_SIZE = 2  # halved from 1B's 4 for d_model=2560 MPS safety per §H4-7


def main() -> None:
    assert_mps_fallback_enabled()
    print(f"Loading Pythia-{SIZE}-deduped @ step{STEP}...", flush=True)
    t0 = time.time()
    model = load_pythia(SIZE, step=STEP)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"  loaded in {time.time() - t0:.1f}s; n_layers={n_layers}, n_heads={n_heads}")

    prompts = build_successor_prompts(model.tokenizer, seed=0)
    print(f"  Built {len(prompts)} prompts (GPT-NeoX BPE first-token mappings)")
    save_successor_prompts(
        prompts,
        OUT_PROMPTS,
        seed=0,
        tokenizer_name=f"pythia-{SIZE}-deduped (GPT-NeoX BPE)",
    )
    print(f"  Wrote {OUT_PROMPTS.relative_to(REPO_ROOT)}")

    print("\nRunning full successor screen...", flush=True)
    t0 = time.time()
    result = successor_screen(model, prompts, batch_size=BATCH_SIZE)
    elapsed = time.time() - t0
    print(f"  Screen complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    rows: list[dict] = []
    for L in range(n_layers):
        for H in range(n_heads):
            for metric_name, value in (
                ("lift_dla", float(result.lift_dla[L, H].item())),
                ("real_dla", float(result.real_dla[L, H].item())),
                ("null_dla", float(result.null_dla[L, H].item())),
            ):
                rows.append(
                    dict(
                        model=f"pythia-{SIZE}-deduped",
                        step=STEP,
                        layer=L,
                        head=H,
                        metric=metric_name,
                        value=value,
                    )
                )
            for cat in CATEGORIES:
                if cat not in result.per_category_real:
                    continue
                real_v = float(result.per_category_real[cat][L, H].item())
                null_v = float(result.per_category_null[cat][L, H].item())
                rows.append(
                    dict(
                        model=f"pythia-{SIZE}-deduped",
                        step=STEP,
                        layer=L,
                        head=H,
                        metric=f"lift_dla_{cat}",
                        value=real_v - null_v,
                    )
                )

    OUT_AGG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT_AGG, index=False)
    np.savez_compressed(
        OUT_RAW,
        real_dla=result.real_dla.numpy(),
        null_dla=result.null_dla.numpy(),
        lift_dla=result.lift_dla.numpy(),
        per_category_real=np.stack(
            [result.per_category_real[c].numpy() for c in CATEGORIES if c in result.per_category_real]
        ),
        per_category_null=np.stack(
            [result.per_category_null[c].numpy() for c in CATEGORIES if c in result.per_category_null]
        ),
        category_order=np.array(
            [c for c in CATEGORIES if c in result.per_category_real]
        ),
        lift_threshold_locked=TAU_LIFT,
        n_prompts=result.n_prompts,
    )
    print(f"\nWrote {OUT_AGG.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_RAW.relative_to(REPO_ROOT)}")

    print("\n=== HYPOTHESIS.md §H4-4 successor anchor gates ===")
    flat = result.lift_dla.flatten()
    ranking = torch.argsort(flat, descending=True)

    print("Top 10 Pythia-2.8B candidates by lift:")
    for rank in range(10):
        idx = int(ranking[rank].item())
        L, H = idx // n_heads, idx % n_heads
        lift = float(flat[idx].item())
        clears = lift >= TAU_LIFT
        flag = " [clears τ_lift]" if clears else ""
        print(f"  #{rank + 1:2d}: L{L:>2d}H{H:<2d}  lift={lift:+.4f}{flag}")

    n_clear = int((flat >= TAU_LIFT).sum().item())
    print(f"\nHeads clearing τ_lift = {TAU_LIFT}: {n_clear}")
    numerical_pass = n_clear >= 1
    print(f"Numerical gate (≥1 head clears τ_lift): {'PASS' if numerical_pass else 'FAIL'}")

    top_idx = int(ranking[0].item())
    top_L, top_H = top_idx // n_heads, top_idx % n_heads
    print(f"\nTop candidate L{top_L}H{top_H} per-category lift breakdown:")
    n_positive_categories = 0
    for cat in CATEGORIES:
        if cat not in result.per_category_real:
            continue
        real_v = float(result.per_category_real[cat][top_L, top_H].item())
        null_v = float(result.per_category_null[cat][top_L, top_H].item())
        lift_v = real_v - null_v
        positive = lift_v > 0
        if positive:
            n_positive_categories += 1
        marker = "(positive)" if positive else "(non-positive)"
        print(
            f"  {cat:>10s}: lift = {lift_v:+.4f}  "
            f"(real {real_v:+.4f}, null {null_v:+.4f})  {marker}"
        )

    qualitative_pass = n_positive_categories >= 3
    print(
        f"\nQualitative gate (≥3 of 4 categories positive lift on top candidate): "
        f"{n_positive_categories}/{len(CATEGORIES)} positive → "
        f"{'PASS' if qualitative_pass else 'FAIL'}"
    )

    print("\n=== Anchor verdict ===")
    print(f"Numerical:    {'PASS' if numerical_pass else 'FAIL'}")
    print(f"Qualitative:  {'PASS' if qualitative_pass else 'FAIL'}")
    print(
        f"Per §H4-4: regardless of anchor outcome, the 40-cell sweep proceeds "
        f"unconditionally. Anchor result is documentation."
    )


if __name__ == "__main__":
    main()
