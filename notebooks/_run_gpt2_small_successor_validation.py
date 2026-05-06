"""GPT-2 small successor detector validation screen (Phase 1.4 §SU-1b-4).

Runs the lift-form successor detector across all 144 heads with the locked
4-category prompt set (days, months, numerals=mixed digit+word, letters)
under L 2023's named target L9H1. Produces per-head real/null/lift scalars,
the per-category breakdown, and the §SU-1b-4 gate verdict.

Outputs:
  - data/exploration/successor_gpt2_small_validation.parquet (long format,
    per-head per-category and aggregate scores)
  - data/exploration/successor_gpt2_small_per_head.npz (raw tensors for the
    proof notebook to reload)
  - data/prompts/successor_prompts_gpt2.tsv (committed prompt set with
    first-token mappings logged per §SU-2)

Pre-condition: HYPOTHESIS.md amendments §SU and §SU-1b committed.
The follow-up §SU-tau amendment (numerical lift threshold lock) waits
until this script's output is reviewed and the gate verifies as PASS.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402

from src.detectors.successor import (  # noqa: E402
    CATEGORIES,
    GPT2_SMALL_L9H1,
    build_successor_prompts,
    gate_verdict,
    save_successor_prompts,
    successor_screen,
)
from src.utils.mps_compat import assert_mps_fallback_enabled  # noqa: E402

OUT_AGG = (
    REPO_ROOT
    / "data"
    / "exploration"
    / "successor_gpt2_small_validation.parquet"
)
OUT_RAW = (
    REPO_ROOT / "data" / "exploration" / "successor_gpt2_small_per_head.npz"
)
OUT_PROMPTS = REPO_ROOT / "data" / "prompts" / "successor_prompts_gpt2.tsv"

SEED = 0
BATCH_SIZE = 16


def main() -> None:
    assert_mps_fallback_enabled()

    print("Loading GPT-2 small...", flush=True)
    model = HookedTransformer.from_pretrained("gpt2")
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"  n_layers={n_layers}, n_heads={n_heads}")

    prompts = build_successor_prompts(model.tokenizer, seed=SEED)
    print(f"  Built {len(prompts)} successor prompts (seed={SEED})")
    save_successor_prompts(
        prompts,
        OUT_PROMPTS,
        seed=SEED,
        tokenizer_name="gpt2",
    )
    print(f"  Wrote {OUT_PROMPTS.relative_to(REPO_ROOT)}")

    print("\nRunning full 144-head screen...", flush=True)
    t0 = time.time()
    result = successor_screen(model, prompts, batch_size=BATCH_SIZE)
    elapsed = time.time() - t0
    print(f"  Screen complete in {elapsed:.1f}s")

    # Long-format parquet: per-head aggregates + per-category breakdown.
    rows: list[dict] = []
    for L in range(n_layers):
        for H in range(n_heads):
            rows.append(
                dict(
                    model="gpt2-small",
                    layer=L,
                    head=H,
                    metric="lift_dla",
                    value=float(result.lift_dla[L, H].item()),
                )
            )
            rows.append(
                dict(
                    model="gpt2-small",
                    layer=L,
                    head=H,
                    metric="real_dla",
                    value=float(result.real_dla[L, H].item()),
                )
            )
            rows.append(
                dict(
                    model="gpt2-small",
                    layer=L,
                    head=H,
                    metric="null_dla",
                    value=float(result.null_dla[L, H].item()),
                )
            )
            for cat in CATEGORIES:
                if cat not in result.per_category_real:
                    continue
                real_v = float(result.per_category_real[cat][L, H].item())
                null_v = float(result.per_category_null[cat][L, H].item())
                rows.append(
                    dict(
                        model="gpt2-small",
                        layer=L,
                        head=H,
                        metric=f"real_dla_{cat}",
                        value=real_v,
                    )
                )
                rows.append(
                    dict(
                        model="gpt2-small",
                        layer=L,
                        head=H,
                        metric=f"null_dla_{cat}",
                        value=null_v,
                    )
                )
                rows.append(
                    dict(
                        model="gpt2-small",
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
        lift_threshold=result.lift_threshold,
        n_prompts=result.n_prompts,
    )
    print(f"\nWrote {OUT_AGG.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_RAW.relative_to(REPO_ROOT)}")

    print("\n=== HYPOTHESIS.md §SU-1b-4 gate verdict ===")
    verdict = gate_verdict(result, GPT2_SMALL_L9H1, top_k=3)
    for k, v in verdict.items():
        print(f"  {k}: {v}")

    print(f"\n=== Top 8 heads by lift ===")
    flat_lift = result.lift_dla.flatten()
    ranking = torch.argsort(flat_lift, descending=True)
    for rank in range(8):
        idx = int(ranking[rank].item())
        L, H = idx // n_heads, idx % n_heads
        marker = " ★ L9H1" if (L, H) == GPT2_SMALL_L9H1 else ""
        lift = float(flat_lift[idx].item())
        real_v = float(result.real_dla[L, H].item())
        null_v = float(result.null_dla[L, H].item())
        print(
            f"  #{rank + 1}: L{L}H{H}  lift={lift:+.4f}  "
            f"(real={real_v:+.4f}, null={null_v:+.4f}){marker}"
        )

    if verdict["passes"]:
        print(
            f"\nGATE PASS. Next: §SU-tau numerical lift-threshold lock = "
            f"{result.lift_threshold:.5f}; commit GPT-2 results + amendment; "
            f"proceed to Pythia anchor."
        )
    else:
        print(
            "\nGATE FAIL. Per §SU-6 hard-stop: re-grill detector spec from §SU-1."
        )


if __name__ == "__main__":
    main()
