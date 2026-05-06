"""GPT-2 small S-inhibition detector validation screen (Phase 1.3 §S-5).

Runs the locked detector across all 144 heads of GPT-2 small with the N=200
IOI prompt set from Phase 1.2. Produces per-head Δ_h scalars + the (sender ×
NM) matrix, plus the gate verdict per HYPOTHESIS.md §S-5.

Outputs:
  - data/exploration/s_inhibition_gpt2_validation.parquet  (long format
    aggregate over all 144 heads + 4 NMs)
  - data/exploration/s_inhibition_gpt2_per_nm.npz  (raw per-NM matrix and
    metadata for the notebook to reload)

Pre-condition: HYPOTHESIS.md amendment §S-inhibition committed (procedure-only).
The follow-up amendment with τ_strict numerical lock waits until this script's
output is reviewed and the gate is verified to pass.
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

from src.detectors.s_inhibition import (  # noqa: E402
    WANG_S_INHIBITION_HEADS,
    build_abc_corrupted_prompts,
    s_inhibition_screen,
)
from src.replication.tigges_ioi import (  # noqa: E402
    WANG_NAMES,
    build_ioi_prompts,
    filter_single_token_names,
)
from src.utils.mps_compat import assert_mps_fallback_enabled  # noqa: E402

OUT_AGG = REPO_ROOT / "data" / "exploration" / "s_inhibition_gpt2_validation.parquet"
OUT_RAW = REPO_ROOT / "data" / "exploration" / "s_inhibition_gpt2_per_nm.npz"

SEED = 0
N_PROMPTS = 200
BATCH_SIZE = 50


def main() -> None:
    assert_mps_fallback_enabled()

    print("Loading GPT-2 small...", flush=True)
    model = HookedTransformer.from_pretrained("gpt2")
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"  n_layers={n_layers}, n_heads={n_heads}")

    # Re-verify all 24 Wang names single-token under GPT-2 BPE.
    gpt2_names = filter_single_token_names(WANG_NAMES, model.tokenizer)
    if len(gpt2_names) != len(WANG_NAMES):
        dropped = tuple(n for n in WANG_NAMES if n not in gpt2_names)
        print(f"  WARNING: {len(dropped)} names dropped under GPT-2 BPE: {dropped}")
    else:
        print(f"  All {len(WANG_NAMES)} Wang names single-token under GPT-2 BPE.")

    # Build N=200 IOI prompts under GPT-2 BPE (deterministic with seed=0).
    clean = build_ioi_prompts(seed=SEED, n=N_PROMPTS, tokenizer=model.tokenizer)
    corrupt = build_abc_corrupted_prompts(clean, model.tokenizer, seed=SEED)
    print(f"  Built {len(clean)} clean+corrupt prompts (seed={SEED})")

    print("\nRunning full 144-head path-patching screen...", flush=True)
    t0 = time.time()
    result = s_inhibition_screen(model, clean, corrupt, batch_size=BATCH_SIZE)
    elapsed = time.time() - t0
    print(f"  Screen complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    print(f"\n=== Component-DLA top-4 NMs (per §S-3) ===")
    for i, (L, H) in enumerate(result.nm_heads):
        print(f"  rank {i+1}: L{L}H{H}")
    wang_nms = {(9, 6), (9, 9), (10, 0)}
    our_nm_set = set(result.nm_heads)
    overlap = our_nm_set & wang_nms
    extra = our_nm_set - wang_nms
    print(f"  Overlap with Wang's published NMs {sorted(wang_nms)}: {sorted(overlap)}")
    if extra:
        print(f"  Extra NMs (not in Wang's set): {sorted(extra)}")

    # Build long-format parquet.
    rows: list[dict] = []
    for L in range(n_layers):
        for H in range(n_heads):
            delta = float(result.delta_h[L, H].item())
            rows.append(
                dict(model="gpt2-small", layer=L, head=H,
                     metric="delta_h", value=delta)
            )
            for nm_idx, nm_lh in enumerate(result.nm_heads):
                rows.append(
                    dict(
                        model="gpt2-small",
                        layer=L,
                        head=H,
                        metric=f"delta_h_nm{nm_idx}_L{nm_lh[0]}H{nm_lh[1]}",
                        value=float(result.per_nm_matrix[L, H, nm_idx].item()),
                    )
                )

    OUT_AGG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT_AGG, index=False)
    print(f"\nWrote {OUT_AGG.relative_to(REPO_ROOT)}")

    np.savez_compressed(
        OUT_RAW,
        delta_h=result.delta_h.numpy(),
        per_nm_matrix=result.per_nm_matrix.numpy(),
        nm_heads=np.array(result.nm_heads, dtype=np.int32),
        n_prompts=result.n_prompts,
    )
    print(f"Wrote {OUT_RAW.relative_to(REPO_ROOT)}")

    print("\n=== HYPOTHESIS.md §S-5 gate verdict ===")
    flat = result.delta_h.flatten()
    ranking = torch.argsort(flat, descending=True)
    wang_set = set(WANG_S_INHIBITION_HEADS)
    wang_ranks: list[tuple[tuple[int, int], int, float]] = []
    for w_lh in WANG_S_INHIBITION_HEADS:
        w_idx = w_lh[0] * n_heads + w_lh[1]
        rank = int((ranking == w_idx).nonzero().item()) + 1
        wang_ranks.append((w_lh, rank, float(flat[w_idx])))

    print("Wang's S-Inhibition heads (rank out of 144):")
    for (L, H), rank, v in wang_ranks:
        marker = "PASS" if rank <= 8 else "FAIL"
        print(f"  L{L}H{H}: rank #{rank}  Δ_h={v:+.4f}  [{marker}]")

    top8_pass = all(rank <= 8 for _, rank, _ in wang_ranks)
    wang_values = torch.tensor([v for _, _, v in wang_ranks], dtype=torch.float32)
    wang_median = float(wang_values.median().item())
    bulk_mean = float(flat.mean().item())
    bulk_std = float(flat.std(unbiased=True).item())
    sigma_above = (wang_median - bulk_mean) / bulk_std if bulk_std > 0 else float("nan")
    twosigma_pass = sigma_above >= 2.0

    print(f"\nBulk distribution mean: {bulk_mean:+.5f}")
    print(f"Bulk distribution std:  {bulk_std:.5f}")
    print(f"Wang heads median:      {wang_median:+.5f}")
    print(f"Sigmas above bulk:      {sigma_above:+.3f}σ")
    print(f"Top-8 inclusion:  {'PASS' if top8_pass else 'FAIL'}")
    print(f"Bulk separation:  {'PASS' if twosigma_pass else 'FAIL'} (need ≥ 2σ)")
    print(f"\nGATE: {'PASS' if (top8_pass and twosigma_pass) else 'FAIL'}")

    print("\nTop 12 by Δ_h:")
    for i in range(12):
        idx = int(ranking[i].item())
        L, H = idx // n_heads, idx % n_heads
        v = float(flat[idx].item())
        marker = " ★" if (L, H) in wang_set else ""
        print(f"  #{i+1:2d}: L{L}H{H}  Δ_h={v:+.4f}{marker}")


if __name__ == "__main__":
    main()
