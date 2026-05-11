"""Pythia-70M @ step143000 S-inhibition anchor inspection (§S-8 analog).

Pre-registered as the §H6-causal Strategy-B prerequisite (HYPOTHESIS.md
§H6-causal-6). Runs the locked §S-1 path-patching detector at clean
70M step143000 to identify the component-DLA top-4 Name Movers, which
are then pinned across all §H6 ablation conditions at 70M.

This script does NOT introduce any new numerical thresholds — it inherits
τ_strict = 0.0372 (§S-tau), τ_permissive = 0.0186 (§S-tau-permissive),
the §S-1 / §S-3 path-patching protocol, and the §S-8 anchor gate
predicates (numerical: ≥1 head clears τ_strict; mechanistic: top
candidate's ≥2 of 4 NMs show positive Δ).

Per the §H6-causal Strategy-B clause: this runner MUST be executed
BEFORE `_run_phase4_h6_induction_70m_anchor.py`. The §H6 70M runner
loads `s_inhibition_pythia_70m_anchor_per_nm.npz`, reads `nm_heads`,
and asserts NM identity bit-for-bit at runtime against the value
recorded by this script.

Outputs:
  - data/exploration/s_inhibition_pythia_70m_anchor.parquet (long format,
    matching the 410m/1b/2.8b anchor parquet schema)
  - data/exploration/s_inhibition_pythia_70m_anchor_per_nm.npz (raw)
"""

from __future__ import annotations

import os

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

from src.detectors.s_inhibition import (  # noqa: E402
    build_abc_corrupted_prompts,
    s_inhibition_screen,
)
from src.replication.tigges_ioi import load_ioi_prompts  # noqa: E402
from src.utils.mps_compat import assert_mps_fallback_enabled  # noqa: E402
from src.utils.pythia_loader import load_pythia  # noqa: E402

OUT_AGG = REPO_ROOT / "data" / "exploration" / "s_inhibition_pythia_70m_anchor.parquet"
OUT_RAW = REPO_ROOT / "data" / "exploration" / "s_inhibition_pythia_70m_anchor_per_nm.npz"
PROMPTS_PATH = REPO_ROOT / "data" / "prompts" / "ioi_prompts.tsv"

SIZE = "70m"
STEP = 143000
TAU_STRICT = 0.0372
TAU_PERMISSIVE = 0.0186
# 70m has d_model=512 (smallest); batch_size can be the default. Path-patching
# at 6 layers × 8 heads = 48 total senders runs in ~30s on M5 Pro MPS.
BATCH_SIZE = 50


def main() -> None:
    assert_mps_fallback_enabled()
    print(f"Loading Pythia-{SIZE}-deduped @ step{STEP}...", flush=True)
    t0 = time.time()
    model = load_pythia(SIZE, step=STEP)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"  loaded in {time.time() - t0:.1f}s; n_layers={n_layers}, n_heads={n_heads}")

    clean = load_ioi_prompts(PROMPTS_PATH)
    corrupt = build_abc_corrupted_prompts(clean, model.tokenizer, seed=0)
    print(f"  N={len(clean)} clean+corrupt prompts (GPT-NeoX BPE)")

    print("\nRunning full path-patching screen (§S-1 protocol)...", flush=True)
    t0 = time.time()
    result = s_inhibition_screen(model, clean, corrupt, batch_size=BATCH_SIZE)
    elapsed = time.time() - t0
    print(f"  Screen complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Component-DLA top-4 NMs: {result.nm_heads}")

    rows: list[dict] = []
    for L in range(n_layers):
        for H in range(n_heads):
            rows.append(
                dict(
                    model=f"pythia-{SIZE}-deduped",
                    step=STEP,
                    layer=L,
                    head=H,
                    metric="delta_h",
                    value=float(result.delta_h[L, H].item()),
                )
            )
            for nm_idx, nm_lh in enumerate(result.nm_heads):
                rows.append(
                    dict(
                        model=f"pythia-{SIZE}-deduped",
                        step=STEP,
                        layer=L,
                        head=H,
                        metric=f"delta_h_nm{nm_idx}_L{nm_lh[0]}H{nm_lh[1]}",
                        value=float(result.per_nm_matrix[L, H, nm_idx].item()),
                    )
                )

    OUT_AGG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT_AGG, index=False)
    np.savez_compressed(
        OUT_RAW,
        delta_h=result.delta_h.numpy(),
        per_nm_matrix=result.per_nm_matrix.numpy(),
        nm_heads=np.array(result.nm_heads, dtype=np.int32),
        n_prompts=result.n_prompts,
    )
    print(f"\nWrote {OUT_AGG.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_RAW.relative_to(REPO_ROOT)}")

    # §S-8 anchor gates — diagnostic only at 70M; per §H2-9-R we already know
    # 70M S-inhibition is censored. Reported here for documentation continuity
    # with the 410m / 1b / 2.8b anchor runners.
    print("\n=== HYPOTHESIS.md §S-8 anchor gates (70M; diagnostic only) ===")
    flat = result.delta_h.flatten()
    ranking = torch.argsort(flat, descending=True)
    top10_idx = ranking[:10]

    print(f"Top 10 Pythia-{SIZE} @ step{STEP} candidates by Δ_h:")
    for rank, idx in enumerate(top10_idx, start=1):
        idx_int = int(idx.item())
        L, H = idx_int // n_heads, idx_int % n_heads
        v = float(flat[idx_int].item())
        clears_strict = v >= TAU_STRICT
        clears_permissive = v >= TAU_PERMISSIVE
        flags = []
        if clears_strict:
            flags.append("τ_strict")
        if clears_permissive and not clears_strict:
            flags.append("τ_permissive")
        flag_str = f" [{'+'.join(flags)}]" if flags else ""
        print(f"  #{rank:2d}: L{L}H{H}  Δ_h={v:+.4f}{flag_str}")

    n_clear_strict = int((flat >= TAU_STRICT).sum().item())
    n_clear_perm = int((flat >= TAU_PERMISSIVE).sum().item())
    print(f"\nHeads clearing τ_strict ({TAU_STRICT}): {n_clear_strict}")
    print(f"Heads clearing τ_permissive ({TAU_PERMISSIVE}): {n_clear_perm}")
    numerical_pass = n_clear_strict >= 1
    print(f"Numerical gate (≥1 head clears τ_strict): {'PASS' if numerical_pass else 'FAIL'}")
    if not numerical_pass:
        print("  (Expected per §H2-9-R: 70M S-inhibition is censored. The NM identity")
        print("   derived here is still meaningful as a structural anchor for §H6's NM-exclusion clause.)")

    top_idx = int(top10_idx[0].item())
    top_L, top_H = top_idx // n_heads, top_idx % n_heads
    top_per_nm = result.per_nm_matrix[top_L, top_H]
    print(f"\nTop candidate L{top_L}H{top_H} per-NM Δ contributions:")
    n_positive_nms = 0
    for nm_idx, nm_lh in enumerate(result.nm_heads):
        v = float(top_per_nm[nm_idx].item())
        positive = v > 0
        if positive:
            n_positive_nms += 1
        print(f"  NM L{nm_lh[0]}H{nm_lh[1]}: Δ = {v:+.4f}  {'(positive)' if positive else '(non-positive)'}")
    mechanistic_pass = n_positive_nms >= 2
    print(
        f"\nMechanistic gate (≥2 of 4 NMs positive on top candidate): "
        f"{n_positive_nms}/4 positive → {'PASS' if mechanistic_pass else 'FAIL'}"
    )

    print("\n=== §H6-causal Strategy-B prerequisite outcome ===")
    print(f"NM identity locked for §H6 70M: {result.nm_heads.tolist() if hasattr(result.nm_heads, 'tolist') else list(result.nm_heads)}")
    print("This npz is the input to §H6-2 NM-exclusion clause at 70M.")
    print("§H6 70M compute can now proceed.")


if __name__ == "__main__":
    main()
