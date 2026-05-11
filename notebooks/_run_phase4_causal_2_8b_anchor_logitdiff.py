"""§H5-causal-3-2.8b anchor experiment: IO−S logit-diff metric at Pythia-2.8B step143000 (Metric B).

Pre-registered in HYPOTHESIS.md §H5-causal-3-2.8b (commit pre-data).
Mirrors `_run_phase4_causal_410m_anchor_logitdiff.py` exactly; only per-size
derivation rules differ. All gate predicates inherited verbatim from §H5-causal-2-6.

Locked sets per §H5-causal-3-2.8b-2 / -4 / -5 (asserted bit-for-bit at runtime):
  - suc:         [(15, 14), (28, 17), (27, 13), (13, 10), (29, 28)]
  - si senders:  [(11, 29), (11, 5), (13, 9)]  (used only for verdict context)
  - NMs:         [(11, 29), (17, 12), (22, 31), (13, 9)]  (used only for verdict context)
ctrl is procedure-locked (bracket-widening, rng=default_rng(0), NM-excluded).

Outputs (data/exploration/, gitignored):
  - phase4_causal_2_8b_anchor_logitdiff.parquet
  - phase4_causal_2_8b_anchor_logitdiff_summary.parquet
  - phase4_causal_2_8b_anchor_logitdiff_verdict.parquet
  - phase4_causal_2_8b_anchor_logitdiff.log
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gc
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

# Reuse helpers verbatim from the 410m sources.
from notebooks._run_phase4_causal_410m_anchor import (  # noqa: E402
    ABLATE_K,
    BRACKET_WIDTH_INIT,
    BRACKET_WIDTH_STEP,
    TAU_LIFT,
    bootstrap_drop_ratio,
    install_mean_ablation_hooks,
    precompute_mean_z_by_length,
)
from notebooks._run_phase4_causal_410m_anchor_logitdiff import (  # noqa: E402
    B_BOOTSTRAP,
    DEP_THRESHOLD,
    GENERIC_THRESHOLD,
    NULL_HI,
    NULL_LO,
    classify_logitdiff,
    compute_logit_diff_per_prompt,
)
from notebooks._lib.sweep_io import read_long  # noqa: E402
from src.replication.tigges_ioi import load_ioi_prompts  # noqa: E402
from src.utils.mps_compat import assert_mps_fallback_enabled  # noqa: E402
from src.utils.pythia_loader import load_pythia  # noqa: E402

EXPL = REPO_ROOT / "data" / "exploration"
OUT_PER_PROMPT = EXPL / "phase4_causal_2_8b_anchor_logitdiff.parquet"
OUT_SUMMARY = EXPL / "phase4_causal_2_8b_anchor_logitdiff_summary.parquet"
OUT_VERDICT = EXPL / "phase4_causal_2_8b_anchor_logitdiff_verdict.parquet"
LOG_PATH = EXPL / "phase4_causal_2_8b_anchor_logitdiff.log"

PROMPTS_PATH = REPO_ROOT / "data" / "prompts" / "ioi_prompts.tsv"
SIZE = "2.8b"
STEP = 143000

# §H5-causal-3-2.8b-2 / -4 / -5: locked sets, asserted at runtime.
LOCKED_SUC_SET = [(15, 14), (28, 17), (27, 13), (13, 10), (29, 28)]
LOCKED_SI_SENDERS = [(11, 29), (11, 5), (13, 9)]
LOCKED_NM_HEADS = [(11, 29), (17, 12), (22, 31), (13, 9)]


def select_top_suc_2_8b(df_suc: pd.DataFrame, k: int) -> list[tuple[int, int, float]]:
    """§H5-3 procedure at 2.8b step143000."""
    sub = df_suc[(df_suc["size"] == SIZE) & (df_suc["step"] == STEP)].copy()
    sub = sub.sort_values(by=["score", "layer", "head"], ascending=[False, True, True])
    head_rows = sub.head(k)
    return [(int(r["layer"]), int(r["head"]), float(r["score"])) for _, r in head_rows.iterrows()]


def select_ctrl_set_2_8b(
    df_suc: pd.DataFrame,
    suc_set: list[tuple[int, int, float]],
    nm_set: list[tuple[int, int]],
    k: int,
    bracket_width_init: float,
    bracket_step: float,
) -> tuple[list[tuple[int, int, float]], float]:
    """§H5-4 + §H5-causal-3-7 NM-exclusion."""
    sub = df_suc[(df_suc["size"] == SIZE) & (df_suc["step"] == STEP)].copy()
    suc_lh = {(l, h) for l, h, _ in suc_set}
    nm_lh = {(l, h) for l, h in nm_set}
    exclude = suc_lh | nm_lh
    bw = bracket_width_init
    while True:
        lo, hi = TAU_LIFT - bw, TAU_LIFT
        mask = (sub["score"] >= lo) & (sub["score"] < hi)
        candidates = sub[mask].copy()
        candidates = candidates[
            ~candidates.apply(lambda r: (int(r["layer"]), int(r["head"])) in exclude, axis=1)
        ]
        if len(candidates) >= k:
            break
        bw += bracket_step
        if bw > 1.0:
            raise RuntimeError(
                f"could not assemble {k} ctrl candidates even with bracket_width={bw}"
            )
    candidates = candidates.sort_values(by=["layer", "head"]).reset_index(drop=True)
    rng = np.random.default_rng(0)
    chosen_idx = rng.choice(len(candidates), size=k, replace=False)
    chosen = candidates.iloc[sorted(chosen_idx.tolist())]
    return (
        [(int(r["layer"]), int(r["head"]), float(r["score"])) for _, r in chosen.iterrows()],
        float(bw),
    )


PAPER_HEADLINES = {
    "NULL": (
        "IOI logit-diff at Pythia-2.8b is independent of the registered top-5 "
        "successor heads. Replicates the §H5-causal-2 410m NULL on Metric B and "
        "extends the causal-disjointness claim at the head-count tier 1024."
    ),
    "DEP": (
        "Successor heads contribute causally to IOI logit-diff at Pythia-2.8b "
        "via a path the §S-1 metric does not read. Substantial reframe of the "
        "410m / 2.8b cross-metric story required."
    ),
    "GENERIC": (
        "Methodological note: IOI logit-diff is not robust to mean-ablation of "
        "any near-threshold head at Pythia-2.8b step143000; metric sensitivity "
        "is insufficient. Verdict deferred pending re-tooling."
    ),
    "MIXED": (
        "Heterogeneous ablation effect on IOI logit-diff at Pythia-2.8b; no "
        "global verdict on suc → IOI dependence. Reported as numerical-only "
        "result with CI bands; deferred for follow-up."
    ),
}


def main() -> None:
    log_lines: list[str] = []

    def log(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        print(msg, **kwargs, flush=True)
        log_lines.append(msg)

    assert_mps_fallback_enabled()
    log(f"=== §H5-causal-3-2.8b anchor experiment, Pythia-{SIZE}-deduped step{STEP} (Metric B) ===")
    log("Pre-registration: HYPOTHESIS.md §H5-causal-3-2.8b (commit before this run).")
    t_start = time.time()

    # --- Re-derive sets and assert ---
    df_suc = read_long(EXPL / "phase4_2_8b_successor_sweep.parquet")
    suc_set = select_top_suc_2_8b(df_suc, ABLATE_K)
    suc_layer_heads = [(l, h) for l, h, _ in suc_set]
    if suc_layer_heads != LOCKED_SUC_SET:
        raise RuntimeError(
            f"§H5-causal-3-2.8b-2 NO-CHERRY-PICKING ASSERTION FAILED: "
            f"re-derived suc set {suc_layer_heads} != locked {LOCKED_SUC_SET}."
        )
    log(f"\n§H5-causal-3-2.8b-2 suc set (locked, asserted match):")
    for l, h, s in suc_set:
        log(f"  L{l}H{h}  score={s:.4f}")

    nm_npz = np.load(EXPL / "s_inhibition_pythia_2_8b_anchor_per_nm.npz")
    nm_heads_pinned: list[tuple[int, int]] = [(int(L), int(H)) for L, H in nm_npz["nm_heads"]]
    if nm_heads_pinned != LOCKED_NM_HEADS:
        raise RuntimeError(
            f"§H5-causal-3-2.8b-5 NO-CHERRY-PICKING ASSERTION FAILED: "
            f"re-read NMs {nm_heads_pinned} != locked {LOCKED_NM_HEADS}."
        )
    log(f"\n§H5-causal-3-2.8b-5 pinned NMs (locked, asserted match): {nm_heads_pinned}")

    ctrl_set, widened_bw = select_ctrl_set_2_8b(
        df_suc, suc_set, nm_heads_pinned, ABLATE_K, BRACKET_WIDTH_INIT, BRACKET_WIDTH_STEP
    )
    ctrl_layer_heads = [(l, h) for l, h, _ in ctrl_set]
    log(
        f"\n§H5-causal-3-2.8b-3 ctrl set (procedure-locked, "
        f"bracket_width={widened_bw:.3f}, seed=0, NM-excluded):"
    )
    for l, h, s in ctrl_set:
        log(f"  L{l}H{h}  score={s:.4f}")

    # --- Load model + prompts ---
    log(f"\nLoading Pythia-{SIZE}-deduped @ step{STEP}...")
    t0 = time.time()
    model = load_pythia(SIZE, step=STEP)
    log(
        f"  loaded in {time.time() - t0:.1f}s; "
        f"n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads}"
    )

    clean = load_ioi_prompts(PROMPTS_PATH)
    log(f"  N={len(clean)} clean IOI prompts (GPT-NeoX BPE)")

    # --- Condition 1: clean ---
    model.reset_hooks()
    log("\n=== Condition 1: clean (no ablation) ===")
    t0 = time.time()
    clean_logit_diff = compute_logit_diff_per_prompt(model, clean)
    log(
        f"  forward pass complete in {time.time() - t0:.1f}s; "
        f"mean Δlogit_clean = {clean_logit_diff.mean():+.4f}, "
        f"std = {clean_logit_diff.std():.4f}"
    )

    # --- Precompute mean_z for both ablation sets ---
    log("\nPrecomputing mean_z for suc set...")
    t0 = time.time()
    mean_z_suc = precompute_mean_z_by_length(model, clean, suc_layer_heads)
    log(f"  done in {time.time() - t0:.1f}s")

    log("Precomputing mean_z for ctrl set...")
    t0 = time.time()
    mean_z_ctrl = precompute_mean_z_by_length(model, clean, ctrl_layer_heads)
    log(f"  done in {time.time() - t0:.1f}s")

    # --- Condition 2: suc_ablated ---
    model.reset_hooks()
    install_mean_ablation_hooks(model, suc_layer_heads, mean_z_suc)
    log("\n=== Condition 2: suc_ablated ===")
    log(f"  perma hooks installed on {len(suc_layer_heads)} suc heads")
    t0 = time.time()
    suc_logit_diff = compute_logit_diff_per_prompt(model, clean)
    log(
        f"  forward pass complete in {time.time() - t0:.1f}s; "
        f"mean Δlogit_suc = {suc_logit_diff.mean():+.4f}, "
        f"std = {suc_logit_diff.std():.4f}"
    )

    # --- Condition 3: ctrl_ablated ---
    model.reset_hooks()
    install_mean_ablation_hooks(model, ctrl_layer_heads, mean_z_ctrl)
    log("\n=== Condition 3: ctrl_ablated ===")
    log(f"  perma hooks installed on {len(ctrl_layer_heads)} ctrl heads")
    t0 = time.time()
    ctrl_logit_diff = compute_logit_diff_per_prompt(model, clean)
    log(
        f"  forward pass complete in {time.time() - t0:.1f}s; "
        f"mean Δlogit_ctrl = {ctrl_logit_diff.mean():+.4f}, "
        f"std = {ctrl_logit_diff.std():.4f}"
    )
    model.reset_hooks()

    # --- Bootstrap and verdict ---
    log("\n=== Bootstrap (B=200 paired, seed=1) ===")
    rng = np.random.default_rng(1)
    ratio_suc, suc_lo, suc_hi = bootstrap_drop_ratio(
        clean_logit_diff, suc_logit_diff, B_BOOTSTRAP, rng
    )
    ratio_ctrl, ctrl_lo, ctrl_hi = bootstrap_drop_ratio(
        clean_logit_diff, ctrl_logit_diff, B_BOOTSTRAP, rng
    )
    log(f"  ratio_suc  = {ratio_suc:.4f}  CI=[{suc_lo:.4f}, {suc_hi:.4f}]")
    log(f"  ratio_ctrl = {ratio_ctrl:.4f}  CI=[{ctrl_lo:.4f}, {ctrl_hi:.4f}]")

    pattern = classify_logitdiff(ratio_suc, suc_lo, suc_hi, ratio_ctrl, ctrl_lo, ctrl_hi)
    headline = PAPER_HEADLINES[pattern]
    log(f"\n=== §H5-causal-3-2.8b-7 verdict (Metric B) ===")
    log(f"pattern: {pattern}")
    log("matched paper headline:")
    log(f"  {headline}")

    # --- Write parquets ---
    EXPL.mkdir(parents=True, exist_ok=True)
    per_prompt_rows: list[dict] = []
    for cond_label, arr in [
        ("clean", clean_logit_diff),
        ("suc_ablated", suc_logit_diff),
        ("ctrl_ablated", ctrl_logit_diff),
    ]:
        for pi, val in enumerate(arr):
            per_prompt_rows.append(
                dict(condition=cond_label, prompt_idx=pi, logit_diff=float(val))
            )
    pd.DataFrame(per_prompt_rows).to_parquet(OUT_PER_PROMPT, index=False)

    summary_rows = []
    for cond_label, mean_val, ratio, lo, hi, ablate_set in [
        ("clean", float(clean_logit_diff.mean()), 1.0, 1.0, 1.0, ""),
        (
            "suc_ablated",
            float(suc_logit_diff.mean()),
            ratio_suc,
            suc_lo,
            suc_hi,
            ",".join(f"L{l}H{h}" for l, h in suc_layer_heads),
        ),
        (
            "ctrl_ablated",
            float(ctrl_logit_diff.mean()),
            ratio_ctrl,
            ctrl_lo,
            ctrl_hi,
            ",".join(f"L{l}H{h}" for l, h in ctrl_layer_heads),
        ),
    ]:
        summary_rows.append(
            dict(
                condition=cond_label,
                logit_diff_mean=mean_val,
                drop_ratio_mean=ratio,
                ratio_ci_low=lo,
                ratio_ci_high=hi,
                ablate_set=ablate_set,
                n_prompts=int(clean_logit_diff.shape[0]),
            )
        )
    pd.DataFrame(summary_rows).to_parquet(OUT_SUMMARY, index=False)

    pd.DataFrame(
        [
            dict(
                pattern=pattern,
                ratio_suc=ratio_suc,
                ratio_suc_ci_low=suc_lo,
                ratio_suc_ci_high=suc_hi,
                ratio_ctrl=ratio_ctrl,
                ratio_ctrl_ci_low=ctrl_lo,
                ratio_ctrl_ci_high=ctrl_hi,
                clean_mean=float(clean_logit_diff.mean()),
                suc_ablated_mean=float(suc_logit_diff.mean()),
                ctrl_ablated_mean=float(ctrl_logit_diff.mean()),
                suc_set=",".join(f"L{l}H{h}" for l, h in suc_layer_heads),
                ctrl_set=",".join(f"L{l}H{h}" for l, h in ctrl_layer_heads),
                widened_bracket_width=float(widened_bw),
                paper_headline=headline,
            )
        ]
    ).to_parquet(OUT_VERDICT, index=False)

    log(f"\nWrote {OUT_PER_PROMPT.relative_to(REPO_ROOT)}")
    log(f"Wrote {OUT_SUMMARY.relative_to(REPO_ROOT)}")
    log(f"Wrote {OUT_VERDICT.relative_to(REPO_ROOT)}")

    LOG_PATH.write_text("\n".join(log_lines) + "\n")
    print(f"\nWall: {(time.time() - t_start) / 60:.1f} min")
    print(f"Log: {LOG_PATH.relative_to(REPO_ROOT)}")

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
