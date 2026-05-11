"""§H5-causal-3-2.8b anchor experiment: causal-dependence ablation at Pythia-2.8B step143000 (Metric A: §S-1 path-patching).

Pre-registered in HYPOTHESIS.md §H5-causal-3-2.8b (commit pre-data).
Mirrors `_run_phase4_causal_410m_anchor.py` exactly; only per-size derivation
rules (suc / SI / NM lists, BATCH_SIZE) differ. All gate predicates, ablation
methods, bootstrap params, and verdict taxonomies are inherited verbatim.

Locked sets per §H5-causal-3-2.8b-2 / -4 / -5 (asserted bit-for-bit at runtime):
  - suc:         [(15, 14), (28, 17), (27, 13), (13, 10), (29, 28)]
  - si senders:  [(11, 29), (11, 5), (13, 9)]
  - NMs:         [(11, 29), (17, 12), (22, 31), (13, 9)]
ctrl is procedure-locked (bracket-widening, rng=default_rng(0)).

Outputs (data/exploration/, all gitignored per *.parquet rule):
  - phase4_causal_2_8b_anchor.parquet
  - phase4_causal_2_8b_anchor_summary.parquet
  - phase4_causal_2_8b_anchor_verdict.parquet
  - phase4_causal_2_8b_anchor.log
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

# Reuse §H5-causal helpers verbatim. Functions that depend on module-level
# SIZE/STEP from the 410m source (select_top_suc / select_ctrl_set /
# select_top_si) are redefined locally below to avoid the closure-over-SIZE bug.
from notebooks._run_phase4_causal_410m_anchor import (  # noqa: E402
    ABLATE_K,
    BRACKET_WIDTH_INIT,
    BRACKET_WIDTH_STEP,
    B_BOOTSTRAP,
    DEP_THRESHOLD,
    NULL_BAND,
    SI_SENDER_K,
    TAU_LIFT,
    aggregate_verdict,
    bootstrap_drop_ratio,
    classify_per_sender,
    install_mean_ablation_hooks,
    precompute_mean_z_by_length,
    run_condition,
)
from notebooks._lib.sweep_io import read_long  # noqa: E402
from src.detectors.s_inhibition import build_abc_corrupted_prompts  # noqa: E402
from src.replication.tigges_ioi import load_ioi_prompts  # noqa: E402
from src.utils.mps_compat import assert_mps_fallback_enabled  # noqa: E402
from src.utils.pythia_loader import load_pythia  # noqa: E402

EXPL = REPO_ROOT / "data" / "exploration"
OUT_PER_PROMPT = EXPL / "phase4_causal_2_8b_anchor.parquet"
OUT_SUMMARY = EXPL / "phase4_causal_2_8b_anchor_summary.parquet"
OUT_VERDICT = EXPL / "phase4_causal_2_8b_anchor_verdict.parquet"

PROMPTS_PATH = REPO_ROOT / "data" / "prompts" / "ioi_prompts.tsv"
SIZE = "2.8b"
STEP = 143000

BATCH_SIZE = 10  # §H4-7 precedent for path-patching at d_model=2560

# §H5-causal-3-2.8b-2 / -4 / -5: locked sets, asserted at runtime.
LOCKED_SUC_SET = [(15, 14), (28, 17), (27, 13), (13, 10), (29, 28)]
LOCKED_SI_SENDERS = [(11, 29), (11, 5), (13, 9)]
LOCKED_NM_HEADS = [(11, 29), (17, 12), (22, 31), (13, 9)]


def select_top_suc_2_8b(df_suc: pd.DataFrame, k: int) -> list[tuple[int, int, float]]:
    """§H5-3 procedure at 2.8b step143000 (local; avoids 410m SIZE closure)."""
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
    """§H5-4 procedure + §H5-causal-3-7 NM-exclusion clause."""
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


def select_top_si_from_anchor(df_si_anchor: pd.DataFrame, k: int) -> list[tuple[int, int, float]]:
    """§H5-5 procedure at 2.8b step143000, reading from the §S-8 anchor parquet
    (long-form with metric=delta_h; not the same schema as a sweep parquet)."""
    sub = df_si_anchor[df_si_anchor["metric"] == "delta_h"].copy()
    sub = sub.rename(columns={"value": "score"})
    sub = sub.sort_values(by=["score", "layer", "head"], ascending=[False, True, True])
    rows = sub.head(k)
    return [(int(r["layer"]), int(r["head"]), float(r["score"])) for _, r in rows.iterrows()]


PAPER_HEADLINES = {
    "NULL": (
        "S-inhibition Δ_h is independent of successor heads at convergence in "
        "Pythia-2.8b. Replicates the §H5-causal 410m NULL on the §S-1 metric "
        "and extends the causal-disjointness claim to head-count tier 1024."
    ),
    "DEP": (
        "S-inhibition causally depends on successor at Pythia-2.8b convergence; "
        "the §H1-C ordering reflects an inference-time chain at the largest "
        "registered size. Substantial reframe of the 410m NULL story required."
    ),
    "GENERIC": (
        "Methodological note: Δ_h is not robust to ablation of any near-threshold "
        "head at this checkpoint; metric sensitivity is insufficient. Verdict "
        "deferred pending re-tooling."
    ),
    "MIXED": (
        "Per-sender heterogeneous dependence at 2.8b; no global verdict. "
        "Reported by sender."
    ),
}


def main() -> None:
    log_path = EXPL / "phase4_causal_2_8b_anchor.log"
    log_lines: list[str] = []

    def log(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        print(msg, **kwargs, flush=True)
        log_lines.append(msg)

    assert_mps_fallback_enabled()
    log(f"=== §H5-causal-3-2.8b anchor experiment, Pythia-{SIZE}-deduped step{STEP} (Metric A) ===")
    log("Pre-registration: HYPOTHESIS.md §H5-causal-3-2.8b (commit before this run).")
    t_start = time.time()

    # --- Read parquets to derive ablation sets per §H5-3 / §H5-4 / §H5-5 ---
    df_suc = read_long(EXPL / "phase4_2_8b_successor_sweep.parquet")
    df_si_anchor = pd.read_parquet(EXPL / "s_inhibition_pythia_2_8b_anchor.parquet")

    suc_set = select_top_suc_2_8b(df_suc, ABLATE_K)
    suc_layer_heads = [(l, h) for l, h, _ in suc_set]
    if suc_layer_heads != LOCKED_SUC_SET:
        raise RuntimeError(
            f"§H5-causal-3-2.8b-2 NO-CHERRY-PICKING ASSERTION FAILED: "
            f"re-derived suc set {suc_layer_heads} != locked {LOCKED_SUC_SET}. "
            f"Halting per §H2-8 — investigate parquet drift before proceeding."
        )
    log(f"\n§H5-causal-3-2.8b-2 suc set (locked, asserted match):")
    for l, h, s in suc_set:
        clears = "≥τ_lift" if s >= TAU_LIFT else "below τ_lift"
        log(f"  L{l}H{h}  score={s:.4f}  ({clears})")

    # --- Read pinned NMs from §S-8 anchor inspection (§H5-6 procedure) ---
    nm_npz = np.load(EXPL / "s_inhibition_pythia_2_8b_anchor_per_nm.npz")
    nm_heads_pinned: list[tuple[int, int]] = [(int(L), int(H)) for L, H in nm_npz["nm_heads"]]
    if nm_heads_pinned != LOCKED_NM_HEADS:
        raise RuntimeError(
            f"§H5-causal-3-2.8b-5 NO-CHERRY-PICKING ASSERTION FAILED: "
            f"re-read NMs {nm_heads_pinned} != locked {LOCKED_NM_HEADS}. "
            f"Halting per §H2-8 — investigate npz drift before proceeding."
        )
    log(f"\n§H5-causal-3-2.8b-5 pinned NMs (locked, asserted match):")
    for nl, nh in nm_heads_pinned:
        log(f"  L{nl}H{nh}")

    ctrl_set, widened_bw = select_ctrl_set_2_8b(
        df_suc, suc_set, nm_heads_pinned, ABLATE_K, BRACKET_WIDTH_INIT, BRACKET_WIDTH_STEP
    )
    log(
        f"\n§H5-causal-3-2.8b-3 ctrl set (procedure-locked; "
        f"bracket [{TAU_LIFT - widened_bw:.4f}, {TAU_LIFT:.4f}), "
        f"bracket_width={widened_bw:.3f}, seed=0, NM-excluded):"
    )
    for l, h, s in ctrl_set:
        log(f"  L{l}H{h}  score={s:.4f}")

    si_set = select_top_si_from_anchor(df_si_anchor, SI_SENDER_K)
    si_lh = [(l, h) for l, h, _ in si_set]
    if si_lh != LOCKED_SI_SENDERS:
        raise RuntimeError(
            f"§H5-causal-3-2.8b-4 NO-CHERRY-PICKING ASSERTION FAILED: "
            f"re-derived SI senders {si_lh} != locked {LOCKED_SI_SENDERS}. "
            f"Halting per §H2-8 — investigate anchor parquet drift before proceeding."
        )
    log(f"\n§H5-causal-3-2.8b-4 SI senders (locked, asserted match):")
    for l, h, s in si_set:
        log(f"  L{l}H{h}  Δ_h_clean_pre={s:.4f}")

    # --- §H5-causal-3-2.8b-6 structural caveat: log per-sender downstream NM count ---
    log("\n§H5-causal-3-2.8b-6 structural caveats (per-sender downstream NM count):")
    for sl, sh in si_lh:
        down = [(nl, nh) for nl, nh in nm_heads_pinned if nl > sl]
        log(f"  SI sender L{sl}H{sh} → {len(down)} downstream NMs: {down}")
    max_nm_layer = max(nl for nl, _ in nm_heads_pinned)
    log(f"  max(NM layer) = {max_nm_layer}")
    for sl, sh, _ in suc_set:
        loc = "structurally MUTE for Metric A" if sl > max_nm_layer else "Metric A sees this ablation"
        log(f"  suc L{sl}H{sh}: {loc}")

    # --- Load model ---
    log(f"\nLoading Pythia-{SIZE}-deduped @ step{STEP}...")
    t0 = time.time()
    model = load_pythia(SIZE, step=STEP)
    log(
        f"  loaded in {time.time() - t0:.1f}s; "
        f"n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads}"
    )

    clean = load_ioi_prompts(PROMPTS_PATH)
    corrupt = build_abc_corrupted_prompts(clean, model.tokenizer, seed=0)
    log(f"  N={len(clean)} clean+corrupt IOI prompts (GPT-NeoX BPE)")

    si_senders = [(l, h) for l, h, _ in si_set]
    suc_layer_heads_list = [(l, h) for l, h, _ in suc_set]
    ctrl_layer_heads = [(l, h) for l, h, _ in ctrl_set]

    # --- Condition 1: clean ---
    model.reset_hooks()
    log("\n=== Condition 1: clean (no ablation) ===")
    clean_per_prompt = run_condition(
        model, clean, corrupt, si_senders, nm_heads_pinned, "clean", log
    )

    # --- Precompute mean_z for suc and ctrl sets (using clean model) ---
    log("\nPrecomputing mean_z for suc set...")
    t0 = time.time()
    mean_z_suc = precompute_mean_z_by_length(model, clean, suc_layer_heads_list)
    log(f"  done in {time.time() - t0:.1f}s; {len(mean_z_suc)} (layer, head, T) entries")

    log("Precomputing mean_z for ctrl set...")
    t0 = time.time()
    mean_z_ctrl = precompute_mean_z_by_length(model, clean, ctrl_layer_heads)
    log(f"  done in {time.time() - t0:.1f}s; {len(mean_z_ctrl)} (layer, head, T) entries")

    # --- Condition 2: suc_ablated ---
    model.reset_hooks()
    install_mean_ablation_hooks(model, suc_layer_heads_list, mean_z_suc)
    log("\n=== Condition 2: suc_ablated ===")
    log(f"  perma hooks installed on {len(suc_layer_heads_list)} suc heads")
    suc_per_prompt = run_condition(
        model, clean, corrupt, si_senders, nm_heads_pinned, "suc_ablated", log
    )

    # --- Condition 3: ctrl_ablated ---
    model.reset_hooks()
    install_mean_ablation_hooks(model, ctrl_layer_heads, mean_z_ctrl)
    log("\n=== Condition 3: ctrl_ablated ===")
    log(f"  perma hooks installed on {len(ctrl_layer_heads)} ctrl heads")
    ctrl_per_prompt = run_condition(
        model, clean, corrupt, si_senders, nm_heads_pinned, "ctrl_ablated", log
    )
    model.reset_hooks()

    # --- Bootstrap and verdict ---
    log("\n=== Bootstrap (B=200 paired, seed=1) and per-sender classification ===")
    rng = np.random.default_rng(1)
    summary_rows: list[dict] = []
    per_prompt_rows: list[dict] = []
    per_sender_pattern: list[str] = []
    for sl, sh in si_senders:
        c = clean_per_prompt[(sl, sh)]
        a_suc = suc_per_prompt[(sl, sh)]
        a_ctrl = ctrl_per_prompt[(sl, sh)]
        for cond_label, arr in [("clean", c), ("suc_ablated", a_suc), ("ctrl_ablated", a_ctrl)]:
            for pi, val in enumerate(arr):
                per_prompt_rows.append(
                    dict(
                        condition=cond_label,
                        si_sender_layer=sl,
                        si_sender_head=sh,
                        prompt_idx=pi,
                        delta_h=float(val),
                    )
                )
        ratio_suc, suc_lo, suc_hi = bootstrap_drop_ratio(c, a_suc, B_BOOTSTRAP, rng)
        ratio_ctrl, ctrl_lo, ctrl_hi = bootstrap_drop_ratio(c, a_ctrl, B_BOOTSTRAP, rng)
        pattern = classify_per_sender(ratio_suc, ratio_ctrl)
        per_sender_pattern.append(pattern)
        log(
            f"  L{sl}H{sh}: clean Δ̄={c.mean():+.4f}  "
            f"suc_ablated Δ̄={a_suc.mean():+.4f} (ratio={ratio_suc:.3f}, CI=[{suc_lo:.3f},{suc_hi:.3f}])  "
            f"ctrl_ablated Δ̄={a_ctrl.mean():+.4f} (ratio={ratio_ctrl:.3f}, CI=[{ctrl_lo:.3f},{ctrl_hi:.3f}])  "
            f"→ {pattern}"
        )
        for cond_label, mean_val, ratio, lo, hi in [
            ("clean", float(c.mean()), 1.0, 1.0, 1.0),
            ("suc_ablated", float(a_suc.mean()), ratio_suc, suc_lo, suc_hi),
            ("ctrl_ablated", float(a_ctrl.mean()), ratio_ctrl, ctrl_lo, ctrl_hi),
        ]:
            summary_rows.append(
                dict(
                    condition=cond_label,
                    si_sender_layer=sl,
                    si_sender_head=sh,
                    delta_h_mean=mean_val,
                    drop_ratio_mean=ratio,
                    ratio_ci_low=lo,
                    ratio_ci_high=hi,
                    ablate_set_repr=(
                        ""
                        if cond_label == "clean"
                        else (
                            ",".join(f"L{l}H{h}" for l, h in suc_layer_heads_list)
                            if cond_label == "suc_ablated"
                            else ",".join(f"L{l}H{h}" for l, h in ctrl_layer_heads)
                        )
                    ),
                    widened_bracket_width=widened_bw if cond_label == "ctrl_ablated" else float("nan"),
                    n_prompts=int(c.shape[0]),
                )
            )

    # --- Aggregate verdict ---
    aggregate = aggregate_verdict(per_sender_pattern)
    headline = PAPER_HEADLINES[aggregate]
    log(f"\n=== §H5-causal-3-2.8b-7 aggregate verdict (Metric A) ===")
    log(f"per-sender patterns: {per_sender_pattern}")
    log(f"aggregate: {aggregate}")
    log(f"matched paper headline:")
    log(f"  {headline}")

    # --- Write parquets ---
    EXPL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_prompt_rows).to_parquet(OUT_PER_PROMPT, index=False)
    pd.DataFrame(summary_rows).to_parquet(OUT_SUMMARY, index=False)
    verdict_row = pd.DataFrame(
        [
            dict(
                pattern=aggregate,
                n_senders=len(per_sender_pattern),
                n_dep=per_sender_pattern.count("DEP"),
                n_null=per_sender_pattern.count("NULL"),
                n_generic=per_sender_pattern.count("GENERIC"),
                n_mixed=per_sender_pattern.count("MIXED"),
                per_sender_patterns=";".join(per_sender_pattern),
                widened_bracket_width=widened_bw,
                suc_set=",".join(f"L{l}H{h}" for l, h in suc_layer_heads_list),
                ctrl_set=",".join(f"L{l}H{h}" for l, h in ctrl_layer_heads),
                si_senders=",".join(f"L{l}H{h}" for l, h in si_senders),
                nm_heads=",".join(f"L{l}H{h}" for l, h in nm_heads_pinned),
                paper_headline=headline,
            )
        ]
    )
    verdict_row.to_parquet(OUT_VERDICT, index=False)
    log(f"\nWrote {OUT_PER_PROMPT.relative_to(REPO_ROOT)}")
    log(f"Wrote {OUT_SUMMARY.relative_to(REPO_ROOT)}")
    log(f"Wrote {OUT_VERDICT.relative_to(REPO_ROOT)}")

    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"\nWall: {(time.time() - t_start) / 60:.1f} min")
    print(f"Log: {log_path.relative_to(REPO_ROOT)}")

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
