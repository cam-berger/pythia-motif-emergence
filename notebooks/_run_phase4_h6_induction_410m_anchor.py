"""§H6-causal induction-ablation anchor at Pythia-410M step 143000 (pilot).

Pre-registered in HYPOTHESIS.md §H6-causal (committed before this run).
Mirrors `_run_phase4_h6_induction_70m_anchor.py` exactly; only the locked
sets and per-size parameters differ.

Locked sets (asserted bit-for-bit at runtime; see drafts/H6_locked_sets.md):
  - LOCKED_NM_HEADS: [(12,12), (17,10), (14,0), (20,15)] — §H5-causal lock
  - LOCKED_SI_SENDERS: [(12,12), (13,13), (14,0)] — §H5-causal lock
  - LOCKED_SUC_RECEIVERS: [(22,6), (22,2), (20,4), (22,10), (12,8)] — §H5-3 suc top-5
  - LOCKED_IND_SET: 7 heads (K_size=7 per §H6-causal-2 K-scaling rule), no exhaustion

Outputs:
  - phase4_h6_induction_410m_anchor_per_prompt.parquet
  - phase4_h6_induction_410m_anchor_summary.parquet
  - phase4_h6_induction_410m_anchor_verdict.parquet
  - phase4_h6_induction_410m_anchor.log
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

from notebooks._run_phase4_causal_410m_anchor import (  # noqa: E402
    B_BOOTSTRAP,
    SI_SENDER_K,
    TAU_STRICT,
    aggregate_verdict,
    bootstrap_drop_ratio,
    classify_per_sender,
    install_mean_ablation_hooks,
    precompute_mean_z_by_length,
    run_condition,
)
from notebooks._run_phase4_causal_410m_anchor_logitdiff import (  # noqa: E402
    classify_logitdiff,
    compute_logit_diff_per_prompt,
)
from notebooks._lib.h6_helpers import (  # noqa: E402
    HARD_K_MIN,
    INDUCTION_THRESHOLD,
    PAPER_HEADLINES,
    aggregate_cross_readout,
    classify_lift,
    compute_successor_lift_per_prompt,
    select_ctrl_induction,
    select_top_induction,
)
from notebooks._lib.sweep_io import read_long  # noqa: E402
from src.detectors.s_inhibition import build_abc_corrupted_prompts  # noqa: E402
from src.detectors.successor import build_successor_prompts  # noqa: E402
from src.replication.tigges_ioi import load_ioi_prompts  # noqa: E402
from src.utils.mps_compat import assert_mps_fallback_enabled  # noqa: E402
from src.utils.pythia_loader import load_pythia  # noqa: E402

EXPL = REPO_ROOT / "data" / "exploration"
OUT_PER_PROMPT = EXPL / "phase4_h6_induction_410m_anchor_per_prompt.parquet"
OUT_SUMMARY = EXPL / "phase4_h6_induction_410m_anchor_summary.parquet"
OUT_VERDICT = EXPL / "phase4_h6_induction_410m_anchor_verdict.parquet"
LOG_PATH = EXPL / "phase4_h6_induction_410m_anchor.log"
PROMPTS_PATH = REPO_ROOT / "data" / "prompts" / "ioi_prompts.tsv"

SIZE = "410m"
STEP = 143000

# §H6-causal-2 locks (per drafts/H6_locked_sets.md from §H5-causal + B4 derivation).
LOCKED_NM_HEADS = [(12, 12), (17, 10), (14, 0), (20, 15)]  # §H5-causal lock
LOCKED_SI_SENDERS = [(12, 12), (13, 13), (14, 0)]  # §H5-causal lock
LOCKED_SUC_RECEIVERS = [(22, 6), (22, 2), (20, 4), (22, 10), (12, 8)]  # §H5-3 suc top-5
LOCKED_IND_SET = [
    (11, 14), (11, 2), (7, 1), (8, 6), (10, 9), (10, 3), (8, 7),
]  # K=7 per K-scaling rule (ceil(0.33 * 19) = 7); clean — no exclusions at 410M
LOCKED_K_LOCKED = 7
LOCKED_K_EFFECTIVE = 7  # no exhaustion at 410M

# Pre-existing §S-8 410M anchor for NM verification
NM_NPZ_PATH_410M = EXPL / "s_inhibition_pythia_410m_anchor_per_nm.npz"


def main() -> None:
    log_lines: list[str] = []

    def log(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        print(msg, **kwargs, flush=True)
        log_lines.append(msg)

    assert_mps_fallback_enabled()
    log(f"=== §H6-causal anchor experiment, Pythia-{SIZE}-deduped step{STEP} ===")
    log("Pre-registration: HYPOTHESIS.md §H6-causal (locked before this run).")
    log(f"Pilot scope: 70M + 410M only. This run is the 410M pilot leg.")
    t_start = time.time()

    # --- Verify §H5-causal NMs (sealed) ---
    nm_npz = np.load(NM_NPZ_PATH_410M)
    nm_heads_pinned: list[tuple[int, int]] = [
        (int(L), int(H)) for L, H in nm_npz["nm_heads"]
    ]
    if nm_heads_pinned != LOCKED_NM_HEADS:
        raise RuntimeError(
            f"§H6-causal NM-lock drift at 410M: read {nm_heads_pinned}, locked "
            f"{LOCKED_NM_HEADS}. Halt per §H2-8 — investigate npz drift."
        )
    log(f"\n§H6-causal NM identity (inherited §H5-causal lock, asserted match): {nm_heads_pinned}")

    # --- Locked sets: re-derive and assert bit-for-bit ---
    df_ind = read_long(EXPL / "phase2_induction_sweep.parquet")
    df_suc = read_long(EXPL / "phase2_successor_sweep.parquet")
    df_si = read_long(EXPL / "phase2_s_inhibition_sweep.parquet")

    si_sub = df_si[(df_si["size"] == SIZE) & (df_si["step"] == STEP)].copy()
    si_sub = si_sub.sort_values(
        by=["score", "layer", "head"], ascending=[False, True, True]
    ).head(SI_SENDER_K)
    si_heads_derived = [
        (int(r["layer"]), int(r["head"])) for _, r in si_sub.iterrows()
    ]
    if si_heads_derived != LOCKED_SI_SENDERS:
        raise RuntimeError(
            f"§H6 SI senders drift at 410M: derived {si_heads_derived}, locked {LOCKED_SI_SENDERS}. "
            f"Halt per §H2-8."
        )
    log(f"\n§H6 SI senders (locked, asserted match):")
    for _, r in si_sub.iterrows():
        log(f"  L{int(r['layer'])}H{int(r['head'])}  Δ_h={float(r['score']):+.4f}")

    suc_sub = df_suc[(df_suc["size"] == SIZE) & (df_suc["step"] == STEP)].copy()
    suc_sub = suc_sub.sort_values(
        by=["score", "layer", "head"], ascending=[False, True, True]
    ).head(5)
    suc_recv_derived = [
        (int(r["layer"]), int(r["head"])) for _, r in suc_sub.iterrows()
    ]
    if suc_recv_derived != LOCKED_SUC_RECEIVERS:
        raise RuntimeError(
            f"§H6 suc receivers drift at 410M: derived {suc_recv_derived}, locked {LOCKED_SUC_RECEIVERS}. "
            f"Halt per §H2-8."
        )
    log(f"\n§H6 suc receivers (locked, asserted match):")
    for _, r in suc_sub.iterrows():
        log(f"  L{int(r['layer'])}H{int(r['head'])}  lift={float(r['score']):+.4f}")

    ind_set, widen_depth, excluded_log = select_top_induction(
        df_ind, SIZE, STEP, LOCKED_K_LOCKED,
        nm_set=LOCKED_NM_HEADS,
        si_set=LOCKED_SI_SENDERS,
        suc_receivers_set=LOCKED_SUC_RECEIVERS,
        hard_k_min=HARD_K_MIN,
    )
    ind_lh = [(l, h) for l, h, _ in ind_set]
    if ind_lh != LOCKED_IND_SET:
        raise RuntimeError(
            f"§H6-causal-2 ablation-set drift at 410M: derived {ind_lh}, locked {LOCKED_IND_SET}. "
            f"Halt per §H2-8."
        )
    k_effective = len(ind_set)
    structural_caveat_k_exhausted = k_effective < LOCKED_K_LOCKED
    log(f"\n§H6-causal-2 ablation set (K_locked={LOCKED_K_LOCKED}, K_effective={k_effective}, "
        f"widen_depth={widen_depth}; structural_caveat_k_exhausted={structural_caveat_k_exhausted}):")
    for l, h, s in ind_set:
        log(f"  L{l}H{h}  prefix_match={s:.4f}")
    if excluded_log:
        log(f"  Excluded heads ({len(excluded_log)}):")
        for (l, h), label, s in excluded_log:
            log(f"    L{l}H{h}  prefix_match={s:.4f}  reason={label}")

    ctrl_set, ctrl_bw = select_ctrl_induction(
        df_ind, SIZE, STEP,
        ind_set=ind_lh,
        nm_set=LOCKED_NM_HEADS,
        si_set=LOCKED_SI_SENDERS,
        suc_receivers_set=LOCKED_SUC_RECEIVERS,
        k=k_effective,
    )
    ctrl_lh = [(l, h) for l, h, _ in ctrl_set]
    log(f"\n§H6-causal-3 ctrl set (K={k_effective}, bracket_width={ctrl_bw:.3f}, seed=0):")
    for l, h, s in ctrl_set:
        log(f"  L{l}H{h}  prefix_match={s:.4f}")

    # --- Load model + 3 prompt sets ---
    log(f"\nLoading Pythia-{SIZE}-deduped @ step{STEP}...")
    t0 = time.time()
    model = load_pythia(SIZE, step=STEP)
    log(f"  loaded in {time.time() - t0:.1f}s; n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads}")

    clean_ioi = load_ioi_prompts(PROMPTS_PATH)
    corrupt_ioi = build_abc_corrupted_prompts(clean_ioi, model.tokenizer, seed=0)
    log(f"  N={len(clean_ioi)} IOI prompts (GPT-NeoX BPE) for Readouts B + C")

    suc_prompts = build_successor_prompts(model.tokenizer, seed=0)
    log(f"  N={len(suc_prompts)} successor prompts for Readout A")

    PER_CONDITION_BUDGET_MIN = 150  # §H4-7-style escape hatch

    # --- Condition 1: clean ---
    model.reset_hooks()
    log("\n=== Condition 1: clean (no ablation) ===")
    t_cond = time.time()
    clean_si = run_condition(model, clean_ioi, corrupt_ioi, LOCKED_SI_SENDERS, LOCKED_NM_HEADS, "clean", log)
    clean_logit_diff = compute_logit_diff_per_prompt(model, clean_ioi)
    clean_lift = compute_successor_lift_per_prompt(model, suc_prompts, LOCKED_SUC_RECEIVERS)
    log(f"  Condition 1 wall: {(time.time() - t_cond) / 60:.1f} min")
    if (time.time() - t_cond) / 60 > 2 * PER_CONDITION_BUDGET_MIN:
        log("§H4-7 escape hatch tripped on Condition 1. Halting.")
        return

    # --- Precompute mean_z (union of IOI + successor prompt lengths) ---
    log("\nPrecomputing mean_z for ind + ctrl sets (union of IOI + successor lengths)...")
    t0 = time.time()
    mean_z_ind = precompute_mean_z_by_length(model, clean_ioi, ind_lh)
    mean_z_ind_suc = precompute_mean_z_by_length(
        model,
        [type("P", (), {"text": p.clean_text})() for p in suc_prompts],
        ind_lh,
    )
    mean_z_ind.update(mean_z_ind_suc)
    mean_z_ctrl = precompute_mean_z_by_length(model, clean_ioi, ctrl_lh)
    mean_z_ctrl_suc = precompute_mean_z_by_length(
        model,
        [type("P", (), {"text": p.clean_text})() for p in suc_prompts],
        ctrl_lh,
    )
    mean_z_ctrl.update(mean_z_ctrl_suc)
    log(f"  done in {time.time() - t0:.1f}s; "
        f"|mean_z_ind|={len(mean_z_ind)}, |mean_z_ctrl|={len(mean_z_ctrl)}")

    # --- Condition 2: ind_ablated ---
    model.reset_hooks()
    install_mean_ablation_hooks(model, ind_lh, mean_z_ind)
    log("\n=== Condition 2: ind_ablated ===")
    t_cond = time.time()
    ind_si = run_condition(model, clean_ioi, corrupt_ioi, LOCKED_SI_SENDERS, LOCKED_NM_HEADS, "ind_ablated", log)
    ind_logit_diff = compute_logit_diff_per_prompt(model, clean_ioi)
    ind_lift = compute_successor_lift_per_prompt(model, suc_prompts, LOCKED_SUC_RECEIVERS)
    log(f"  Condition 2 wall: {(time.time() - t_cond) / 60:.1f} min")
    if (time.time() - t_cond) / 60 > 2 * PER_CONDITION_BUDGET_MIN:
        log("§H4-7 escape hatch tripped on Condition 2. Halting.")
        return

    # --- Condition 3: ctrl_ablated ---
    model.reset_hooks()
    install_mean_ablation_hooks(model, ctrl_lh, mean_z_ctrl)
    log("\n=== Condition 3: ctrl_ablated ===")
    t_cond = time.time()
    ctrl_si = run_condition(model, clean_ioi, corrupt_ioi, LOCKED_SI_SENDERS, LOCKED_NM_HEADS, "ctrl_ablated", log)
    ctrl_logit_diff = compute_logit_diff_per_prompt(model, clean_ioi)
    ctrl_lift = compute_successor_lift_per_prompt(model, suc_prompts, LOCKED_SUC_RECEIVERS)
    log(f"  Condition 3 wall: {(time.time() - t_cond) / 60:.1f} min")
    if (time.time() - t_cond) / 60 > 2 * PER_CONDITION_BUDGET_MIN:
        log("§H4-7 escape hatch tripped on Condition 3. Halting.")
        return
    model.reset_hooks()

    # --- Bootstrap + classify per readout ---
    log("\n=== Bootstrap (B=200 paired, seed=1) and classification ===")
    rng = np.random.default_rng(1)

    def _agg_lift(lift_dict):
        arr = np.stack([lift_dict[lh] for lh in LOCKED_SUC_RECEIVERS], axis=0)
        return arr.mean(axis=0)
    clean_lift_agg = _agg_lift(clean_lift)
    ind_lift_agg = _agg_lift(ind_lift)
    ctrl_lift_agg = _agg_lift(ctrl_lift)
    ratio_A_ind, A_ind_lo, A_ind_hi = bootstrap_drop_ratio(clean_lift_agg, ind_lift_agg, B_BOOTSTRAP, rng)
    ratio_A_ctrl, A_ctrl_lo, A_ctrl_hi = bootstrap_drop_ratio(clean_lift_agg, ctrl_lift_agg, B_BOOTSTRAP, rng)
    verdict_A = classify_lift(ratio_A_ind, A_ind_lo, A_ind_hi, ratio_A_ctrl, A_ctrl_lo, A_ctrl_hi)
    log(f"\nReadout A (successor lift):")
    log(f"  ratio_ind  = {ratio_A_ind:.4f}  CI=[{A_ind_lo:.4f}, {A_ind_hi:.4f}]")
    log(f"  ratio_ctrl = {ratio_A_ctrl:.4f}  CI=[{A_ctrl_lo:.4f}, {A_ctrl_hi:.4f}]")
    log(f"  verdict_A = {verdict_A}")

    per_sender_pattern: list[str] = []
    per_sender_log: list[dict] = []
    for (sl, sh) in LOCKED_SI_SENDERS:
        c = clean_si[(sl, sh)]
        a_ind = ind_si[(sl, sh)]
        a_ctrl = ctrl_si[(sl, sh)]
        r_ind, ind_lo, ind_hi = bootstrap_drop_ratio(c, a_ind, B_BOOTSTRAP, rng)
        r_ctrl, ctrl_lo, ctrl_hi = bootstrap_drop_ratio(c, a_ctrl, B_BOOTSTRAP, rng)
        pattern = classify_per_sender(r_ind, r_ctrl)
        per_sender_pattern.append(pattern)
        per_sender_log.append(dict(
            sl=sl, sh=sh,
            clean_mean=float(c.mean()),
            ind_mean=float(a_ind.mean()), ctrl_mean=float(a_ctrl.mean()),
            r_ind=r_ind, r_ind_lo=ind_lo, r_ind_hi=ind_hi,
            r_ctrl=r_ctrl, r_ctrl_lo=ctrl_lo, r_ctrl_hi=ctrl_hi,
            pattern=pattern,
        ))
        log(f"  Readout B L{sl}H{sh}: r_ind={r_ind:.3f} CI=[{ind_lo:.3f},{ind_hi:.3f}]  "
            f"r_ctrl={r_ctrl:.3f} CI=[{ctrl_lo:.3f},{ctrl_hi:.3f}] → {pattern}")
    verdict_B = aggregate_verdict(per_sender_pattern)
    log(f"  verdict_B (aggregate) = {verdict_B}")

    ratio_C_ind, C_ind_lo, C_ind_hi = bootstrap_drop_ratio(clean_logit_diff, ind_logit_diff, B_BOOTSTRAP, rng)
    ratio_C_ctrl, C_ctrl_lo, C_ctrl_hi = bootstrap_drop_ratio(clean_logit_diff, ctrl_logit_diff, B_BOOTSTRAP, rng)
    verdict_C = classify_logitdiff(ratio_C_ind, C_ind_lo, C_ind_hi, ratio_C_ctrl, C_ctrl_lo, C_ctrl_hi)
    log(f"\nReadout C (logit-diff):")
    log(f"  ratio_ind  = {ratio_C_ind:.4f}  CI=[{C_ind_lo:.4f}, {C_ind_hi:.4f}]")
    log(f"  ratio_ctrl = {ratio_C_ctrl:.4f}  CI=[{C_ctrl_lo:.4f}, {C_ctrl_hi:.4f}]")
    log(f"  verdict_C = {verdict_C}")

    aggregate = aggregate_cross_readout(verdict_A, verdict_B, verdict_C)
    headline = PAPER_HEADLINES[aggregate].replace("{size}", SIZE)

    log(f"\n=== §H6-causal-7-agg cross-readout aggregate: {aggregate} ===")
    log(f"  verdict_A={verdict_A}  verdict_B={verdict_B}  verdict_C={verdict_C}")
    log(f"  Paper headline:")
    log(f"    {headline}")

    # --- Write outputs ---
    EXPL.mkdir(parents=True, exist_ok=True)
    per_prompt_rows: list[dict] = []
    for cond_label, lift_dict in (
        ("clean", clean_lift), ("ind_ablated", ind_lift), ("ctrl_ablated", ctrl_lift),
    ):
        for (rl, rh) in LOCKED_SUC_RECEIVERS:
            arr = lift_dict[(rl, rh)]
            for pi, val in enumerate(arr):
                per_prompt_rows.append(dict(
                    condition=cond_label, readout="A_lift",
                    receiver_layer=int(rl), receiver_head=int(rh),
                    prompt_idx=int(pi), value=float(val),
                ))
    for cond_label, si_dict in (
        ("clean", clean_si), ("ind_ablated", ind_si), ("ctrl_ablated", ctrl_si),
    ):
        for (rl, rh) in LOCKED_SI_SENDERS:
            arr = si_dict[(rl, rh)]
            for pi, val in enumerate(arr):
                per_prompt_rows.append(dict(
                    condition=cond_label, readout="B_delta_h",
                    receiver_layer=int(rl), receiver_head=int(rh),
                    prompt_idx=int(pi), value=float(val),
                ))
    for cond_label, ld_arr in (
        ("clean", clean_logit_diff), ("ind_ablated", ind_logit_diff), ("ctrl_ablated", ctrl_logit_diff),
    ):
        for pi, val in enumerate(ld_arr):
            per_prompt_rows.append(dict(
                condition=cond_label, readout="C_logit_diff",
                receiver_layer=-1, receiver_head=-1,
                prompt_idx=int(pi), value=float(val),
            ))
    pd.DataFrame(per_prompt_rows).to_parquet(OUT_PER_PROMPT, index=False)

    summary_rows: list[dict] = []
    for cond_label, r_A, A_lo, A_hi in (
        ("clean", 1.0, 1.0, 1.0),
        ("ind_ablated", ratio_A_ind, A_ind_lo, A_ind_hi),
        ("ctrl_ablated", ratio_A_ctrl, A_ctrl_lo, A_ctrl_hi),
    ):
        summary_rows.append(dict(
            condition=cond_label, readout="A_lift",
            ratio_mean=r_A, ratio_ci_low=A_lo, ratio_ci_high=A_hi,
        ))
    for row in per_sender_log:
        for cond_label, r, lo, hi in (
            ("clean", 1.0, 1.0, 1.0),
            ("ind_ablated", row["r_ind"], row["r_ind_lo"], row["r_ind_hi"]),
            ("ctrl_ablated", row["r_ctrl"], row["r_ctrl_lo"], row["r_ctrl_hi"]),
        ):
            summary_rows.append(dict(
                condition=cond_label, readout="B_delta_h",
                receiver_layer=row["sl"], receiver_head=row["sh"],
                ratio_mean=r, ratio_ci_low=lo, ratio_ci_high=hi,
            ))
    for cond_label, r_C, C_lo, C_hi in (
        ("clean", 1.0, 1.0, 1.0),
        ("ind_ablated", ratio_C_ind, C_ind_lo, C_ind_hi),
        ("ctrl_ablated", ratio_C_ctrl, C_ctrl_lo, C_ctrl_hi),
    ):
        summary_rows.append(dict(
            condition=cond_label, readout="C_logit_diff",
            ratio_mean=r_C, ratio_ci_low=C_lo, ratio_ci_high=C_hi,
        ))
    pd.DataFrame(summary_rows).to_parquet(OUT_SUMMARY, index=False)

    verdict_row = dict(
        size=SIZE, step=STEP,
        verdict_A=verdict_A, verdict_B=verdict_B, verdict_C=verdict_C,
        aggregate=aggregate,
        ratio_A_ind=ratio_A_ind, ratio_A_ind_lo=A_ind_lo, ratio_A_ind_hi=A_ind_hi,
        ratio_A_ctrl=ratio_A_ctrl, ratio_A_ctrl_lo=A_ctrl_lo, ratio_A_ctrl_hi=A_ctrl_hi,
        ratio_B_per_sender=";".join(
            f"L{r['sl']}H{r['sh']}:{r['r_ind']:.3f}|{r['r_ctrl']:.3f}" for r in per_sender_log
        ),
        per_sender_patterns=";".join(per_sender_pattern),
        ratio_C_ind=ratio_C_ind, ratio_C_ind_lo=C_ind_lo, ratio_C_ind_hi=C_ind_hi,
        ratio_C_ctrl=ratio_C_ctrl, ratio_C_ctrl_lo=C_ctrl_lo, ratio_C_ctrl_hi=C_ctrl_hi,
        ind_set=",".join(f"L{l}H{h}" for l, h in ind_lh),
        ctrl_set=",".join(f"L{l}H{h}" for l, h in ctrl_lh),
        si_senders=",".join(f"L{l}H{h}" for l, h in LOCKED_SI_SENDERS),
        nm_heads=",".join(f"L{l}H{h}" for l, h in LOCKED_NM_HEADS),
        suc_receivers=",".join(f"L{l}H{h}" for l, h in LOCKED_SUC_RECEIVERS),
        K_locked=LOCKED_K_LOCKED, K_effective=k_effective,
        widen_depth=widen_depth,
        ctrl_bracket_width=ctrl_bw,
        structural_caveat_k_exhausted=structural_caveat_k_exhausted,
        structural_caveat_70m_weak_si=False,
        n_excluded_total=len(excluded_log),
        paper_headline=headline,
    )
    pd.DataFrame([verdict_row]).to_parquet(OUT_VERDICT, index=False)

    log(f"\nWrote {OUT_PER_PROMPT.relative_to(REPO_ROOT)}")
    log(f"Wrote {OUT_SUMMARY.relative_to(REPO_ROOT)}")
    log(f"Wrote {OUT_VERDICT.relative_to(REPO_ROOT)}")

    LOG_PATH.write_text("\n".join(log_lines) + "\n")
    print(f"\nWall: {(time.time() - t_start) / 60:.1f} min")
    print(f"Log: {LOG_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
