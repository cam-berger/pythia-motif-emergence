"""§H6-causal-ksweep — induction K-dose-response at Pythia-410M step 143000.

Pre-registered in HYPOTHESIS.md §H6-causal-ksweep (committed before this run).
Loops K ∈ {5, 10, 15, 19} (locked); for each K, derives the induction
ablation set (with §H6-2 + §H6-2-bis 3-way exclusion) and a matched ctrl
set, then runs the same three readouts (suc lift / §S-1 Δ_h / IO−S
logit-diff) used by the §H6 pilot.

Optimisation: model is loaded once and clean condition is run once;
ablated conditions (ind / ctrl) loop per K. Total: 1 clean + 2 × 4 = 9
ablated condition-equivalents.

Outputs:
  - phase4_h6_induction_410m_ksweep.parquet            per-(K, cond, readout, prompt) values
  - phase4_h6_induction_410m_ksweep_verdict.parquet    per-K verdict table
  - phase4_h6_induction_410m_ksweep.log                captured stdout
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

from notebooks._run_phase4_causal_410m_anchor import (  # noqa: E402
    B_BOOTSTRAP,
    SI_SENDER_K,
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
OUT_PER_PROMPT = EXPL / "phase4_h6_induction_410m_ksweep.parquet"
OUT_VERDICT = EXPL / "phase4_h6_induction_410m_ksweep_verdict.parquet"
LOG_PATH = EXPL / "phase4_h6_induction_410m_ksweep.log"
PROMPTS_PATH = REPO_ROOT / "data" / "prompts" / "ioi_prompts.tsv"

SIZE = "410m"
STEP = 143000

# §H6-causal-ksweep-2: K values locked
K_VALUES = [5, 10, 15, 19]

# §H6-causal-2 / §H5-causal locked receiver sets (inherited verbatim)
LOCKED_NM_HEADS = [(12, 12), (17, 10), (14, 0), (20, 15)]
LOCKED_SI_SENDERS = [(12, 12), (13, 13), (14, 0)]
LOCKED_SUC_RECEIVERS = [(22, 6), (22, 2), (20, 4), (22, 10), (12, 8)]


def main() -> None:
    log_lines: list[str] = []

    def log(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        print(msg, **kwargs, flush=True)
        log_lines.append(msg)

    assert_mps_fallback_enabled()
    log("=== §H6-causal-ksweep — Pythia-410M induction K-dose-response ===")
    log(f"K values: {K_VALUES}")
    t_start = time.time()

    df_ind = read_long(EXPL / "phase2_induction_sweep.parquet")

    # Pre-derive each K's ind + ctrl sets and assert they're internally consistent
    log("\n--- Pre-deriving K-specific ablation + ctrl sets ---")
    per_K = {}
    for K in K_VALUES:
        ind_set, widen, excluded = select_top_induction(
            df_ind, SIZE, STEP, K,
            nm_set=LOCKED_NM_HEADS, si_set=LOCKED_SI_SENDERS,
            suc_receivers_set=LOCKED_SUC_RECEIVERS,
            hard_k_min=HARD_K_MIN,
        )
        ind_lh = [(l, h) for l, h, _ in ind_set]
        ctrl_set, ctrl_bw = select_ctrl_induction(
            df_ind, SIZE, STEP,
            ind_set=ind_lh,
            nm_set=LOCKED_NM_HEADS, si_set=LOCKED_SI_SENDERS,
            suc_receivers_set=LOCKED_SUC_RECEIVERS,
            k=K,
        )
        ctrl_lh = [(l, h) for l, h, _ in ctrl_set]
        per_K[K] = dict(ind_set=ind_set, ind_lh=ind_lh,
                         ctrl_set=ctrl_set, ctrl_lh=ctrl_lh,
                         ctrl_bw=ctrl_bw, widen=widen,
                         excluded=excluded)
        log(f"\nK={K}:")
        log(f"  ind  set ({len(ind_set)}): {', '.join(f'L{l}H{h}' for l, h, _ in ind_set)}")
        log(f"  ctrl set ({len(ctrl_set)}, bw={ctrl_bw:.3f}): {', '.join(f'L{l}H{h}' for l, h, _ in ctrl_set)}")
        if excluded:
            log(f"  excluded ({len(excluded)}): {excluded}")

    # Load model + prompts once
    log(f"\n--- Loading Pythia-{SIZE}-deduped @ step{STEP} ---")
    t0 = time.time()
    model = load_pythia(SIZE, step=STEP)
    log(f"  loaded in {time.time() - t0:.1f}s")

    clean_ioi = load_ioi_prompts(PROMPTS_PATH)
    corrupt_ioi = build_abc_corrupted_prompts(clean_ioi, model.tokenizer, seed=0)
    suc_prompts = build_successor_prompts(model.tokenizer, seed=0)
    log(f"  N_ioi={len(clean_ioi)}, N_suc={len(suc_prompts)}")

    # Run clean condition once
    log("\n=== Clean condition (shared across all K) ===")
    t_cond = time.time()
    model.reset_hooks()
    clean_si = run_condition(model, clean_ioi, corrupt_ioi, LOCKED_SI_SENDERS, LOCKED_NM_HEADS, "clean", log)
    clean_logit_diff = compute_logit_diff_per_prompt(model, clean_ioi)
    clean_lift = compute_successor_lift_per_prompt(model, suc_prompts, LOCKED_SUC_RECEIVERS)
    log(f"  clean wall: {(time.time() - t_cond) / 60:.1f} min")

    # Loop over K
    per_prompt_rows: list[dict] = []
    verdict_rows: list[dict] = []

    # Helper to dump per_prompt rows
    def dump_rows(K: int, cond: str, lift_dict, si_dict, ld_arr):
        for (rl, rh) in LOCKED_SUC_RECEIVERS:
            arr = lift_dict[(rl, rh)]
            for pi, v in enumerate(arr):
                per_prompt_rows.append(dict(
                    K=K, condition=cond, readout="A_lift",
                    receiver_layer=int(rl), receiver_head=int(rh),
                    prompt_idx=int(pi), value=float(v),
                ))
        for (rl, rh) in LOCKED_SI_SENDERS:
            arr = si_dict[(rl, rh)]
            for pi, v in enumerate(arr):
                per_prompt_rows.append(dict(
                    K=K, condition=cond, readout="B_delta_h",
                    receiver_layer=int(rl), receiver_head=int(rh),
                    prompt_idx=int(pi), value=float(v),
                ))
        for pi, v in enumerate(ld_arr):
            per_prompt_rows.append(dict(
                K=K, condition=cond, readout="C_logit_diff",
                receiver_layer=-1, receiver_head=-1,
                prompt_idx=int(pi), value=float(v),
            ))

    # Write clean rows once per K=anchor so the parquet has a complete picture
    # (clean is identical across K; for plot reconstruction we replicate K=0 sentinel)
    dump_rows(0, "clean", clean_lift, clean_si, clean_logit_diff)

    PER_CONDITION_BUDGET_MIN = 150

    for K in K_VALUES:
        log(f"\n========== K = {K} ==========")
        K_t0 = time.time()
        ind_lh = per_K[K]["ind_lh"]
        ctrl_lh = per_K[K]["ctrl_lh"]

        # Precompute mean_z (union of IOI + successor lengths)
        log(f"\n[K={K}] Precomputing mean_z for ind + ctrl sets...")
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
        log(f"  done in {time.time() - t0:.1f}s")

        # ind_ablated
        log(f"\n[K={K}] Condition: ind_ablated")
        t_cond = time.time()
        model.reset_hooks()
        install_mean_ablation_hooks(model, ind_lh, mean_z_ind)
        ind_si = run_condition(model, clean_ioi, corrupt_ioi, LOCKED_SI_SENDERS, LOCKED_NM_HEADS, f"K{K}_ind", log)
        ind_logit_diff = compute_logit_diff_per_prompt(model, clean_ioi)
        ind_lift = compute_successor_lift_per_prompt(model, suc_prompts, LOCKED_SUC_RECEIVERS)
        log(f"  ind_ablated wall: {(time.time() - t_cond) / 60:.1f} min")
        if (time.time() - t_cond) / 60 > 2 * PER_CONDITION_BUDGET_MIN:
            log(f"§H4-7 escape hatch tripped on K={K} ind_ablated. Halting.")
            return

        # ctrl_ablated
        log(f"\n[K={K}] Condition: ctrl_ablated")
        t_cond = time.time()
        model.reset_hooks()
        install_mean_ablation_hooks(model, ctrl_lh, mean_z_ctrl)
        ctrl_si = run_condition(model, clean_ioi, corrupt_ioi, LOCKED_SI_SENDERS, LOCKED_NM_HEADS, f"K{K}_ctrl", log)
        ctrl_logit_diff = compute_logit_diff_per_prompt(model, clean_ioi)
        ctrl_lift = compute_successor_lift_per_prompt(model, suc_prompts, LOCKED_SUC_RECEIVERS)
        log(f"  ctrl_ablated wall: {(time.time() - t_cond) / 60:.1f} min")
        model.reset_hooks()

        dump_rows(K, "ind_ablated", ind_lift, ind_si, ind_logit_diff)
        dump_rows(K, "ctrl_ablated", ctrl_lift, ctrl_si, ctrl_logit_diff)

        # Bootstrap + classify per readout
        log(f"\n[K={K}] Bootstrap + classify")
        rng = np.random.default_rng(1)

        def _agg_lift(d):
            return np.stack([d[lh] for lh in LOCKED_SUC_RECEIVERS], axis=0).mean(axis=0)
        clean_A = _agg_lift(clean_lift)
        ind_A = _agg_lift(ind_lift)
        ctrl_A = _agg_lift(ctrl_lift)
        r_A_ind, A_ind_lo, A_ind_hi = bootstrap_drop_ratio(clean_A, ind_A, B_BOOTSTRAP, rng)
        r_A_ctrl, A_ctrl_lo, A_ctrl_hi = bootstrap_drop_ratio(clean_A, ctrl_A, B_BOOTSTRAP, rng)
        verdict_A = classify_lift(r_A_ind, A_ind_lo, A_ind_hi, r_A_ctrl, A_ctrl_lo, A_ctrl_hi)

        per_sender_pattern = []
        per_sender_ratios = []
        for (sl, sh) in LOCKED_SI_SENDERS:
            c = clean_si[(sl, sh)]
            a_ind = ind_si[(sl, sh)]
            a_ctrl = ctrl_si[(sl, sh)]
            r_ind, ind_lo, ind_hi = bootstrap_drop_ratio(c, a_ind, B_BOOTSTRAP, rng)
            r_ctrl, ctrl_lo, ctrl_hi = bootstrap_drop_ratio(c, a_ctrl, B_BOOTSTRAP, rng)
            pat = classify_per_sender(r_ind, r_ctrl)
            per_sender_pattern.append(pat)
            per_sender_ratios.append(f"L{sl}H{sh}:{r_ind:.3f}|{r_ctrl:.3f}")
        verdict_B = aggregate_verdict(per_sender_pattern)

        r_C_ind, C_ind_lo, C_ind_hi = bootstrap_drop_ratio(clean_logit_diff, ind_logit_diff, B_BOOTSTRAP, rng)
        r_C_ctrl, C_ctrl_lo, C_ctrl_hi = bootstrap_drop_ratio(clean_logit_diff, ctrl_logit_diff, B_BOOTSTRAP, rng)
        verdict_C = classify_logitdiff(r_C_ind, C_ind_lo, C_ind_hi, r_C_ctrl, C_ctrl_lo, C_ctrl_hi)

        aggregate = aggregate_cross_readout(verdict_A, verdict_B, verdict_C)

        log(f"\n[K={K}] Per-readout verdicts:")
        log(f"  A: {verdict_A}  ratio_ind={r_A_ind:.4f} [{A_ind_lo:.4f},{A_ind_hi:.4f}]  "
            f"ratio_ctrl={r_A_ctrl:.4f} [{A_ctrl_lo:.4f},{A_ctrl_hi:.4f}]")
        log(f"  B: {verdict_B}  per-sender: {per_sender_pattern}")
        log(f"  C: {verdict_C}  ratio_ind={r_C_ind:.4f} [{C_ind_lo:.4f},{C_ind_hi:.4f}]  "
            f"ratio_ctrl={r_C_ctrl:.4f} [{C_ctrl_lo:.4f},{C_ctrl_hi:.4f}]")
        log(f"  aggregate: {aggregate}")

        verdict_rows.append(dict(
            K_target=K,
            K_effective=len(ind_lh),
            n_induction_pop=19,
            pct_ablated=100.0 * len(ind_lh) / 19,
            structural_caveat_k_exhausted=(len(ind_lh) < K),
            verdict_A=verdict_A, verdict_B=verdict_B, verdict_C=verdict_C,
            aggregate=aggregate,
            ratio_A_ind=r_A_ind, ratio_A_ind_lo=A_ind_lo, ratio_A_ind_hi=A_ind_hi,
            ratio_A_ctrl=r_A_ctrl, ratio_A_ctrl_lo=A_ctrl_lo, ratio_A_ctrl_hi=A_ctrl_hi,
            ratio_B_per_sender=";".join(per_sender_ratios),
            per_sender_patterns=";".join(per_sender_pattern),
            ratio_C_ind=r_C_ind, ratio_C_ind_lo=C_ind_lo, ratio_C_ind_hi=C_ind_hi,
            ratio_C_ctrl=r_C_ctrl, ratio_C_ctrl_lo=C_ctrl_lo, ratio_C_ctrl_hi=C_ctrl_hi,
            ind_set=",".join(f"L{l}H{h}" for l, h in ind_lh),
            ctrl_set=",".join(f"L{l}H{h}" for l, h in ctrl_lh),
            ctrl_bracket_width=per_K[K]["ctrl_bw"],
            widen_depth=per_K[K]["widen"],
        ))

        log(f"[K={K}] total wall: {(time.time() - K_t0) / 60:.1f} min")

    # Write outputs
    EXPL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_prompt_rows).to_parquet(OUT_PER_PROMPT, index=False)
    pd.DataFrame(verdict_rows).to_parquet(OUT_VERDICT, index=False)
    log(f"\nWrote {OUT_PER_PROMPT.relative_to(REPO_ROOT)}")
    log(f"Wrote {OUT_VERDICT.relative_to(REPO_ROOT)}")

    LOG_PATH.write_text("\n".join(log_lines) + "\n")
    print(f"\nTotal wall: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
