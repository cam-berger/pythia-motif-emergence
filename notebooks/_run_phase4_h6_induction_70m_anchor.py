"""§H6-causal induction-ablation anchor at Pythia-70M step 143000 (pilot).

Pre-registered in HYPOTHESIS.md §H6-causal (committed before this run).
Tests whether mean-ablating the top induction heads at convergence disrupts
three downstream readouts:
  - Readout A: successor lift_dla on the locked top-5 suc receivers
  - Readout B: §S-1 path-patching Δ_h on the locked top-3 SI senders
  - Readout C: IO−S logit-diff at END

§H6-causal-2 + §H6-causal-2-bis 3-way exclusion (NM + SI senders + suc
receivers) at 70M exhausts the 6-head induction-detected population to 4
surviving heads. Per user Decision (a), runner proceeds at K_effective=4
with `structural_caveat_k_exhausted=True` flag in the verdict parquet.

Locked sets (asserted bit-for-bit at runtime; see drafts/H6_locked_sets.md):
  - LOCKED_NM_HEADS: read from s_inhibition_pythia_70m_anchor_per_nm.npz
    (§H6-causal-6 Strategy-B prerequisite). Expected: [(4,2), (5,5), (4,7), (0,3)]
  - LOCKED_SI_SENDERS: [(4,2), (4,0), (3,5)]
  - LOCKED_SUC_RECEIVERS: [(4,0), (4,1), (2,7), (0,4), (0,6)]
  - LOCKED_IND_SET: [(3,1), (3,3), (3,6), (3,0)] — 4 surviving after 3-way exclusion
  - LOCKED_CTRL_SET: derived at runtime per §H6-causal-3 procedure with seed=0

Outputs (under data/exploration/, *.parquet gitignored):
  - phase4_h6_induction_70m_anchor_per_prompt.parquet
  - phase4_h6_induction_70m_anchor_summary.parquet
  - phase4_h6_induction_70m_anchor_verdict.parquet
  - phase4_h6_induction_70m_anchor.log
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

# Reuse §H5-causal helpers (model-independent infra).
from notebooks._run_phase4_causal_410m_anchor import (  # noqa: E402
    B_BOOTSTRAP,
    DEP_THRESHOLD as DEP_THRESHOLD_SI,
    NULL_BAND as NULL_BAND_SI,
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
OUT_PER_PROMPT = EXPL / "phase4_h6_induction_70m_anchor_per_prompt.parquet"
OUT_SUMMARY = EXPL / "phase4_h6_induction_70m_anchor_summary.parquet"
OUT_VERDICT = EXPL / "phase4_h6_induction_70m_anchor_verdict.parquet"
LOG_PATH = EXPL / "phase4_h6_induction_70m_anchor.log"
PROMPTS_PATH = REPO_ROOT / "data" / "prompts" / "ioi_prompts.tsv"

SIZE = "70m"
STEP = 143000

# §H6-causal-2 locks (per drafts/H6_locked_sets.md, derived from sealed parquets).
LOCKED_NM_HEADS = [(4, 2), (5, 5), (4, 7), (0, 3)]  # from §S-8 70M anchor (Strategy B prerequisite)
LOCKED_SI_SENDERS = [(4, 2), (4, 0), (3, 5)]  # top-3 Δ_h at 70m step143000
LOCKED_SUC_RECEIVERS = [(4, 0), (4, 1), (2, 7), (0, 4), (0, 6)]  # top-5 lift_dla at 70m step143000
LOCKED_IND_SET = [(3, 1), (3, 3), (3, 6), (3, 0)]  # 4 surviving after 3-way exclusion
LOCKED_K_LOCKED = 5
LOCKED_K_EFFECTIVE = 4  # K_locked=5 floor not reachable; §H6-causal-2 Decision (a)


def main() -> None:
    log_lines: list[str] = []

    def log(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        print(msg, **kwargs, flush=True)
        log_lines.append(msg)

    assert_mps_fallback_enabled()
    log(f"=== §H6-causal anchor experiment, Pythia-{SIZE}-deduped step{STEP} ===")
    log("Pre-registration: HYPOTHESIS.md §H6-causal (locked before this run).")
    log(f"Pilot scope: 70M + 410M only. This run is the 70M pilot leg.")
    t_start = time.time()

    # --- Strategy-B prerequisite check ---
    nm_npz_path = EXPL / "s_inhibition_pythia_70m_anchor_per_nm.npz"
    if not nm_npz_path.exists():
        raise RuntimeError(
            f"§H6-causal-6 Strategy-B prerequisite missing: {nm_npz_path}. "
            f"Run notebooks/_run_pythia_70m_anchor_s_inhibition.py first."
        )
    nm_npz = np.load(nm_npz_path)
    nm_heads_pinned: list[tuple[int, int]] = [
        (int(L), int(H)) for L, H in nm_npz["nm_heads"]
    ]
    if nm_heads_pinned != LOCKED_NM_HEADS:
        raise RuntimeError(
            f"§H6-causal-6 NM-lock drift: read {nm_heads_pinned} from npz, "
            f"locked {LOCKED_NM_HEADS}. Re-run §S-8 70M anchor; investigate "
            f"per §H2-8."
        )
    log(f"\n§H6-causal-6 NM identity (locked, asserted match): {nm_heads_pinned}")

    # --- Locked sets: re-derive and assert bit-for-bit ---
    df_ind = read_long(EXPL / "phase2_induction_sweep.parquet")
    df_suc = read_long(EXPL / "phase2_successor_sweep.parquet")
    df_si = read_long(EXPL / "phase2_s_inhibition_sweep.parquet")

    # SI senders top-3 (re-derive and assert)
    si_sub = df_si[(df_si["size"] == SIZE) & (df_si["step"] == STEP)].copy()
    si_sub = si_sub.sort_values(
        by=["score", "layer", "head"], ascending=[False, True, True]
    ).head(SI_SENDER_K)
    si_heads_derived = [
        (int(r["layer"]), int(r["head"])) for _, r in si_sub.iterrows()
    ]
    if si_heads_derived != LOCKED_SI_SENDERS:
        raise RuntimeError(
            f"§H6 SI senders drift: derived {si_heads_derived}, locked {LOCKED_SI_SENDERS}. "
            f"Halt per §H2-8."
        )
    log(f"\n§H6 SI senders (locked, asserted match):")
    for _, r in si_sub.iterrows():
        log(f"  L{int(r['layer'])}H{int(r['head'])}  Δ_h={float(r['score']):+.4f}")

    # Suc receivers top-5 (re-derive and assert)
    suc_sub = df_suc[(df_suc["size"] == SIZE) & (df_suc["step"] == STEP)].copy()
    suc_sub = suc_sub.sort_values(
        by=["score", "layer", "head"], ascending=[False, True, True]
    ).head(5)
    suc_recv_derived = [
        (int(r["layer"]), int(r["head"])) for _, r in suc_sub.iterrows()
    ]
    if suc_recv_derived != LOCKED_SUC_RECEIVERS:
        raise RuntimeError(
            f"§H6 suc receivers drift: derived {suc_recv_derived}, locked {LOCKED_SUC_RECEIVERS}. "
            f"Halt per §H2-8."
        )
    log(f"\n§H6 suc receivers — Readout A targets (locked, asserted match):")
    for _, r in suc_sub.iterrows():
        log(f"  L{int(r['layer'])}H{int(r['head'])}  lift={float(r['score']):+.4f}")

    # Induction ablation set with 3-way exclusion
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
            f"§H6-causal-2 ablation-set drift: derived {ind_lh}, locked {LOCKED_IND_SET}. "
            f"Halt per §H2-8."
        )
    k_effective = len(ind_set)
    structural_caveat_k_exhausted = k_effective < LOCKED_K_LOCKED
    log(f"\n§H6-causal-2 ablation set (K_locked={LOCKED_K_LOCKED}, K_effective={k_effective}, "
        f"widen_depth={widen_depth}; structural_caveat_k_exhausted={structural_caveat_k_exhausted}):")
    for l, h, s in ind_set:
        log(f"  L{l}H{h}  prefix_match={s:.4f}")
    log(f"  Excluded heads ({len(excluded_log)}):")
    for (l, h), label, s in excluded_log:
        log(f"    L{l}H{h}  prefix_match={s:.4f}  reason={label}")

    # Ctrl set: bracket-widening per §H6-causal-3
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

    # --- 70M weak-SI structural-caveat preview (clean SI Δ_h on SI senders) ---
    log(f"\nLocked SI senders' clean-sweep Δ_h (pre-flight diagnostic):")
    for sl, sh in LOCKED_SI_SENDERS:
        sweep_val = float(df_si[(df_si["size"] == SIZE) & (df_si["step"] == STEP)
                                & (df_si["layer"] == sl) & (df_si["head"] == sh)]["score"].iloc[0])
        below_tau = abs(sweep_val) < TAU_STRICT * 0.5
        log(f"  L{sl}H{sh}: Δ_h={sweep_val:+.4f}{' (BELOW 0.5×τ_strict — Readout B noise-floor risk)' if below_tau else ''}")
    structural_caveat_70m_weak_si = any(
        abs(float(df_si[(df_si["size"] == SIZE) & (df_si["step"] == STEP)
                        & (df_si["layer"] == sl) & (df_si["head"] == sh)]["score"].iloc[0])) < TAU_STRICT * 0.5
        for sl, sh in LOCKED_SI_SENDERS
    )
    if structural_caveat_70m_weak_si:
        log("STRUCTURAL CAVEAT: at least one SI sender at 70M is below 0.5×τ_strict in clean sweep. "
            "Readout B is expected to produce GENERIC/MIXED verdict from numerical noise, not from "
            "genuine ablation insensitivity. Headline weighting prioritizes Readouts A and C.")

    # --- Load model + all 3 prompt sets ---
    log(f"\nLoading Pythia-{SIZE}-deduped @ step{STEP}...")
    t0 = time.time()
    model = load_pythia(SIZE, step=STEP)
    log(f"  loaded in {time.time() - t0:.1f}s; n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads}")

    clean_ioi = load_ioi_prompts(PROMPTS_PATH)
    corrupt_ioi = build_abc_corrupted_prompts(clean_ioi, model.tokenizer, seed=0)
    log(f"  N={len(clean_ioi)} IOI prompts (GPT-NeoX BPE) for Readouts B + C")

    suc_prompts = build_successor_prompts(model.tokenizer, seed=0)
    log(f"  N={len(suc_prompts)} successor prompts for Readout A")

    # --- Condition 1: clean ---
    model.reset_hooks()
    log("\n=== Condition 1: clean (no ablation) ===")
    t_cond = time.time()
    clean_si = run_condition(model, clean_ioi, corrupt_ioi, LOCKED_SI_SENDERS, LOCKED_NM_HEADS, "clean", log)
    clean_logit_diff = compute_logit_diff_per_prompt(model, clean_ioi)
    clean_lift = compute_successor_lift_per_prompt(model, suc_prompts, LOCKED_SUC_RECEIVERS)
    log(f"  Condition 1 wall: {time.time() - t_cond:.1f}s")

    # --- Precompute mean_z (union of IOI + successor prompt lengths) ---
    log("\nPrecomputing mean_z for ind + ctrl sets (union of IOI + successor lengths)...")
    t0 = time.time()
    # We need mean_z keyed by length over the *union* of prompt-length sets the
    # ablation will see. The hook reads `hook_z[B, T, head, :]` — we need an entry
    # for every T appearing in either prompt set.
    # We use IOI prompts as the primary input (path-patching uses them); we extend
    # the cache by computing mean_z on successor prompts separately and merging.
    mean_z_ind = precompute_mean_z_by_length(model, clean_ioi, ind_lh)
    mean_z_ind_suc = precompute_mean_z_by_length(
        model,
        # Wrap successor prompts with a `.text` attribute so the helper can call to_tokens
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
    log(f"  Condition 2 wall: {time.time() - t_cond:.1f}s")

    # --- Condition 3: ctrl_ablated ---
    model.reset_hooks()
    install_mean_ablation_hooks(model, ctrl_lh, mean_z_ctrl)
    log("\n=== Condition 3: ctrl_ablated ===")
    t_cond = time.time()
    ctrl_si = run_condition(model, clean_ioi, corrupt_ioi, LOCKED_SI_SENDERS, LOCKED_NM_HEADS, "ctrl_ablated", log)
    ctrl_logit_diff = compute_logit_diff_per_prompt(model, clean_ioi)
    ctrl_lift = compute_successor_lift_per_prompt(model, suc_prompts, LOCKED_SUC_RECEIVERS)
    log(f"  Condition 3 wall: {time.time() - t_cond:.1f}s")
    model.reset_hooks()

    # --- Bootstrap + classify per readout ---
    log("\n=== Bootstrap (B=200 paired, seed=1) and classification ===")
    rng = np.random.default_rng(1)

    # Readout A: aggregate over 5 receivers per prompt, then bootstrap over prompts.
    def _agg_lift(lift_dict):
        # 5 receivers × 70 prompts → average across receivers → 70 scalars
        arr = np.stack([lift_dict[lh] for lh in LOCKED_SUC_RECEIVERS], axis=0)  # (5, 70)
        return arr.mean(axis=0)  # (70,)
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

    # Readout B: per-sender classification (§H5-7), aggregate via §H5-7 priority.
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

    # Readout C: IO−S logit-diff at END
    ratio_C_ind, C_ind_lo, C_ind_hi = bootstrap_drop_ratio(clean_logit_diff, ind_logit_diff, B_BOOTSTRAP, rng)
    ratio_C_ctrl, C_ctrl_lo, C_ctrl_hi = bootstrap_drop_ratio(clean_logit_diff, ctrl_logit_diff, B_BOOTSTRAP, rng)
    verdict_C = classify_logitdiff(ratio_C_ind, C_ind_lo, C_ind_hi, ratio_C_ctrl, C_ctrl_lo, C_ctrl_hi)
    log(f"\nReadout C (logit-diff):")
    log(f"  ratio_ind  = {ratio_C_ind:.4f}  CI=[{C_ind_lo:.4f}, {C_ind_hi:.4f}]")
    log(f"  ratio_ctrl = {ratio_C_ctrl:.4f}  CI=[{C_ctrl_lo:.4f}, {C_ctrl_hi:.4f}]")
    log(f"  verdict_C = {verdict_C}")

    # Cross-readout aggregate
    aggregate = aggregate_cross_readout(verdict_A, verdict_B, verdict_C)
    headline_template = PAPER_HEADLINES[aggregate]
    headline = headline_template.replace("{size}", SIZE)
    if structural_caveat_k_exhausted:
        headline = f"[K_effective={k_effective} < K_locked={LOCKED_K_LOCKED}; structural exhaustion at 70M — 4 surviving induction heads after 3-way exclusion] " + headline

    log(f"\n=== §H6-causal-7-agg cross-readout aggregate: {aggregate} ===")
    log(f"  verdict_A={verdict_A}  verdict_B={verdict_B}  verdict_C={verdict_C}")
    log(f"  Paper headline:")
    log(f"    {headline}")

    # --- Write outputs ---
    EXPL.mkdir(parents=True, exist_ok=True)

    # Per-prompt parquet (unified schema)
    per_prompt_rows: list[dict] = []
    # Readout A
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
    # Readout B
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
    # Readout C
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

    # Summary parquet
    summary_rows: list[dict] = []
    for cond_label, ratio_A, A_lo, A_hi in (
        ("clean", 1.0, 1.0, 1.0),
        ("ind_ablated", ratio_A_ind, A_ind_lo, A_ind_hi),
        ("ctrl_ablated", ratio_A_ctrl, A_ctrl_lo, A_ctrl_hi),
    ):
        summary_rows.append(dict(
            condition=cond_label, readout="A_lift",
            ratio_mean=ratio_A, ratio_ci_low=A_lo, ratio_ci_high=A_hi,
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
    for cond_label, ratio_C, C_lo, C_hi in (
        ("clean", 1.0, 1.0, 1.0),
        ("ind_ablated", ratio_C_ind, C_ind_lo, C_ind_hi),
        ("ctrl_ablated", ratio_C_ctrl, C_ctrl_lo, C_ctrl_hi),
    ):
        summary_rows.append(dict(
            condition=cond_label, readout="C_logit_diff",
            ratio_mean=ratio_C, ratio_ci_low=C_lo, ratio_ci_high=C_hi,
        ))
    pd.DataFrame(summary_rows).to_parquet(OUT_SUMMARY, index=False)

    # Verdict parquet (1 row)
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
        structural_caveat_70m_weak_si=structural_caveat_70m_weak_si,
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
