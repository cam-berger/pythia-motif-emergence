"""§H5-causal-3 anchor: scale extension to Pythia-1B step143000, both metrics.

Pre-registered in HYPOTHESIS.md §H5-causal-3 (commit `9e9a200`, 2026-05-07).
Tests inference-time independence of S-inhibition from successor at Pythia-1B
— the scale where the registered temporal emergence ordering reverses
(μ_si^1B < μ_suc^1B). Runs both Metric A (§S-1 path-patching scalar, parallel
to §H5-causal) and Metric B (IO−S logit-diff at END, parallel to §H5-causal-2)
in one model load for compute efficiency.

All four locked sets (suc top-5, si senders top-3, NM top-4, ctrl 5) are
asserted bit-for-bit at runtime per §H5-causal-3-12 / §H5-causal-3-14
no-cherry-picking discipline. Halt on divergence.

Outputs (under data/exploration/, gitignored):
- phase4_causal_1b_anchor.parquet                     Metric A per-prompt
- phase4_causal_1b_anchor_summary.parquet             Metric A per-(condition, sender)
- phase4_causal_1b_anchor_verdict.parquet             Metric A verdict
- phase4_causal_1b_anchor_logitdiff.parquet           Metric B per-prompt
- phase4_causal_1b_anchor_logitdiff_summary.parquet   Metric B per-condition
- phase4_causal_1b_anchor_logitdiff_verdict.parquet   Metric B verdict
- phase4_causal_1b_anchor_h5causal3_verdict.parquet   cross-metric pattern + headline
- phase4_causal_1b_anchor.log                         captured stdout
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

# Reuse §H5-causal helpers verbatim (compute primitives are size-agnostic).
from notebooks._run_phase4_causal_410m_anchor import (  # noqa: E402
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
from notebooks._lib.sweep_io import read_long  # noqa: E402
from src.replication.tigges_ioi import load_ioi_prompts  # noqa: E402
from src.utils.mps_compat import assert_mps_fallback_enabled  # noqa: E402
from src.utils.pythia_loader import load_pythia  # noqa: E402

EXPL = REPO_ROOT / "data" / "exploration"
PROMPTS_PATH = REPO_ROOT / "data" / "prompts" / "ioi_prompts.tsv"

SIZE = "1b"
STEP = 143000

# §SU-tau / §S-tau locks (unchanged across sizes per the registered detector)
TAU_LIFT = 0.13496
TAU_STRICT = 0.0372

# §H5-causal-3 amendment locks (re-derived from 1B parquets at amendment time)
LOCKED_SUC_SET = [(11, 6), (14, 2), (12, 3), (15, 7), (15, 1)]
LOCKED_SI_SENDERS = [(8, 7), (9, 1), (10, 4)]
LOCKED_NM_SET = [(11, 0), (11, 5), (14, 2), (11, 2)]
LOCKED_CTRL_SET = [(11, 1), (11, 3), (11, 4), (12, 2), (13, 0)]
LOCKED_CTRL_BRACKET_WIDTH = 0.125

# §H5-causal-3-9 / §H5-causal-3-10 inherited
ABLATE_K = 5
SI_SENDER_K = 3
B_BOOTSTRAP = 200
BRACKET_WIDTH_INIT = 0.05
BRACKET_WIDTH_STEP = 0.025

OUT_A_PER_PROMPT = EXPL / "phase4_causal_1b_anchor.parquet"
OUT_A_SUMMARY = EXPL / "phase4_causal_1b_anchor_summary.parquet"
OUT_A_VERDICT = EXPL / "phase4_causal_1b_anchor_verdict.parquet"
OUT_B_PER_PROMPT = EXPL / "phase4_causal_1b_anchor_logitdiff.parquet"
OUT_B_SUMMARY = EXPL / "phase4_causal_1b_anchor_logitdiff_summary.parquet"
OUT_B_VERDICT = EXPL / "phase4_causal_1b_anchor_logitdiff_verdict.parquet"
OUT_H5C3_VERDICT = EXPL / "phase4_causal_1b_anchor_h5causal3_verdict.parquet"
LOG_PATH = EXPL / "phase4_causal_1b_anchor.log"

# §H5-causal-3-13 cross-metric pattern → paper-headline
CROSS_METRIC_HEADLINES = {
    ("NULL", "NULL"): (
        "CONVERGING-NULL",
        "S-inhibition's circuit is causally disjoint from successor's at "
        "inference time at both Pythia-410m and Pythia-1B, despite the temporal "
        "emergence ordering reversing between scales (μ_suc < μ_si at 410m vs "
        "μ_si < μ_suc at 1B). The §H1-C ordering is a temporal coincidence of "
        "independent maturations, not a compositional chain. Strong direct "
        "mechanistic refutation of the inference-time-routing reading of §H1-C, "
        "robust across scale and across temporal-ordering direction.",
    ),
    ("NULL", "DEP"): (
        "B-DEP-A-NULL",
        "At Pythia-1B, successor heads contribute causally to IOI logit-diff "
        "but the contribution does not route through the registered "
        "S-inhibition mechanism (Metric A NULL stands). Successor → IOI is "
        "causal; successor → S-inhibition is not. The §H1-C ordering at 1B is "
        "therefore decoupled from the suc → si compositional reading even when "
        "suc → IOI is causal — refining the §H5-causal-2 410m result.",
    ),
    ("DEP", "NULL"): (
        "A-DEP-B-NULL",
        "Methodological note at Pythia-1B: §S-1 path-patching scalar drops "
        "under suc-ablation while IOI logit-diff is unchanged. This is a "
        "peculiar split — successor's effect on the §S-1 measurement is not "
        "on IOI task performance, suggesting §S-1 is detecting incidental "
        "in-circuit attention-pattern shifts rather than IOI-relevant signal. "
        "Verdict deferred pending mechanistic investigation; do not claim "
        "suc → si dependence on §S-1 alone.",
    ),
    ("DEP", "DEP"): (
        "BOTH-DEP",
        "At Pythia-1B, successor heads contribute causally to S-inhibition's "
        "circuit AND to IOI logit-diff. The §H1-C ordering at 1B IS a "
        "compositional chain at inference time, contrary to the 410m finding. "
        "Scale-dependent compositional structure: successor → S-inhibition → "
        "IOI is causal at 1B but not at 410m. Substantial reframe of §H1-C "
        "narrative required, with explicit scale-dependent causality.",
    ),
}
MIXED_HEADLINE = (
    "MIXED",
    "Heterogeneous Metric A and B verdicts at 1B; no global conclusion on "
    "suc → si causal dependence. Reported per-metric with per-sender CIs; "
    "deferred for follow-up.",
)


def select_ctrl_set_with_exclusion(
    df_suc: pd.DataFrame,
    suc_set: set[tuple[int, int]],
    nm_set: set[tuple[int, int]],
    k: int,
    bw_init: float,
    bw_step: float,
) -> tuple[list[tuple[int, int, float]], float]:
    """§H5-causal-3-7: ctrl candidates exclude both suc set AND NM set.

    The NM-exclusion clause is the methodological refinement at §H5-causal-3-7;
    at 410m it would have been a no-op (no overlap), at 1B it is required
    because L11H0 / L11H5 / L11H2 sit within the natural ctrl bracket.
    """
    sub = df_suc[(df_suc["size"] == SIZE) & (df_suc["step"] == STEP)].copy()
    exclude = suc_set | nm_set
    bw = bw_init
    while True:
        lo, hi = TAU_LIFT - bw, TAU_LIFT
        mask = (sub["score"] >= lo) & (sub["score"] < hi)
        cands = sub[mask].copy()
        cands = cands[
            ~cands.apply(
                lambda r: (int(r["layer"]), int(r["head"])) in exclude, axis=1
            )
        ]
        if len(cands) >= k:
            break
        bw += bw_step
        if bw > 1.0:
            raise RuntimeError(
                f"could not assemble {k} ctrl candidates even with bracket_width={bw}"
            )
    cands = cands.sort_values(by=["layer", "head"]).reset_index(drop=True)
    rng = np.random.default_rng(0)
    chosen_idx = rng.choice(len(cands), size=k, replace=False)
    chosen = cands.iloc[sorted(chosen_idx.tolist())]
    out = []
    for _, r in chosen.iterrows():
        out.append((int(r["layer"]), int(r["head"]), float(r["score"])))
    return out, float(bw)


def assert_locked(label: str, derived: list, locked: list) -> None:
    """No-cherry-picking assertion per §H5-causal-3-12 / §H5-causal-3-14."""
    if derived != locked:
        raise RuntimeError(
            f"§H5-causal-3 NO-CHERRY-PICKING ASSERTION FAILED for {label}: "
            f"re-derived {derived} != locked {locked}. "
            f"Halting per §H5-causal-3-14 — investigate parquet drift before proceeding."
        )


def main() -> None:
    log_lines: list[str] = []

    def log(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        print(msg, **kwargs, flush=True)
        log_lines.append(msg)

    assert_mps_fallback_enabled()
    log(f"=== §H5-causal-3 anchor experiment, Pythia-{SIZE}-deduped step{STEP} ===")
    log("Pre-registration: HYPOTHESIS.md §H5-causal-3 (commit `9e9a200`).")
    log("Both metrics in one model load. All sets asserted bit-for-bit.")
    t_start = time.time()

    # --- Re-derive sets and assert against locked lists ---
    df_suc = read_long(EXPL / "phase3_1b_successor_sweep.parquet")
    df_si = read_long(EXPL / "phase3_1b_s_inhibition_sweep.parquet")

    sub_s = df_suc[(df_suc["size"] == SIZE) & (df_suc["step"] == STEP)].copy()
    sub_s = sub_s.sort_values(by=["score", "layer", "head"], ascending=[False, True, True])
    suc_top5 = []
    for _, r in sub_s.head(ABLATE_K).iterrows():
        suc_top5.append((int(r["layer"]), int(r["head"]), float(r["score"])))
    suc_layer_heads = [(l, h) for l, h, _ in suc_top5]
    assert_locked("suc top-5", suc_layer_heads, LOCKED_SUC_SET)
    log(f"\n§H5-causal-3-4 suc top-5 (locked, asserted match):")
    for l, h, s in suc_top5:
        log(f"  L{l}H{h}  score={s:.4f}")

    sub_si = df_si[(df_si["size"] == SIZE) & (df_si["step"] == STEP)].copy()
    sub_si = sub_si.sort_values(by=["score", "layer", "head"], ascending=[False, True, True])
    si_top3 = []
    for _, r in sub_si.head(SI_SENDER_K).iterrows():
        si_top3.append((int(r["layer"]), int(r["head"]), float(r["score"])))
    si_senders = [(l, h) for l, h, _ in si_top3]
    assert_locked("si senders top-3", si_senders, LOCKED_SI_SENDERS)
    log(f"\n§H5-causal-3-5 si senders top-3 (locked, asserted match):")
    for l, h, s in si_top3:
        clears = "≥τ_strict" if s >= TAU_STRICT else "below τ_strict"
        log(f"  L{l}H{h}  Δ_h_clean_pre={s:.4f}  ({clears})")

    nm_npz = np.load(EXPL / "s_inhibition_pythia_1b_anchor_per_nm.npz")
    nm_heads_pinned: list[tuple[int, int]] = [
        (int(L), int(H)) for L, H in nm_npz["nm_heads"]
    ]
    assert_locked("NM top-4", nm_heads_pinned, LOCKED_NM_SET)
    log(f"\n§H5-causal-3-6 NMs (locked, asserted match):")
    for nl, nh in nm_heads_pinned:
        log(f"  L{nl}H{nh}")

    # Note dual-role L14H2 transparency
    suc_set_lh = set(suc_layer_heads)
    nm_set_lh = set(nm_heads_pinned)
    overlap = suc_set_lh & nm_set_lh
    if overlap:
        log(f"\n  NOTE: dual-role heads (in both suc and NM): {sorted(overlap)}")
        log(f"  Per §H5-causal-3-4, ablating these affects Metric B fully (z → logits)")
        log(f"  but not Metric A's pattern read (patterns are pre-z).")

    ctrl_top5, widened_bw = select_ctrl_set_with_exclusion(
        df_suc, suc_set_lh, nm_set_lh, ABLATE_K, BRACKET_WIDTH_INIT, BRACKET_WIDTH_STEP
    )
    ctrl_layer_heads = [(l, h) for l, h, _ in ctrl_top5]
    assert_locked("ctrl 5", ctrl_layer_heads, LOCKED_CTRL_SET)
    if abs(widened_bw - LOCKED_CTRL_BRACKET_WIDTH) > 1e-9:
        raise RuntimeError(
            f"§H5-causal-3-7 NO-CHERRY-PICKING ASSERTION FAILED on bracket_width: "
            f"re-derived {widened_bw:.3f} != locked {LOCKED_CTRL_BRACKET_WIDTH:.3f}"
        )
    log(f"\n§H5-causal-3-7 ctrl set (locked, asserted match; bracket_width={widened_bw:.3f}):")
    for l, h, s in ctrl_top5:
        log(f"  L{l}H{h}  score={s:.4f}")

    # --- Load model + prompts ---
    log(f"\nLoading Pythia-{SIZE}-deduped @ step{STEP}...")
    t0 = time.time()
    model = load_pythia(SIZE, step=STEP)
    log(f"  loaded in {time.time()-t0:.1f}s; "
        f"n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads}, "
        f"d_model={model.cfg.d_model}")

    clean = load_ioi_prompts(PROMPTS_PATH)
    log(f"  N={len(clean)} clean IOI prompts")

    from src.detectors.s_inhibition import build_abc_corrupted_prompts  # noqa: E402
    corrupt = build_abc_corrupted_prompts(clean, model.tokenizer, seed=0)
    log(f"  Built ABC-corrupted twins (seed=0)")

    # ============================================================
    # Condition 1: clean (no ablation) — both metrics
    # ============================================================
    model.reset_hooks()
    log("\n=== Condition 1: clean (no ablation) ===")
    log("  [Metric A] §S-1 path-patching screen with pinned NMs...")
    t0 = time.time()
    clean_A = run_condition(
        model, clean, corrupt, si_senders, nm_heads_pinned, "clean", log
    )
    log(f"  [Metric A] complete in {time.time()-t0:.1f}s")
    log("  [Metric B] IO−S logit-diff at END...")
    t0 = time.time()
    clean_B = compute_logit_diff_per_prompt(model, clean)
    log(f"  [Metric B] complete in {time.time()-t0:.1f}s; "
        f"mean Δlogit_clean = {clean_B.mean():+.4f}")

    # --- Precompute mean_z for both ablation sets ---
    log("\nPrecomputing mean_z for suc set...")
    t0 = time.time()
    mean_z_suc = precompute_mean_z_by_length(model, clean, suc_layer_heads)
    log(f"  done in {time.time()-t0:.1f}s")

    log("Precomputing mean_z for ctrl set...")
    t0 = time.time()
    mean_z_ctrl = precompute_mean_z_by_length(model, clean, ctrl_layer_heads)
    log(f"  done in {time.time()-t0:.1f}s")

    # ============================================================
    # Condition 2: suc_ablated — both metrics
    # ============================================================
    model.reset_hooks()
    install_mean_ablation_hooks(model, suc_layer_heads, mean_z_suc)
    log(f"\n=== Condition 2: suc_ablated ===")
    log(f"  perma hooks installed on {len(suc_layer_heads)} suc heads")
    log("  [Metric A] §S-1 path-patching screen...")
    t0 = time.time()
    suc_A = run_condition(
        model, clean, corrupt, si_senders, nm_heads_pinned, "suc_ablated", log
    )
    log(f"  [Metric A] complete in {time.time()-t0:.1f}s")
    log("  [Metric B] IO−S logit-diff at END...")
    t0 = time.time()
    suc_B = compute_logit_diff_per_prompt(model, clean)
    log(f"  [Metric B] complete in {time.time()-t0:.1f}s; "
        f"mean Δlogit_suc = {suc_B.mean():+.4f}")

    # ============================================================
    # Condition 3: ctrl_ablated — both metrics
    # ============================================================
    model.reset_hooks()
    install_mean_ablation_hooks(model, ctrl_layer_heads, mean_z_ctrl)
    log(f"\n=== Condition 3: ctrl_ablated ===")
    log(f"  perma hooks installed on {len(ctrl_layer_heads)} ctrl heads")
    log("  [Metric A] §S-1 path-patching screen...")
    t0 = time.time()
    ctrl_A = run_condition(
        model, clean, corrupt, si_senders, nm_heads_pinned, "ctrl_ablated", log
    )
    log(f"  [Metric A] complete in {time.time()-t0:.1f}s")
    log("  [Metric B] IO−S logit-diff at END...")
    t0 = time.time()
    ctrl_B = compute_logit_diff_per_prompt(model, clean)
    log(f"  [Metric B] complete in {time.time()-t0:.1f}s; "
        f"mean Δlogit_ctrl = {ctrl_B.mean():+.4f}")
    model.reset_hooks()

    # ============================================================
    # Bootstrap and verdict — both metrics
    # ============================================================
    log("\n=== Bootstrap (B=200 paired, seed=1) and per-metric classification ===")
    rng = np.random.default_rng(1)

    # ---- Metric A: per-sender ----
    summary_A_rows: list[dict] = []
    per_prompt_A_rows: list[dict] = []
    per_sender_A_pattern: list[str] = []
    log("\n  [Metric A] §S-1 path-patching scalar:")
    for sl, sh in si_senders:
        c = clean_A[(sl, sh)]
        a_suc = suc_A[(sl, sh)]
        a_ctrl = ctrl_A[(sl, sh)]
        for cond_label, arr in [("clean", c), ("suc_ablated", a_suc), ("ctrl_ablated", a_ctrl)]:
            for pi, val in enumerate(arr):
                per_prompt_A_rows.append(dict(
                    condition=cond_label, si_sender_layer=sl, si_sender_head=sh,
                    prompt_idx=pi, delta_h=float(val),
                ))
        ratio_suc, suc_lo, suc_hi = bootstrap_drop_ratio(c, a_suc, B_BOOTSTRAP, rng)
        ratio_ctrl, ctrl_lo, ctrl_hi = bootstrap_drop_ratio(c, a_ctrl, B_BOOTSTRAP, rng)
        pattern = classify_per_sender(ratio_suc, ratio_ctrl)
        per_sender_A_pattern.append(pattern)
        log(f"    L{sl}H{sh}: clean Δ̄={c.mean():+.4f}  "
            f"suc Δ̄={a_suc.mean():+.4f} (r={ratio_suc:.3f} [{suc_lo:.3f},{suc_hi:.3f}])  "
            f"ctrl Δ̄={a_ctrl.mean():+.4f} (r={ratio_ctrl:.3f} [{ctrl_lo:.3f},{ctrl_hi:.3f}])  "
            f"→ {pattern}")
        for cond_label, mean_val, ratio, lo, hi in [
            ("clean", float(c.mean()), 1.0, 1.0, 1.0),
            ("suc_ablated", float(a_suc.mean()), ratio_suc, suc_lo, suc_hi),
            ("ctrl_ablated", float(a_ctrl.mean()), ratio_ctrl, ctrl_lo, ctrl_hi),
        ]:
            summary_A_rows.append(dict(
                condition=cond_label,
                si_sender_layer=sl, si_sender_head=sh,
                delta_h_mean=mean_val,
                drop_ratio_mean=ratio, ratio_ci_low=lo, ratio_ci_high=hi,
                ablate_set_repr=(
                    "" if cond_label == "clean"
                    else (",".join(f"L{l}H{h}" for l, h in suc_layer_heads) if cond_label == "suc_ablated"
                          else ",".join(f"L{l}H{h}" for l, h in ctrl_layer_heads))
                ),
                widened_bracket_width=widened_bw if cond_label == "ctrl_ablated" else float("nan"),
                n_prompts=int(c.shape[0]),
            ))

    pattern_A = aggregate_verdict(per_sender_A_pattern)
    log(f"  [Metric A] per-sender: {per_sender_A_pattern}  → aggregate: {pattern_A}")

    # ---- Metric B: single condition ----
    log("\n  [Metric B] IO−S logit-diff:")
    ratio_suc_B, suc_lo_B, suc_hi_B = bootstrap_drop_ratio(clean_B, suc_B, B_BOOTSTRAP, rng)
    ratio_ctrl_B, ctrl_lo_B, ctrl_hi_B = bootstrap_drop_ratio(clean_B, ctrl_B, B_BOOTSTRAP, rng)
    pattern_B = classify_logitdiff(
        ratio_suc_B, suc_lo_B, suc_hi_B, ratio_ctrl_B, ctrl_lo_B, ctrl_hi_B
    )
    log(f"    ratio_suc  = {ratio_suc_B:.4f} CI=[{suc_lo_B:.4f}, {suc_hi_B:.4f}]")
    log(f"    ratio_ctrl = {ratio_ctrl_B:.4f} CI=[{ctrl_lo_B:.4f}, {ctrl_hi_B:.4f}]")
    log(f"  [Metric B] verdict: {pattern_B}")

    # ---- Cross-metric pattern ----
    if (pattern_A, pattern_B) in CROSS_METRIC_HEADLINES:
        h5c3_pattern, h5c3_headline = CROSS_METRIC_HEADLINES[(pattern_A, pattern_B)]
    else:
        h5c3_pattern, h5c3_headline = MIXED_HEADLINE
    log(f"\n=== §H5-causal-3-13 cross-metric pattern ===")
    log(f"  (Metric A, Metric B) = ({pattern_A}, {pattern_B})")
    log(f"  matched §H5-causal-3 pattern: {h5c3_pattern}")
    log(f"  matched paper headline:")
    log(f"    {h5c3_headline}")

    # ============================================================
    # Write parquets
    # ============================================================
    EXPL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_prompt_A_rows).to_parquet(OUT_A_PER_PROMPT, index=False)
    pd.DataFrame(summary_A_rows).to_parquet(OUT_A_SUMMARY, index=False)
    pd.DataFrame([dict(
        pattern=pattern_A,
        n_senders=len(per_sender_A_pattern),
        n_dep=per_sender_A_pattern.count("DEP"),
        n_null=per_sender_A_pattern.count("NULL"),
        n_generic=per_sender_A_pattern.count("GENERIC"),
        n_mixed=per_sender_A_pattern.count("MIXED"),
        per_sender_patterns=";".join(per_sender_A_pattern),
        widened_bracket_width=widened_bw,
        suc_set=",".join(f"L{l}H{h}" for l, h in suc_layer_heads),
        ctrl_set=",".join(f"L{l}H{h}" for l, h in ctrl_layer_heads),
        si_senders=",".join(f"L{l}H{h}" for l, h in si_senders),
        nm_heads=",".join(f"L{l}H{h}" for l, h in nm_heads_pinned),
    )]).to_parquet(OUT_A_VERDICT, index=False)

    per_prompt_B_rows = []
    for cond_label, arr in [("clean", clean_B), ("suc_ablated", suc_B), ("ctrl_ablated", ctrl_B)]:
        for pi, val in enumerate(arr):
            per_prompt_B_rows.append(dict(condition=cond_label, prompt_idx=pi, logit_diff=float(val)))
    pd.DataFrame(per_prompt_B_rows).to_parquet(OUT_B_PER_PROMPT, index=False)

    pd.DataFrame([
        dict(condition="clean",        logit_diff_mean=float(clean_B.mean()),
             drop_ratio_mean=1.0, ratio_ci_low=1.0, ratio_ci_high=1.0,
             ablate_set="", n_prompts=int(clean_B.shape[0])),
        dict(condition="suc_ablated",  logit_diff_mean=float(suc_B.mean()),
             drop_ratio_mean=ratio_suc_B, ratio_ci_low=suc_lo_B, ratio_ci_high=suc_hi_B,
             ablate_set=",".join(f"L{l}H{h}" for l, h in suc_layer_heads),
             n_prompts=int(clean_B.shape[0])),
        dict(condition="ctrl_ablated", logit_diff_mean=float(ctrl_B.mean()),
             drop_ratio_mean=ratio_ctrl_B, ratio_ci_low=ctrl_lo_B, ratio_ci_high=ctrl_hi_B,
             ablate_set=",".join(f"L{l}H{h}" for l, h in ctrl_layer_heads),
             n_prompts=int(clean_B.shape[0])),
    ]).to_parquet(OUT_B_SUMMARY, index=False)

    pd.DataFrame([dict(
        pattern=pattern_B,
        ratio_suc=ratio_suc_B, ratio_suc_ci_low=suc_lo_B, ratio_suc_ci_high=suc_hi_B,
        ratio_ctrl=ratio_ctrl_B, ratio_ctrl_ci_low=ctrl_lo_B, ratio_ctrl_ci_high=ctrl_hi_B,
        clean_mean=float(clean_B.mean()),
        suc_ablated_mean=float(suc_B.mean()),
        ctrl_ablated_mean=float(ctrl_B.mean()),
        suc_set=",".join(f"L{l}H{h}" for l, h in suc_layer_heads),
        ctrl_set=",".join(f"L{l}H{h}" for l, h in ctrl_layer_heads),
        widened_bracket_width=float(widened_bw),
    )]).to_parquet(OUT_B_VERDICT, index=False)

    pd.DataFrame([dict(
        h5causal3_pattern=h5c3_pattern,
        metric_A_pattern=pattern_A,
        metric_B_pattern=pattern_B,
        metric_A_per_sender=";".join(per_sender_A_pattern),
        metric_B_ratio_suc=ratio_suc_B,
        metric_B_ratio_ctrl=ratio_ctrl_B,
        suc_set=",".join(f"L{l}H{h}" for l, h in suc_layer_heads),
        ctrl_set=",".join(f"L{l}H{h}" for l, h in ctrl_layer_heads),
        si_senders=",".join(f"L{l}H{h}" for l, h in si_senders),
        nm_heads=",".join(f"L{l}H{h}" for l, h in nm_heads_pinned),
        widened_bracket_width=float(widened_bw),
        paper_headline=h5c3_headline,
    )]).to_parquet(OUT_H5C3_VERDICT, index=False)

    log(f"\nWrote {OUT_A_PER_PROMPT.relative_to(REPO_ROOT)}")
    log(f"Wrote {OUT_A_SUMMARY.relative_to(REPO_ROOT)}")
    log(f"Wrote {OUT_A_VERDICT.relative_to(REPO_ROOT)}")
    log(f"Wrote {OUT_B_PER_PROMPT.relative_to(REPO_ROOT)}")
    log(f"Wrote {OUT_B_SUMMARY.relative_to(REPO_ROOT)}")
    log(f"Wrote {OUT_B_VERDICT.relative_to(REPO_ROOT)}")
    log(f"Wrote {OUT_H5C3_VERDICT.relative_to(REPO_ROOT)}")

    LOG_PATH.write_text("\n".join(log_lines) + "\n")
    print(f"\nWall: {(time.time()-t_start)/60:.1f} min")
    print(f"Log: {LOG_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
