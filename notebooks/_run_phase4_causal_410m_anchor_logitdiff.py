"""§H5-causal-2 anchor: IO−S logit-diff metric for causal-dependence ablation at Pythia-410m step143000.

Pre-registered in HYPOTHESIS.md §H5-causal-2 (commit `51caa35`, 2026-05-07).
Tests whether the IO−S logit-difference at the END token depends on the
top-5 successor heads being live (mean-ablated on hook_z), vs the same
score-bracket-matched random control set used in §H5-causal.

Reuses the §H5-3 suc set and §H5-4 ctrl set verbatim — both are re-derived
from the parquets and asserted bit-for-bit against the locked lists per
§H5-causal-2-3 / §H5-causal-2-4 / §H5-causal-2-10 (no cherry-picking).

Reuses §H5-causal helpers (mean-ablation, bootstrap) verbatim by importing
from the sibling runner.

Outputs (under data/exploration/, gitignored):
- phase4_causal_410m_anchor_logitdiff.parquet            per-prompt logit-diff
- phase4_causal_410m_anchor_logitdiff_summary.parquet    per-condition + CI
- phase4_causal_410m_anchor_logitdiff_verdict.parquet    §H5-causal-2-6 verdict
- phase4_causal_410m_anchor_logitdiff.log                captured stdout
"""

from __future__ import annotations

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

# Reuse §H5-causal helpers verbatim per §H5-causal-2-9 lock.
from notebooks._run_phase4_causal_410m_anchor import (  # noqa: E402
    BRACKET_WIDTH_INIT,
    BRACKET_WIDTH_STEP,
    ABLATE_K,
    SIZE,
    STEP,
    TAU_LIFT,
    bootstrap_drop_ratio,
    install_mean_ablation_hooks,
    precompute_mean_z_by_length,
    select_ctrl_set,
    select_top_suc,
)
from notebooks._lib.sweep_io import read_long  # noqa: E402
from src.detectors.s_inhibition import build_abc_corrupted_prompts  # noqa: E402  (only for log alignment; not used here)
from src.replication.tigges_ioi import load_ioi_prompts  # noqa: E402
from src.utils.mps_compat import assert_mps_fallback_enabled  # noqa: E402
from src.utils.pythia_loader import load_pythia  # noqa: E402

EXPL = REPO_ROOT / "data" / "exploration"
OUT_PER_PROMPT = EXPL / "phase4_causal_410m_anchor_logitdiff.parquet"
OUT_SUMMARY = EXPL / "phase4_causal_410m_anchor_logitdiff_summary.parquet"
OUT_VERDICT = EXPL / "phase4_causal_410m_anchor_logitdiff_verdict.parquet"
LOG_PATH = EXPL / "phase4_causal_410m_anchor_logitdiff.log"

PROMPTS_PATH = REPO_ROOT / "data" / "prompts" / "ioi_prompts.tsv"

# §H5-causal-2-3 / §H5-causal-2-4: locked lists, asserted bit-for-bit.
LOCKED_SUC_SET = [(22, 6), (22, 2), (20, 4), (22, 10), (12, 8)]
LOCKED_CTRL_SET = [(17, 12), (20, 6), (22, 11), (23, 10), (23, 13)]
LOCKED_BRACKET_WIDTH = 0.100

# §H5-causal-2-7 bootstrap parameters
B_BOOTSTRAP = 200

# §H5-causal-2-6 gate thresholds
NULL_LO, NULL_HI = 0.8, 1.2
DEP_THRESHOLD = 0.5
GENERIC_THRESHOLD = 0.7


def compute_logit_diff_per_prompt(model, clean_prompts) -> np.ndarray:
    """Return per-prompt IO−S logit-diff at END position. Length-grouped forward.

    Assumes clean_prompts share a common ordering with their indices in the
    returned array. Reads io_token_id and s_token_id from each IOIPrompt.
    """
    device = next(model.parameters()).device
    by_len: dict[int, list[int]] = defaultdict(list)
    token_rows = [model.to_tokens(p.text, prepend_bos=True)[0] for p in clean_prompts]
    for i, t in enumerate(token_rows):
        by_len[int(t.shape[0])].append(i)

    out = np.zeros(len(clean_prompts), dtype=np.float32)
    with torch.no_grad():
        for length, idxs in sorted(by_len.items()):
            tokens = torch.stack([token_rows[i] for i in idxs]).to(device)
            logits = model(tokens, return_type="logits")  # (B, T, V)
            end_logits = logits[:, length - 1, :].cpu()  # (B, V)
            for batch_i, prompt_i in enumerate(idxs):
                io = clean_prompts[prompt_i].io_token_id
                s = clean_prompts[prompt_i].s_token_id
                out[prompt_i] = float(end_logits[batch_i, io] - end_logits[batch_i, s])
    return out


def classify_logitdiff(
    ratio_suc: float, suc_lo: float, suc_hi: float,
    ratio_ctrl: float, ctrl_lo: float, ctrl_hi: float,
) -> str:
    """§H5-causal-2-6 gate classifier."""
    suc_in_null = (NULL_LO <= ratio_suc <= NULL_HI) and (NULL_LO <= suc_lo) and (suc_hi <= NULL_HI)
    ctrl_in_null = (NULL_LO <= ratio_ctrl <= NULL_HI) and (NULL_LO <= ctrl_lo) and (ctrl_hi <= NULL_HI)
    suc_dep = (ratio_suc < DEP_THRESHOLD) and (suc_hi < DEP_THRESHOLD)
    ctrl_in_null_loose = NULL_LO <= ratio_ctrl <= NULL_HI
    suc_below_generic = ratio_suc < GENERIC_THRESHOLD
    ctrl_below_generic = ratio_ctrl < GENERIC_THRESHOLD

    if suc_in_null and ctrl_in_null:
        return "NULL"
    if suc_dep and ctrl_in_null_loose:
        return "DEP"
    if suc_below_generic and ctrl_below_generic:
        return "GENERIC"
    return "MIXED"


PAPER_HEADLINES = {
    "NULL": (
        "IOI logit-diff at Pythia-410m is independent of the registered top-5 "
        "successor heads. Combined with the §H5-causal NULL on the §S-1 metric, "
        "this is converging evidence that S-inhibition's circuit is causally "
        "disjoint from successor's at inference time. The temporal emergence "
        "ordering ind→suc→si is decoupled from any architectural causal chain."
    ),
    "DEP": (
        "Successor heads contribute causally to IOI logit-diff at Pythia-410m via "
        "a path the §S-1 metric does not read. The §H5-causal NULL is therefore "
        "an instance of metric insensitivity (b), not genuine independence (a). "
        "The §H1-C compositional reading is partially supported: successor → IOI "
        "is causal at inference time, but not via the registered S-inhibition "
        "mechanism. Substantial reframe of the §H1-C narrative required."
    ),
    "GENERIC": (
        "Methodological note: IOI logit-diff is not robust to mean-ablation of "
        "any layer-22-cluster head at this checkpoint; metric sensitivity is "
        "insufficient. Verdict deferred pending re-tooling (e.g., per-head "
        "individual ablation, or non-mean ablation method)."
    ),
    "MIXED": (
        "Heterogeneous ablation effect on IOI logit-diff; no global verdict on "
        "suc → IOI dependence. Reported as numerical-only result with CI bands; "
        "deferred for follow-up."
    ),
}


def main() -> None:
    log_lines: list[str] = []

    def log(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        print(msg, **kwargs, flush=True)
        log_lines.append(msg)

    assert_mps_fallback_enabled()
    log(f"=== §H5-causal-2 anchor experiment, Pythia-{SIZE}-deduped step{STEP} ===")
    log("Pre-registration: HYPOTHESIS.md §H5-causal-2 (commit `51caa35`).")
    log("Reuses §H5-3 suc set and §H5-4 ctrl set verbatim (asserted bit-for-bit).")
    t_start = time.time()

    # --- Re-derive sets and assert against locked lists ---
    df_suc = read_long(EXPL / "phase2_successor_sweep.parquet")
    suc_set_full = select_top_suc(df_suc, ABLATE_K)
    suc_layer_heads = [(l, h) for l, h, _ in suc_set_full]
    if suc_layer_heads != LOCKED_SUC_SET:
        raise RuntimeError(
            f"§H5-causal-2-3 NO-CHERRY-PICKING ASSERTION FAILED: "
            f"re-derived suc set {suc_layer_heads} != locked {LOCKED_SUC_SET}. "
            f"Halting per §H5-causal-2-10 — investigate parquet drift before proceeding."
        )
    log(f"\n§H5-causal-2-3 suc set (locked, asserted match):")
    for l, h, s in suc_set_full:
        log(f"  L{l}H{h}  score={s:.4f}")

    ctrl_set_full, widened_bw = select_ctrl_set(
        df_suc, suc_set_full, ABLATE_K, BRACKET_WIDTH_INIT, BRACKET_WIDTH_STEP
    )
    ctrl_layer_heads = [(l, h) for l, h, _ in ctrl_set_full]
    if ctrl_layer_heads != LOCKED_CTRL_SET or abs(widened_bw - LOCKED_BRACKET_WIDTH) > 1e-9:
        raise RuntimeError(
            f"§H5-causal-2-4 NO-CHERRY-PICKING ASSERTION FAILED: "
            f"re-derived ctrl set {ctrl_layer_heads} (bw={widened_bw:.3f}) != "
            f"locked {LOCKED_CTRL_SET} (bw={LOCKED_BRACKET_WIDTH:.3f}). "
            f"Halting per §H5-causal-2-10 — investigate parquet drift before proceeding."
        )
    log(f"\n§H5-causal-2-4 ctrl set (locked, asserted match; bracket_width={widened_bw:.3f}):")
    for l, h, s in ctrl_set_full:
        log(f"  L{l}H{h}  score={s:.4f}")

    # --- Load model + prompts ---
    log(f"\nLoading Pythia-{SIZE}-deduped @ step{STEP}...")
    t0 = time.time()
    model = load_pythia(SIZE, step=STEP)
    log(f"  loaded in {time.time()-t0:.1f}s; n_layers={model.cfg.n_layers}, n_heads={model.cfg.n_heads}")

    clean = load_ioi_prompts(PROMPTS_PATH)
    log(f"  N={len(clean)} clean IOI prompts (GPT-NeoX BPE)")

    # --- Condition 1: clean ---
    model.reset_hooks()
    log("\n=== Condition 1: clean (no ablation) ===")
    t0 = time.time()
    clean_logit_diff = compute_logit_diff_per_prompt(model, clean)
    log(f"  forward pass complete in {time.time()-t0:.1f}s; "
        f"mean Δlogit_clean = {clean_logit_diff.mean():+.4f}, "
        f"std = {clean_logit_diff.std():.4f}")

    # --- Precompute mean_z for both ablation sets ---
    log("\nPrecomputing mean_z for suc set...")
    t0 = time.time()
    mean_z_suc = precompute_mean_z_by_length(model, clean, suc_layer_heads)
    log(f"  done in {time.time()-t0:.1f}s")

    log("Precomputing mean_z for ctrl set...")
    t0 = time.time()
    mean_z_ctrl = precompute_mean_z_by_length(model, clean, ctrl_layer_heads)
    log(f"  done in {time.time()-t0:.1f}s")

    # --- Condition 2: suc_ablated ---
    model.reset_hooks()
    install_mean_ablation_hooks(model, suc_layer_heads, mean_z_suc)
    log("\n=== Condition 2: suc_ablated ===")
    log(f"  perma hooks installed on {len(suc_layer_heads)} suc heads")
    t0 = time.time()
    suc_logit_diff = compute_logit_diff_per_prompt(model, clean)
    log(f"  forward pass complete in {time.time()-t0:.1f}s; "
        f"mean Δlogit_suc = {suc_logit_diff.mean():+.4f}, "
        f"std = {suc_logit_diff.std():.4f}")

    # --- Condition 3: ctrl_ablated ---
    model.reset_hooks()
    install_mean_ablation_hooks(model, ctrl_layer_heads, mean_z_ctrl)
    log("\n=== Condition 3: ctrl_ablated ===")
    log(f"  perma hooks installed on {len(ctrl_layer_heads)} ctrl heads")
    t0 = time.time()
    ctrl_logit_diff = compute_logit_diff_per_prompt(model, clean)
    log(f"  forward pass complete in {time.time()-t0:.1f}s; "
        f"mean Δlogit_ctrl = {ctrl_logit_diff.mean():+.4f}, "
        f"std = {ctrl_logit_diff.std():.4f}")
    model.reset_hooks()

    # --- Bootstrap and verdict ---
    log("\n=== Bootstrap (B=200 paired, seed=1) ===")
    rng = np.random.default_rng(1)
    ratio_suc, suc_lo, suc_hi = bootstrap_drop_ratio(clean_logit_diff, suc_logit_diff, B_BOOTSTRAP, rng)
    ratio_ctrl, ctrl_lo, ctrl_hi = bootstrap_drop_ratio(clean_logit_diff, ctrl_logit_diff, B_BOOTSTRAP, rng)
    log(f"  ratio_suc  = {ratio_suc:.4f}  CI=[{suc_lo:.4f}, {suc_hi:.4f}]")
    log(f"  ratio_ctrl = {ratio_ctrl:.4f}  CI=[{ctrl_lo:.4f}, {ctrl_hi:.4f}]")

    pattern = classify_logitdiff(ratio_suc, suc_lo, suc_hi, ratio_ctrl, ctrl_lo, ctrl_hi)
    headline = PAPER_HEADLINES[pattern]
    log(f"\n=== §H5-causal-2-6 verdict ===")
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
            per_prompt_rows.append(dict(
                condition=cond_label, prompt_idx=pi, logit_diff=float(val),
            ))
    pd.DataFrame(per_prompt_rows).to_parquet(OUT_PER_PROMPT, index=False)

    summary_rows = []
    for cond_label, mean_val, ratio, lo, hi, ablate_set in [
        ("clean", float(clean_logit_diff.mean()), 1.0, 1.0, 1.0, ""),
        ("suc_ablated", float(suc_logit_diff.mean()), ratio_suc, suc_lo, suc_hi,
         ",".join(f"L{l}H{h}" for l, h in suc_layer_heads)),
        ("ctrl_ablated", float(ctrl_logit_diff.mean()), ratio_ctrl, ctrl_lo, ctrl_hi,
         ",".join(f"L{l}H{h}" for l, h in ctrl_layer_heads)),
    ]:
        summary_rows.append(dict(
            condition=cond_label,
            logit_diff_mean=mean_val,
            drop_ratio_mean=ratio, ratio_ci_low=lo, ratio_ci_high=hi,
            ablate_set=ablate_set,
            n_prompts=int(clean_logit_diff.shape[0]),
        ))
    pd.DataFrame(summary_rows).to_parquet(OUT_SUMMARY, index=False)

    pd.DataFrame([dict(
        pattern=pattern,
        ratio_suc=ratio_suc, ratio_suc_ci_low=suc_lo, ratio_suc_ci_high=suc_hi,
        ratio_ctrl=ratio_ctrl, ratio_ctrl_ci_low=ctrl_lo, ratio_ctrl_ci_high=ctrl_hi,
        clean_mean=float(clean_logit_diff.mean()),
        suc_ablated_mean=float(suc_logit_diff.mean()),
        ctrl_ablated_mean=float(ctrl_logit_diff.mean()),
        suc_set=",".join(f"L{l}H{h}" for l, h in suc_layer_heads),
        ctrl_set=",".join(f"L{l}H{h}" for l, h in ctrl_layer_heads),
        widened_bracket_width=float(widened_bw),
        paper_headline=headline,
    )]).to_parquet(OUT_VERDICT, index=False)

    log(f"\nWrote {OUT_PER_PROMPT.relative_to(REPO_ROOT)}")
    log(f"Wrote {OUT_SUMMARY.relative_to(REPO_ROOT)}")
    log(f"Wrote {OUT_VERDICT.relative_to(REPO_ROOT)}")

    LOG_PATH.write_text("\n".join(log_lines) + "\n")
    print(f"\nWall: {(time.time()-t_start)/60:.1f} min")
    print(f"Log: {LOG_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
