"""§H4-supersede analysis: (A.timing) + (A.count) gate + verdict.

Pre-registered in HYPOTHESIS.md §H4-supersede-5 (deliverable 4).

Inputs:
  - data/exploration/phase4_2_8b_s_inhibition_supersede_sweep.parquet (10 cells)
  - data/exploration/phase4_2_8b_s_inhibition_supersede_per_prompt/*.npz (10 cells)
  - data/exploration/phase2_s_inhibition_sweep.parquet (410m full 40-cell sweep)
  - data/exploration/phase2_s_inhibition_per_prompt/410m_step*.npz (410m 40 cells)

Outputs:
  - data/exploration/phase4_2_8b_h4supersede_verdict.parquet  (single-row verdict)
  - data/exploration/phase4_2_8b_h4supersede_bootstrap_mu.parquet  (per-bootstrap μ pairs)

Gate per §H4-2 (inherited verbatim by §H4-supersede-2):
  (A.timing) P(mu_si^2.8B < mu_si^410m) >= 0.95 over B=1000 paired per-prompt bootstrap
  (A.count)  max_count_si^2.8B >= 5 over the 10-cell supersede grid

Verdict taxonomy per §H4-5 + §H4-7-supersede (priority `DEFERRED > TOOLING > NEITHER >
COUNT-ONLY > TIMING-ONLY > PASS`).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

EXPL = REPO_ROOT / "data" / "exploration"

# §S-tau / §H4-2 locks
TAU_STRICT = 0.0372
COUNT_GATE = 5
TIMING_GATE = 0.95
B_BOOTSTRAP = 1000

# §H4-supersede-1 grid
SUPERSEDE_STEPS = (5000, 7000, 10000, 14000, 20000, 29000, 41000, 49000, 59000, 70000)


def logistic(log_step: np.ndarray, L: float, k: float, mu_log: float) -> np.ndarray:
    return L / (1.0 + np.exp(-k * (log_step - mu_log)))


def fit_mu(steps: np.ndarray, counts: np.ndarray) -> float:
    """Return mu_si (in step space). NaN on fit failure."""
    log_steps = np.log10(steps + 1.0)
    L0 = max(float(counts.max()), 1.0)
    try:
        popt, _ = curve_fit(logistic, log_steps, counts, p0=[L0, 2.0, 4.0], maxfev=5000)
        _, _, mu_log = popt
        return float(10**mu_log - 1.0)
    except Exception:
        return float("nan")


def per_head_score_from_npz(per_prompt_delta: np.ndarray, prompt_idx: np.ndarray) -> np.ndarray:
    """per_prompt_delta shape (n_heads, k_nm, n_prompts). Returns per-head scalar after
    averaging over NMs and over resampled prompts.

    NOTE: this matches s_inhibition_screen's reported `delta_h` aggregation —
    mean over NMs then mean over prompts.
    """
    # Average over NMs first → (n_heads, n_prompts), then resample prompts → mean over those
    by_nm_mean = per_prompt_delta.mean(axis=1)  # (n_heads, n_prompts)
    return by_nm_mean[:, prompt_idx].mean(axis=1)  # (n_heads,)


def main() -> None:
    print("=== §H4-supersede analysis ===")
    print(f"  τ_strict = {TAU_STRICT}; (A.count) gate ≥ {COUNT_GATE}; "
          f"(A.timing) gate ≥ {TIMING_GATE}; B = {B_BOOTSTRAP}")

    # --- (A.count) point estimate ---
    df_28b = pd.read_parquet(EXPL / "phase4_2_8b_s_inhibition_supersede_sweep.parquet")
    counts_28b_point = (
        df_28b[df_28b.score >= TAU_STRICT]
        .groupby("step")
        .size()
        .reindex(SUPERSEDE_STEPS, fill_value=0)
    )
    max_count_28b = int(counts_28b_point.max())
    a_count_pass = max_count_28b >= COUNT_GATE
    print(f"\n(A.count) point estimate: max_count_si^2.8B = {max_count_28b}")
    for step, c in counts_28b_point.items():
        print(f"    step{step:6d}: count={c}")
    print(f"    → (A.count) {'PASS' if a_count_pass else 'FAIL'}")

    # --- (A.timing) point estimate ---
    df_410m_all = pd.read_parquet(EXPL / "phase2_s_inhibition_sweep.parquet")
    df_410m = df_410m_all[df_410m_all["size"] == "410m"].copy()
    counts_410m_point = (
        df_410m[df_410m.score >= TAU_STRICT].groupby("step").size()
    )
    all_410m_steps = sorted(df_410m.step.unique())
    counts_410m_point = counts_410m_point.reindex(all_410m_steps, fill_value=0)

    mu_28b_point = fit_mu(
        np.array(SUPERSEDE_STEPS, dtype=float), counts_28b_point.values.astype(float)
    )
    mu_410m_point = fit_mu(
        np.array(all_410m_steps, dtype=float), counts_410m_point.values.astype(float)
    )
    print(f"\n(A.timing) point estimates:")
    print(f"    μ_si^2.8B  = {mu_28b_point:.1f}")
    print(f"    μ_si^410m  = {mu_410m_point:.1f}")

    # --- Bootstrap reversal-rate per §H2-2 / §H4-2 ---
    # Load per-prompt npz for both sizes at all relevant cells
    print(f"\nLoading per-prompt npz for {len(SUPERSEDE_STEPS)} cells × 2 sizes...")
    npz_28b: dict[int, np.ndarray] = {}
    for step in SUPERSEDE_STEPS:
        p = EXPL / "phase4_2_8b_s_inhibition_supersede_per_prompt" / f"2.8b_step{step}.npz"
        npz_28b[step] = np.load(p)["per_prompt_delta"]  # (1024, 4, 200)

    npz_410m: dict[int, np.ndarray] = {}
    for step in all_410m_steps:
        p = EXPL / "phase2_s_inhibition_per_prompt" / f"410m_step{step}.npz"
        if p.exists():
            npz_410m[step] = np.load(p)["per_prompt_delta"]  # (384, 4, 200)

    n_prompts = 200  # locked §S-1 / §H5-5
    rng = np.random.default_rng(0)
    mu_pairs: list[tuple[float, float]] = []
    a_timing_count = 0
    fit_fail_28b = 0
    fit_fail_410m = 0
    for b in range(B_BOOTSTRAP):
        idx = rng.integers(0, n_prompts, size=n_prompts)

        # 2.8B counts
        counts_28b_b = np.zeros(len(SUPERSEDE_STEPS), dtype=float)
        for i, step in enumerate(SUPERSEDE_STEPS):
            scores = per_head_score_from_npz(npz_28b[step], idx)
            counts_28b_b[i] = int((scores >= TAU_STRICT).sum())

        # 410m counts (paired — same idx)
        counts_410m_b = np.zeros(len(all_410m_steps), dtype=float)
        for i, step in enumerate(all_410m_steps):
            scores = per_head_score_from_npz(npz_410m[step], idx)
            counts_410m_b[i] = int((scores >= TAU_STRICT).sum())

        mu_28b_b = fit_mu(np.array(SUPERSEDE_STEPS, dtype=float), counts_28b_b)
        mu_410m_b = fit_mu(np.array(all_410m_steps, dtype=float), counts_410m_b)
        mu_pairs.append((mu_28b_b, mu_410m_b))

        if np.isnan(mu_28b_b):
            fit_fail_28b += 1
            continue
        if np.isnan(mu_410m_b):
            fit_fail_410m += 1
            continue
        if mu_28b_b < mu_410m_b:
            a_timing_count += 1

        if (b + 1) % 100 == 0:
            print(
                f"  [{b+1}/{B_BOOTSTRAP}] reversal_rate so far = "
                f"{a_timing_count / (b+1 - fit_fail_28b - fit_fail_410m):.3f} "
                f"(fit_fail 2.8B={fit_fail_28b}, 410m={fit_fail_410m})"
            )

    valid = B_BOOTSTRAP - fit_fail_28b - fit_fail_410m
    reversal_rate = a_timing_count / valid if valid > 0 else 0.0
    a_timing_pass = reversal_rate >= TIMING_GATE

    print(f"\n(A.timing) bootstrap:")
    print(f"    B = {B_BOOTSTRAP}, valid = {valid} "
          f"(fit_fail 2.8B = {fit_fail_28b}, 410m = {fit_fail_410m})")
    print(f"    reversal_rate = {reversal_rate:.4f}  (gate ≥ {TIMING_GATE})")
    print(f"    → (A.timing) {'PASS' if a_timing_pass else 'FAIL'}")

    # --- §H4-5 + §H4-7-supersede verdict ---
    if a_timing_pass and a_count_pass:
        pattern = "PASS"
        headline = (
            "§H4-supersede passes: at head-count tier 1024 (Pythia-2.8B), S-inhibition "
            "timing accelerates beyond 410m (paired bootstrap reversal-rate = "
            f"{reversal_rate:.3f} ≥ 0.95) AND count exceeds the 410m saturation cap "
            f"(max_count = {max_count_28b} ≥ 5). Scaling argument on the head-count "
            "axis confirmed."
        )
    elif a_timing_pass and not a_count_pass:
        pattern = "TIMING-ONLY"
        headline = (
            f"§H4-supersede TIMING-ONLY: timing accelerates at 2.8B (rev_rate = "
            f"{reversal_rate:.3f}) but count saturates (max_count = {max_count_28b} < 5). "
            "Count saturation appears to be a fundamental property, not head-count-rate-limited."
        )
    elif not a_timing_pass and a_count_pass:
        pattern = "COUNT-ONLY"
        headline = (
            f"§H4-supersede COUNT-ONLY: count exceeds 410m cap at 2.8B (max_count = "
            f"{max_count_28b} ≥ 5) but timing saturates (rev_rate = {reversal_rate:.3f} < 0.95)."
        )
    else:
        pattern = "NEITHER"
        headline = (
            f"§H4-supersede NEITHER: both legs fail (rev_rate = {reversal_rate:.3f}, "
            f"max_count = {max_count_28b}). Scaling argument on the head-count axis "
            "falsified at 2.8B."
        )

    print(f"\n§H4-supersede VERDICT: {pattern}")
    print(f"  {headline}")

    # --- Write parquets ---
    pd.DataFrame(
        [
            dict(
                pattern=pattern,
                a_count_pass=a_count_pass,
                max_count_28b=max_count_28b,
                a_count_gate=COUNT_GATE,
                a_timing_pass=a_timing_pass,
                reversal_rate=reversal_rate,
                a_timing_gate=TIMING_GATE,
                mu_28b_point=mu_28b_point,
                mu_410m_point=mu_410m_point,
                B=B_BOOTSTRAP,
                valid=valid,
                fit_fail_28b=fit_fail_28b,
                fit_fail_410m=fit_fail_410m,
                tau_strict=TAU_STRICT,
                paper_headline=headline,
            )
        ]
    ).to_parquet(EXPL / "phase4_2_8b_h4supersede_verdict.parquet", index=False)

    pd.DataFrame(mu_pairs, columns=["mu_28b", "mu_410m"]).to_parquet(
        EXPL / "phase4_2_8b_h4supersede_bootstrap_mu.parquet", index=False
    )

    print(f"\nWrote {(EXPL / 'phase4_2_8b_h4supersede_verdict.parquet').relative_to(REPO_ROOT)}")
    print(f"Wrote {(EXPL / 'phase4_2_8b_h4supersede_bootstrap_mu.parquet').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
