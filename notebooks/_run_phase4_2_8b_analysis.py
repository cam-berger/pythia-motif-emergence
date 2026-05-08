"""Phase 4 2.8B bootstrap + §H4-scaling verdict analysis (HYPOTHESIS.md §H4-scaling).

Runs the §H2-2 per-prompt bootstrap (B=1000, 95% CI) on all 5 sizes
(70m, 160m, 410m, 1b, 2.8b) × 3 motifs, computes the §H4-scaling
conjunctive gate ((A.timing) AND (A.count)) and matches the §H4-5
failure-mode pattern, and persists results for the 5-size notebook
extensions.

Outputs:
  - data/exploration/phase4_2_8b_bootstrap_mu.parquet
  - data/exploration/phase4_2_8b_bootstrap_mus.npz
  - data/exploration/phase4_2_8b_h4scaling_verdict.parquet
"""

from __future__ import annotations

import os

# Must be set before any huggingface_hub import (analysis script doesn't load
# models, but the import chain pulls in modules that may depend on this).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.phase2_bootstrap import (  # noqa: E402
    bootstrap_induction,
    bootstrap_s_inhibition,
    bootstrap_successor,
    summarize_bootstrap,
)
from src.analysis.phase2_logistic import (  # noqa: E402
    tiered_fit,
)

# Locked thresholds
INDUCTION_THRESHOLD = 0.3
TAU_LIFT = 0.13496
TAU_STRICT = 0.0372

ALL_SIZES = ("70m", "160m", "410m", "1b", "2.8b")
DATA_DIR = REPO_ROOT / "data" / "exploration"
OUT_MU = DATA_DIR / "phase4_2_8b_bootstrap_mu.parquet"
OUT_MUS = DATA_DIR / "phase4_2_8b_bootstrap_mus.npz"
OUT_VERDICT = DATA_DIR / "phase4_2_8b_h4scaling_verdict.parquet"


def _parquet_path(size: str, motif: str) -> Path:
    """phase2_*_sweep.parquet for {70m,160m,410m}; phase3_1b_*; phase4_2_8b_*."""
    if size == "1b":
        return DATA_DIR / f"phase3_1b_{motif}_sweep.parquet"
    if size == "2.8b":
        return DATA_DIR / f"phase4_2_8b_{motif}_sweep.parquet"
    return DATA_DIR / f"phase2_{motif}_sweep.parquet"


def _per_cell_dir(size: str, motif: str) -> Path:
    if motif == "induction":
        suffix = "induction_per_seq"
    elif motif == "successor":
        suffix = "successor_per_prompt"
    else:
        suffix = "s_inhibition_per_prompt"
    if size == "1b":
        return DATA_DIR / f"phase3_1b_{suffix}"
    if size == "2.8b":
        return DATA_DIR / f"phase4_2_8b_{suffix}"
    return DATA_DIR / f"phase2_{suffix}"


def _load_size_curve(size: str, motif: str, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(_parquet_path(size, motif))
    sub = df[df["size"] == size].copy()
    if motif == "induction":
        passes = (sub["score"] > threshold).astype(int)
    else:
        passes = (sub["score"] >= threshold).astype(int)
    sub["passes"] = passes
    grouped = sub.groupby("step")["passes"].sum().sort_index()
    return grouped.index.values, grouped.values


def _run_bootstrap(size: str, motif: str, threshold: float, steps: np.ndarray, rng: np.random.Generator):
    cache_dir = _per_cell_dir(size, motif)
    if motif == "induction":
        per_seq = {
            int(s): np.load(cache_dir / f"{size}_step{s}.npz")["per_sequence_scores"]
            for s in steps
        }
        mus, mu_point = bootstrap_induction(per_seq, threshold, rng, B=1000)
    elif motif == "successor":
        per_pp = {}
        for s in steps:
            npz = np.load(cache_dir / f"{size}_step{s}.npz")
            per_pp[int(s)] = {
                "real": npz["per_prompt_real"],
                "null": npz["per_prompt_null"],
                "cats": npz["prompt_categories"],
            }
        mus, mu_point = bootstrap_successor(per_pp, threshold, rng, B=1000)
    else:  # s_inhibition
        per_pp: dict[int, np.ndarray] = {}
        nm_layers: dict[int, list[int]] = {}
        for s in steps:
            npz = np.load(cache_dir / f"{size}_step{s}.npz")
            per_pp[int(s)] = npz["per_prompt_delta"]
            nm_layers[int(s)] = npz["nm_heads"][:, 0].tolist()
        df = pd.read_parquet(_parquet_path(size, motif))
        a_grp = df[(df["size"] == size) & (df["step"] == int(steps[0]))]
        n_layers = int(a_grp["layer"].max() + 1)
        n_heads = int(a_grp["head"].max() + 1)
        sender_layers = np.array([L for L in range(n_layers) for _ in range(n_heads)])
        mus, mu_point = bootstrap_s_inhibition(
            per_pp, threshold, nm_layers, sender_layers, rng, B=1000
        )
    return mus, mu_point


def main() -> None:
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    bootstrap_mus: dict[str, np.ndarray] = {}
    overall_t0 = time.time()
    motifs = (
        ("induction", INDUCTION_THRESHOLD),
        ("successor", TAU_LIFT),
        ("s_inhibition", TAU_STRICT),
    )
    for size in ALL_SIZES:
        for motif, threshold in motifs:
            t0 = time.time()
            print(f"  Bootstrap {size:>4s} × {motif:>13s} ...", flush=True, end="")
            steps, counts = _load_size_curve(size, motif, threshold)
            mus, mu_point = _run_bootstrap(size, motif, threshold, steps, rng)
            bres = summarize_bootstrap(size, motif, threshold, mus, mu_point)
            bootstrap_mus[f"{size}_{motif}"] = mus
            fr = tiered_fit(
                size, motif, steps, counts, bootstrap_median_mu=bres.mu_bootstrap_median
            )
            rows.append(
                dict(
                    size=size,
                    motif=motif,
                    threshold=threshold,
                    regime=fr.regime,
                    max_count=fr.max_count,
                    mu_point=fr.mu,
                    mu_bootstrap_median=bres.mu_bootstrap_median,
                    mu_ci_low=bres.mu_ci_low,
                    mu_ci_high=bres.mu_ci_high,
                )
            )
            print(
                f" max={fr.max_count:>2d} regime={fr.regime:>9s} "
                f"μ={fr.mu:.0f} CI=[{bres.mu_ci_low:.0f}, {bres.mu_ci_high:.0f}] "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    mu_df = pd.DataFrame(rows)
    mu_df.to_parquet(OUT_MU, index=False)
    np.savez_compressed(OUT_MUS, **bootstrap_mus)
    print(f"\nWrote {OUT_MU.relative_to(REPO_ROOT)}")
    print(f"Wrote {OUT_MUS.relative_to(REPO_ROOT)}")

    # ============================================================
    # §H4-scaling verdict
    # ============================================================
    print()
    print("=" * 78)
    print("§H4-scaling verdict (HYPOTHESIS.md §H4-2 through §H4-5)")
    print("=" * 78)

    def _row(size: str, motif: str) -> pd.Series:
        return mu_df[(mu_df["size"] == size) & (mu_df["motif"] == motif)].iloc[0]

    # ---- (A.timing) bootstrap reversal-rate P(μ_si^2.8B < μ_si^410m) ≥ 0.95 ----
    mus_2_8b = bootstrap_mus["2.8b_s_inhibition"]
    mus_410m = bootstrap_mus["410m_s_inhibition"]
    finite_mask = np.isfinite(mus_2_8b) & np.isfinite(mus_410m)
    paired_lt = (mus_2_8b[finite_mask] < mus_410m[finite_mask]).astype(np.float64)
    timing_rate = float(paired_lt.mean())
    timing_pass = timing_rate >= 0.95
    print(
        f"\n(A.timing) P(μ_si^2.8B < μ_si^410m) = {timing_rate:.3f}  "
        f"[requirement ≥ 0.95]  → {'PASS' if timing_pass else 'FAIL'}"
    )

    # ---- (A.count) max_count_si^2.8B ≥ 5 ----
    si_2_8b = _row("2.8b", "s_inhibition")
    count_max = int(si_2_8b["max_count"])
    count_pass = count_max >= 5
    print(
        f"(A.count)  max_count_si^2.8B = {count_max}  [requirement ≥ 5]  → "
        f"{'PASS' if count_pass else 'FAIL'}"
    )

    # ---- Joint gate (§H4-3) ----
    gate_pass = timing_pass and count_pass
    print(f"\nJoint §H4-scaling gate (both legs): {'PASS' if gate_pass else 'FAIL'}")

    # ---- §H4-5 failure-mode pattern ----
    # Priority: TOOLING > NEITHER > COUNT-ONLY > TIMING-ONLY > PASS
    # TOOLING is detected externally (NaN/sign-flip patterns); skipped here.
    if gate_pass:
        pattern = "PASS"
    elif not timing_pass and not count_pass:
        pattern = "NEITHER"
    elif count_pass and not timing_pass:
        pattern = "COUNT-ONLY"
    elif timing_pass and not count_pass:
        pattern = "TIMING-ONLY"
    else:
        pattern = "UNKNOWN"
    print(f"§H4-5 failure-mode pattern: {pattern}")

    headlines = {
        "PASS": "Scaling argument confirmed: at Pythia-2.8B's 1024-head architecture, S-inhibition timing accelerates beyond 410m and head count exceeds the 410m saturation cap.",
        "TIMING-ONLY": "Timing-axis scaling holds at 2.8B; count-axis saturation extends from 1B's narrow architecture to 2.8B's 1024-head architecture, suggesting count saturation is fundamental rather than head-count-rate-limited.",
        "COUNT-ONLY": "Count-axis scaling unlocks at 1024 heads; timing-axis saturates between 410m and 2.8B.",
        "NEITHER": "Scaling argument falsified at 2.8B: both timing and count saturate beyond 410m on the head-count axis.",
    }
    print(f"Matched paper headline:\n  {headlines.get(pattern, '(custom)')}")

    # ---- Cross-size diagnostic: P(μ_si^X < μ_si^410m) for all sizes ----
    print()
    print("=" * 78)
    print("Reversal-rate diagnostics: P(μ_si^X < μ_si^410m) across all sizes")
    print("=" * 78)
    for size in ALL_SIZES:
        if size == "410m":
            continue
        mus = bootstrap_mus[f"{size}_s_inhibition"]
        mask = np.isfinite(mus) & np.isfinite(mus_410m)
        rate = float((mus[mask] < mus_410m[mask]).mean()) if mask.sum() else float("nan")
        print(f"  P(μ_si^{size:>4s} < μ_si^410m) = {rate:.3f}")

    # ---- Persist verdict parquet ----
    verdict_rows = [
        dict(leg="A.timing", pass_=timing_pass, value=f"{timing_rate:.3f}", requirement="≥ 0.95"),
        dict(leg="A.count", pass_=count_pass, value=str(count_max), requirement="≥ 5"),
        dict(leg="JOINT", pass_=gate_pass, value=str(int(timing_pass) + int(count_pass)), requirement="2/2"),
        dict(leg="PATTERN", pass_=False, value=pattern, requirement="(see §H4-5 priority)"),
    ]
    pd.DataFrame(verdict_rows).to_parquet(OUT_VERDICT, index=False)
    print(f"\nWrote {OUT_VERDICT.relative_to(REPO_ROOT)}")
    print(f"\nTotal analysis wall time: {(time.time() - overall_t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
