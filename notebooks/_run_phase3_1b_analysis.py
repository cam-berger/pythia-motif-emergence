"""Phase 3 1B bootstrap + §H3-scale verdict analysis (HYPOTHESIS.md §H3-scale).

Runs the §H2-2 per-prompt bootstrap (B=1000, 95% CI) on all 4 sizes
(70m, 160m, 410m, 1b) × 3 motifs, computes the §H3-scale conjunctive
gate (A.i, A.ii, A.iii, B.i, B.ii), and persists results for the
new 1B notebooks + h1c_ordering_test.ipynb extension.

Outputs:
  - data/exploration/phase3_1b_bootstrap_mu.parquet — per-(size, motif) μ
    point estimate, bootstrap median, 95% CI, max_count, regime
  - data/exploration/phase3_1b_bootstrap_mus.npz — raw B=1000 mus_arrays
    keyed by '{size}_{motif}', for reversal-rate and CI-overlap downstream
  - data/exploration/phase3_1b_h3scale_verdict.parquet — leg-by-leg pass/fail
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

from src.analysis.phase2_bootstrap import (  # noqa: E402
    bootstrap_induction,
    bootstrap_pair_reversal_rate,
    bootstrap_s_inhibition,
    bootstrap_successor,
    summarize_bootstrap,
)
from src.analysis.phase2_logistic import (  # noqa: E402
    RIGHT_CENSOR_STEP,
    evaluate_ordering,
    tiered_fit,
)

# Locked thresholds
INDUCTION_THRESHOLD = 0.3
TAU_LIFT = 0.13496
TAU_STRICT = 0.0372

ALL_SIZES = ("70m", "160m", "410m", "1b")
DATA_DIR = REPO_ROOT / "data" / "exploration"
OUT_MU = DATA_DIR / "phase3_1b_bootstrap_mu.parquet"
OUT_MUS = DATA_DIR / "phase3_1b_bootstrap_mus.npz"
OUT_VERDICT = DATA_DIR / "phase3_1b_h3scale_verdict.parquet"


def _parquet_path(size: str, motif: str) -> Path:
    """Phase 2 (3-size) sweeps live in `phase2_*.parquet`; 1B in `phase3_1b_*.parquet`."""
    if size == "1b":
        return DATA_DIR / f"phase3_1b_{motif}_sweep.parquet"
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
    return DATA_DIR / f"phase2_{suffix}"


def _load_size_curve(size: str, motif: str, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, counts) for a single (size, motif) cell."""
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
    # §H3-scale verdict
    # ============================================================
    print()
    print("=" * 78)
    print("§H3-scale verdict (HYPOTHESIS.md §H3-scale-2 through §H3-scale-6)")
    print("=" * 78)

    def _row(size: str, motif: str) -> pd.Series:
        return mu_df[(mu_df["size"] == size) & (mu_df["motif"] == motif)].iloc[0]

    # ---- (A.i) max_count_si^1B ≥ 5 ----
    si_1b = _row("1b", "s_inhibition")
    Ai_max = int(si_1b["max_count"])
    Ai_pass = Ai_max >= 5
    print(
        f"\n(A.i)  max_count_si^1B = {Ai_max}  [requirement ≥ 5]  → "
        f"{'PASS' if Ai_pass else 'FAIL'}"
    )

    # ---- (A.ii) bootstrap reversal-rate P(μ_si^1B < μ_si^410m) ≥ 0.95 ----
    mus_1b = bootstrap_mus["1b_s_inhibition"]
    mus_410m = bootstrap_mus["410m_s_inhibition"]
    finite_mask = np.isfinite(mus_1b) & np.isfinite(mus_410m)
    paired_lt = (mus_1b[finite_mask] < mus_410m[finite_mask]).astype(np.float64)
    Aii_rate = float(paired_lt.mean())
    Aii_pass = Aii_rate >= 0.95
    print(
        f"(A.ii) P(μ_si^1B < μ_si^410m) = {Aii_rate:.3f}  [requirement ≥ 0.95]  → "
        f"{'PASS' if Aii_pass else 'FAIL'}"
    )

    # ---- (A.iii) within-1B disjoint CI on μ_si^1B vs μ_suc^1B ----
    suc_1b = _row("1b", "successor")
    Aiii_si_low = float(si_1b["mu_ci_low"])
    Aiii_si_high = float(si_1b["mu_ci_high"])
    Aiii_suc_low = float(suc_1b["mu_ci_low"])
    Aiii_suc_high = float(suc_1b["mu_ci_high"])
    Aiii_disjoint = Aiii_si_low > Aiii_suc_high  # μ_si entirely above μ_suc
    Aiii_pass = Aiii_disjoint
    print(
        f"(A.iii) μ_si^1B  CI = [{Aiii_si_low:.0f}, {Aiii_si_high:.0f}]"
    )
    print(
        f"        μ_suc^1B CI = [{Aiii_suc_low:.0f}, {Aiii_suc_high:.0f}]  → "
        f"{'PASS (disjoint, si above)' if Aiii_pass else 'FAIL (CIs overlap or wrong order)'}"
    )

    # ---- (B.i) all 3 motifs at 1B in full-fit regime ----
    ind_1b = _row("1b", "induction")
    regimes_1b = (ind_1b["regime"], suc_1b["regime"], si_1b["regime"])
    Bi_pass = all(r == "emerged" for r in regimes_1b)
    print(
        f"\n(B.i)  1B regimes: ind={ind_1b['regime']}, suc={suc_1b['regime']}, "
        f"si={si_1b['regime']}  → {'PASS' if Bi_pass else 'FAIL'}"
    )

    # ---- (B.ii) strict ordering μ_ind^1B < μ_suc^1B < μ_si^1B ----
    mu_ind = float(ind_1b["mu_point"])
    mu_suc = float(suc_1b["mu_point"])
    mu_si = float(si_1b["mu_point"])
    ord_1b = evaluate_ordering("1b", mu_ind, mu_suc, mu_si)
    Bii_pass = ord_1b.holds_strict
    print(
        f"(B.ii) μ_ind={mu_ind:.0f} < μ_suc={mu_suc:.0f} < μ_si={mu_si:.0f}  → "
        f"{'PASS' if Bii_pass else 'FAIL'}"
    )
    print(
        f"       pair (ind→suc): {'HOLDS' if ord_1b.pair_ind_suc_holds else 'FAILS'}"
    )
    print(
        f"       pair (suc→si):  {'HOLDS' if ord_1b.pair_suc_si_holds else 'FAILS'}"
    )

    # ---- Joint gate (§H3-scale-4) ----
    legs = {
        "A.i": Ai_pass,
        "A.ii": Aii_pass,
        "A.iii": Aiii_pass,
        "B.i": Bi_pass,
        "B.ii": Bii_pass,
    }
    gate_pass = all(legs.values())
    failed = [k for k, v in legs.items() if not v]
    print()
    print(f"Joint §H3-scale gate (5/5 must hold): {'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        print(f"  Failed legs: {', '.join(failed)}")

    # ---- §H3-scale-6 failure-mode pattern ----
    # Priority: TOOLING > REGR > ORD-BREAK > WIDE-CI > SAT > PASS
    if gate_pass:
        pattern = "PASS"
    elif not Ai_pass:
        pattern = "REGR"
    elif not Bii_pass:
        pattern = "ORD-BREAK"
    elif not Aiii_pass:
        pattern = "WIDE-CI"
    elif not Aii_pass:
        pattern = "SAT"
    else:
        pattern = "UNKNOWN"
    print(f"§H3-scale-6 failure-mode pattern: {pattern}")

    headlines = {
        "PASS": "Scale-dependent S-inhibition emergence holds across Pythia 70m → 1B.",
        "SAT": "S-inhibition emergence time saturates between 410m and 1B; head count continues to scale.",
        "REGR": "Non-monotonic S-inhibition emergence in Pythia: regression at 1B.",
        "ORD-BREAK": "H1-C ordering is scale-bounded in Pythia: breaks at 1B.",
        "WIDE-CI": "1B reproduces 160m's marginal-overlap pattern, not 410m's clean separation; scale-dependence is non-monotonic in CI-axis.",
    }
    print(f"Matched paper headline:\n  {headlines.get(pattern, '(custom)')}")

    # ---- Reversal-rate diagnostics for the existing 3 sizes (also valuable) ----
    print()
    print("=" * 78)
    print("Reversal-rate diagnostics: P(μ_si < μ_si^410m) across sizes")
    print("=" * 78)
    for size in ALL_SIZES:
        if size == "410m":
            continue
        mus = bootstrap_mus[f"{size}_s_inhibition"]
        mask = np.isfinite(mus) & np.isfinite(mus_410m)
        if mask.sum() == 0:
            print(f"  {size:>4s}: no finite paired bootstraps")
            continue
        rate = float((mus[mask] < mus_410m[mask]).mean())
        print(f"  P(μ_si^{size:>4s} < μ_si^410m) = {rate:.3f}")

    # ---- Persist verdict parquet ----
    verdict_rows = [
        dict(leg="A.i", pass_=Ai_pass, value=str(Ai_max), requirement="≥ 5"),
        dict(leg="A.ii", pass_=Aii_pass, value=f"{Aii_rate:.3f}", requirement="≥ 0.95"),
        dict(leg="A.iii", pass_=Aiii_pass, value=f"{Aiii_si_low - Aiii_suc_high:.0f}", requirement="μ_si CI strictly above μ_suc CI"),
        dict(leg="B.i", pass_=Bi_pass, value=str(sum(r == "emerged" for r in regimes_1b)), requirement="3/3 motifs in 'emerged' regime"),
        dict(leg="B.ii", pass_=Bii_pass, value=str(int(ord_1b.holds_strict)), requirement="μ_ind < μ_suc < μ_si"),
        dict(leg="JOINT", pass_=gate_pass, value=str(sum(legs.values())), requirement="5/5"),
        dict(leg="PATTERN", pass_=False, value=pattern, requirement="(see §H3-scale-6 priority)"),
    ]
    pd.DataFrame(verdict_rows).to_parquet(OUT_VERDICT, index=False)
    print(f"\nWrote {OUT_VERDICT.relative_to(REPO_ROOT)}")

    print(f"\nTotal analysis wall time: {(time.time() - overall_t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
