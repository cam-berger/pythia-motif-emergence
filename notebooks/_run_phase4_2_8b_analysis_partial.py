"""Phase 4 2.8B PARTIAL bootstrap analysis (HYPOTHESIS.md §H4-7-supersede).

Per §H4-7-supersede (committed 2026-05-08), the 2.8B S-inhibition sweep was
halted at 8 of 40 cells under the §H4-7 per-cell-cost escape hatch (observed
~57 min/cell vs ~6 min/cell projection). The §H4 conjunctive gate is therefore
DEFERRED: (A.timing) and (A.count) cannot be evaluated.

This script computes the cross-size bootstrap analysis for the **5 sizes ×
2 motifs (induction + successor) only**. S-inhibition is explicitly skipped.
Induction and successor at 2.8B are reported as side observations per §H4-1.

Outputs:
  - data/exploration/phase4_2_8b_bootstrap_mu_partial.parquet (5 sizes × 2 motifs)
  - data/exploration/phase4_2_8b_bootstrap_mus_partial.npz
  - data/exploration/phase4_2_8b_h4scaling_verdict_partial.parquet (DEFERRED)
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

from src.analysis.phase2_bootstrap import (  # noqa: E402
    bootstrap_induction,
    bootstrap_successor,
    summarize_bootstrap,
)
from src.analysis.phase2_logistic import tiered_fit  # noqa: E402

INDUCTION_THRESHOLD = 0.3
TAU_LIFT = 0.13496

ALL_SIZES = ("70m", "160m", "410m", "1b", "2.8b")
DATA_DIR = REPO_ROOT / "data" / "exploration"
OUT_MU = DATA_DIR / "phase4_2_8b_bootstrap_mu_partial.parquet"
OUT_MUS = DATA_DIR / "phase4_2_8b_bootstrap_mus_partial.npz"
OUT_VERDICT = DATA_DIR / "phase4_2_8b_h4scaling_verdict_partial.parquet"


def _parquet_path(size: str, motif: str) -> Path:
    if size == "1b":
        return DATA_DIR / f"phase3_1b_{motif}_sweep.parquet"
    if size == "2.8b":
        return DATA_DIR / f"phase4_2_8b_{motif}_sweep.parquet"
    return DATA_DIR / f"phase2_{motif}_sweep.parquet"


def _per_cell_dir(size: str, motif: str) -> Path:
    suffix = "induction_per_seq" if motif == "induction" else "successor_per_prompt"
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
    else:  # successor
        per_pp = {}
        for s in steps:
            npz = np.load(cache_dir / f"{size}_step{s}.npz")
            per_pp[int(s)] = {
                "real": npz["per_prompt_real"],
                "null": npz["per_prompt_null"],
                "cats": npz["prompt_categories"],
            }
        mus, mu_point = bootstrap_successor(per_pp, threshold, rng, B=1000)
    return mus, mu_point


def main() -> None:
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    bootstrap_mus: dict[str, np.ndarray] = {}
    overall_t0 = time.time()
    motifs = (
        ("induction", INDUCTION_THRESHOLD),
        ("successor", TAU_LIFT),
    )
    for size in ALL_SIZES:
        for motif, threshold in motifs:
            t0 = time.time()
            print(f"  Bootstrap {size:>4s} × {motif:>9s} ...", flush=True, end="")
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

    print()
    print("=" * 78)
    print("§H4-scaling verdict (HYPOTHESIS.md §H4-7-supersede DEFERRED state)")
    print("=" * 78)
    print()
    print("(A.timing) P(μ_si^2.8B < μ_si^410m) = UNDETERMINABLE")
    print("           — S-inhibition sweep halted at 8/40 cells; μ_si^2.8B not extractable.")
    print("(A.count)  max_count_si^2.8B = UNDETERMINABLE")
    print("           — emergence-relevant cells (steps 5000+) not sampled.")
    print()
    print("Joint §H4-scaling gate: DEFERRED")
    print("Matched §H4-5 pattern: DEFERRED (registered in §H4-7-supersede 2026-05-08)")
    print()
    print("Side observations (induction + successor, 5 sizes, full sweeps):")
    for size in ALL_SIZES:
        ind = mu_df[(mu_df["size"] == size) & (mu_df["motif"] == "induction")].iloc[0]
        suc = mu_df[(mu_df["size"] == size) & (mu_df["motif"] == "successor")].iloc[0]
        print(
            f"  {size:>4s}: ind μ={ind['mu_point']:>7.0f} (max={int(ind['max_count'])}, regime={ind['regime']:>9s})  "
            f"suc μ={suc['mu_point']:>7.0f} (max={int(suc['max_count'])}, regime={suc['regime']:>9s})"
        )

    verdict_rows = [
        dict(leg="A.timing", pass_=False, value="UNDETERMINABLE", requirement="≥ 0.95"),
        dict(leg="A.count", pass_=False, value="UNDETERMINABLE", requirement="≥ 5"),
        dict(leg="JOINT", pass_=False, value="DEFERRED", requirement="2/2"),
        dict(
            leg="PATTERN",
            pass_=False,
            value="DEFERRED",
            requirement="(see §H4-5 priority + §H4-7-supersede)",
        ),
    ]
    pd.DataFrame(verdict_rows).to_parquet(OUT_VERDICT, index=False)
    print(f"\nWrote {OUT_VERDICT.relative_to(REPO_ROOT)}")
    print(f"\nTotal analysis wall time: {(time.time() - overall_t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
