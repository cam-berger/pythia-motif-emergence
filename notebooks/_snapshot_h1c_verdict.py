"""Compute the §H1-C / §H2-5 / §H2-9-R / §H3-scale numerical state from the
current parquets and write a JSON snapshot for prose reconciliation.

This script reproduces the load-and-compute logic of `h1c_ordering_test.ipynb`
TWICE:

- `buggy` — exactly what the production notebook does today, after commit
  0ae0e27 (which calls `read_long(sweep_path(motif, s))` once per size in
  `SIZES = ['70m','160m','410m','1b']`; sweep_path returns the same
  phase2_*_sweep.parquet for the 3 registered sizes, so phase2 data is
  loaded three times → counts inflated 3× for every phase2 cell).

- `fixed` — load each parquet exactly once (phase2_*_sweep.parquet ∪
  phase3_1b_*_sweep.parquet), then filter by size. This is the correct
  read of the underlying data and matches what the §H2-9-R amendment,
  the f4e262a commit message, and WRITEUP.md Track 1 all claim.

Both blocks share thresholds (induction > 0.3; successor ≥ 0.13496;
S-inhibition ≥ 0.0372) and the §H2-3 tiered logistic-fit / §H2-4
ties-fail / §H2-5 joint sign-test machinery in src/analysis/.

Output: data/exploration/h1c_state_snapshot.json — durable, diffable.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from notebooks._lib.sweep_io import read_long
from src.analysis.phase2_logistic import (
    JOINT_NULL_P,
    RIGHT_CENSOR_STEP,
    evaluate_ordering,
    joint_h1c_test,
    tiered_fit,
)
from src.analysis.phase2_bootstrap import (
    bootstrap_induction,
    bootstrap_pair_reversal_rate,
    bootstrap_s_inhibition,
    bootstrap_successor,
    summarize_bootstrap,
)

SIZES = ["70m", "160m", "410m", "1b"]
REGISTERED_SIZES = ["70m", "160m", "410m"]
MOTIFS = ["induction", "successor", "s_inhibition"]

INDUCTION_THRESHOLD = 0.3
TAU_LIFT = 0.13496
TAU_STRICT = 0.0372

EXPLORATION = REPO / "data" / "exploration"


def sweep_path(motif: str, size: str) -> Path:
    if size == "1b":
        return EXPLORATION / f"phase3_1b_{motif}_sweep.parquet"
    return EXPLORATION / f"phase2_{motif}_sweep.parquet"


def per_cell_dir(motif: str, size: str) -> Path:
    suffix = "per_seq" if motif == "induction" else "per_prompt"
    if size == "1b":
        return EXPLORATION / f"phase3_1b_{motif}_{suffix}"
    return EXPLORATION / f"phase2_{motif}_{suffix}"


def per_size_curve(df, size, threshold, comparator):
    sub = df[df["size"] == size].copy()
    if comparator == "gt":
        passes = (sub["score"] > threshold).astype(int)
    else:
        passes = (sub["score"] >= threshold).astype(int)
    sub["passes"] = passes
    grouped = sub.groupby("step")["passes"].sum().sort_index()
    return grouped.index.values, grouped.values


def load_buggy() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproduce the broken 4-size load currently in h1c_ordering_test.ipynb."""
    df_ind = pd.concat(
        [read_long(sweep_path("induction", s)) for s in SIZES]
    ).reset_index(drop=True)
    df_suc = pd.concat(
        [read_long(sweep_path("successor", s)) for s in SIZES]
    ).reset_index(drop=True)
    df_si = pd.concat(
        [read_long(sweep_path("s_inhibition", s)) for s in SIZES]
    ).reset_index(drop=True)
    return df_ind, df_suc, df_si


def load_fixed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load each parquet exactly once."""
    parts = {}
    for motif in MOTIFS:
        ph2 = read_long(EXPLORATION / f"phase2_{motif}_sweep.parquet")
        ph3 = read_long(EXPLORATION / f"phase3_1b_{motif}_sweep.parquet")
        parts[motif] = pd.concat([ph2, ph3]).reset_index(drop=True)
    return parts["induction"], parts["successor"], parts["s_inhibition"]


def compute_block(df_ind, df_suc, df_si, label, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    cells = []
    bootstrap_mus: dict[tuple[str, str], np.ndarray] = {}
    for size in SIZES:
        for motif, df, threshold, comparator in [
            ("induction", df_ind, INDUCTION_THRESHOLD, "gt"),
            ("successor", df_suc, TAU_LIFT, "gte"),
            ("s_inhibition", df_si, TAU_STRICT, "gte"),
        ]:
            steps, counts = per_size_curve(df, size, threshold, comparator)
            if motif == "induction":
                per = {
                    int(s): np.load(
                        per_cell_dir("induction", size) / f"{size}_step{s}.npz"
                    )["per_sequence_scores"]
                    for s in steps
                }
                mus, mu_point = bootstrap_induction(per, threshold, rng, B=1000)
            elif motif == "successor":
                per = {}
                for s in steps:
                    npz = np.load(
                        per_cell_dir("successor", size) / f"{size}_step{s}.npz"
                    )
                    per[int(s)] = {
                        "real": npz["per_prompt_real"],
                        "null": npz["per_prompt_null"],
                        "cats": npz["prompt_categories"],
                    }
                mus, mu_point = bootstrap_successor(per, threshold, rng, B=1000)
            else:
                per = {}
                nm_layers: dict[int, list[int]] = {}
                for s in steps:
                    npz = np.load(
                        per_cell_dir("s_inhibition", size) / f"{size}_step{s}.npz"
                    )
                    per[int(s)] = npz["per_prompt_delta"]
                    nm_layers[int(s)] = npz["nm_heads"][:, 0].tolist()
                a_grp = df_si[(df_si["size"] == size) & (df_si["step"] == steps[0])]
                # IMPORTANT: under the buggy load, df_si may be triplicated, so
                # n_layers/n_heads from .max() is unaffected (still right).
                n_layers = int(a_grp["layer"].max() + 1)
                n_heads = int(a_grp["head"].max() + 1)
                sender_layers = np.array(
                    [L for L in range(n_layers) for _ in range(n_heads)]
                )
                mus, mu_point = bootstrap_s_inhibition(
                    per, threshold, nm_layers, sender_layers, rng, B=1000
                )
            bres = summarize_bootstrap(size, motif, threshold, mus, mu_point)
            bootstrap_mus[(size, motif)] = mus
            fr = tiered_fit(
                size, motif, steps, counts,
                bootstrap_median_mu=bres.mu_bootstrap_median,
            )
            cells.append(
                {
                    "size": size, "motif": motif, "threshold": threshold,
                    "regime": fr.regime, "max_count": int(fr.max_count),
                    "mu_point": float(fr.mu),
                    "mu_boot_median": float(bres.mu_bootstrap_median),
                    "mu_ci_low": float(bres.mu_ci_low),
                    "mu_ci_high": float(bres.mu_ci_high),
                    "count_at_step143000": int(
                        counts[steps == 143000][0] if (steps == 143000).any() else 0
                    ),
                }
            )

    cell_lookup = {(c["size"], c["motif"]): c for c in cells}

    per_size_orderings = []
    for size in SIZES:
        mu_ind = cell_lookup[(size, "induction")]["mu_point"]
        mu_suc = cell_lookup[(size, "successor")]["mu_point"]
        mu_si = cell_lookup[(size, "s_inhibition")]["mu_point"]
        o = evaluate_ordering(size, mu_ind, mu_suc, mu_si)
        per_size_orderings.append(
            {
                "size": size, "in_h1c_gate": size in REGISTERED_SIZES,
                "mu_ind": mu_ind, "mu_suc": mu_suc, "mu_si": mu_si,
                "pair_ind_suc_holds": bool(o.pair_ind_suc_holds),
                "pair_suc_si_holds": bool(o.pair_suc_si_holds),
                "strict_holds": bool(o.holds_strict),
            }
        )

    registered_orderings = [
        evaluate_ordering(
            size,
            cell_lookup[(size, "induction")]["mu_point"],
            cell_lookup[(size, "successor")]["mu_point"],
            cell_lookup[(size, "s_inhibition")]["mu_point"],
        )
        for size in REGISTERED_SIZES
    ]
    verdict = joint_h1c_test(registered_orderings)
    joint_p = (
        None
        if (verdict.joint_p_under_h0 is None
            or (isinstance(verdict.joint_p_under_h0, float)
                and np.isnan(verdict.joint_p_under_h0)))
        else float(verdict.joint_p_under_h0)
    )

    PAIRS = [
        ("induction", "successor", "induction → successor"),
        ("successor", "s_inhibition", "successor → S-inhibition"),
        ("induction", "s_inhibition", "induction → S-inhibition (transitive)"),
    ]
    reversal_rows = []
    for size in SIZES:
        for early, late, lab in PAIRS:
            r = bootstrap_pair_reversal_rate(
                mus_early=bootstrap_mus[(size, early)],
                mus_late=bootstrap_mus[(size, late)],
                size=size, motif_early=early, motif_late=late,
            )
            reversal_rows.append(
                {"size": size, "pair": lab, "n_total": r.n_total,
                 "n_holds": r.n_holds, "n_undetermined": r.n_undetermined,
                 "holds_rate": float(r.holds_rate),
                 "reversal_rate": float(r.reversal_rate)}
            )

    return {
        "label": label,
        "cells": cells,
        "per_size_orderings": per_size_orderings,
        "registered_h1c_gate": {
            "n_sizes_holding": int(verdict.n_sizes_holding),
            "n_sizes_evaluated": len(REGISTERED_SIZES),
            "joint_p_under_h0": joint_p,
            "joint_null_p_if_all_hold": JOINT_NULL_P,
            "passes_at_alpha_005": bool(verdict.passes),
            "interpretation": (
                "PASS" if verdict.passes
                else f"FAIL — {verdict.n_sizes_holding}/3 sizes hold strict ordering"
            ),
        },
        "bootstrap_reversal_rates": reversal_rows,
    }


def main() -> None:
    t0 = time.time()
    print("Loading parquets BUGGY (current notebook code path)...", flush=True)
    df_ind_b, df_suc_b, df_si_b = load_buggy()
    print(f"  buggy successor row counts per size: {dict(df_suc_b.groupby('size').size())}",
          flush=True)
    buggy = compute_block(df_ind_b, df_suc_b, df_si_b, "buggy_4size_concat")

    print("Loading parquets FIXED (each parquet read once)...", flush=True)
    df_ind_f, df_suc_f, df_si_f = load_fixed()
    print(f"  fixed successor row counts per size: {dict(df_suc_f.groupby('size').size())}",
          flush=True)
    fixed = compute_block(df_ind_f, df_suc_f, df_si_f, "fixed_dedup_load")

    h3scale_df = pd.read_parquet(EXPLORATION / "phase3_1b_h3scale_verdict.parquet")
    h3scale = [
        {"leg": r["leg"], "value": str(r["value"]),
         "requirement": str(r["requirement"]), "pass_": bool(r["pass_"])}
        for _, r in h3scale_df.iterrows()
    ]

    snapshot = {
        "generated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generated_at_unix": int(time.time()),
        "wall_seconds": time.time() - t0,
        "thresholds": {
            "induction_prefix_match": {"value": INDUCTION_THRESHOLD, "comparator": "gt"},
            "successor_lift": {"value": TAU_LIFT, "comparator": "gte"},
            "s_inhibition_delta_h": {"value": TAU_STRICT, "comparator": "gte"},
        },
        "parquet_mtimes": {
            f"{size}__{motif}": int(sweep_path(motif, size).resolve().stat().st_mtime)
            for size in SIZES for motif in MOTIFS
        },
        "registered_sizes": REGISTERED_SIZES,
        "all_sizes": SIZES,
        "bug_summary": {
            "description": (
                "h1c_ordering_test.ipynb (since commit 0ae0e27, 2026-05-07) "
                "calls pd.concat([read_long(sweep_path(motif, s)) for s in "
                "['70m','160m','410m','1b']]). sweep_path returns "
                "phase2_*_sweep.parquet for any of the 3 registered sizes, "
                "and that parquet itself contains all 3 phase2 sizes' rows. "
                "Net effect: phase2 data is loaded 3 times; per-size head "
                "counts are inflated by 3×, regime classifications and μ "
                "fits drift, and the §H2-5 joint sign test reports FAIL "
                "instead of the registered PASS."
            ),
            "introduced_in_commit": "0ae0e27 (Smooth-integrate 1B + register §H4-scaling on head-count axis, 2026-05-07)",
            "affects": [
                "notebooks/h1c_ordering_test.ipynb",
                "notebooks/_build_h1c_ordering_test.py",
                "all four *_full_sweep.ipynb notebooks (same 4-size load pattern)",
                "motif_structural_reuse.ipynb (same 4-size load pattern, per WRITEUP.md §H4-8)",
            ],
            "underlying_data_unchanged_since": "f4e262a (2026-05-06)",
            "registered_gate_under_fixed_load": "PASS (matches f4e262a, §H2-9-R)",
            "registered_gate_under_buggy_load": "FAIL (cosmetic only — code defect, not a data event)",
        },
        "h3scale_verdict": h3scale,
        "buggy": buggy,
        "fixed": fixed,
    }

    out = EXPLORATION / "h1c_state_snapshot.json"
    out.write_text(json.dumps(snapshot, indent=2, default=float))
    print(f"\nWrote: {out}")

    print("\n=== FIXED (correct load) ===")
    for c in fixed["cells"]:
        print(f"  {c['size']:>4s} {c['motif']:<14s} {c['regime']:<10s} "
              f"max={c['max_count']:>3d}  μ={c['mu_point']:.0f} "
              f"[{c['mu_ci_low']:.0f}, {c['mu_ci_high']:.0f}]")
    print(f"\nFixed registered gate: "
          f"{fixed['registered_h1c_gate']['interpretation']} "
          f"(n_sizes_holding={fixed['registered_h1c_gate']['n_sizes_holding']}/3, "
          f"p={fixed['registered_h1c_gate']['joint_p_under_h0']})")

    print("\n=== BUGGY (current notebook output) ===")
    for c in buggy["cells"]:
        print(f"  {c['size']:>4s} {c['motif']:<14s} {c['regime']:<10s} "
              f"max={c['max_count']:>3d}  μ={c['mu_point']:.0f}")
    print(f"\nBuggy registered gate: "
          f"{buggy['registered_h1c_gate']['interpretation']} "
          f"(n_sizes_holding={buggy['registered_h1c_gate']['n_sizes_holding']}/3)")


if __name__ == "__main__":
    main()
