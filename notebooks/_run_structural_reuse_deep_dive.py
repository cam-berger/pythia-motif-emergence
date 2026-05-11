"""Structural-reuse deep dive (A12 stretch deliverable, 2026-05-11).

Pure analytical extraction; no new compute. For each (size, step) cell in the
5-size sweep, counts heads above each motif's locked detection threshold and
intersects the populations pairwise.

Question (per the 3-week plan §A12 stretch): Is the §H5 NULL × NULL
causal-disjointness consistent with the cross-motif head-population overlap
observed in the sweep data?

One-sentence answer: yes — successor and S-inhibition operate on structurally
disjoint head populations at every (size, step) cell in our 5-size sweep, and
the §H5 NULL × NULL is the natural readout when two disjoint populations are
independently probed.

Inputs (all sealed):
  - phase2_{induction,successor,s_inhibition}_sweep.parquet
  - phase3_1b_{induction,successor,s_inhibition}_sweep.parquet
  - phase4_2_8b_{induction,successor}_sweep.parquet
  - phase4_2_8b_s_inhibition_supersede_sweep.parquet

Outputs:
  - data/exploration/structural_reuse_deep_dive.parquet  (170 rows)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPL = REPO_ROOT / "data" / "exploration"

# Locked thresholds per HYPOTHESIS.md
TAU_IND = 0.30
TAU_LIFT = 0.13496
TAU_STRICT = 0.0372

SIZES = ["70m", "160m", "410m", "1b", "2.8b"]


def load_long(motif: str) -> pd.DataFrame:
    """Concatenate phase2 / phase3_1b / phase4_2_8b sweeps for one motif.

    S-inhibition's 2.8b parquet is the §H4-supersede 10-cell reduced grid;
    other motifs use the full 40-cell §H4-scaling sweep.
    """
    dfs = []
    if motif == "s_inhibition":
        dfs.append(pd.read_parquet(EXPL / "phase2_s_inhibition_sweep.parquet"))
        dfs.append(pd.read_parquet(EXPL / "phase3_1b_s_inhibition_sweep.parquet"))
        dfs.append(
            pd.read_parquet(EXPL / "phase4_2_8b_s_inhibition_supersede_sweep.parquet")
        )
    else:
        dfs.append(pd.read_parquet(EXPL / f"phase2_{motif}_sweep.parquet"))
        dfs.append(pd.read_parquet(EXPL / f"phase3_1b_{motif}_sweep.parquet"))
        dfs.append(pd.read_parquet(EXPL / f"phase4_2_8b_{motif}_sweep.parquet"))
    return pd.concat(dfs).reset_index(drop=True)


def heads_above(df: pd.DataFrame, size: str, step: int, tau: float) -> set[tuple[int, int]]:
    sub = df.loc[
        (df["size"] == size) & (df["step"] == step) & (df["score"] >= tau),
        ["layer", "head"],
    ]
    return set(map(tuple, sub.values.tolist()))


def main() -> None:
    df_ind = load_long("induction")
    df_suc = load_long("successor")
    df_si = load_long("s_inhibition")
    print(f"Loaded sweeps: ind={df_ind.shape}, suc={df_suc.shape}, si={df_si.shape}")

    rows: list[dict] = []
    for size in SIZES:
        si_steps = set(df_si.loc[df_si["size"] == size, "step"].unique())
        ind_steps = set(df_ind.loc[df_ind["size"] == size, "step"].unique())
        suc_steps = set(df_suc.loc[df_suc["size"] == size, "step"].unique())
        common = sorted(si_steps & ind_steps & suc_steps)
        for step in common:
            ind_h = heads_above(df_ind, size, step, TAU_IND)
            suc_h = heads_above(df_suc, size, step, TAU_LIFT)
            si_h = heads_above(df_si, size, step, TAU_STRICT)
            rows.append(
                dict(
                    size=size,
                    step=int(step),
                    n_ind=len(ind_h),
                    n_suc=len(suc_h),
                    n_si=len(si_h),
                    n_ind_suc=len(ind_h & suc_h),
                    n_ind_si=len(ind_h & si_h),
                    n_suc_si=len(suc_h & si_h),
                    n_triple=len(ind_h & suc_h & si_h),
                    suc_si_heads=";".join(f"L{l}H{h}" for l, h in sorted(suc_h & si_h)),
                    ind_si_heads=";".join(f"L{l}H{h}" for l, h in sorted(ind_h & si_h)),
                )
            )
    out = pd.DataFrame(rows)

    out_path = EXPL / "structural_reuse_deep_dive.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}  ({len(out):,} rows)")

    # Per-size summary
    print("\n=== suc ∩ si trajectory per size ===")
    for size in SIZES:
        sub = out[out["size"] == size]
        nonzero = sub[sub["n_suc_si"] > 0]
        if len(nonzero) > 0:
            heads_union = set()
            for h_str in sub["suc_si_heads"]:
                if h_str:
                    for h in h_str.split(";"):
                        heads_union.add(h)
            print(
                f"  {size:>5s}: {len(nonzero)}/{len(sub)} steps with suc∩si>0; "
                f"heads ever in overlap: {sorted(heads_union)}"
            )
        else:
            print(f"  {size:>5s}: ZERO suc∩si across all {len(sub)} steps")


if __name__ == "__main__":
    main()
