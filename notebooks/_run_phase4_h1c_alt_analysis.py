"""§H1-C-altdetectors robustness analysis under §H1-C-altdetectors-2-rr-supersede.

Implements the within-Pythia frozen-threshold + rank-only secondary view per
the amendment. Produces:

  1. Pre-validation table: top-K(locked) ∩ top-K(alt) at step 143000 per (size, motif).
  2. Frozen-threshold trajectory: per-cell pass count under threshold = K-th-
     largest alt-score at step 143000 (K = locked-detector final count).
  3. Top-5% sensitivity-of-sensitivity: same but K' = ceil(n_layers*n_heads*0.05).
  4. Rank-only secondary: per-cell |alt_top_K_step_t ∩ top_K_final| / K trajectory.
  5. Emergence step under each scheme: smallest step with metric ≥ half-final.
  6. Joint sign-test under all four detector triples × {frozen-final-K, top-5%,
     rank-only} schemes.

Outputs:
  - data/exploration/phase4_h1c_alt_prevalidation.parquet
  - data/exploration/phase4_h1c_alt_trajectories.parquet (long: size, motif, detector_kind, scheme, step, metric)
  - data/exploration/phase4_h1c_alt_emergence_steps.parquet (per scheme + triple)
  - data/exploration/phase4_h1c_alt_joint_verdict.parquet (joint sign-test outputs)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import math

import numpy as np
import pandas as pd

OUT_PREVAL = REPO_ROOT / "data" / "exploration" / "phase4_h1c_alt_prevalidation.parquet"
OUT_TRAJ = REPO_ROOT / "data" / "exploration" / "phase4_h1c_alt_trajectories.parquet"
OUT_EMERG = REPO_ROOT / "data" / "exploration" / "phase4_h1c_alt_emergence_steps.parquet"
OUT_VERDICT = REPO_ROOT / "data" / "exploration" / "phase4_h1c_alt_joint_verdict.parquet"

FINAL_STEP = 143000

# Locked detector thresholds for pass-count computation
LOCKED_TAU = {
    "induction": ("score > 0.30", lambda s: s > 0.30),
    "successor": ("score >= 0.13496", lambda s: s >= 0.13496),
    "s_inhibition": ("score >= 0.0372", lambda s: s >= 0.0372),
}

# Locked detector parquet paths
LOCKED_PARQUET = {
    "induction":    REPO_ROOT / "data" / "exploration" / "phase2_induction_sweep.parquet",
    "successor":    REPO_ROOT / "data" / "exploration" / "phase2_successor_sweep.parquet",
    "s_inhibition": REPO_ROOT / "data" / "exploration" / "phase2_s_inhibition_sweep.parquet",
}

# Alt detector parquet paths
ALT_PARQUET = {
    "induction":    REPO_ROOT / "data" / "exploration" / "phase4_h1c_alt_induction_ov.parquet",
    "successor":    REPO_ROOT / "data" / "exploration" / "phase4_h1c_alt_successor_argmax.parquet",
    "s_inhibition": REPO_ROOT / "data" / "exploration" / "phase4_h1c_alt_s_inhibition_compdla.parquet",
}

SIZES = ("70m", "160m", "410m")
N_HEADS_BY_SIZE = {"70m": 48, "160m": 144, "410m": 384}


def top_k_heads(df_cell: pd.DataFrame, score_col: str, k: int) -> set[tuple[int, int]]:
    """Top-k (layer, head) heads by score."""
    if k <= 0 or len(df_cell) == 0:
        return set()
    g = df_cell.nlargest(k, score_col, keep="first")
    return set(zip(g["layer"].astype(int), g["head"].astype(int)))


def locked_final_K(motif: str, size: str) -> int:
    """K = locked-detector pass count at step 143000."""
    df = pd.read_parquet(LOCKED_PARQUET[motif])
    g = df[(df["size"] == size) & (df["step"] == FINAL_STEP)]
    _, pred = LOCKED_TAU[motif]
    return int(pred(g["score"]).sum())


def locked_top_K_final(motif: str, size: str, k: int) -> set[tuple[int, int]]:
    df = pd.read_parquet(LOCKED_PARQUET[motif])
    g = df[(df["size"] == size) & (df["step"] == FINAL_STEP)]
    return top_k_heads(g, "score", k)


def main() -> None:
    OUT_PREVAL.parent.mkdir(parents=True, exist_ok=True)

    print("=== §H1-C-altdetectors robustness analysis (§H1-C-altdetectors-2-rr) ===")

    # ------------------------------------------------------------------
    # 1. Pre-validation: top-K(locked) ∩ top-K(alt) at step 143000
    # ------------------------------------------------------------------
    print("\n=== [1/4] Pre-validation at step 143000 ===")
    preval_rows = []
    for motif in ("induction", "successor", "s_inhibition"):
        alt = pd.read_parquet(ALT_PARQUET[motif])
        for size in SIZES:
            K = locked_final_K(motif, size)
            locked_topK = locked_top_K_final(motif, size, K)
            alt_final = alt[(alt["size"] == size) & (alt["step"] == FINAL_STEP)]
            alt_topK = top_k_heads(alt_final, "score", K)
            overlap = len(locked_topK & alt_topK)
            pct = overlap / K * 100 if K > 0 else float("nan")
            disposition = "usable" if pct >= 50 else ("partial" if pct >= 30 else "unreliable")
            preval_rows.append(dict(
                motif=motif, size=size, K=K,
                overlap=overlap, overlap_pct=pct,
                disposition=disposition,
                locked_topK=str(sorted(locked_topK)),
                alt_topK=str(sorted(alt_topK)),
            ))
            print(f"  {motif:>13} {size:>5}  K={K:>2}  overlap={overlap}/{K} ({pct:>4.0f}%)  [{disposition}]")
    preval_df = pd.DataFrame(preval_rows)
    preval_df.to_parquet(OUT_PREVAL, index=False)
    print(f"  Wrote {OUT_PREVAL.relative_to(REPO_ROOT)}")

    # ------------------------------------------------------------------
    # 2. Frozen-threshold + rank-only + top-5% trajectories
    # ------------------------------------------------------------------
    print("\n=== [2/4] Per-cell trajectories under 3 schemes ===")
    traj_rows = []
    for motif in ("induction", "successor", "s_inhibition"):
        alt = pd.read_parquet(ALT_PARQUET[motif])
        for size in SIZES:
            K = locked_final_K(motif, size)
            top_K_final_set = top_k_heads(
                alt[(alt["size"] == size) & (alt["step"] == FINAL_STEP)],
                "score", K,
            )
            n_h = N_HEADS_BY_SIZE[size]
            n_layers_x_heads = alt[(alt["size"] == size) & (alt["step"] == FINAL_STEP)].shape[0]
            K_top5 = max(1, math.ceil(n_layers_x_heads * 0.05))
            # Frozen threshold value: K-th-largest alt-score at final step
            final_cell = alt[(alt["size"] == size) & (alt["step"] == FINAL_STEP)].copy()
            if K > 0 and len(final_cell) > 0:
                tau_frozen = float(final_cell.nlargest(K, "score", keep="first")["score"].iloc[-1])
            else:
                tau_frozen = float("inf")
            # Top-5% threshold value
            if K_top5 > 0 and len(final_cell) > 0:
                tau_top5 = float(final_cell.nlargest(K_top5, "score", keep="first")["score"].iloc[-1])
            else:
                tau_top5 = float("inf")
            print(f"  {motif:>13} {size:>5}  K={K:>2}  K_top5={K_top5:>3}  tau_frozen={tau_frozen:+.4f}  tau_top5={tau_top5:+.4f}")
            for step in sorted(alt[alt["size"] == size]["step"].unique()):
                cell = alt[(alt["size"] == size) & (alt["step"] == step)]
                n_pass_frozen = int((cell["score"] >= tau_frozen).sum())
                n_pass_top5 = int((cell["score"] >= tau_top5).sum())
                alt_topK_t = top_k_heads(cell, "score", K)
                rank_overlap = len(alt_topK_t & top_K_final_set) / K if K > 0 else float("nan")
                traj_rows.append(dict(
                    motif=motif, size=size, step=int(step),
                    K_final=K, K_top5=K_top5,
                    tau_frozen=tau_frozen, tau_top5=tau_top5,
                    n_pass_frozen=n_pass_frozen,
                    n_pass_top5=n_pass_top5,
                    rank_overlap=rank_overlap,
                ))
    traj_df = pd.DataFrame(traj_rows)
    traj_df.to_parquet(OUT_TRAJ, index=False)
    print(f"  Wrote {OUT_TRAJ.relative_to(REPO_ROOT)}")

    # ------------------------------------------------------------------
    # 3. Emergence steps per scheme
    # ------------------------------------------------------------------
    print("\n=== [3/4] Emergence steps (half-final crossing) ===")

    def first_step_meeting(grp: pd.DataFrame, col: str, half_target: float) -> int | None:
        """Smallest step where grp[col] >= half_target."""
        g = grp.sort_values("step")
        hits = g[g[col] >= half_target]["step"]
        return int(hits.iloc[0]) if len(hits) > 0 else None

    emerg_rows = []
    for (motif, size), grp in traj_df.groupby(["motif", "size"]):
        K = int(grp["K_final"].iloc[0])
        K_top5 = int(grp["K_top5"].iloc[0])
        half_K = K / 2
        half_K_top5 = K_top5 / 2
        # Frozen-threshold emergence step (count ≥ K/2)
        es_frozen = first_step_meeting(grp, "n_pass_frozen", half_K)
        es_top5 = first_step_meeting(grp, "n_pass_top5", half_K_top5)
        # Rank-only emergence step (rank_overlap ≥ 0.5)
        es_rank = first_step_meeting(grp, "rank_overlap", 0.5)
        # Locked-detector emergence step for comparison
        locked_df = pd.read_parquet(LOCKED_PARQUET[motif])
        locked_size = locked_df[locked_df["size"] == size]
        _, pred = LOCKED_TAU[motif]
        per_step_locked = locked_size.groupby("step").apply(
            lambda d: int(pred(d["score"]).sum()),
            include_groups=False,
        )
        locked_traj = per_step_locked.reset_index().rename(columns={0: "n_pass"})
        es_locked = first_step_meeting(locked_traj, "n_pass", half_K)
        emerg_rows.append(dict(
            motif=motif, size=size, K_final=K, K_top5=K_top5,
            es_locked=es_locked,
            es_alt_frozen=es_frozen,
            es_alt_top5=es_top5,
            es_alt_rank=es_rank,
        ))
        print(f"  {motif:>13} {size:>5}  locked={es_locked}  frozen={es_frozen}  top5={es_top5}  rank={es_rank}")
    emerg_df = pd.DataFrame(emerg_rows)
    emerg_df.to_parquet(OUT_EMERG, index=False)
    print(f"  Wrote {OUT_EMERG.relative_to(REPO_ROOT)}")

    # ------------------------------------------------------------------
    # 4. Joint sign-test under 4 detector triples × 3 schemes
    # ------------------------------------------------------------------
    print("\n=== [4/4] Joint sign-test under detector triples ===")
    # The §H2-5 joint sign-test asks: at each of 3 sizes (70m, 160m, 410m),
    # does the ordering μ_ind < μ_suc < μ_si hold? Under independent direction
    # per size with H0: each ordering is random ((1/6)^3 = 1/216 = p ≈ 0.00463
    # for all 3 sizes match the predicted ordering).

    def ordering_holds(emerg: dict[str, int | None]) -> bool:
        """True iff es_ind < es_suc < es_si AND all three are not None."""
        i, s, h = emerg.get("induction"), emerg.get("successor"), emerg.get("s_inhibition")
        if i is None or s is None or h is None:
            return False
        return i < s < h

    # Build per-(scheme, triple) verdict
    # Triples: 4 substitutions (1 alt + 2 locked, alt placement varies; plus 3-alt all)
    # We compute emergence steps for the relevant combination at each size.
    # locked emergence per (motif, size) is es_locked.
    # alt emergence per (motif, size, scheme) is one of es_alt_frozen / es_alt_top5 / es_alt_rank.

    schemes = ("locked_only", "frozen", "top5", "rank")
    triples = (
        ("locked", "locked", "locked"),  # baseline (= primary, p=0.00463 already)
        ("alt", "locked", "locked"),
        ("locked", "alt", "locked"),
        ("locked", "locked", "alt"),
        ("alt", "alt", "alt"),
    )

    verdict_rows = []
    emerg_idx = {(r["motif"], r["size"]): r for r in emerg_df.to_dict("records")}
    for scheme in schemes:
        if scheme == "locked_only":
            alt_col = None
        elif scheme == "frozen":
            alt_col = "es_alt_frozen"
        elif scheme == "top5":
            alt_col = "es_alt_top5"
        else:
            alt_col = "es_alt_rank"
        for triple in triples:
            # Skip "all locked" except under locked_only scheme (it's redundant)
            if triple == ("locked", "locked", "locked") and scheme != "locked_only":
                continue
            n_pass_sizes = 0
            per_size = {}
            for size in SIZES:
                emerg_at_size = {}
                for motif, choice in zip(("induction", "successor", "s_inhibition"), triple):
                    row = emerg_idx[(motif, size)]
                    if choice == "locked":
                        emerg_at_size[motif] = row["es_locked"]
                    else:
                        emerg_at_size[motif] = row[alt_col] if alt_col else None
                holds = ordering_holds(emerg_at_size)
                per_size[size] = (emerg_at_size, holds)
                if holds:
                    n_pass_sizes += 1
            joint_p = (1 / 6) ** n_pass_sizes if n_pass_sizes > 0 else 1.0
            verdict_rows.append(dict(
                scheme=scheme,
                triple="_".join(triple),
                n_pass_sizes=n_pass_sizes,
                joint_p_approx=joint_p,
                detail=str({k: v[0] for k, v in per_size.items()}),
            ))
            print(
                f"  scheme={scheme:>8}  triple={triple}  "
                f"sizes_holding={n_pass_sizes}/3  p≈{joint_p:.5f}"
            )
    verdict_df = pd.DataFrame(verdict_rows)
    verdict_df.to_parquet(OUT_VERDICT, index=False)
    print(f"  Wrote {OUT_VERDICT.relative_to(REPO_ROOT)}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== ROBUSTNESS VERDICT ===")
    print(f"  Primary (locked-only): see baseline row above")
    for scheme in ("frozen", "top5", "rank"):
        all_alt = verdict_df[
            (verdict_df["scheme"] == scheme) & (verdict_df["triple"] == "alt_alt_alt")
        ]
        if len(all_alt) > 0:
            r = all_alt.iloc[0]
            n = int(r["n_pass_sizes"])
            p = float(r["joint_p_approx"])
            survives = "PASSES" if (n == 3 and p < 0.005) else (
                "PARTIAL" if n >= 2 else "FAILS"
            )
            print(f"  All-alt triple under {scheme:>8}: {n}/3 sizes hold, p≈{p:.5f}  → {survives}")


if __name__ == "__main__":
    main()
