"""Build a figure showing motif emergence trajectories alongside training loss.

For each Pythia size (70m, 160m, 410m, 1b, 2.8b), plot per-cell pass counts for
the three motifs (induction QK>0.30, successor lift≥0.13496, S-inhibition
Δ_h≥0.0372) against the published checkpoint step axis. Overlay the training-
loss curve (pulled from EleutherAI's public wandb) on a secondary axis.

Output:
  - notebooks/figures/F6_loss_vs_emergence.png
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_PNG = REPO_ROOT / "notebooks" / "figures" / "F6_loss_vs_emergence.png"

SIZES = ("70m", "160m", "410m", "1b", "2.8b")

LOCKED = {
    "induction":    ("phase2_induction_sweep.parquet",
                     "phase4_2_8b_induction_sweep.parquet",
                     "phase3_1b_induction_sweep.parquet",
                     0.30, "gt"),
    "successor":    ("phase2_successor_sweep.parquet",
                     "phase4_2_8b_successor_sweep.parquet",
                     "phase3_1b_successor_sweep.parquet",
                     0.13496, "ge"),
    "s_inhibition": ("phase2_s_inhibition_sweep.parquet",
                     None,  # 2.8b S-inh assembled from multiple sub-grids below
                     "phase3_1b_s_inhibition_sweep.parquet",
                     0.0372, "ge"),
}

MOTIF_COLORS = {
    "induction":    "#1f77b4",  # blue
    "successor":    "#ff7f0e",  # orange
    "s_inhibition": "#2ca02c",  # green
}

MOTIF_LABELS = {
    "induction":    "induction  (Olsson QK > 0.30)",
    "successor":    "successor  (lift_dla ≥ 0.135)",
    "s_inhibition": "S-inhibition  (Δ_h ≥ 0.037)",
}


def load_motif_size(motif: str, size: str) -> pd.DataFrame:
    """Return per-cell DataFrame with columns [step, n_pass] for (motif, size)."""
    track1_pq, _2_8b_pq, _1b_pq, thr, op = LOCKED[motif]
    if size in ("70m", "160m", "410m"):
        df = pd.read_parquet(REPO_ROOT / "data" / "exploration" / track1_pq)
        df = df[df["size"] == size]
    elif size == "1b":
        if _1b_pq is None:
            return pd.DataFrame(columns=["step", "n_pass"])
        df = pd.read_parquet(REPO_ROOT / "data" / "exploration" / _1b_pq)
        if "size" in df.columns:
            df = df[df["size"] == size]
    elif size == "2.8b":
        if motif == "s_inhibition":
            # Merge 3 sub-grids
            pieces = []
            for f in (
                "phase4_2_8b_s_inhibition_supersede_sweep.parquet",
                "phase4_2_8b_s_inhibition_fullgrid_sweep.parquet",
            ):
                p = REPO_ROOT / "data" / "exploration" / f
                if p.exists():
                    pieces.append(pd.read_parquet(p))
            df = pd.concat(pieces, ignore_index=True).drop_duplicates(
                ["size", "step", "layer", "head"], keep="first"
            )
        else:
            df = pd.read_parquet(REPO_ROOT / "data" / "exploration" / _2_8b_pq)
            if "size" in df.columns:
                df = df[df["size"] == size]

    if op == "gt":
        per_step = df.groupby("step").apply(lambda g: (g["score"] > thr).sum(), include_groups=False)
    else:
        per_step = df.groupby("step").apply(lambda g: (g["score"] >= thr).sum(), include_groups=False)
    out = per_step.reset_index()
    out.columns = ["step", "n_pass"]
    return out.sort_values("step")


def downsample_loss(loss_df: pd.DataFrame, n: int = 500) -> pd.DataFrame:
    """Downsample to ~n points evenly spaced on log scale for cleaner plotting."""
    if len(loss_df) <= n:
        return loss_df
    log_steps = np.geomspace(max(1, loss_df["step"].min()), loss_df["step"].max(), n)
    idx = []
    seen = set()
    for ls in log_steps:
        i = (loss_df["step"] - ls).abs().idxmin()
        if i not in seen:
            seen.add(i)
            idx.append(i)
    return loss_df.loc[idx].sort_values("step")


def main() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    loss_df = pd.read_parquet(REPO_ROOT / "data" / "exploration" / "pythia_training_loss_wandb.parquet")
    print(f"Loss data: {len(loss_df):,} rows across {loss_df['size'].nunique()} sizes")

    fig, axes = plt.subplots(len(SIZES), 1, figsize=(11, 13), sharex=True)
    if len(SIZES) == 1:
        axes = [axes]

    for ax, size in zip(axes, SIZES):
        # Per-size loss curve
        loss_sz = downsample_loss(
            loss_df[loss_df["size"] == size][["step", "train_lm_loss"]].sort_values("step"),
            n=400,
        )
        ax_loss = ax.twinx()
        if len(loss_sz) > 0:
            ax_loss.plot(
                loss_sz["step"].values + 1,  # +1 to handle step=0 in log scale
                loss_sz["train_lm_loss"].values,
                color="#888", linestyle="-", linewidth=1.2, alpha=0.85,
                label="train lm_loss (wandb)",
                zorder=1,
            )
        ax_loss.set_ylabel("train lm_loss", color="#666", fontsize=9)
        ax_loss.tick_params(axis="y", labelcolor="#666", labelsize=8)
        # Auto-scale; clip to converged range to avoid initial-loss dominance
        if len(loss_sz):
            lo = max(0.5, float(loss_sz["train_lm_loss"].iloc[-100:].min()) - 0.5)
            hi = min(11.5, float(loss_sz["train_lm_loss"].iloc[-100:].max()) + 6.0)
            ax_loss.set_ylim(lo, hi)

        # Per-motif emergence curves on primary axis
        for motif in ("induction", "successor", "s_inhibition"):
            traj = load_motif_size(motif, size)
            if len(traj) == 0:
                continue
            ax.plot(
                traj["step"].values + 1,
                traj["n_pass"].values,
                "o-",
                color=MOTIF_COLORS[motif],
                linewidth=1.6,
                markersize=3,
                label=MOTIF_LABELS[motif] if size == SIZES[0] else None,
                zorder=3,
            )

        ax.set_ylabel(f"pass-count\n(Pythia-{size})", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xscale("log")
        ax.set_xlim(0.8, 200_000)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("Training step (log scale)", fontsize=11)
    axes[0].set_title(
        "Motif emergence vs. training loss across Pythia (deduped)",
        fontsize=12,
    )
    # Legend on the top axis (motif lines only)
    handles_motif, labels_motif = axes[0].get_legend_handles_labels()
    handles_loss, labels_loss = axes[0].twinx().get_legend_handles_labels()
    # Combined legend on figure
    fig.legend(
        handles_motif + [plt.Line2D([0], [0], color="#888", linewidth=1.2)],
        labels_motif + ["train lm_loss (wandb)"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        fontsize=9,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUT_PNG.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
