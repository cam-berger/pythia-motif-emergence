"""Build the cross-size integrated-atlas side-by-side figure + animation.

Produces:
  - notebooks/figures/atlas_v1/cross_size/C6_integrated_4size_final.png
    (static 4-panel figure at step 143000)
  - notebooks/figures/atlas_v1/cross_size/atlas_v1_integrated_4size.gif
    (animated 4-panel GIF advancing through all 40 §H2-1 checkpoints in
     lockstep across 70m / 160m / 410m / 2.8b)

Each panel shows the (layer × head) grid for that size, with each cell
colored by its primary family — the family with the highest
normalized score (score / threshold) among those whose thresholds it
meets, or gray if no family passes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from src.locked_thresholds import INDUCTION_QK, SUCCESSOR_LIFT, S_INHIBITION_DELTA


SIZES = ["70m", "160m", "410m", "2.8b"]
SIZE_FS = {"70m": "70m", "160m": "160m", "410m": "410m", "2.8b": "2_8b"}
TOTAL_HEADS = {"70m": 48, "160m": 144, "410m": 384, "2.8b": 1024}

THRESHOLDS = {
    "induction":         (INDUCTION_QK.value,         INDUCTION_QK.comparator),
    "successor":         (SUCCESSOR_LIFT.value,       SUCCESSOR_LIFT.comparator),
    "s_inhibition":      (S_INHIBITION_DELTA.value,   S_INHIBITION_DELTA.comparator),
    "previous_token":    (0.20, "ge"),
    "duplicate_token":   (0.20, "ge"),
    "positional_offset": (0.15, "ge"),
    "bos_attention":     (0.50, "ge"),
    "delimiter":         (0.40, "ge"),
}
MOTIF_ORDER = list(THRESHOLDS.keys())
MOTIF_COLORS = {
    "induction":         "#1f77b4",
    "successor":         "#ff7f0e",
    "s_inhibition":      "#2ca02c",
    "previous_token":    "#d62728",
    "duplicate_token":   "#9467bd",
    "positional_offset": "#8c564b",
    "bos_attention":     "#e377c2",
    "delimiter":         "#17becf",
}

LOCKED_PARQUETS = {
    "induction":    ["phase2_induction_sweep.parquet", "phase4_2_8b_induction_sweep.parquet"],
    "successor":    ["phase2_successor_sweep.parquet", "phase4_2_8b_successor_sweep.parquet"],
    "s_inhibition": ["phase2_s_inhibition_sweep.parquet",
                     "phase4_2_8b_s_inhibition_supersede_sweep.parquet",
                     "phase4_2_8b_s_inhibition_fullgrid_sweep.parquet"],
}


def load_combined_for_size(size: str) -> pd.DataFrame:
    atlas = pd.read_parquet(REPO_ROOT / "data" / "atlas" / f"atlas_v1_{SIZE_FS[size]}_sweep.parquet")
    parts: list[pd.DataFrame] = []
    for motif, fnames in LOCKED_PARQUETS.items():
        pieces = []
        for fname in fnames:
            p = REPO_ROOT / "data" / "exploration" / fname
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            if "size" in df.columns:
                df = df[df["size"] == size]
            if df.empty:
                continue
            pieces.append(df[["size", "step", "layer", "head", "score"]])
        if not pieces:
            continue
        d = pd.concat(pieces, ignore_index=True).drop_duplicates(["size", "step", "layer", "head"], keep="first")
        d["motif"] = motif
        parts.append(d[["size", "step", "layer", "head", "motif", "score"]])
    locked = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return pd.concat([locked, atlas[["size", "step", "layer", "head", "motif", "score"]]], ignore_index=True)


def primary_family_grid(combined: pd.DataFrame, step: int, n_layers: int, n_heads: int) -> np.ndarray:
    """0 = no family; 1..len(MOTIF_ORDER) = family index (in MOTIF_ORDER)."""
    grid = np.zeros((n_layers, n_heads), dtype=int)
    best_norm = np.full((n_layers, n_heads), -np.inf)
    cell_step = combined[combined["step"] == step]
    for fi, motif in enumerate(MOTIF_ORDER, start=1):
        thr_val, op = THRESHOLDS[motif]
        m_rows = cell_step[cell_step["motif"] == motif]
        if m_rows.empty:
            continue
        if op == "gt":
            passes = m_rows["score"] > thr_val
        else:
            passes = m_rows["score"] >= thr_val
        m_pass = m_rows[passes]
        for _, r in m_pass.iterrows():
            L, H = int(r["layer"]), int(r["head"])
            if 0 <= L < n_layers and 0 <= H < n_heads:
                norm_score = float(r["score"]) / thr_val
                if norm_score > best_norm[L, H]:
                    best_norm[L, H] = norm_score
                    grid[L, H] = fi
    return grid


def main() -> None:
    out_dir = REPO_ROOT / "notebooks" / "figures" / "atlas_v1" / "cross_size"
    out_dir.mkdir(parents=True, exist_ok=True)

    palette = ["#d9d9d9"] + [MOTIF_COLORS[m] for m in MOTIF_ORDER]
    cmap = ListedColormap(palette)
    bounds = list(range(len(palette) + 1))
    norm = BoundaryNorm(bounds, cmap.N)

    print("Loading per-size combined data ...")
    t0 = time.time()
    combined_by_size: dict[str, pd.DataFrame] = {}
    dims: dict[str, tuple[int, int]] = {}
    for sz in SIZES:
        combined_by_size[sz] = load_combined_for_size(sz)
        d = combined_by_size[sz]
        nl = int(d["layer"].max() + 1)
        nh = int(d["head"].max() + 1)
        dims[sz] = (nl, nh)
        print(f"  {sz}: {len(d):,} rows, dims=({nl}, {nh}), total_heads={nl*nh}")
    print(f"  total load: {time.time()-t0:.1f}s")

    STEPS = sorted(combined_by_size["70m"]["step"].unique())
    print(f"steps: {len(STEPS)}, range [{STEPS[0]}, {STEPS[-1]}]")

    legend_handles = [Patch(facecolor=palette[0], edgecolor="black", linewidth=0.3, label="(no family)")]
    for i, motif in enumerate(MOTIF_ORDER, start=1):
        legend_handles.append(Patch(facecolor=palette[i], edgecolor="black", linewidth=0.3, label=motif))

    # -------------------- Static figure: final step --------------------
    final_step = 143000
    print(f"\nbuilding static side-by-side figure at step {final_step} ...")
    fig, axes = plt.subplots(1, len(SIZES), figsize=(20, 6),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1]))
    for ax, sz in zip(axes, SIZES):
        nl, nh = dims[sz]
        grid = primary_family_grid(combined_by_size[sz], final_step, nl, nh)
        ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title(f"{sz}  ({TOTAL_HEADS[sz]} heads)", fontsize=11)
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Atlas-v1 integrated — primary family per (layer, head) at step {final_step}",
                 fontsize=13, y=1.0)
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    out_static = out_dir / "C6_integrated_4size_final.png"
    plt.savefig(out_static, dpi=140, bbox_inches="tight")
    print(f"  saved {out_static.relative_to(REPO_ROOT)}")
    plt.close(fig)

    # -------------------- Animated GIF: all 40 steps in lockstep --------------------
    print(f"\nbuilding animated 4-size GIF ({len(STEPS)} frames) ...")
    t0 = time.time()
    frames_by_size: dict[str, list[np.ndarray]] = {}
    for sz in SIZES:
        nl, nh = dims[sz]
        frames_by_size[sz] = [
            primary_family_grid(combined_by_size[sz], step, nl, nh)
            for step in STEPS
        ]
        print(f"  {sz}: built {len(frames_by_size[sz])} frames  ({time.time()-t0:.1f}s elapsed)")

    fig, axes = plt.subplots(1, len(SIZES), figsize=(20, 6),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1]))
    ims = []
    for ax, sz in zip(axes, SIZES):
        im = ax.imshow(frames_by_size[sz][0], cmap=cmap, norm=norm,
                       aspect="auto", interpolation="nearest")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title(f"{sz}  ({TOTAL_HEADS[sz]} heads)", fontsize=11)
        ims.append(im)
    title = fig.suptitle(
        f"Atlas-v1 integrated, 4 sizes side-by-side — step {STEPS[0]}  (frame 1/{len(STEPS)})",
        fontsize=13, y=1.0,
    )
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))

    def update(i):
        for im, sz in zip(ims, SIZES):
            im.set_array(frames_by_size[sz][i])
        title.set_text(
            f"Atlas-v1 integrated, 4 sizes side-by-side — step {STEPS[i]}  "
            f"(frame {i+1}/{len(STEPS)})"
        )
        return ims + [title]

    ani = animation.FuncAnimation(fig, update, frames=len(STEPS), interval=333, blit=False)
    out_gif = out_dir / "atlas_v1_integrated_4size.gif"
    ani.save(out_gif, writer=animation.PillowWriter(fps=3))
    print(f"  saved {out_gif.relative_to(REPO_ROOT)}  ({time.time()-t0:.1f}s total)")
    plt.close(fig)


if __name__ == "__main__":
    main()
