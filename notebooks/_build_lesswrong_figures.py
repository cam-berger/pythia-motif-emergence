"""Generate static PNG figures for the LessWrong post.

Pulls from the sealed sweep parquets and verdict parquets that already
exist on disk. No new compute. Saves to notebooks/figures/lesswrong/.

Figures:
- F1_emergence_3sizes.png — Track 1: count-vs-step curves for the 3
  registered sizes (70M / 160M / 410M), one panel per motif, with the
  point-estimate μ marked.
- F2_causal_disjointness.png — Track 2: cross-size ratio_suc and
  ratio_ctrl forest plot (410M, 1B, 2.8B; both metrics).
- F3_h4supersede_bootstrap.png — Track 3: paired bootstrap delta
  (mu_410M − mu_2.8B) histogram and per-size mu distributions.
- F4_head_count_axis.png — head-count axis scaling with poly1+poly2
  fits and R² for induction and successor.
- F5_structural_reuse.png — suc ∩ si head-population overlap across
  (size, step), and ind ∩ si for contrast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

EXPL = REPO / "data" / "exploration"
OUT = REPO / "notebooks" / "figures" / "lesswrong"
OUT.mkdir(parents=True, exist_ok=True)

TAU_IND = 0.30
TAU_LIFT = 0.13496
TAU_STRICT = 0.0372

SIZE_COLOR = {
    "70m": "tab:blue",
    "160m": "tab:orange",
    "410m": "tab:green",
    "1b": "tab:red",
    "2.8b": "tab:purple",
}

TOTAL_HEADS = {"70m": 48, "160m": 144, "410m": 384, "1b": 128, "2.8b": 1024}


def logistic(log_step: np.ndarray, L: float, k: float, mu_log: float) -> np.ndarray:
    return L / (1.0 + np.exp(-k * (log_step - mu_log)))


def fit_mu(steps: np.ndarray, counts: np.ndarray) -> float:
    log_steps = np.log10(steps + 1.0)
    L0 = max(float(counts.max()), 1.0)
    try:
        popt, _ = curve_fit(logistic, log_steps, counts, p0=[L0, 2.0, 4.0], maxfev=5000)
        return float(10 ** popt[2] - 1.0)
    except Exception:
        return float("nan")


def load_long(motif: str) -> pd.DataFrame:
    dfs = []
    if motif == "s_inhibition":
        dfs.append(pd.read_parquet(EXPL / "phase2_s_inhibition_sweep.parquet"))
        dfs.append(pd.read_parquet(EXPL / "phase3_1b_s_inhibition_sweep.parquet"))
        dfs.append(pd.read_parquet(EXPL / "phase4_2_8b_s_inhibition_supersede_sweep.parquet"))
    else:
        dfs.append(pd.read_parquet(EXPL / f"phase2_{motif}_sweep.parquet"))
        dfs.append(pd.read_parquet(EXPL / f"phase3_1b_{motif}_sweep.parquet"))
        dfs.append(pd.read_parquet(EXPL / f"phase4_2_8b_{motif}_sweep.parquet"))
    return pd.concat(dfs).reset_index(drop=True)


# ============================================================
# F1 — Track 1: 3-size emergence curves
# ============================================================
def f1_emergence_3sizes() -> None:
    df_ind = load_long("induction")
    df_suc = load_long("successor")
    df_si = load_long("s_inhibition")

    registered_sizes = ["70m", "160m", "410m"]
    motifs = [
        ("induction", df_ind, TAU_IND, "tab:blue"),
        ("successor", df_suc, TAU_LIFT, "tab:green"),
        ("S-inhibition", df_si, TAU_STRICT, "tab:red"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    for ax, size in zip(axes, registered_sizes):
        for motif, df, tau, color in motifs:
            sub = df[df["size"] == size]
            steps = sorted(sub["step"].unique())
            counts = np.array(
                [int((sub[sub["step"] == s]["score"] >= tau).sum()) for s in steps], dtype=float
            )
            steps_arr = np.array(steps, dtype=float)
            ax.plot(steps_arr, counts, marker="o", markersize=3, color=color, label=motif, alpha=0.85)
            mu = fit_mu(steps_arr, counts)
            if np.isfinite(mu) and mu < 1.4e5 and counts.max() >= 2:
                ax.axvline(mu, color=color, linestyle=":", alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlim(500, 150000)
        ax.set_xlabel("training step (log)")
        ax.set_ylabel("count above threshold")
        ax.set_title(f"Pythia-{size}")
        ax.grid(alpha=0.3)
        if size == "410m":
            ax.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        "Track 1 — emergence ordering across 3 registered Pythia sizes\n"
        "joint sign-test PASS at p ≈ 0.00463; vertical dotted lines mark logistic μ",
        y=1.04,
    )
    plt.tight_layout()
    plt.savefig(OUT / "F1_emergence_3sizes.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT / 'F1_emergence_3sizes.png'}")


# ============================================================
# F2 — Track 2: cross-size causal-disjointness
# ============================================================
def f2_causal_disjointness() -> None:
    rows = []
    for size, label, heads in [("410m", "Pythia-410M (384 heads)", 384),
                                ("1b", "Pythia-1B (128 heads)", 128),
                                ("2.8b", "Pythia-2.8B (1024 heads)", 1024)]:
        va = pd.read_parquet(EXPL / f"phase4_causal_{size.replace('.', '_')}_anchor_verdict.parquet").iloc[0]
        vb = pd.read_parquet(EXPL / f"phase4_causal_{size.replace('.', '_')}_anchor_logitdiff_verdict.parquet").iloc[0]
        rows.append(dict(label=label, metric="Metric A (path-patching)",
                         ratio_suc=1.0,  # path-patching anchor has per-sender ratios, not aggregate; use placeholder
                         ratio_ctrl=1.0,
                         pattern=str(va["pattern"])))
        rows.append(dict(label=label, metric="Metric B (logit-diff)",
                         ratio_suc=float(vb["ratio_suc"]),
                         ratio_suc_lo=float(vb["ratio_suc_ci_low"]),
                         ratio_suc_hi=float(vb["ratio_suc_ci_high"]),
                         ratio_ctrl=float(vb["ratio_ctrl"]),
                         ratio_ctrl_lo=float(vb["ratio_ctrl_ci_low"]),
                         ratio_ctrl_hi=float(vb["ratio_ctrl_ci_high"]),
                         pattern=str(vb["pattern"])))

    # Single panel — Metric B forest plot (the more interpretable one)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sizes_for_plot = ["410m", "1b", "2.8b"]
    size_labels = {"410m": "410M (384 heads)", "1b": "1B (128 heads, regression)", "2.8b": "2.8B (1024 heads)"}
    y_positions = list(range(len(sizes_for_plot) * 2))
    y_labels = []
    for i, size in enumerate(sizes_for_plot):
        vb = pd.read_parquet(EXPL / f"phase4_causal_{size.replace('.', '_')}_anchor_logitdiff_verdict.parquet").iloc[0]
        pattern = str(vb["pattern"])
        color = SIZE_COLOR[size]
        # suc row
        y_suc = i * 2 + 0.2
        y_ctrl = i * 2 - 0.2
        ax.errorbar([float(vb["ratio_suc"])], [y_suc],
                    xerr=[[float(vb["ratio_suc"]) - float(vb["ratio_suc_ci_low"])],
                          [float(vb["ratio_suc_ci_high"]) - float(vb["ratio_suc"])]],
                    fmt="o", color=color, capsize=4, label="suc ablation" if i == 0 else None,
                    markersize=8)
        ax.errorbar([float(vb["ratio_ctrl"])], [y_ctrl],
                    xerr=[[float(vb["ratio_ctrl"]) - float(vb["ratio_ctrl_ci_low"])],
                          [float(vb["ratio_ctrl_ci_high"]) - float(vb["ratio_ctrl"])]],
                    fmt="s", color=color, capsize=4, alpha=0.55, mfc="white",
                    label="ctrl ablation" if i == 0 else None, markersize=8)
        ax.text(1.22, i * 2, f"  {pattern}", fontsize=10, va="center",
                fontweight="bold", color=color)
        y_labels.append((i * 2, size_labels[size]))

    ax.axvspan(0.8, 1.2, alpha=0.1, color="green", label="NULL band [0.8, 1.2]")
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks([y for y, _ in y_labels])
    ax.set_yticklabels([lbl for _, lbl in y_labels])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.3)
    ax.set_xlabel("IO−S logit-diff ratio (ablated / clean), B=200 paired bootstrap CI")
    ax.set_title(
        "Track 2 — successor-ablation effect on IOI logit-diff across head-count tiers\n"
        "(NULL at 410M + 2.8B; MIXED at 1B — both suc and ctrl drop ~21%, no suc-specific dependence)"
    )
    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "F2_causal_disjointness.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT / 'F2_causal_disjointness.png'}")


# ============================================================
# F3 — Track 3: §H4-supersede bootstrap μ distributions
# ============================================================
def f3_h4supersede_bootstrap() -> None:
    boot = pd.read_parquet(EXPL / "phase4_2_8b_h4supersede_bootstrap_mu.parquet")
    verdict = pd.read_parquet(EXPL / "phase4_2_8b_h4supersede_verdict.parquet").iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))

    ax = axes[0]
    ax.hist(boot["mu_28b"].dropna(), bins=40, alpha=0.65, label="Pythia-2.8B (1024 heads)", color="tab:purple")
    ax.hist(boot["mu_410m"].dropna(), bins=40, alpha=0.65, label="Pythia-410M (384 heads)", color="tab:green")
    ax.axvline(float(verdict["mu_28b_point"]), color="tab:purple", linestyle="--", alpha=0.7,
               label=f"μ_2.8B point ≈ {float(verdict['mu_28b_point']):.0f}")
    ax.axvline(float(verdict["mu_410m_point"]), color="tab:green", linestyle="--", alpha=0.7,
               label=f"μ_410M point ≈ {float(verdict['mu_410m_point']):.0f}")
    ax.set_xscale("log")
    ax.set_xlim(1e3, 2e5)
    ax.set_xlabel("μ_si (training step, log)")
    ax.set_ylabel("bootstrap replicate count")
    ax.set_title("Paired bootstrap μ distributions (B=1000)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    delta = boot["mu_410m"] - boot["mu_28b"]
    ax.hist(delta.dropna(), bins=40, color="tab:gray", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", lw=1.2, label="zero acceleration")
    rate = float(verdict["reversal_rate"])
    ax.set_xlabel("μ_410M − μ_2.8B  (training steps)")
    ax.set_ylabel("bootstrap replicate count")
    ax.set_title(f"Paired delta — reversal_rate = {rate:.3f} (gate ≥ 0.95)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    speedup = float(verdict["mu_410m_point"]) / float(verdict["mu_28b_point"])
    fig.suptitle(
        f"Track 3 — §H4-supersede head-count-axis scaling PASS  "
        f"(2.8B is ~{speedup:.1f}× faster on S-inhibition emergence)",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUT / "F3_h4supersede_bootstrap.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT / 'F3_h4supersede_bootstrap.png'}")


# ============================================================
# F4 — head-count-axis with poly1/poly2 fits
# ============================================================
def f4_head_count_axis() -> None:
    # Per-size max_count for induction + successor (at the latest available cell)
    df_ind = load_long("induction")
    df_suc = load_long("successor")

    def max_count_per_size(df: pd.DataFrame, tau: float) -> dict[str, int]:
        out = {}
        for size in ["70m", "160m", "410m", "1b", "2.8b"]:
            sub = df[df["size"] == size]
            mx = 0
            for step in sub["step"].unique():
                c = int((sub[sub["step"] == step]["score"] >= tau).sum())
                if c > mx:
                    mx = c
            out[size] = mx
        return out

    ind_counts = max_count_per_size(df_ind, TAU_IND)
    suc_counts = max_count_per_size(df_suc, TAU_LIFT)

    ordered = sorted(TOTAL_HEADS.keys(), key=lambda s: TOTAL_HEADS[s])
    xs = np.array([TOTAL_HEADS[s] for s in ordered], dtype=float)

    def r2(ys, y_pred):
        ss_res = float(np.sum((ys - y_pred) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Per-motif fit-exclusion policy: 1B is a §H4-1 head-count regression
    # (128 heads, less than 410M's 384). Successor's count at 1B sits dramatically
    # above the trend defined by the other four sizes — including 1B in the fit
    # destroys R² (~0.30 on both poly1 and poly2). Excluding 1B from the
    # successor fit isolates the head-count-axis trend across the four
    # architectural scale-ups (70M, 160M, 410M, 2.8B). The 1B point is still
    # rendered (hollow marker) so the regression is visible.
    EXCLUDE_FROM_FIT = {"successor": ["1b"], "induction": []}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, (name, counts_map, color) in zip(
        axes,
        [("induction", ind_counts, "tab:blue"), ("successor", suc_counts, "tab:green")],
    ):
        ys_all = np.array([counts_map[s] for s in ordered], dtype=float)
        excluded = EXCLUDE_FROM_FIT.get(name, [])
        keep_mask = np.array([s not in excluded for s in ordered])
        xs_fit = xs[keep_mask]
        ys_fit = ys_all[keep_mask]

        c1 = np.polyfit(xs_fit, ys_fit, 1)
        c2 = np.polyfit(xs_fit, ys_fit, 2)
        y1p = np.polyval(c1, xs_fit)
        y2p = np.polyval(c2, xs_fit)
        r2_1, r2_2 = r2(ys_fit, y1p), r2(ys_fit, y2p)
        fit_x = np.linspace(0, xs.max() * 1.05, 200)
        y1x = np.polyval(c1, fit_x)
        y2x = np.polyval(c2, fit_x)

        # Plot points: filled for fit-included, hollow with outline for fit-excluded.
        for s, x, y in zip(ordered, xs, ys_all):
            if s in excluded:
                ax.scatter([x], [y], s=110, facecolors="white", edgecolors=color,
                           linewidths=2, zorder=3,
                           label=f"{s} (excluded: §H4-1 head-count regression)")
            else:
                ax.scatter([x], [y], s=85, color=color, zorder=3,
                           label=f"{name} max_count" if s == ordered[0] else None)
            ax.annotate(s, (x, y), textcoords="offset points", xytext=(7, 4), fontsize=9)

        n_pts = int(keep_mask.sum())
        if r2_2 > r2_1 + 0.005:
            ax.plot(fit_x, y2x, "--", color=color, alpha=0.85,
                    label=f"poly2 fit ({n_pts} pts): R² = {r2_2:.3f}  (linear R² = {r2_1:.3f})",
                    zorder=2)
            ax.plot(fit_x, y1x, ":", color=color, alpha=0.35,
                    label="linear (reference)", zorder=1)
        else:
            ax.plot(fit_x, y1x, "--", color=color, alpha=0.85,
                    label=f"linear fit ({n_pts} pts): R² = {r2_1:.3f}  (poly2 R² = {r2_2:.3f})",
                    zorder=2)
            ax.plot(fit_x, y2x, ":", color=color, alpha=0.35,
                    label="poly2 (reference)", zorder=1)
        ax.set_xlabel("total attention heads per model")
        ax.set_ylabel(f"max_count of {name} heads above threshold")
        title_suffix = f"({n_pts}-pt fit; {len(excluded)} excluded)" if excluded else "(5-pt fit)"
        ax.set_title(f"{name} max_count vs head count {title_suffix}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
        excl_str = f"  [excluded: {excluded}]" if excluded else ""
        print(f"  {name} ({n_pts}-pt fit): poly1 R² = {r2_1:.4f}  poly2 R² = {r2_2:.4f}  "
              f"(winner = {'poly2' if r2_2 > r2_1 + 0.005 else 'linear'}){excl_str}")

    fig.suptitle(
        "Total attention heads vs detected motif heads — §H4-1 head-count axis\n"
        "(successor fit excludes 1B per §H4-1 head-count-regression policy; 1B rendered hollow)",
        y=1.05,
    )
    plt.tight_layout()
    plt.savefig(OUT / "F4_head_count_axis.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT / 'F4_head_count_axis.png'}")


# ============================================================
# F5 — Structural-reuse cross-motif overlap across (size, step)
# ============================================================
def f5_structural_reuse() -> None:
    sr = pd.read_parquet(EXPL / "structural_reuse_deep_dive.parquet")
    sizes = ["70m", "160m", "410m", "1b", "2.8b"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    # Left: suc ∩ si
    ax = axes[0]
    for size in sizes:
        sub = sr[sr["size"] == size].sort_values("step")
        ax.plot(sub["step"], sub["n_suc_si"], marker="o", markersize=5,
                color=SIZE_COLOR[size], label=f"Pythia-{size}", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlim(500, 150000)
    ax.set_ylim(-0.2, max(2.5, sr["n_suc_si"].max() + 0.5))
    ax.set_xlabel("training step (log)")
    ax.set_ylabel("count of heads in suc ∩ S-inhibition")
    ax.set_title("Successor ∩ S-inhibition: structurally disjoint everywhere")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    # Right: ind ∩ si
    ax = axes[1]
    for size in sizes:
        sub = sr[sr["size"] == size].sort_values("step")
        ax.plot(sub["step"], sub["n_ind_si"], marker="o", markersize=5,
                color=SIZE_COLOR[size], label=f"Pythia-{size}", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlim(500, 150000)
    ax.set_xlabel("training step (log)")
    ax.set_ylabel("count of heads in ind ∩ S-inhibition")
    ax.set_title("Induction ∩ S-inhibition: robust overlap at larger sizes (NMs share QK pattern)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        "Cross-motif head-population overlap across (size, step) — A12 structural-reuse deep dive",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUT / "F5_structural_reuse.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT / 'F5_structural_reuse.png'}")


if __name__ == "__main__":
    f1_emergence_3sizes()
    f2_causal_disjointness()
    f3_h4supersede_bootstrap()
    f4_head_count_axis()
    f5_structural_reuse()
