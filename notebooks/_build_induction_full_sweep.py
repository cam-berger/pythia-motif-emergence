"""Build notebooks/induction_full_sweep.ipynb.

Phase 2 induction full-sweep notebook (40 cells × 3 sizes per §H2-1).
Mirrors the Phase 1 6-cell induction_emergence_exploration.ipynb but adds
prompt-bootstrap CIs on μ (§H2-2) and threshold-sensitivity bracket.
The Phase 1 preview notebook remains as the 6-cell historical artifact.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "induction_full_sweep.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            "# Induction emergence — Phase 2 full sweep (40 cells × 3 sizes)\n"
            "\n"
            "**Phase 2 deliverable.** 40-cell extension of the Phase 1 6-cell "
            "preview (`induction_emergence_exploration.ipynb`, kept as a "
            "historical artifact). Same Olsson prefix-matching detector "
            "(threshold > 0.3 per `PROJECT_BRIEF.md` §4); 40 log-spaced "
            "checkpoints from `step0` to `step143000` per `HYPOTHESIS.md` §H2-1; "
            "prompt-bootstrap CIs on μ per §H2-2; threshold-sensitivity bracket "
            "± 25% in 5 increments per §H2-2.\n"
            "\n"
            "**Locked thresholds:**\n"
            "- Detection threshold: prefix-matching score > 0.3 (locked in brief).\n"
            "- Bootstrap: per-sequence resampling, B = 1000, 95% percentile CI.\n"
            "- Tiered fit handling: emerged (max ≥ 5), marginal (2 ≤ max < 5, bootstrap-median), censored (max < 2 → μ = step143000).\n"
        ),
        md("## Setup"),
        code(
            "import os\n"
            "os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')\n"
            "\n"
            "import sys\n"
            "from pathlib import Path\n"
            "REPO = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "if str(REPO) not in sys.path:\n"
            "    sys.path.insert(0, str(REPO))\n"
            "\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from notebooks._lib.sweep_io import read_long\n"
            "from src.analysis.phase2_logistic import tiered_fit, RIGHT_CENSOR_STEP\n"
            "from src.analysis.phase2_bootstrap import (\n"
            "    bootstrap_induction, summarize_bootstrap, threshold_sensitivity_curve,\n"
            "    THRESHOLD_SENSITIVITY_FRACTIONS,\n"
            ")\n"
            "\n"
            "SIZES = ['70m', '160m', '410m']\n"
            "SIZE_COLOR = {'70m': 'tab:blue', '160m': 'tab:orange', '410m': 'tab:green'}\n"
            "INDUCTION_THRESHOLD = 0.3"
        ),
        md(
            "## Load sweep data\n"
            "\n"
            "Long-format parquet from `_run_phase2_induction_sweep.py`. Per-cell "
            "`.npz` with per-sequence scores for bootstrap is in "
            "`data/exploration/phase2_induction_per_seq/`."
        ),
        code(
            "df = read_long(REPO / 'data' / 'exploration' / 'phase2_induction_sweep.parquet')\n"
            "print(f'Total rows: {len(df):,}')\n"
            "print(f'Sizes: {sorted(df[\"size\"].unique().tolist())}')\n"
            "print(f'Steps: {sorted(df.step.unique().tolist())[:5]}...{sorted(df.step.unique().tolist())[-3:]}')\n"
            "STEPS = sorted(df.step.unique().tolist())\n"
            "print(f'N cells per size: {len(STEPS)}')\n"
            "print(f'Heads per (size, step):')\n"
            "print(df.groupby(['size', 'step']).size().unstack().iloc[:, :3].to_string())"
        ),
        md(
            "## Per-cell summary: count of induction heads (score > 0.3)"
        ),
        code(
            "summary_rows = []\n"
            "for (size, step), grp in df.groupby(['size', 'step']):\n"
            "    scores = grp['score'].values\n"
            "    n_above = int((scores > INDUCTION_THRESHOLD).sum())\n"
            "    max_idx = int(np.argmax(scores))\n"
            "    top_row = grp.iloc[max_idx]\n"
            "    summary_rows.append(dict(\n"
            "        size=size, step=step,\n"
            "        n_above_03=n_above,\n"
            "        max_score=float(scores.max()),\n"
            "        top_layer=int(top_row['layer']), top_head=int(top_row['head']),\n"
            "    ))\n"
            "summary = pd.DataFrame(summary_rows)\n"
            "summary['size'] = pd.Categorical(summary['size'], categories=SIZES, ordered=True)\n"
            "summary = summary.sort_values(['size', 'step']).reset_index(drop=True)\n"
            "print(summary.to_string(index=False))"
        ),
        md(
            "## Emergence curves with bootstrap CI on μ\n"
            "\n"
            "Per Pythia size, count of heads with prefix-matching score > 0.3 over "
            "training. Logistic fit applied per §H2-3 tiered handling; emergence "
            "step μ extracted; B = 1000 prompt-bootstrap CI on μ at 95%."
        ),
        code(
            "rng = np.random.default_rng(0)\n"
            "fits = {}\n"
            "boots = {}\n"
            "for size in SIZES:\n"
            "    sub = summary[summary['size'] == size].sort_values('step')\n"
            "    steps = sub['step'].values\n"
            "    counts = sub['n_above_03'].values\n"
            "    # Load per-seq cache\n"
            "    per_seq_by_step = {}\n"
            "    for s in steps:\n"
            "        npz = np.load(REPO / 'data' / 'exploration' / 'phase2_induction_per_seq' / f'{size}_step{s}.npz')\n"
            "        per_seq_by_step[int(s)] = npz['per_sequence_scores']\n"
            "    mus, mu_point = bootstrap_induction(per_seq_by_step, INDUCTION_THRESHOLD, rng, B=1000)\n"
            "    bres = summarize_bootstrap(size, 'induction', INDUCTION_THRESHOLD, mus, mu_point)\n"
            "    fr = tiered_fit(size, 'induction', steps, counts,\n"
            "                    bootstrap_median_mu=bres.mu_bootstrap_median)\n"
            "    fits[size] = fr\n"
            "    boots[size] = bres\n"
            "    print(f'{size}: regime={fr.regime}, μ_point={fr.mu:.0f}, '\n"
            "          f'μ_boot=[{bres.mu_ci_low:.0f}, {bres.mu_ci_high:.0f}] (95% CI)')"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "for size in SIZES:\n"
            "    sub = summary[summary['size'] == size]\n"
            "    ax.plot(sub['step'], sub['n_above_03'], marker='o', alpha=0.6,\n"
            "            color=SIZE_COLOR[size], label=f'Pythia-{size}')\n"
            "    bres = boots[size]\n"
            "    if np.isfinite(bres.mu_point_estimate) and bres.mu_point_estimate < RIGHT_CENSOR_STEP - 1:\n"
            "        ax.axvspan(bres.mu_ci_low, bres.mu_ci_high, alpha=0.15,\n"
            "                   color=SIZE_COLOR[size])\n"
            "        ax.axvline(bres.mu_point_estimate, color=SIZE_COLOR[size],\n"
            "                   linestyle='--', alpha=0.7,\n"
            "                   label=f'  μ = {bres.mu_point_estimate:.0f}')\n"
            "ax.set_xscale('symlog', linthresh=1)\n"
            "ax.set_xlim(0.5, 200000)\n"
            "ax.set_xlabel('training step (symlog)')\n"
            "ax.set_ylabel('count of heads with prefix-matching score > 0.3')\n"
            "ax.set_title('Induction-head count emergence with bootstrap CI on μ (Phase 2 full sweep)')\n"
            "ax.legend(fontsize=9)\n"
            "ax.grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## (Layer, head) heatmap grid over training\n"
            "\n"
            "Per-cell (layer, head) prefix-matching score. Bright cells = "
            "induction-candidate heads. With 40 cells the spatial-emergence "
            "trajectory is visible at finer resolution than the 6-cell preview."
        ),
        code(
            "# 3 sizes × 40 steps is too dense for one grid; show every 4th step\n"
            "PLOT_STEPS = STEPS[::4]\n"
            "vmax = float(df['score'].max())\n"
            "fig, axes = plt.subplots(\n"
            "    nrows=len(SIZES), ncols=len(PLOT_STEPS),\n"
            "    figsize=(2 * len(PLOT_STEPS), 2.5 * len(SIZES)),\n"
            ")\n"
            "for row, size in enumerate(SIZES):\n"
            "    for col, step in enumerate(PLOT_STEPS):\n"
            "        ax = axes[row][col]\n"
            "        sub = df[(df['size'] == size) & (df['step'] == step)]\n"
            "        if sub.empty:\n"
            "            ax.set_visible(False); continue\n"
            "        n_layers = int(sub['layer'].max() + 1)\n"
            "        n_heads = int(sub['head'].max() + 1)\n"
            "        grid = np.zeros((n_layers, n_heads))\n"
            "        for _, r in sub.iterrows():\n"
            "            grid[int(r['layer']), int(r['head'])] = r['score']\n"
            "        im = ax.imshow(grid, vmin=0, vmax=vmax, cmap='viridis', aspect='auto')\n"
            "        ax.set_title(f'{size} step{step}', fontsize=8)\n"
            "        if col == 0: ax.set_ylabel('layer')\n"
            "        if row == len(SIZES) - 1: ax.set_xlabel('head')\n"
            "fig.suptitle('Induction prefix-matching across (layer, head) over training (every-4th cell)', y=1.0)\n"
            "fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02, label='prefix-matching')\n"
            "plt.show()"
        ),
        md(
            "## Threshold-sensitivity bracket per §H2-2\n"
            "\n"
            "Vary the locked threshold (0.3) by ± 25% in 5 increments and report "
            "the resulting μ for each. Documents how robust the emergence-step "
            "estimate is to threshold mis-specification."
        ),
        code(
            "rng = np.random.default_rng(1)\n"
            "rows = []\n"
            "for size in SIZES:\n"
            "    sub = summary[summary['size'] == size].sort_values('step')\n"
            "    steps = sub['step'].values\n"
            "    per_seq_by_step = {}\n"
            "    for s in steps:\n"
            "        npz = np.load(REPO / 'data' / 'exploration' / 'phase2_induction_per_seq' / f'{size}_step{s}.npz')\n"
            "        per_seq_by_step[int(s)] = npz['per_sequence_scores']\n"
            "    for frac in THRESHOLD_SENSITIVITY_FRACTIONS:\n"
            "        thr = INDUCTION_THRESHOLD * (1.0 + frac)\n"
            "        mus, mu_point = bootstrap_induction(per_seq_by_step, thr, rng, B=1)\n"
            "        rows.append(dict(size=size, threshold=thr, fraction=frac, mu_point=mu_point))\n"
            "sens_df = pd.DataFrame(rows)\n"
            "print(sens_df.to_string(index=False))"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(8, 5))\n"
            "for size in SIZES:\n"
            "    sub = sens_df[sens_df['size'] == size].sort_values('threshold')\n"
            "    ax.plot(sub['threshold'], sub['mu_point'], marker='o',\n"
            "            color=SIZE_COLOR[size], label=f'Pythia-{size}')\n"
            "ax.axvline(INDUCTION_THRESHOLD, color='gray', linestyle='--', alpha=0.5,\n"
            "           label=f'locked threshold = {INDUCTION_THRESHOLD}')\n"
            "ax.set_xlabel('detection threshold (prefix-matching score)')\n"
            "ax.set_ylabel('emergence step μ (point estimate)')\n"
            "ax.set_yscale('symlog', linthresh=1)\n"
            "ax.set_title('Threshold sensitivity of induction emergence step μ')\n"
            "ax.legend()\n"
            "ax.grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Verdict\n"
            "\n"
            "Phase 2 deliverable for induction: 40-cell sweep complete, μ_{s,induction} extracted with bootstrap CI per (size). The induction emergence step provides the *first* term in the H1-C ordering test; the joint H1-C verdict is in `h1c_ordering_test.ipynb`.\n"
            "\n"
            "Phase 1's 6-cell preview (`induction_emergence_exploration.ipynb`) remains on disk as a historical artifact for the chronology."
        ),
    ]
    return nb


def main() -> None:
    nb = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
