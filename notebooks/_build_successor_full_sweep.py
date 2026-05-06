"""Build notebooks/successor_full_sweep.ipynb.

Phase 2 successor full-sweep notebook (40 cells × 3 sizes per §H2-1).
40-cell extension of the Phase 1 6-cell preview (`successor_emergence_
exploration.ipynb`, kept as historical artifact).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "successor_full_sweep.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            "# Successor emergence — Phase 2 full sweep (40 cells × 3 sizes)\n"
            "\n"
            "**Phase 2 deliverable.** 40-cell extension of the Phase 1 6-cell "
            "preview (`successor_emergence_exploration.ipynb`, kept as "
            "historical artifact). §SU-1b lift-form cross-category DLA "
            "detector; locked threshold τ_lift = 0.13496 (§SU-tau); 40 "
            "log-spaced checkpoints per §H2-1; prompt-bootstrap CIs on μ "
            "per §H2-2; threshold-sensitivity bracket ± 25%.\n"
            "\n"
            "**Locked thresholds:**\n"
            "- Detection threshold: lift ≥ τ_lift = 0.13496 (§SU-tau).\n"
            "- Bootstrap: per-prompt resampling, B = 1000, 95% percentile CI.\n"
            "- Tiered fit handling: emerged (max ≥ 5), marginal (2-4), censored (max < 2)."
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
            "    bootstrap_successor, summarize_bootstrap, THRESHOLD_SENSITIVITY_FRACTIONS,\n"
            ")\n"
            "\n"
            "SIZES = ['70m', '160m', '410m']\n"
            "SIZE_COLOR = {'70m': 'tab:blue', '160m': 'tab:orange', '410m': 'tab:green'}\n"
            "TAU_LIFT = 0.13496"
        ),
        md("## Load sweep data"),
        code(
            "df = read_long(REPO / 'data' / 'exploration' / 'phase2_successor_sweep.parquet')\n"
            "STEPS = sorted(df.step.unique().tolist())\n"
            "print(f'Total rows: {len(df):,}; sizes: {sorted(df[\"size\"].unique().tolist())}; n cells per size: {len(STEPS)}')"
        ),
        md("## Per-cell summary: count of successor heads (lift ≥ τ_lift)"),
        code(
            "summary_rows = []\n"
            "for (size, step), grp in df.groupby(['size', 'step']):\n"
            "    scores = grp['score'].values\n"
            "    n_above = int((scores >= TAU_LIFT).sum())\n"
            "    summary_rows.append(dict(\n"
            "        size=size, step=step,\n"
            "        n_above_tau=n_above,\n"
            "        max_lift=float(scores.max()),\n"
            "    ))\n"
            "summary = pd.DataFrame(summary_rows)\n"
            "summary['size'] = pd.Categorical(summary['size'], categories=SIZES, ordered=True)\n"
            "summary = summary.sort_values(['size', 'step']).reset_index(drop=True)\n"
            "print(summary.to_string(index=False))"
        ),
        md("## Emergence curves with bootstrap CI on μ"),
        code(
            "rng = np.random.default_rng(0)\n"
            "fits = {}\n"
            "boots = {}\n"
            "for size in SIZES:\n"
            "    sub = summary[summary['size'] == size].sort_values('step')\n"
            "    steps = sub['step'].values\n"
            "    counts = sub['n_above_tau'].values\n"
            "    per_prompt_by_step = {}\n"
            "    for s in steps:\n"
            "        npz = np.load(REPO / 'data' / 'exploration' / 'phase2_successor_per_prompt' / f'{size}_step{s}.npz')\n"
            "        per_prompt_by_step[int(s)] = {\n"
            "            'real': npz['per_prompt_real'],\n"
            "            'null': npz['per_prompt_null'],\n"
            "            'cats': npz['prompt_categories'],\n"
            "        }\n"
            "    mus, mu_point = bootstrap_successor(per_prompt_by_step, TAU_LIFT, rng, B=1000)\n"
            "    bres = summarize_bootstrap(size, 'successor', TAU_LIFT, mus, mu_point)\n"
            "    fr = tiered_fit(size, 'successor', steps, counts)\n"
            "    fits[size] = fr\n"
            "    boots[size] = bres\n"
            "    print(f'{size}: regime={fr.regime}, max_count={fr.max_count}, μ_point={fr.mu:.0f}, '\n"
            "          f'μ_boot=[{bres.mu_ci_low:.0f}, {bres.mu_ci_high:.0f}]')"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "for size in SIZES:\n"
            "    sub = summary[summary['size'] == size]\n"
            "    ax.plot(sub['step'], sub['n_above_tau'], marker='o', alpha=0.6,\n"
            "            color=SIZE_COLOR[size], label=f'Pythia-{size}')\n"
            "    bres = boots[size]\n"
            "    if np.isfinite(bres.mu_point_estimate) and bres.mu_point_estimate < RIGHT_CENSOR_STEP - 1:\n"
            "        ax.axvspan(bres.mu_ci_low, bres.mu_ci_high, alpha=0.15, color=SIZE_COLOR[size])\n"
            "        ax.axvline(bres.mu_point_estimate, color=SIZE_COLOR[size],\n"
            "                   linestyle='--', alpha=0.7,\n"
            "                   label=f'  μ = {bres.mu_point_estimate:.0f}')\n"
            "ax.set_xscale('symlog', linthresh=1)\n"
            "ax.set_xlim(0.5, 200000)\n"
            "ax.set_xlabel('training step (symlog)')\n"
            "ax.set_ylabel(f'count of heads with lift ≥ {TAU_LIFT}')\n"
            "ax.set_title('Successor emergence with bootstrap CI on μ (Phase 2 full sweep)')\n"
            "ax.legend(fontsize=9)\n"
            "ax.grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Verdict\n"
            "\n"
            "Phase 2 deliverable for successor: 40-cell sweep complete; μ_{s,successor} extracted per (size). The successor emergence step provides the *middle* term in the H1-C ordering test; joint verdict is in `h1c_ordering_test.ipynb`."
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
