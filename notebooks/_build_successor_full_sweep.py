"""Build notebooks/successor_full_sweep.ipynb.

Successor full-sweep notebook: 40 cells × 4 Pythia sizes (70m, 160m, 410m, 1b)
per §H2-1 grid. Original 3-size Phase 2 deliverable + 1B added in-place under
§H3-scale (parquet artifacts split: `phase2_*_sweep.parquet` for original 3
sizes, `phase3_1b_*_sweep.parquet` for 1B; the notebook loads both
transparently). §SU-1b lift-form cross-category DLA detector with τ_lift =
0.13496 (§SU-tau).
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
            "# Successor emergence — full sweep (40 cells × 4 Pythia sizes)\n"
            "\n"
            "§SU-1b lift-form cross-category DLA detector across Pythia-70m, "
            "160m, 410m, 1b; locked threshold τ_lift = 0.13496 (§SU-tau); 40 "
            "log-spaced checkpoints per §H2-1; prompt-bootstrap CIs on μ per "
            "§H2-2; threshold-sensitivity bracket ± 25%.\n"
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
            "SIZES = ['70m', '160m', '410m', '1b']\n"
            "SIZE_COLOR = {'70m': 'tab:blue', '160m': 'tab:orange', '410m': 'tab:green', '1b': 'tab:red'}\n"
            "TAU_LIFT = 0.13496\n"
            "\n"
            "def sweep_path(size: str) -> Path:\n"
            "    if size == '1b':\n"
            "        return REPO / 'data' / 'exploration' / 'phase3_1b_successor_sweep.parquet'\n"
            "    return REPO / 'data' / 'exploration' / 'phase2_successor_sweep.parquet'\n"
            "\n"
            "def per_prompt_dir(size: str) -> Path:\n"
            "    if size == '1b':\n"
            "        return REPO / 'data' / 'exploration' / 'phase3_1b_successor_per_prompt'\n"
            "    return REPO / 'data' / 'exploration' / 'phase2_successor_per_prompt'"
        ),
        md("## Load sweep data"),
        code(
            "df = pd.concat([read_long(sweep_path(s)) for s in SIZES]).reset_index(drop=True)\n"
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
            "        npz = np.load(per_prompt_dir(size) / f'{size}_step{s}.npz')\n"
            "        per_prompt_by_step[int(s)] = {\n"
            "            'real': npz['per_prompt_real'],\n"
            "            'null': npz['per_prompt_null'],\n"
            "            'cats': npz['prompt_categories'],\n"
            "        }\n"
            "    mus, mu_point = bootstrap_successor(per_prompt_by_step, TAU_LIFT, rng, B=1000)\n"
            "    bres = summarize_bootstrap(size, 'successor', TAU_LIFT, mus, mu_point)\n"
            "    fr = tiered_fit(size, 'successor', steps, counts,\n"
            "                    bootstrap_median_mu=bres.mu_bootstrap_median)\n"
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
            "ax.set_title('Successor emergence with bootstrap CI on μ (4 Pythia sizes)')\n"
            "ax.legend(fontsize=9)\n"
            "ax.grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Head identity at final checkpoint (step143000)\n"
            "\n"
            "Per-size enumeration of every (layer, head) pair passing τ_lift = "
            "0.13496 at step143000, sorted by lift score. The inspectable "
            "circuit a reviewer needs in order to compare against Pythia-410m "
            "L22H6 (the §SU-2 anchor) and the successor heads identified in "
            "the Phase 1 6-cell preview."
        ),
        code(
            "rows = []\n"
            "for size in SIZES:\n"
            "    final = df[(df['size'] == size) & (df['step'] == 143000)]\n"
            "    n_layers = int(final['layer'].max() + 1)\n"
            "    passes = final[final['score'] >= TAU_LIFT].sort_values('score', ascending=False)\n"
            "    if passes.empty:\n"
            "        rows.append(dict(size=size, layer=None, head=None,\n"
            "                         lift_score=float('nan'),\n"
            "                         norm_depth=float('nan'),\n"
            "                         note='no heads above τ_lift'))\n"
            "        continue\n"
            "    for _, r in passes.iterrows():\n"
            "        rows.append(dict(\n"
            "            size=size,\n"
            "            layer=int(r['layer']), head=int(r['head']),\n"
            "            lift_score=float(r['score']),\n"
            "            norm_depth=int(r['layer']) / max(n_layers - 1, 1),\n"
            "            note='',\n"
            "        ))\n"
            "id_df = pd.DataFrame(rows)\n"
            "print('Successor heads at step143000, by size (sorted by lift within size):')\n"
            "print(id_df.to_string(index=False, float_format=lambda v: f'{v:.4f}'))"
        ),
        md(
            "## Verdict\n"
            "\n"
            "Successor full-sweep complete across 4 Pythia sizes; μ_{s,successor} extracted per (size). The successor emergence step provides the *middle* term in the H1-C ordering test (and the §H3-scale (B.ii) check at 1B, where ordering reverses); joint verdict is in `h1c_ordering_test.ipynb`."
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
