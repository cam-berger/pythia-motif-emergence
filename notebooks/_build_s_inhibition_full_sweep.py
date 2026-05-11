"""Build notebooks/s_inhibition_full_sweep.ipynb.

S-inhibition full-sweep notebook: 5 Pythia sizes (70m, 160m, 410m, 1b, 2.8b).
- 70m / 160m / 410m: 40 cells per §H2-1 grid (`phase2_s_inhibition_sweep.parquet`).
- 1b: 40 cells per §H2-1 grid (`phase3_1b_s_inhibition_sweep.parquet`) per §H3-scale-8-vis.
- 2.8b: **10 cells** per §H4-supersede reduced grid
  (`phase4_2_8b_s_inhibition_supersede_sweep.parquet`), authorized for
  5-size visualization by §writeup-conv-4 (§H4-supersede-vis). The original 40-cell
  §H4-scaling sweep was halted at 8/40 cells under §H4-7-supersede (DEFERRED);
  §H4-supersede (2026-05-10) ran the reduced 10-cell grid
  `[5000, 7000, 10000, 14000, 20000, 29000, 41000, 49000, 59000, 70000]`
  and PASSed both (A.timing) and (A.count) gates on 2026-05-11.

§S-1 path-patching Δ_h detector with τ_strict = 0.0372 (§S-tau).

§writeup-conv-4 caveat: 2.8B is presentation-only and does NOT extend the
§H1-C registered emergence claim. The registered Track-1 evidence is in the
70m / 160m / 410m columns.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "s_inhibition_full_sweep.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            "# S-inhibition emergence — full sweep (5 Pythia sizes)\n"
            "\n"
            "Goldowsky-Dill 2023 path-patching detector with frozen paths "
            "across Pythia-70m, 160m, 410m, 1b, 2.8b; locked threshold "
            "τ_strict = 0.0372 (§S-tau).\n"
            "\n"
            "- **70m / 160m / 410m / 1b:** 40 log-spaced checkpoints per §H2-1.\n"
            "- **2.8b:** **10 cells** per §H4-supersede reduced grid `[5000, 7000, 10000, 14000, 20000, 29000, 41000, 49000, 59000, 70000]`. The original §H4-scaling 40-cell sweep was halted at 8/40 cells under §H4-7-supersede (DEFERRED); §H4-supersede (2026-05-10) registered the reduced grid and PASSed both (A.timing) and (A.count) gates on 2026-05-11. The 2.8b column is authorized for 5-size visualization by §writeup-conv-4 (§H4-supersede-vis).\n"
            "\n"
            "Prompt-bootstrap CIs on μ per §H2-2 are computed for sizes with full 40-cell sweeps; the 2.8b column's μ is computed against the 10-cell §H4-supersede grid only (this is what §H4-supersede registered as the canonical 2.8b detection grid).\n"
            "\n"
            "**§writeup-conv-4 caveat:** 2.8b is presentation-only. The registered §H1-C Track-1 emergence claim remains locked to 70m / 160m / 410m per §H2-1.\n"
            "\n"
            "**Locked thresholds:**\n"
            "- Detection threshold: Δ_h ≥ τ_strict = 0.0372 (§S-tau).\n"
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
            "    bootstrap_s_inhibition, summarize_bootstrap, THRESHOLD_SENSITIVITY_FRACTIONS,\n"
            ")\n"
            "\n"
            "SIZES = ['70m', '160m', '410m', '1b', '2.8b']\n"
            "SIZE_COLOR = {'70m': 'tab:blue', '160m': 'tab:orange', '410m': 'tab:green', '1b': 'tab:red', '2.8b': 'tab:purple'}\n"
            "TAU_STRICT = 0.0372\n"
            "\n"
            "def sweep_path(size: str) -> Path:\n"
            "    if size == '1b':\n"
            "        return REPO / 'data' / 'exploration' / 'phase3_1b_s_inhibition_sweep.parquet'\n"
            "    if size == '2.8b':\n"
            "        # §writeup-conv-4 / §H4-supersede-vis: 2.8b uses the §H4-supersede 10-cell reduced grid.\n"
            "        return REPO / 'data' / 'exploration' / 'phase4_2_8b_s_inhibition_supersede_sweep.parquet'\n"
            "    return REPO / 'data' / 'exploration' / 'phase2_s_inhibition_sweep.parquet'\n"
            "\n"
            "def per_prompt_dir(size: str) -> Path:\n"
            "    if size == '1b':\n"
            "        return REPO / 'data' / 'exploration' / 'phase3_1b_s_inhibition_per_prompt'\n"
            "    if size == '2.8b':\n"
            "        return REPO / 'data' / 'exploration' / 'phase4_2_8b_s_inhibition_supersede_per_prompt'\n"
            "    return REPO / 'data' / 'exploration' / 'phase2_s_inhibition_per_prompt'"
        ),
        md("## Load sweep data"),
        code(
            "# Dedup parquet paths: phase2_s_inhibition_sweep.parquet holds all 3 phase2 sizes,\n"
            "# so sweep_path('70m') == sweep_path('410m'). Iterating SIZES naïvely triple-counts.\n"
            "_paths = list(dict.fromkeys(sweep_path(s) for s in SIZES))\n"
            "df = pd.concat([read_long(p) for p in _paths]).reset_index(drop=True)\n"
            "STEPS = sorted(df.step.unique().tolist())\n"
            "print(f'Total rows: {len(df):,}; sizes: {sorted(df[\"size\"].unique().tolist())}; n cells per size: {len(STEPS)}')"
        ),
        md("## Per-cell summary: count of S-inhibition heads (Δ_h ≥ τ_strict)"),
        code(
            "summary_rows = []\n"
            "for (size, step), grp in df.groupby(['size', 'step']):\n"
            "    scores = grp['score'].values\n"
            "    n_above = int((scores >= TAU_STRICT).sum())\n"
            "    summary_rows.append(dict(\n"
            "        size=size, step=step,\n"
            "        n_above_tau=n_above,\n"
            "        max_delta=float(scores.max()),\n"
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
            "    nm_layers_by_step = {}\n"
            "    for s in steps:\n"
            "        npz = np.load(per_prompt_dir(size) / f'{size}_step{s}.npz')\n"
            "        per_prompt_by_step[int(s)] = npz['per_prompt_delta']\n"
            "        nm_layers_by_step[int(s)] = npz['nm_heads'][:, 0].tolist()\n"
            "    # Sender layers: first column of (layer, head) for n_layers × n_heads.\n"
            "    a_step = steps[0]\n"
            "    n_senders = per_prompt_by_step[a_step].shape[0]\n"
            "    # Sender layer = senders[i][0]; runner used row-major (L, H) over all heads.\n"
            "    # Reconstruct: senders = [(L, H) for L in range(n_layers) for H in range(n_heads)].\n"
            "    a_grp = df[(df['size'] == size) & (df['step'] == a_step)]\n"
            "    n_layers = int(a_grp['layer'].max() + 1)\n"
            "    n_heads = int(a_grp['head'].max() + 1)\n"
            "    sender_layers = np.array([L for L in range(n_layers) for _ in range(n_heads)])\n"
            "    mus, mu_point = bootstrap_s_inhibition(\n"
            "        per_prompt_by_step, TAU_STRICT, nm_layers_by_step, sender_layers, rng, B=1000,\n"
            "    )\n"
            "    bres = summarize_bootstrap(size, 's_inhibition', TAU_STRICT, mus, mu_point)\n"
            "    fr = tiered_fit(size, 's_inhibition', steps, counts,\n"
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
            "ax.set_xscale('symlog', linthresh=100)\n"
            "ax.set_xlim(100, 150000)\n"
            "ax.set_xlabel('training step (symlog)')\n"
            "ax.set_ylabel(f'count of heads with Δ_h ≥ {TAU_STRICT}')\n"
            "ax.set_title('S-inhibition emergence with bootstrap CI on μ (4 Pythia sizes)')\n"
            "ax.legend(fontsize=9)\n"
            "ax.grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Sender-head identity at final checkpoint (step143000 where available)\n"
            "\n"
            "Per-size enumeration of every sender (layer, head) pair passing "
            "τ_strict = 0.0372 at step143000, sorted by Δ_h. Note: the "
            "S-inhibition score is per-*sender*; the receiver NM heads are "
            "fixed by the IOI circuit definition (Wang 2022). Cross-reference "
            "with `h1c_ordering_test.ipynb` sub-deliverable 5 for cross-motif "
            "depth comparison and §S-tau for the threshold derivation.\n"
            "\n"
            "**70m note:** if the table is empty for 70m, that is the "
            "scale-dependence finding made concrete — S-inhibition does not "
            "emerge at the smallest size during training (max_count = 1 head, "
            "right-censored at step143000).\n"
            "\n"
            "**2.8b note (§writeup-conv-4 / §H4-supersede-vis):** the §H4-supersede "
            "10-cell grid stops at step 70000 — there is no step143000 cell in "
            "this notebook's 2.8b column. We report the **latest available 2.8b "
            "cell (step 70000)** as the closest analog of a terminal-anchor view. "
            "For the registered §H5-causal-3-2.8b causal-dependence anchor at "
            "step143000 (which uses a separate sealed S-inhibition anchor parquet, "
            "not the §H4-supersede sweep), see `causal_dependence.ipynb`."
        ),
        code(
            "rows = []\n"
            "for size in SIZES:\n"
            "    sub_size = df[df['size'] == size]\n"
            "    if sub_size.empty:\n"
            "        continue\n"
            "    # 2.8b uses the §H4-supersede 10-cell grid (last cell = step 70000);\n"
            "    # all other sizes use the §H2-1 40-cell grid (last cell = step 143000).\n"
            "    target_step = 143000 if 143000 in set(sub_size['step'].unique()) else int(sub_size['step'].max())\n"
            "    final = sub_size[sub_size['step'] == target_step]\n"
            "    if final.empty:\n"
            "        continue\n"
            "    n_layers = int(final['layer'].max() + 1)\n"
            "    passes = final[final['score'] >= TAU_STRICT].sort_values('score', ascending=False)\n"
            "    step_note = '' if target_step == 143000 else f' (latest 2.8b cell; §H4-supersede grid stops at step 70000)'\n"
            "    if passes.empty:\n"
            "        rows.append(dict(size=size, step=target_step, sender_layer=None, sender_head=None,\n"
            "                         delta_h=float('nan'),\n"
            "                         norm_depth=float('nan'),\n"
            "                         note='no senders above τ_strict (S-inhibition did not emerge at this scale)' + step_note))\n"
            "        continue\n"
            "    for _, r in passes.iterrows():\n"
            "        rows.append(dict(\n"
            "            size=size,\n"
            "            step=target_step,\n"
            "            sender_layer=int(r['layer']), sender_head=int(r['head']),\n"
            "            delta_h=float(r['score']),\n"
            "            norm_depth=int(r['layer']) / max(n_layers - 1, 1),\n"
            "            note=step_note.strip(' ()') if step_note else '',\n"
            "        ))\n"
            "id_df = pd.DataFrame(rows)\n"
            "print('S-inhibition sender heads at last available cell per size (sorted by Δ_h within size):')\n"
            "print(id_df.to_string(index=False, float_format=lambda v: f'{v:.4f}'))"
        ),
        md(
            "## Verdict\n"
            "\n"
            "S-inhibition full-sweep complete across 4 Pythia sizes; μ_{s,S-inhibition} extracted per (size). The S-inhibition emergence step provides the *third* term in the H1-C ordering test (registered, 3 sizes) and the (A.i)/(A.ii)/(A.iii) legs of §H3-scale (1B scale-extension); the joint verdict combining both is in `h1c_ordering_test.ipynb`."
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
