"""Build notebooks/induction_emergence_exploration.ipynb.

Sparse exploratory sweep across Pythia 70M / 160M / 410M to answer:
  - When do induction heads emerge in each size?
  - How do they accumulate over training?
  - When does the count saturate?

This is exploration, NOT the pre-registered Phase 2 sweep. The Phase 2
schedule is 40 checkpoints per size from checkpoints.yaml; here we use
6 log-spaced steps to keep wall time reasonable for a preview.

Run:
    uv run python notebooks/_build_emergence_exploration.py
    uv run jupyter nbconvert --to notebook --execute --inplace \
        notebooks/induction_emergence_exploration.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "induction_emergence_exploration.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            "# Exploration: induction-head emergence dynamics in Pythia\n"
            "\n"
            "Three questions:\n"
            "\n"
            "1. **When do induction heads emerge** in each Pythia size?\n"
            "2. **How do they accumulate** over training — gradual ramp, sudden jump, "
            "stepped pattern?\n"
            "3. **When does the count saturate** — i.e., at what point do *new* "
            "induction heads stop emerging?\n"
            "\n"
            "**This is exploration, not the pre-registered sweep.** The Phase 2 sweep "
            "(`PROJECT_BRIEF.md` §5) uses 40 checkpoints per size from "
            "`checkpoints.yaml`. Here we use 6 log-spaced steps per size as a preview, "
            "so total wall-clock fits in a single afternoon. The exploration output is "
            "saved to `data/exploration/` and is *not* the canonical sweep dataset.\n"
            "\n"
            "Detector: same Olsson prefix-matching score we validated on Day 1. Sample "
            "size reduced to 20 random sequences (vs. 50) for speed; score variance "
            "across that many sequences is small enough for the trends we're looking for."
        ),
        md("## Setup"),
        code(
            "import os\n"
            "os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'\n"
            "\n"
            "import sys, time\n"
            "from pathlib import Path\n"
            "REPO = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "if str(REPO) not in sys.path:\n"
            "    sys.path.insert(0, str(REPO))\n"
            "\n"
            "import torch\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from scipy.optimize import curve_fit\n"
            "\n"
            "from src.utils.pythia_loader import load_pythia, prefetch_pythia\n"
            "from src.detectors.induction import prefix_matching_score\n"
            "\n"
            "torch.set_grad_enabled(False)\n"
            "\n"
            "def free_mps_cache():\n"
            "    if torch.backends.mps.is_available():\n"
            "        try:\n"
            "            torch.mps.empty_cache()\n"
            "        except Exception:\n"
            "            pass\n"
            "\n"
            "OUT_DIR = REPO / 'data' / 'exploration'\n"
            "OUT_DIR.mkdir(parents=True, exist_ok=True)\n"
            "OUT_PARQUET = OUT_DIR / 'induction_emergence_preview.parquet'"
        ),
        md(
            "## Sweep configuration\n"
            "\n"
            "6 checkpoints per size, chosen to bracket the expected emergence transition. "
            "Step 0 is initialization; steps 1000–25000 cover the documented Pythia "
            "induction-emergence window from Olsson 2022 / Tigges 2024; step 143000 is "
            "the final checkpoint."
        ),
        code(
            "SIZES = ['70m', '160m', '410m']\n"
            "CHECKPOINTS = [0, 1000, 3000, 8000, 25000, 143000]\n"
            "N_SEQUENCES = 20\n"
            "SEQ_LEN = 100\n"
            "THRESHOLD_CANDIDATE = 0.3\n"
            "THRESHOLD_STRONG = 0.5\n"
            "\n"
            "n_runs = len(SIZES) * len(CHECKPOINTS)\n"
            "print(f'sweep: {len(SIZES)} sizes x {len(CHECKPOINTS)} checkpoints = {n_runs} runs')\n"
            "print(f'detector: prefix_matching_score, n_sequences={N_SEQUENCES}, seq_len={SEQ_LEN}')"
        ),
        md(
            "## Prefetch checkpoints\n"
            "\n"
            "Pull each (size, step) snapshot to the local HF cache. First-time downloads "
            "can take several minutes; subsequent calls are no-ops."
        ),
        code(
            "t0 = time.time()\n"
            "for size in SIZES:\n"
            "    for step in CHECKPOINTS:\n"
            "        t1 = time.time()\n"
            "        prefetch_pythia(size, step)\n"
            "        print(f'  pythia-{size} @ step{step:>6} ready ({time.time() - t1:.1f}s)')\n"
            "print(f'\\nprefetch total: {time.time() - t0:.1f}s')"
        ),
        md(
            "## Run the sweep\n"
            "\n"
            "For each (size, step), load the model, score every (layer, head) with the "
            "induction detector, and append to a long-format DataFrame. Persist to parquet."
        ),
        code(
            "if OUT_PARQUET.exists():\n"
            "    df = pd.read_parquet(OUT_PARQUET)\n"
            "    print(f'loaded cached results from {OUT_PARQUET.relative_to(REPO)} '\n"
            "          f'({len(df)} rows)')\n"
            "else:\n"
            "    records = []\n"
            "    t_total = time.time()\n"
            "    for size in SIZES:\n"
            "        for step in CHECKPOINTS:\n"
            "            t1 = time.time()\n"
            "            model = load_pythia(size, step=step)\n"
            "            result = prefix_matching_score(\n"
            "                model, n_sequences=N_SEQUENCES, seq_len=SEQ_LEN\n"
            "            )\n"
            "            scores = result.scores.numpy()\n"
            "            n_layers, n_heads = scores.shape\n"
            "            for layer in range(n_layers):\n"
            "                for head in range(n_heads):\n"
            "                    records.append({\n"
            "                        'size': size, 'step': int(step),\n"
            "                        'layer': int(layer), 'head': int(head),\n"
            "                        'score': float(scores[layer, head]),\n"
            "                    })\n"
            "            del model\n"
            "            free_mps_cache()\n"
            "            top = scores.max()\n"
            "            print(f'  pythia-{size} @ step{step:>6}: max score {top:.3f} '\n"
            "                  f'({time.time() - t1:.1f}s)')\n"
            "    df = pd.DataFrame(records)\n"
            "    df.to_parquet(OUT_PARQUET)\n"
            "    print(f'\\nsweep total: {time.time() - t_total:.1f}s '\n"
            "          f'-> {OUT_PARQUET.relative_to(REPO)} ({len(df)} rows)')"
        ),
        md(
            "## Q1+Q2: when do induction heads emerge, and how do they accumulate?\n"
            "\n"
            "Count of heads above each threshold, plotted against training step. Use a "
            "symmetric-log x-axis so step 0 (init) is visible alongside the log-spaced "
            "later checkpoints."
        ),
        code(
            "counts = (\n"
            "    df.groupby(['size', 'step'])\n"
            "      .agg(\n"
            "          n_above_03=('score', lambda s: (s > THRESHOLD_CANDIDATE).sum()),\n"
            "          n_above_05=('score', lambda s: (s > THRESHOLD_STRONG).sum()),\n"
            "          max_score=('score', 'max'),\n"
            "          mean_score=('score', 'mean'),\n"
            "      )\n"
            "      .reset_index()\n"
            "      .sort_values(['size', 'step'])\n"
            ")\n"
            "print(counts.to_string(index=False))"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "size_color = {'70m': 'tab:blue', '160m': 'tab:orange', '410m': 'tab:green'}\n"
            "for ax, (col, thresh) in zip(\n"
            "    axes,\n"
            "    [('n_above_03', THRESHOLD_CANDIDATE), ('n_above_05', THRESHOLD_STRONG)],\n"
            "):\n"
            "    for size in SIZES:\n"
            "        sub = counts[counts['size'] == size]\n"
            "        ax.plot(sub['step'], sub[col], marker='o',\n"
            "                color=size_color[size], label=f'Pythia-{size}')\n"
            "    ax.set_xscale('symlog', linthresh=1)\n"
            "    ax.set_xlim(100, 150000)\n"
            "    ax.set_xlabel('training step (symlog)')\n"
            "    ax.set_ylabel(f'count of heads with score > {thresh}')\n"
            "    ax.set_title(f'Induction-head count above {thresh}')\n"
            "    ax.grid(alpha=0.3)\n"
            "    ax.legend()\n"
            "fig.suptitle('Induction-head emergence across Pythia sizes', y=1.02)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "Reading the curves:\n"
            "\n"
            "- The y-axis is *count of heads*, not the strength of any one head. Both "
            "panels should rise from ~0 at step 0 to some saturated value by step 143000.\n"
            "- The transition between near-zero and saturated is the **emergence "
            "window**. If the rise is steep, induction emerges as a phase transition; if "
            "gradual, it accumulates head-by-head over many steps.\n"
            "- Comparing across sizes: do smaller models emerge earlier (in absolute "
            "step count) or later? The Olsson 2022 paper documents earlier emergence in "
            "smaller models for some scales — does this hold here too?"
        ),
        md(
            "## Q1 quantified: estimate $\\mu$ per size via logistic fit\n"
            "\n"
            "Fit `count(step) ≈ L / (1 + exp(-k · (log(step) - μ)))` per size to the "
            "*candidate* count (threshold 0.3, more data points than 0.5). $\\mu$ is the "
            "log-step at which count reaches half its final value — the operational "
            "definition of emergence step from `HYPOTHESIS.md`.\n"
            "\n"
            "**Caveat:** with only 6 checkpoints per size, the logistic fit has very few "
            "degrees of freedom and is more anecdotal than rigorous. Phase 2's 40-point "
            "sweep is what gives this number actual statistical weight."
        ),
        code(
            "def logistic(log_step, L, k, mu):\n"
            "    return L / (1.0 + np.exp(-k * (log_step - mu)))\n"
            "\n"
            "fit_rows = []\n"
            "for size in SIZES:\n"
            "    sub = counts[counts['size'] == size].copy()\n"
            "    log_steps = np.log(sub['step'].clip(lower=1).to_numpy())\n"
            "    y = sub['n_above_03'].to_numpy().astype(float)\n"
            "    if y.max() < 1:\n"
            "        fit_rows.append({'size': size, 'L': float('nan'),\n"
            "                         'k': float('nan'), 'mu_log': float('nan'),\n"
            "                         'mu_step': float('nan')})\n"
            "        continue\n"
            "    p0 = (max(y.max(), 1.0), 1.0, np.median(log_steps))\n"
            "    try:\n"
            "        popt, _ = curve_fit(logistic, log_steps, y, p0=p0, maxfev=10000)\n"
            "        L_fit, k_fit, mu_fit = popt\n"
            "    except Exception as e:\n"
            "        print(f'  fit failed for {size}: {e}')\n"
            "        L_fit = k_fit = mu_fit = float('nan')\n"
            "    fit_rows.append({'size': size, 'L': L_fit, 'k': k_fit,\n"
            "                     'mu_log': mu_fit, 'mu_step': float(np.exp(mu_fit))})\n"
            "fit_df = pd.DataFrame(fit_rows)\n"
            "print(fit_df.to_string(index=False))"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(9, 5))\n"
            "for size in SIZES:\n"
            "    sub = counts[counts['size'] == size]\n"
            "    ax.plot(sub['step'], sub['n_above_03'], 'o',\n"
            "            color=size_color[size], label=f'Pythia-{size}')\n"
            "    fit_row = fit_df[fit_df['size'] == size].iloc[0]\n"
            "    if not np.isnan(fit_row['L']):\n"
            "        log_grid = np.linspace(0, np.log(143000), 200)\n"
            "        y_fit = logistic(log_grid, fit_row['L'], fit_row['k'],\n"
            "                          fit_row['mu_log'])\n"
            "        ax.plot(np.exp(log_grid), y_fit, '-',\n"
            "                color=size_color[size], alpha=0.6,\n"
            "                label=f'fit: μ ≈ step {fit_row[\"mu_step\"]:.0f}')\n"
            "        ax.axvline(fit_row['mu_step'], color=size_color[size],\n"
            "                   linestyle=':', alpha=0.5)\n"
            "ax.set_xscale('symlog', linthresh=1)\n"
            "ax.set_xlim(0, 170000)\n"
            "ax.set_xlabel('training step (symlog)')\n"
            "ax.set_ylabel('count of heads with score > 0.3')\n"
            "ax.set_title('Logistic fit: emergence step μ per Pythia size (count > 0.3)')\n"
            "ax.legend()\n"
            "ax.grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Q3: do new induction heads stop appearing?\n"
            "\n"
            "If the count saturates by some step `s*`, then `count(s*) ≈ count(143000)` "
            "for all later checkpoints. We measure how close the *penultimate* checkpoint "
            "is to the final, per size. Closer to 1.0 means the model has finished "
            "growing induction heads well before training ends; substantially less than "
            "1.0 means new heads are still emerging late in training."
        ),
        code(
            "satur_rows = []\n"
            "for size in SIZES:\n"
            "    sub = counts[counts['size'] == size].sort_values('step')\n"
            "    final = sub.iloc[-1]\n"
            "    penult = sub.iloc[-2]\n"
            "    halfway = sub.iloc[len(sub) // 2]\n"
            "    satur_rows.append({\n"
            "        'size': size,\n"
            "        'count_final': int(final['n_above_03']),\n"
            "        'count_penult': int(penult['n_above_03']),\n"
            "        'penult_over_final': (\n"
            "            float(penult['n_above_03']) / max(final['n_above_03'], 1)\n"
            "        ),\n"
            "        'count_halfway': int(halfway['n_above_03']),\n"
            "        'halfway_over_final': (\n"
            "            float(halfway['n_above_03']) / max(final['n_above_03'], 1)\n"
            "        ),\n"
            "        'penult_step': int(penult['step']),\n"
            "        'halfway_step': int(halfway['step']),\n"
            "    })\n"
            "satur_df = pd.DataFrame(satur_rows)\n"
            "print(satur_df.to_string(index=False))"
        ),
        md(
            "Reading the table:\n"
            "\n"
            "- `penult_over_final` close to 1.0 → induction-head growth is essentially done "
            "by the penultimate checkpoint.\n"
            "- `halfway_over_final` tells the same story for an earlier reference point.\n"
            "- If both are near 1.0 across all sizes, the answer to Q3 is *yes, new heads "
            "stop emerging well before final*. If 410M trails 70M, larger models keep "
            "growing induction heads later.\n"
            "\n"
            "Caveat: this 6-checkpoint preview gives a coarse picture. The Phase 2 sweep "
            "with denser late-training sampling will resolve whether saturation is sharp "
            "or there's a long tail of late-emerging weak induction heads."
        ),
        md(
            "## Bonus: where in the model do induction heads live?\n"
            "\n"
            "Spatial-emergence heatmaps. For each size and checkpoint, color the "
            "`(layer, head)` grid by score. The pattern shows whether induction "
            "concentrates in specific layers and how that spatial distribution sharpens "
            "across training."
        ),
        code(
            "fig, axes = plt.subplots(\n"
            "    nrows=len(SIZES), ncols=len(CHECKPOINTS),\n"
            "    figsize=(3 * len(CHECKPOINTS), 3 * len(SIZES)),\n"
            ")\n"
            "vmax = float(df['score'].max())\n"
            "for row, size in enumerate(SIZES):\n"
            "    for col, step in enumerate(CHECKPOINTS):\n"
            "        ax = axes[row][col]\n"
            "        sub = df[(df['size'] == size) & (df['step'] == step)]\n"
            "        n_layers = sub['layer'].max() + 1\n"
            "        n_heads = sub['head'].max() + 1\n"
            "        grid = np.zeros((n_layers, n_heads))\n"
            "        for _, r in sub.iterrows():\n"
            "            grid[int(r['layer']), int(r['head'])] = r['score']\n"
            "        im = ax.imshow(grid, vmin=0, vmax=vmax, cmap='viridis', aspect='auto')\n"
            "        ax.set_title(f'pythia-{size} @ step{step}', fontsize=9)\n"
            "        if col == 0:\n"
            "            ax.set_ylabel('layer')\n"
            "        if row == len(SIZES) - 1:\n"
            "            ax.set_xlabel('head')\n"
            "fig.suptitle('Induction-head score across (layer, head) over training', y=1.0)\n"
            "fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02, label='prefix-matching score')\n"
            "plt.show()"
        ),
        md(
            "Reading the heatmap grid: rows are sizes, columns are checkpoints. Bright "
            "cells are induction heads. Watch for a sharp turn-on at one or two specific "
            "`(layer, head)` cells around the emergence step, vs a gradual brightening "
            "of multiple cells. The former would suggest a *circuit* discovery moment; "
            "the latter, gradual head-by-head accumulation.\n"
            "\n"
            "Smaller models have fewer heads (smaller grids). 70M has 6 layers × 8 "
            "heads = 48 heads; 160M has 12 × 12 = 144; 410M has 24 × 16 = 384."
        ),
        md(
            "## Bonus: trajectories of the strongest final-checkpoint heads\n"
            "\n"
            "Pick the top-3 heads (by final-checkpoint score) per size and plot their "
            "score across training. Does each head turn on at roughly the same step, "
            "or do they emerge sequentially?"
        ),
        code(
            "fig, axes = plt.subplots(1, len(SIZES), figsize=(15, 4.5), sharey=True)\n"
            "for ax, size in zip(axes, SIZES):\n"
            "    final = df[(df['size'] == size) & (df['step'] == 143000)]\n"
            "    top3 = final.nlargest(3, 'score')[['layer', 'head']].values\n"
            "    for layer, head in top3:\n"
            "        traj = df[(df['size'] == size) & (df['layer'] == layer) & (df['head'] == head)]\n"
            "        traj = traj.sort_values('step')\n"
            "        ax.plot(traj['step'], traj['score'], marker='o',\n"
            "                label=f'L{int(layer)}H{int(head)}')\n"
            "    ax.set_xscale('symlog', linthresh=1)\n"
            "    ax.set_xlim(10, 150000)\n"
            "    ax.set_xlabel('training step (symlog)')\n"
            "    if size == SIZES[0]:\n"
            "        ax.set_ylabel('prefix-matching score')\n"
            "    ax.set_title(f'Pythia-{size}: top-3 final heads')\n"
            "    ax.axhline(THRESHOLD_CANDIDATE, color='gray', linestyle='--', alpha=0.4)\n"
            "    ax.axhline(THRESHOLD_STRONG, color='red', linestyle='--', alpha=0.4)\n"
            "    ax.legend(fontsize=8)\n"
            "    ax.grid(alpha=0.3)\n"
            "fig.suptitle('Trajectories of the strongest final-checkpoint induction heads', y=1.02)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Conclusions and open questions\n"
            "\n"
            "From this 6-checkpoint preview:\n"
            "\n"
            "- **Q1 (when):** the logistic fits give one estimate of μ per size. Take "
            "these with a large grain of salt — Phase 2's denser sampling is what "
            "produces a defensible number.\n"
            "- **Q2 (how):** look at whether the curves are step-function-like (phase "
            "transition) or smoothly rising (gradual accumulation), and whether the "
            "trajectory plot shows simultaneous or sequential head turn-ons.\n"
            "- **Q3 (when stop):** read the saturation table. If `penult_over_final` is "
            "near 1.0 across sizes, growth is essentially done well before final.\n"
            "\n"
            "**Open questions** the Phase 2 sweep will resolve:\n"
            "\n"
            "- What does the *shape* of the emergence curve look like at higher temporal "
            "resolution? With only 6 points, we can't distinguish a sharp phase "
            "transition from a smooth ramp.\n"
            "- Does emergence step decrease, increase, or stay flat with model size?\n"
            "- How much does the threshold choice (0.3 vs 0.5 vs 0.7) shift μ?\n"
            "\n"
            "**Pre-registration discipline reminder:** results above are *exploratory*. "
            "They MUST NOT motivate a change to `HYPOTHESIS.md`'s ordering claim "
            "(induction → successor → suppression), since induction is the only "
            "motif this preview measures. Phase 2 will measure all three motifs with "
            "the registered schedule and analysis."
        ),
    ]
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    return nb


def main() -> None:
    nb = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
