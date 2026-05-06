"""Build notebooks/s_inhibition_emergence_exploration.ipynb.

Phase 1.3 deliverable (iii): emergence preview of S-inhibition heads across
Pythia 70m/160m/410m × 6 checkpoints, applying the locked Δ_h thresholds
(τ_strict = 0.0372, τ_permissive = 0.0186) calibrated from GPT-2 small.

Mirrors the induction emergence notebook (size palette tab:blue/orange/green,
gray dashed thresholds) and the copy-suppression emergence notebook (strict
+ permissive count tracking). Cross-motif comparison checks whether the
induction-before-S-inhibition prediction holds in this preview grid.

Run:
    uv run python notebooks/_run_s_inhibition_sweep.py
    uv run python notebooks/_build_s_inhibition_emergence.py
    uv run jupyter nbconvert --to notebook --execute --inplace \
        notebooks/s_inhibition_emergence_exploration.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "s_inhibition_emergence_exploration.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            "# S-inhibition emergence preview — Pythia 70m/160m/410m\n"
            "\n"
            "**Phase 1.3 deliverable (iii).** 18-cell exploration sweep: "
            "Pythia `{70m, 160m, 410m}` × `{0, 1k, 3k, 8k, 25k, 143k}`. Per "
            "cell, NMs are re-derived via component-DLA top-4 (so the NM set "
            "tracks the model's current state, not the final-checkpoint set), "
            "and the path-patching detector runs across all heads producing "
            "`Δ_h = mean over k=4 NMs of (patched−clean) attention shift at "
            "S2 minus the same shift at IO`.\n"
            "\n"
            "**Locked thresholds (HYPOTHESIS.md §S-tau):**\n"
            "- τ_strict = `0.0372` (= min Δ_h of Wang's 4 GPT-2 S-Inhibition "
            "heads).\n"
            "- τ_permissive = `0.0186` (= τ_strict / 2).\n"
            "\n"
            "**Per Q9-(b) sweep gating policy**, this runs unconditionally — "
            "the Pythia anchor (deliverable ii) result documents either way. "
            "The anchor passed both Q8 gates (numerical: 2 heads clear "
            "τ_strict; mechanistic: 2 of 4 NMs positive on top candidate "
            "L12H12), so the emergence figures here are interpretable as "
            "*the* S-inhibition emergence trajectory rather than as Path-C-"
            "style negative-result documentation.\n"
            "\n"
            "**This is exploration, not the pre-registered Phase 2 sweep.** "
            "Phase 2 will use 40 checkpoints per size and bootstrap CIs; this "
            "preview uses 6 log-spaced steps for shape inspection.\n"
            "\n"
            "**Three questions:**\n"
            "1. When do strict-criterion-passing S-inhibition heads first "
            "emerge in each Pythia size?\n"
            "2. How does the count grow over training (sigmoid? plateau? "
            "monotone?), and does the permissive bracket converge to the "
            "strict count by step143000?\n"
            "3. **H1-C ordering check:** does induction emerge before "
            "S-inhibition in each size? The hypothesis predicts μ_induction "
            "< μ_S-inhibition. With 6 checkpoints we can't fit a logistic "
            "robustly, but we can compare counts cell-by-cell."
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
            "from notebooks._lib.sweep_io import read_long, to_wide\n"
            "\n"
            "SIZES = ['70m', '160m', '410m']\n"
            "STEPS = [0, 1000, 3000, 8000, 25000, 143000]\n"
            "TAU_STRICT = 0.0372\n"
            "TAU_PERMISSIVE = 0.0186\n"
            "SIZE_COLOR = {'70m': 'tab:blue', '160m': 'tab:orange', '410m': 'tab:green'}"
        ),
        md(
            "## Load sweep data\n"
            "\n"
            "Long-format parquet from `_run_s_inhibition_sweep.py` (canonical "
            "schema `(size, step, layer, head, motif, score)`). Reuses the "
            "existing `notebooks._lib.sweep_io` helpers."
        ),
        code(
            "df_si = read_long(REPO / 'data' / 'exploration' / 's_inhibition_emergence_preview.parquet')\n"
            "print(f'Total rows: {len(df_si):,}')\n"
            "print(f'Unique motifs: {df_si.motif.unique().tolist()}')\n"
            "print(f'Unique sizes: {sorted(df_si[\"size\"].unique().tolist())}')\n"
            "print(f'Unique steps: {sorted(df_si.step.unique().tolist())}')\n"
            "print(f'Heads per (size, step) cell:')\n"
            "print(df_si.groupby(['size','step']).size().unstack().to_string())"
        ),
        md(
            "## Per-cell summary\n"
            "\n"
            "For each (size, step), compute: count of heads above τ_strict, "
            "count of heads above τ_permissive, max Δ_h, and (sender_layer, "
            "sender_head) of the top candidate."
        ),
        code(
            "summary_rows = []\n"
            "for (size, step), grp in df_si.groupby(['size', 'step']):\n"
            "    scores = grp['score'].values\n"
            "    strict = int((scores >= TAU_STRICT).sum())\n"
            "    permissive = int((scores >= TAU_PERMISSIVE).sum())\n"
            "    max_idx = int(np.argmax(scores))\n"
            "    top_row = grp.iloc[max_idx]\n"
            "    summary_rows.append(dict(\n"
            "        size=size, step=step,\n"
            "        n_strict=strict, n_permissive=permissive,\n"
            "        max_delta=float(scores.max()),\n"
            "        top_layer=int(top_row['layer']), top_head=int(top_row['head']),\n"
            "    ))\n"
            "summary = pd.DataFrame(summary_rows)\n"
            "summary['size'] = pd.Categorical(summary['size'], categories=SIZES, ordered=True)\n"
            "summary = summary.sort_values(['size', 'step']).reset_index(drop=True)\n"
            "print(summary.to_string(index=False))"
        ),
        md(
            "## Headline plot — emergence count across training\n"
            "\n"
            "Per Pythia size, count of heads clearing each threshold over "
            "training. Strict (solid) and permissive (dashed) curves; if the "
            "permissive count rises before strict, that's the emergence-"
            "approach phase where heads are getting close but haven't crossed "
            "the calibrated GPT-2 threshold yet."
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)\n"
            "for ax, (col, thresh, title) in zip(\n"
            "    axes,\n"
            "    [\n"
            "        ('n_strict', TAU_STRICT, f'Heads clearing τ_strict = {TAU_STRICT}'),\n"
            "        ('n_permissive', TAU_PERMISSIVE, f'Heads clearing τ_permissive = {TAU_PERMISSIVE}'),\n"
            "    ],\n"
            "):\n"
            "    for size in SIZES:\n"
            "        sub = summary[summary['size'] == size]\n"
            "        ax.plot(sub['step'], sub[col], marker='o',\n"
            "                color=SIZE_COLOR[size], label=f'Pythia-{size}')\n"
            "    ax.set_xscale('symlog', linthresh=1)\n"
            "    ax.set_xlim(10, 200000)\n"
            "    ax.set_xlabel('training step (symlog)')\n"
            "    ax.set_ylabel('count of S-inhibition heads')\n"
            "    ax.set_title(title)\n"
            "    ax.grid(alpha=0.3)\n"
            "    ax.legend()\n"
            "fig.suptitle('S-inhibition head count emergence across Pythia sizes', y=1.02)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Max Δ_h trajectory across training\n"
            "\n"
            "What's the *strongest* S-inhibition signal at each cell? If max "
            "Δ_h grows monotonically with training step, S-inhibition is "
            "*sharpening* over training; if it plateaus mid-training, the "
            "strongest head is set by some earlier step."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(8, 5))\n"
            "for size in SIZES:\n"
            "    sub = summary[summary['size'] == size]\n"
            "    ax.plot(sub['step'], sub['max_delta'], marker='o',\n"
            "            color=SIZE_COLOR[size], label=f'Pythia-{size}')\n"
            "ax.axhline(TAU_STRICT, color='gray', linestyle='--', alpha=0.4,\n"
            "           label=f'τ_strict = {TAU_STRICT}')\n"
            "ax.axhline(TAU_PERMISSIVE, color='gray', linestyle=':', alpha=0.4,\n"
            "           label=f'τ_permissive = {TAU_PERMISSIVE}')\n"
            "ax.set_xscale('symlog', linthresh=1)\n"
            "ax.set_xlim(10, 200000)\n"
            "ax.set_xlabel('training step (symlog)')\n"
            "ax.set_ylabel('max Δ_h across all heads')\n"
            "ax.set_title('Strongest S-inhibition signal per cell')\n"
            "ax.legend(loc='lower right')\n"
            "ax.grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## H1-C ordering check — induction vs S-inhibition\n"
            "\n"
            "The registered hypothesis H1-C predicts induction emerges before "
            "S-inhibition in each Pythia size. The induction emergence preview "
            "is in `data/exploration/induction_emergence_preview.parquet` "
            "(motif `induction_prefix_match`, threshold > 0.3). Plot both "
            "counts on the same axes per size."
        ),
        code(
            "ind_path = REPO / 'data' / 'exploration' / 'induction_emergence_preview.parquet'\n"
            "if ind_path.exists():\n"
            "    df_ind = read_long(ind_path)\n"
            "    ind_summary = []\n"
            "    for (size, step), grp in df_ind.groupby(['size', 'step']):\n"
            "        ind_summary.append(dict(\n"
            "            size=size, step=step,\n"
            "            n_induction=int((grp['score'] > 0.3).sum()),\n"
            "        ))\n"
            "    ind_df = pd.DataFrame(ind_summary)\n"
            "    fig, axes = plt.subplots(1, len(SIZES), figsize=(15, 4.5), sharey=False)\n"
            "    for ax, size in zip(axes, SIZES):\n"
            "        si_sub = summary[summary['size'] == size]\n"
            "        ind_sub = ind_df[ind_df['size'] == size]\n"
            "        ax.plot(si_sub['step'], si_sub['n_strict'], marker='o',\n"
            "                color=SIZE_COLOR[size],\n"
            "                label='S-inhibition (strict)')\n"
            "        ax.plot(si_sub['step'], si_sub['n_permissive'], marker='s',\n"
            "                color=SIZE_COLOR[size], linestyle=':',\n"
            "                label='S-inhibition (permissive)')\n"
            "        if not ind_sub.empty:\n"
            "            ax.plot(ind_sub['step'], ind_sub['n_induction'], marker='^',\n"
            "                    color='gray', alpha=0.7,\n"
            "                    label='induction (>0.3)')\n"
            "        ax.set_xscale('symlog', linthresh=1)\n"
            "        ax.set_xlim(10, 200000)\n"
            "        ax.set_xlabel('training step (symlog)')\n"
            "        if size == SIZES[0]:\n"
            "            ax.set_ylabel('count of heads passing threshold')\n"
            "        ax.set_title(f'Pythia-{size}')\n"
            "        ax.grid(alpha=0.3)\n"
            "        ax.legend(fontsize=8, loc='upper left')\n"
            "    fig.suptitle('H1-C ordering: induction vs S-inhibition emergence per size', y=1.02)\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "else:\n"
            "    print(f'Induction sweep parquet not found at {ind_path}; skipping comparison.')"
        ),
        md(
            "## Top-3 trajectories per size — which heads emerge as S-inhibition senders?\n"
            "\n"
            "For each size, identify the 3 heads with highest final-checkpoint "
            "Δ_h, then trace their per-cell scores across training. This shows "
            "whether the strongest S-inhibition head at step143000 was already "
            "the strongest at earlier steps (continuity) or whether different "
            "heads played the role at different stages of training (identity "
            "switching, like Phase 1.0's L1H4 → L2H8 finding for the strict "
            "copy-suppression detector)."
        ),
        code(
            "fig, axes = plt.subplots(1, len(SIZES), figsize=(15, 4.5), sharey=False)\n"
            "for ax, size in zip(axes, SIZES):\n"
            "    final = df_si[(df_si['size'] == size) & (df_si['step'] == 143000)]\n"
            "    top3 = final.nlargest(3, 'score')[['layer', 'head']].values\n"
            "    for layer, head in top3:\n"
            "        traj = df_si[(df_si['size'] == size)\n"
            "                      & (df_si['layer'] == layer)\n"
            "                      & (df_si['head'] == head)]\n"
            "        traj = traj.sort_values('step')\n"
            "        ax.plot(traj['step'], traj['score'], marker='o',\n"
            "                label=f'L{int(layer)}H{int(head)}')\n"
            "    ax.set_xscale('symlog', linthresh=1)\n"
            "    ax.set_xlim(10, 200000)\n"
            "    ax.set_xlabel('training step (symlog)')\n"
            "    if size == SIZES[0]:\n"
            "        ax.set_ylabel('Δ_h')\n"
            "    ax.set_title(f'Pythia-{size}: top-3 final-step heads')\n"
            "    ax.axhline(TAU_STRICT, color='gray', linestyle='--', alpha=0.4)\n"
            "    ax.axhline(TAU_PERMISSIVE, color='gray', linestyle=':', alpha=0.4)\n"
            "    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)\n"
            "    ax.legend(fontsize=8, loc='upper left')\n"
            "    ax.grid(alpha=0.3)\n"
            "fig.suptitle('Trajectories of strongest final-checkpoint S-inhibition heads', y=1.02)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Component-DLA NM evolution\n"
            "\n"
            "NM identification is re-derived per cell via component-DLA top-4. "
            "If the same heads appear as NMs across checkpoints, the "
            "downstream IOI circuit is stable; if different heads play NM "
            "roles at different stages, the receiver set itself is part of "
            "the emergence story."
        ),
        code(
            "raw = np.load(REPO / 'data' / 'exploration' / 's_inhibition_emergence_per_cell.npz', allow_pickle=False)\n"
            "nm_records = []\n"
            "for size in SIZES:\n"
            "    for step in STEPS:\n"
            "        key = f'nm_heads__{size}_step{step}'\n"
            "        if key not in raw.files:\n"
            "            continue\n"
            "        nms = raw[key]\n"
            "        for rank, (L, H) in enumerate(nms, start=1):\n"
            "            nm_records.append(dict(\n"
            "                size=size, step=step, nm_rank=rank,\n"
            "                nm_layer=int(L), nm_head=int(H),\n"
            "            ))\n"
            "nm_df = pd.DataFrame(nm_records)\n"
            "wide = nm_df.assign(\n"
            "    nm_label=lambda d: 'L' + d['nm_layer'].astype(str) + 'H' + d['nm_head'].astype(str)\n"
            ").pivot(index=['size', 'nm_rank'], columns='step', values='nm_label')\n"
            "wide['size'] = pd.Categorical(wide.index.get_level_values(0), categories=SIZES, ordered=True)\n"
            "print('Component-DLA top-4 NMs per cell:')\n"
            "print(wide.drop(columns=['size']).to_string())"
        ),
        md(
            "## Verdict\n"
            "\n"
            "The 18-cell sweep characterizes S-inhibition emergence across "
            "Pythia sizes under the locked detector. Specific findings depend "
            "on the data; populate this section after re-execution. Key "
            "questions the sweep answers:\n"
            "\n"
            "1. **Does S-inhibition emerge in all three Pythia sizes by "
            "step143000?** Per the anchor result for Pythia-410M (deliverable "
            "ii), at least 2 heads clear τ_strict at the final checkpoint. "
            "For 70m and 160m, count whether ≥1 head clears τ_strict at "
            "step143000.\n"
            "\n"
            "2. **When does S-inhibition first cross τ_permissive?** The "
            "permissive curve answers: at what step does the *first* head "
            "show measurable S-inhibition behaviour? This is the lower bound "
            "of the emergence window.\n"
            "\n"
            "3. **H1-C ordering**: comparing induction count and S-inhibition "
            "count per cell, does induction reach a non-zero count *before* "
            "S-inhibition does? With 6 checkpoints this is a coarse check, "
            "but if the ordering is consistent across all 3 sizes, it's a "
            "useful Phase 1.3 finding to carry into Phase 2.\n"
            "\n"
            "Phase 1.3 deliverable (iii) status: complete pending re-execution. "
            "Phase 2 sweep design will refine the checkpoint grid and add "
            "bootstrap CIs."
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
