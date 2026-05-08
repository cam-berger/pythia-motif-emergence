"""Build notebooks/motif_structural_reuse.ipynb.

Phase 3 / Extension B deliverable: cross-motif structural reuse analysis.
Asks whether the same physical (layer, head) appears in the top-K of
multiple motifs across training. Uses the existing Phase 2 sweep parquets
(no new compute) to surface heads with multi-motif activity.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "motif_structural_reuse.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            "# Cross-motif structural reuse — Extension B\n"
            "\n"
            "**Phase 3 deliverable, prof-recommended.** Question: across "
            "training, does the same physical (layer, head) appear in the "
            "top-K of multiple motifs, or are the three motif populations "
            "disjoint?\n"
            "\n"
            "**Why this matters.** The H1-C compositional account predicts "
            "that S-inhibition heads *consume* successor/induction outputs at "
            "later layers — i.e., they are structurally distinct heads. If we "
            "instead find substantial overlap between motif top-K populations "
            "(e.g., a head that scores high on both successor and S-inhibition), "
            "that would suggest *circuit reuse* rather than compositional layering.\n"
            "\n"
            "**Method.** For each (size, step), rank heads by motif score and "
            "take top-K (K=5). Compute pairwise Jaccard similarity across "
            "motif top-K sets. Plot Jaccard over training. Surface any heads "
            "that appear in ≥ 2 motifs' top-K at any step (the candidate "
            "multi-motif heads).\n"
            "\n"
            "**Caveat.** Top-K membership uses score rank, not threshold-pass. "
            "A head can be in successor's top-K with a low absolute score (if "
            "the motif fails to emerge in that cell). The threshold-pass "
            "version is in the per-motif `_full_sweep` notebooks; this one "
            "uses rank-based overlap to track *relative* identity churn.\n"
            "\n"
            "**2.8B exclusion.** The cross-motif Jaccard analysis requires "
            "data for all 3 motifs at every (size, step) cell. Per "
            "§H4-7-supersede (committed 2026-05-08), the 2.8B S-inhibition "
            "sweep was halted at 8/40 cells under the §H4-7 per-cell-cost "
            "escape hatch. 2.8B is therefore EXCLUDED from this notebook's "
            "time-trajectory Jaccard analysis. The 5-size induction + "
            "successor cross-size view is in `h1c_ordering_test.ipynb` "
            "§H4-scaling DEFERRED section (parameter-size-vs-max-count "
            "best-fit plot). The step143000 anchor data for all 3 motifs at "
            "2.8B IS available and is used in `motif_attention_inspection.ipynb`."
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
            "\n"
            "SIZES = ['70m', '160m', '410m', '1b']\n"
            "MOTIFS = ['induction', 'successor', 's_inhibition']\n"
            "MOTIF_LABEL = {'induction': 'induction', 'successor': 'successor', 's_inhibition': 'S-inhibition'}\n"
            "SIZE_COLOR = {'70m': 'tab:blue', '160m': 'tab:orange', '410m': 'tab:green', '1b': 'tab:red'}\n"
            "PAIR_COLOR = {\n"
            "    ('induction', 'successor'): 'tab:purple',\n"
            "    ('induction', 's_inhibition'): 'tab:olive',\n"
            "    ('successor', 's_inhibition'): 'tab:cyan',\n"
            "}\n"
            "K_TOP = 5"
        ),
        md("## Load all sweep parquets (3 Phase 2 sizes + 1 Phase 3 1B)"),
        code(
            "# Phase 2 (3 sizes) and Phase 3 (1B) parquets are concatenated for the\n"
            "# 4-size structural-reuse extension per §H3-scale-8 (#6).\n"
            "df_ind = pd.concat([\n"
            "    read_long(REPO / 'data' / 'exploration' / 'phase2_induction_sweep.parquet'),\n"
            "    read_long(REPO / 'data' / 'exploration' / 'phase3_1b_induction_sweep.parquet'),\n"
            "])\n"
            "df_suc = pd.concat([\n"
            "    read_long(REPO / 'data' / 'exploration' / 'phase2_successor_sweep.parquet'),\n"
            "    read_long(REPO / 'data' / 'exploration' / 'phase3_1b_successor_sweep.parquet'),\n"
            "])\n"
            "df_si = pd.concat([\n"
            "    read_long(REPO / 'data' / 'exploration' / 'phase2_s_inhibition_sweep.parquet'),\n"
            "    read_long(REPO / 'data' / 'exploration' / 'phase3_1b_s_inhibition_sweep.parquet'),\n"
            "])\n"
            "DFS = {'induction': df_ind, 'successor': df_suc, 's_inhibition': df_si}\n"
            "STEPS = sorted(df_ind.step.unique().tolist())\n"
            "print(f'Loaded sweeps: {len(STEPS)} cells per size, {len(SIZES)} sizes, {len(MOTIFS)} motifs.')"
        ),
        md(
            "## Top-K head sets per (size, step, motif)\n"
            "\n"
            "For each cell, take the K=5 highest-scoring (layer, head) pairs "
            "per motif. The set of these pairs is the motif's \"top-K identity\" "
            "at that cell. We then ask: do these sets overlap across motifs?"
        ),
        code(
            "topk = {}  # {(size, step, motif): set of (layer, head) tuples}\n"
            "for motif, df in DFS.items():\n"
            "    for size in SIZES:\n"
            "        for step in STEPS:\n"
            "            sub = df[(df['size'] == size) & (df['step'] == step)]\n"
            "            if sub.empty:\n"
            "                topk[(size, step, motif)] = set()\n"
            "                continue\n"
            "            top = sub.nlargest(K_TOP, 'score')[['layer', 'head']]\n"
            "            topk[(size, step, motif)] = set(map(tuple, top.values.tolist()))\n"
            "print(f'Built top-{K_TOP} sets for {len(topk):,} (size, step, motif) cells.')"
        ),
        md(
            "## Pairwise Jaccard overlap across motifs over training\n"
            "\n"
            "Jaccard(A, B) = |A ∩ B| / |A ∪ B|. For top-K sets of the same "
            "size K, this ranges 0 (no overlap) to 1 (identical heads). "
            "Three pair-curves per panel: (induction, successor), (induction, "
            "S-inhibition), (successor, S-inhibition)."
        ),
        code(
            "PAIRS = [\n"
            "    ('induction', 'successor'),\n"
            "    ('induction', 's_inhibition'),\n"
            "    ('successor', 's_inhibition'),\n"
            "]\n"
            "rows = []\n"
            "for size in SIZES:\n"
            "    for step in STEPS:\n"
            "        for ma, mb in PAIRS:\n"
            "            sa = topk[(size, step, ma)]\n"
            "            sb = topk[(size, step, mb)]\n"
            "            union = sa | sb\n"
            "            jac = (len(sa & sb) / len(union)) if union else 0.0\n"
            "            rows.append(dict(\n"
            "                size=size, step=step,\n"
            "                pair=f'{ma}↔{mb}',\n"
            "                jaccard=jac,\n"
            "                intersection_size=len(sa & sb),\n"
            "            ))\n"
            "jac_df = pd.DataFrame(rows)"
        ),
        code(
            "fig, axes = plt.subplots(1, len(SIZES), figsize=(6 * len(SIZES), 5), sharey=True)\n"
            "for ax, size in zip(axes, SIZES):\n"
            "    sub = jac_df[jac_df['size'] == size]\n"
            "    for (ma, mb) in PAIRS:\n"
            "        line = sub[sub['pair'] == f'{ma}↔{mb}'].sort_values('step')\n"
            "        ax.plot(line['step'], line['jaccard'],\n"
            "                marker='o', alpha=0.7, markersize=4,\n"
            "                color=PAIR_COLOR[(ma, mb)],\n"
            "                label=f'{MOTIF_LABEL[ma]} ↔ {MOTIF_LABEL[mb]}')\n"
            "    ax.set_xscale('symlog', linthresh=1)\n"
            "    ax.set_xlim(0.5, 200000)\n"
            "    ax.set_ylim(-0.05, 1.05)\n"
            "    ax.set_xlabel('training step (symlog)')\n"
            "    if size == SIZES[0]:\n"
            "        ax.set_ylabel(f'Jaccard similarity of top-{K_TOP} head sets')\n"
            "    ax.set_title(f'Pythia-{size}')\n"
            "    ax.grid(alpha=0.3)\n"
            "    ax.legend(fontsize=8, loc='upper right')\n"
            "fig.suptitle(f'Cross-motif top-{K_TOP} head overlap across training', y=1.02)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "**Reading the plot.** A Jaccard ≈ 0 line indicates the two motifs "
            "are implemented by structurally disjoint head populations — the "
            "compositional account's prediction. A Jaccard ≈ 1/K (with K=5, "
            "that's 0.2) indicates ~ 1 head consistently shared. Higher Jaccard "
            "indicates substantial circuit reuse.\n"
            "\n"
            "Interpret carefully near the early-training pre-emergence regime: "
            "before motif emergence, top-K is whatever heads happen to score "
            "highest on noise. Post-emergence (after each motif's μ), the "
            "overlap reading is meaningful."
        ),
        md(
            "## Multi-motif heads — single-head appearances in top-K of ≥ 2 motifs\n"
            "\n"
            "List every (size, layer, head) that appears in the top-K of at "
            "least 2 motifs at the final checkpoint (step143000). These are "
            "the candidate multi-purpose heads."
        ),
        code(
            "rows = []\n"
            "for size in SIZES:\n"
            "    head_motif_count = {}  # (layer, head) -> set of motifs\n"
            "    for motif in MOTIFS:\n"
            "        for lh in topk[(size, 143000, motif)]:\n"
            "            head_motif_count.setdefault(lh, set()).add(motif)\n"
            "    multi = {lh: ms for lh, ms in head_motif_count.items() if len(ms) >= 2}\n"
            "    if not multi:\n"
            "        rows.append(dict(size=size, layer=None, head=None,\n"
            "                         motifs='(none)',\n"
            "                         note='no multi-motif heads at step143000'))\n"
            "        continue\n"
            "    for (layer, head), motifs in sorted(multi.items()):\n"
            "        rows.append(dict(\n"
            "            size=size, layer=int(layer), head=int(head),\n"
            "            motifs=' + '.join(sorted(motifs)),\n"
            "            note='',\n"
            "        ))\n"
            "multi_df = pd.DataFrame(rows)\n"
            "print('Multi-motif heads at step143000 (top-K of ≥ 2 motifs simultaneously):')\n"
            "print(multi_df.to_string(index=False))"
        ),
        md(
            "## Persistence: multi-motif heads across training\n"
            "\n"
            "If a head fires in multiple motifs at the final checkpoint but only "
            "in one motif during early training, the multi-motif behavior is a "
            "developmental phenomenon, not a stable property. Conversely, if a "
            "head fires in multiple motifs across most checkpoints, the multi-"
            "motif role is structural.\n"
            "\n"
            "Per (size, layer, head) seen in multi_df above, plot the score "
            "trajectory across training in each motif it participates in."
        ),
        code(
            "if multi_df['layer'].notna().any():\n"
            "    rows_for_plot = multi_df[multi_df['layer'].notna()].copy()\n"
            "    n_panels = len(rows_for_plot)\n"
            "    if n_panels > 0:\n"
            "        fig, axes = plt.subplots(\n"
            "            1, n_panels,\n"
            "            figsize=(5 * n_panels, 4),\n"
            "            squeeze=False,\n"
            "        )\n"
            "        for col, (_, row) in enumerate(rows_for_plot.iterrows()):\n"
            "            ax = axes[0, col]\n"
            "            size, layer, head, motifs_str = row['size'], int(row['layer']), int(row['head']), row['motifs']\n"
            "            for motif in MOTIFS:\n"
            "                df = DFS[motif]\n"
            "                trace = df[(df['size'] == size) & (df['layer'] == layer) & (df['head'] == head)].sort_values('step')\n"
            "                if trace.empty:\n"
            "                    continue\n"
            "                ax.plot(trace['step'], trace['score'], marker='o', markersize=3,\n"
            "                        label=MOTIF_LABEL[motif], alpha=0.8)\n"
            "            ax.set_xscale('symlog', linthresh=1)\n"
            "            ax.set_xlim(0.5, 200000)\n"
            "            ax.set_xlabel('training step (symlog)')\n"
            "            ax.set_ylabel('motif score')\n"
            "            ax.set_title(f'{size} L{layer}H{head}\\n{motifs_str}')\n"
            "            ax.grid(alpha=0.3)\n"
            "            ax.legend(fontsize=8)\n"
            "        plt.tight_layout()\n"
            "        plt.show()\n"
            "    else:\n"
            "        print('No multi-motif heads to plot.')\n"
            "else:\n"
            "    print('No multi-motif heads identified at step143000 in any size.')"
        ),
        md(
            "## Verdict — what does structural reuse tell us about H1-C?\n"
            "\n"
            "Three readings, depending on what the Jaccard plot and multi-motif table show:\n"
            "\n"
            "**Reading A — disjoint populations (Jaccard ≈ 0 post-emergence; multi_df empty).** Each motif lives in its own head population. This is consistent with the compositional account in the registered hypothesis (different motifs implement different functions; corrective heads at later layers consume copying-head outputs at earlier layers). The depth-vs-temporal asymmetry observed in `h1c_ordering_test.ipynb` sub-deliverable 4b becomes more puzzling — disjoint populations *and* depth-reversal points away from the simple compositional reading.\n"
            "\n"
            "**Reading B — sustained overlap (Jaccard > 0.2 post-emergence; multi_df populated).** Some heads persistently fire in multiple motifs. This argues *against* the compositional reading and *for* a circuit-reuse account where individual heads serve multiple functions. The H1-C temporal ordering would then be a property of when heads' multiple roles pass detection thresholds, not a property of distinct circuit components emerging in sequence.\n"
            "\n"
            "**Reading C — late-training drift to overlap.** Jaccard rises during training. This suggests that as training progresses, heads broaden their role from specialist to multi-purpose. The temporal ordering hypothesis would still be coherent (heads first specialize, then broaden), but the late-checkpoint multi-motif behavior would caution against treating step143000 as a clean snapshot of a compositional circuit.\n"
            "\n"
            "**For the BlackboxNLP writeup.** Whichever reading the data supports, it sharpens the H1-C interpretation. Reading A reinforces the registered architecture story but deepens the depth-asymmetry puzzle. Reading B reframes the result more aggressively. Reading C suggests caution about static late-checkpoint inspection."
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
