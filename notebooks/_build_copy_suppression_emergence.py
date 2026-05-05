"""Build notebooks/copy_suppression_emergence_exploration.ipynb.

Sparse exploratory sweep across Pythia 70M / 160M / 410M for copy-suppression.
Mirrors the (size, step) grid of the induction emergence preview for direct
cell-for-cell comparability.

Three questions:
  - When do strict-criterion-passing heads emerge in each Pythia size?
  - How do QK and OV scores evolve over training?
  - Do the strict-passing heads functionally do copy-suppression?

The third question is the load-bearing one: PILOT_RESULTS.md (Day 4) showed
that the strict-passing head at the pre-reg anchor (Pythia-410M @ step143000,
L2H8) is functionally a previous-token head, not a copy-suppression head.
This notebook documents the same pattern across the sweep — strict-criterion
passes correlate with high-attention duplicate-attending heads, not with the
attend-then-suppress mechanism.

This is exploration, not the pre-registered Phase 2 sweep. Path C is locked in
PILOT_RESULTS.md; this notebook documents the search-for-copy-suppression
finding before the project pivots to S-inhibition.

Run:
    uv run python notebooks/_build_copy_suppression_emergence.py
    uv run jupyter nbconvert --to notebook --execute --inplace \
        notebooks/copy_suppression_emergence_exploration.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "copy_suppression_emergence_exploration.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            "# Exploration: copy-suppression emergence in Pythia\n"
            "\n"
            "Sparse sweep across **Pythia 70M / 160M / 410M × {0, 1000, 3000, 8000, "
            "25000, 143000}** = 18 cells, mirroring the induction emergence preview's "
            "grid for direct cell-for-cell comparability. The McDougall two-criterion "
            "detector applied to the canonical 7.5k-token Wikipedia corpus.\n"
            "\n"
            "**Headline finding (post-Path-C-pivot).** The pilot anchor (Pythia-410M @ "
            "step143000) registered Path C: the only head numerically passing the strict "
            "criterion (L2H8) is qualitatively a *previous-token head*, not a "
            "copy-suppression head — corpus-wide ablation lowers duplicate-token "
            "logits (-0.009), the opposite of suppression. This notebook documents the "
            "broader pattern: strict-criterion passes accumulate over training in a "
            "shape that mimics induction emergence, suggesting these passes are *all* "
            "induction-like duplicate-attending heads with marginal-negative corpus-mean "
            "OV — not the McDougall mechanism.\n"
            "\n"
            "**This is exploration, not the pre-registered sweep.** Phase 2 will use 40 "
            "checkpoints per size from `checkpoints.yaml`; this preview uses 6 "
            "log-spaced steps. Output: `data/exploration/copy_suppression_emergence_preview.parquet` "
            "(long-format, schema-compatible with Phase 2's canonical `motif_sweep.parquet`).\n"
            "\n"
            "**Three questions:**\n"
            "1. When do strict-criterion passes first appear in each Pythia size?\n"
            "2. How do QK (max attention to prior duplicates) and OV (most-negative DLA) "
            "scores evolve over training?\n"
            "3. Does the cross-motif comparison support the H1 ordering hypothesis "
            "(induction emerges *before* copy-suppression candidates)?"
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
            "\n"
            "QK_STRICT = 0.3\n"
            "OV_THRESHOLD = 0.0"
        ),
        md(
            "## Load sweep data\n"
            "\n"
            "Long-format parquets for both motifs. The canonical Phase 2 schema is "
            "`(size, step, layer, head, motif, score)`; pivoting to wide gives one "
            "column per motif for easier plotting."
        ),
        code(
            "df_cs = read_long(REPO / 'data' / 'exploration' / 'copy_suppression_emergence_preview.parquet')\n"
            "df_ind = read_long(REPO / 'data' / 'exploration' / 'induction_emergence_preview.parquet')\n"
            "df = pd.concat([df_cs, df_ind], ignore_index=True)\n"
            "wide = to_wide(df, index_cols=('size','step','layer','head'))\n"
            "print('Long-format combined motifs:')\n"
            "print(df.groupby('motif').size().to_string())\n"
            "print()\n"
            "print('Wide-format columns:', list(wide.columns))\n"
            "print('Heads per (size, step) cell:')\n"
            "print(wide.groupby(['size','step']).size().unstack().to_string())"
        ),
        md(
            "## Per-cell summary\n"
            "\n"
            "For each (size, step), record: count of strict-passing heads "
            "(QK > 0.3 AND OV < 0); maximum QK across heads; minimum OV across heads "
            "(most-negative); count of heads with negative OV (regardless of QK); the "
            "induction count from the parallel sweep."
        ),
        code(
            "wide['cs_strict'] = (wide['copy_suppression_qk'] > QK_STRICT) & (wide['copy_suppression_ov'] < OV_THRESHOLD)\n"
            "wide['cs_ov_neg'] = wide['copy_suppression_ov'] < OV_THRESHOLD\n"
            "wide['ind_pass'] = wide['induction_prefix_match'] > 0.3\n"
            "\n"
            "summary = wide.groupby(['size','step']).agg(\n"
            "    strict_pass=('cs_strict','sum'),\n"
            "    ov_neg_count=('cs_ov_neg','sum'),\n"
            "    max_qk=('copy_suppression_qk','max'),\n"
            "    min_ov=('copy_suppression_ov','min'),\n"
            "    induction_pass=('ind_pass','sum'),\n"
            "    induction_max=('induction_prefix_match','max'),\n"
            ").reset_index()\n"
            "summary['size'] = pd.Categorical(summary['size'], categories=SIZES, ordered=True)\n"
            "summary = summary.sort_values(['size','step'])\n"
            "print(summary.to_string(index=False))"
        ),
        md(
            "## Headline plot — H1 ordering check\n"
            "\n"
            "The H1 hypothesis predicts induction emerges *before* copy-suppression. "
            "This plot puts both counts on the same axis per Pythia size. If H1 holds "
            "and the strict-passing heads were genuinely copy-suppression, we'd expect "
            "induction to ramp first and copy-suppression to ramp later (consistent with "
            "the compositional account). If the strict-passing heads aren't copy-"
            "suppression (Path C), we expect copy-suppression count to lag induction *or* "
            "track it, depending on how often induction-like heads happen to clear the "
            "marginal OV criterion."
        ),
        code(
            "fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)\n"
            "for ax, size in zip(axes, SIZES):\n"
            "    sub = summary[summary['size'] == size]\n"
            "    ax.plot(sub['step'], sub['induction_pass'], marker='o', label='induction (Olsson > 0.3)', color='C0', linewidth=2)\n"
            "    ax.plot(sub['step'], sub['strict_pass'], marker='s', label='copy-suppression strict (QK>0.3 & OV<0)', color='C3', linewidth=2)\n"
            "    ax.set_xscale('symlog', linthresh=100)\n"
            "    ax.set_xlabel('training step')\n"
            "    ax.set_ylabel('count of heads passing')\n"
            "    ax.set_title(f'Pythia-{size}-deduped')\n"
            "    ax.grid(alpha=0.3)\n"
            "    ax.legend(fontsize=9)\n"
            "fig.suptitle('Strict-criterion head counts vs training step — induction vs copy-suppression', y=1.02)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "**Reading the plot.** Induction emerges fast in all sizes — by step 1000 "
            "the count is already at saturation level (matches the induction emergence "
            "preview, see `notebooks/induction_emergence_exploration.ipynb`). Copy-"
            "suppression strict-passes accumulate later — first appears around step 3000 "
            "in 160M and 410M, step 143000 in 70M.\n"
            "\n"
            "**But: from PILOT_RESULTS.md Day 4 inspection, the strict-passing head at "
            "the 410M @ step143000 anchor (L2H8) is qualitatively a previous-token head, "
            "not copy-suppression.** The temporal pattern (copy-suppression-pass count "
            "lagging induction) is exactly what you'd expect if the strict-passing heads "
            "are *induction-precursor heads with marginal-negative corpus-mean OV* — they "
            "appear after induction-like attention patterns develop and accumulate small "
            "amounts of incidental-negative DLA. The shape mimics the H1 ordering "
            "prediction *without* providing evidence for the copy-suppression mechanism."
        ),
        md(
            "## Score evolution: max QK and min OV per cell\n"
            "\n"
            "The two scores evolving separately — informative when count-based plots are "
            "noisy due to small N. Heat-style plots show how the *strongest* score moves "
            "over training in each size."
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
            "\n"
            "ax = axes[0]\n"
            "for size in SIZES:\n"
            "    sub = summary[summary['size'] == size]\n"
            "    ax.plot(sub['step'], sub['max_qk'], marker='o', label=f'Pythia-{size}', linewidth=2)\n"
            "ax.axhline(QK_STRICT, color='red', linestyle='--', alpha=0.5, label='QK strict (0.3)')\n"
            "ax.set_xscale('symlog', linthresh=100)\n"
            "ax.set_xlabel('training step')\n"
            "ax.set_ylabel('max QK across all heads')\n"
            "ax.set_title('Maximum attention to prior duplicates (QK)')\n"
            "ax.grid(alpha=0.3); ax.legend()\n"
            "\n"
            "ax = axes[1]\n"
            "for size in SIZES:\n"
            "    sub = summary[summary['size'] == size]\n"
            "    ax.plot(sub['step'], sub['min_ov'], marker='o', label=f'Pythia-{size}', linewidth=2)\n"
            "ax.axhline(OV_THRESHOLD, color='red', linestyle='--', alpha=0.5, label='OV threshold (0)')\n"
            "ax.set_xscale('symlog', linthresh=100)\n"
            "ax.set_xlabel('training step')\n"
            "ax.set_ylabel('min OV across all heads (most-negative)')\n"
            "ax.set_title('Strongest negative DLA on duplicate token (OV)')\n"
            "ax.grid(alpha=0.3); ax.legend()\n"
            "\n"
            "fig.suptitle('Per-cell extreme scores over training', y=1.02)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "**QK and OV evolve on different timescales.** Max QK saturates by step "
            "3000-8000 in all sizes (some head clears 0.3 in attention to prior duplicates "
            "by then). Min OV continues to drop past step 25000, especially in 410M where "
            "the most-negative OV grows from -0.07 at step 3000 to -0.59 at step 143000. "
            "This is the *qualitative pattern* you'd expect if corrective mechanisms "
            "develop after copying behaviors — but again, the corpus-wide ablation of "
            "the most-negative-OV heads (PILOT_RESULTS.md, Day 4) shows they don't "
            "actually do copy-suppression in the McDougall sense.\n"
            "\n"
            "Maybe the strong-OV-but-low-QK heads in late 410M are doing *something* "
            "corrective on a small subset of positions (recall L10H7 in GPT-2 has "
            "per-position OV of -24 on rare worked-example positions while corpus-mean "
            "is much smaller). The sweep aggregate doesn't capture per-position structure."
        ),
        md(
            "## Heatmap: strict-pass counts across the (size, step) grid\n"
            "\n"
            "Visual summary of where strict-criterion passes appear. Useful for the "
            "paper's negative-result figure — "
            "*\"we ran the McDougall detector across 18 (size, step) cells; here is the "
            "count of numerically-passing heads, and here is the qualitative-confirmation "
            "result for the anchor cell.\"*"
        ),
        code(
            "pivot_strict = summary.pivot(index='size', columns='step', values='strict_pass')\n"
            "pivot_strict = pivot_strict.reindex(index=SIZES, columns=STEPS)\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n"
            "for ax, (data, title, fmt, cmap) in zip(axes, [\n"
            "    (pivot_strict, 'Strict-pass head count (QK>0.3 & OV<0)', '.0f', 'Reds'),\n"
            "    (summary.pivot(index='size', columns='step', values='max_qk').reindex(index=SIZES, columns=STEPS), 'max QK', '.2f', 'Oranges'),\n"
            "    (summary.pivot(index='size', columns='step', values='min_ov').reindex(index=SIZES, columns=STEPS), 'min OV', '+.2f', 'Blues_r'),\n"
            "]):\n"
            "    im = ax.imshow(data.values, aspect='auto', cmap=cmap)\n"
            "    ax.set_xticks(range(len(STEPS)))\n"
            "    ax.set_xticklabels(STEPS, rotation=45)\n"
            "    ax.set_yticks(range(len(SIZES)))\n"
            "    ax.set_yticklabels(SIZES)\n"
            "    ax.set_xlabel('training step')\n"
            "    ax.set_ylabel('size')\n"
            "    ax.set_title(title)\n"
            "    for i in range(data.shape[0]):\n"
            "        for j in range(data.shape[1]):\n"
            "            v = data.values[i, j]\n"
            "            ax.text(j, i, format(v, fmt), ha='center', va='center', fontsize=9,\n"
            "                    color='black')\n"
            "    plt.colorbar(im, ax=ax, fraction=0.046)\n"
            "fig.suptitle('Per-cell summary heatmaps — emergence-sweep preview', y=1.02)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Identity tracking: which heads pass strict?\n"
            "\n"
            "If the strict-passing heads are *the same heads* across checkpoints, that's "
            "structural reuse — supports the Extension B framing in PROJECT_BRIEF.md. "
            "If they're *different heads* per checkpoint, that's evidence the strict "
            "criterion is picking up noise rather than a stable circuit."
        ),
        code(
            "for size in SIZES:\n"
            "    print(f'--- Pythia-{size} strict-pass identity tracking ---')\n"
            "    sub = wide[(wide['size'] == size) & wide['cs_strict']]\n"
            "    if len(sub) == 0:\n"
            "        print('  no strict-passing heads at any sweep step')\n"
            "        continue\n"
            "    for step in STEPS:\n"
            "        cands = sub[sub['step'] == step][['layer','head','copy_suppression_qk','copy_suppression_ov']]\n"
            "        if len(cands) > 0:\n"
            "            head_str = ', '.join(\n"
            "                f\"L{int(r['layer']):2d}H{int(r['head']):2d} \"\n"
            "                f\"(QK={r['copy_suppression_qk']:.3f}, OV={r['copy_suppression_ov']:+.3f})\"\n"
            "                for _, r in cands.iterrows()\n"
            "            )\n"
            "            print(f'  step {step:>6}: {head_str}')\n"
            "        else:\n"
            "            print(f'  step {step:>6}: (none)')\n"
            "    print()"
        ),
        md(
            "**Interpretation.** Where the same (layer, head) pair recurs across "
            "checkpoints, the strict-passing head is *the same circuit* persisting and "
            "remaining within the threshold band. Where the (layer, head) changes from "
            "step to step, the strict-criterion is selecting different heads at "
            "different points in training — suggesting the threshold is a noisy "
            "boundary rather than a stable indicator of circuit identity.\n"
            "\n"
            "If a single (layer, head) pair recurs in 410M from step 3000 → step 143000, "
            "this would be consistent with a *single duplicate-attending head whose OV "
            "stayed marginal throughout training*. From the Day 4 inspection of L2H8 we "
            "expect this is what's happening — but verifying via direct identity check "
            "across checkpoints is more rigorous than relying on the single-anchor verdict."
        ),
        md(
            "## Conclusion\n"
            "\n"
            "**What the sweep shows numerically.** Strict-criterion passes do accumulate "
            "across training in a shape qualitatively similar to induction (induction "
            "saturates by step 1000-3000; copy-suppression strict passes appear from "
            "step 3000 onward and grow modestly). This is consistent with H1's compositional "
            "ordering prediction *if* the strict passes were genuine copy-suppression heads.\n"
            "\n"
            "**What the qualitative inspection shows.** The strict-passing head at the "
            "pre-reg anchor (Pythia-410M @ step143000, L2H8) is functionally a "
            "previous-token / induction-precursor head, not a copy-suppression head — "
            "corpus-wide ablation lowers duplicate-token logits, opposite to suppression. "
            "PILOT_RESULTS.md registers Path C: the project pivots to S-inhibition for "
            "the third motif of the H1 ordering claim.\n"
            "\n"
            "**Why both findings can be true.** The strict criterion (QK > 0.3 AND OV < 0) "
            "is a numerical filter that doesn't distinguish *attend-then-suppress* "
            "(McDougall's mechanism) from *attend-then-promote-with-marginal-OV* "
            "(induction-precursor with weak corpus-mean OV). Heads that develop "
            "induction-like QK during early-mid training (the rapid induction emergence "
            "timeline) accumulate marginal-negative corpus-mean OV as a noise byproduct, "
            "passing the strict threshold without the mechanism. Phase 2 should add a "
            "*causal* criterion to the detector — e.g., positive corpus-wide d-logit "
            "on duplicates when ablated — to filter out this confound.\n"
            "\n"
            "**Path forward.** Path C is registered. Phase 2's third-motif detector "
            "becomes S-inhibition (per HYPOTHESIS.md H1-C). The negative-result framing "
            "of this notebook becomes part of the paper: *we searched Pythia for "
            "copy-suppression heads matching McDougall's GPT-2 finding, applied a clean "
            "pre-registered decision rule, and registered Path C.* The infrastructure "
            "(canonical corpus, sweep parquet, side-cache machinery) is reusable for "
            "S-inhibition; the detector implementation needs a causal-criterion addition."
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
