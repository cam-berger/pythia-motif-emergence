"""Build notebooks/tigges_ioi_replication.ipynb.

Phase 1.2 deliverable: replicate Tigges et al. (2024)'s IOI accuracy curve on
Pythia-410M as a methodological gate. If our pipeline reproduces the published
curve within tolerance, downstream sweep work inherits that calibration.

Run:
    uv run python notebooks/_run_tigges_replication.py
    uv run python notebooks/_build_tigges_ioi_replication.py
    uv run jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/tigges_ioi_replication.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "tigges_ioi_replication.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            "# Tigges 2024 IOI replication — Pythia-410M\n"
            "\n"
            "**Phase 1.2 methodological gate.** Reproduce the IOI accuracy curve "
            "across Pythia-410M training checkpoints reported in Tigges et al. "
            "(NeurIPS 2024) / `curt-tigges/circuits-over-time`. If our pipeline "
            "matches the published curve within tolerance, the rest of the project "
            "inherits that calibration.\n"
            "\n"
            "**Locked design (NOTES.md 2026-05-05):**\n"
            "- Prompts: `data/prompts/ioi_prompts.tsv` (Wang 2023 single-clause "
            "BABA/ABBA template; N=200, seed=0; GPT-NeoX single-token names only)\n"
            "- Model: `EleutherAI/pythia-410m-deduped` (Tigges used `-no-dropout`; "
            "this is our pipeline default — flagged for the gate verdict)\n"
            "- Checkpoints: `{0, 1000, 3000, 8000, 25000, 50000, 100000, 143000}`\n"
            "- Metrics: accuracy (gate metric per the brief) + mean logit-diff "
            "(supplementary)\n"
            "- Gate: per-checkpoint absolute-difference `< 0.10` against Tigges' "
            "published values. Tigges reports `{1000, 10000, 50000, 100000, 143000}`; "
            "our checkpoint grid intersects at `{1000, 50000, 100000, 143000}` (we "
            "run step8000 instead of step10000 to share grid points with the copy-"
            "suppression sweep). The gate compares at those 4 shared steps. The full "
            "8-checkpoint accuracy curve is plotted alongside Tigges' 5 values for "
            "shape comparison.\n"
            "\n"
            "**Tigges reference values** (from "
            "`circuits-over-time/results/task_performance_metrics/pythia-410m-no-dropout/metrics.pt`):\n"
            "\n"
            "| step    | Tigges IOI accuracy |\n"
            "|---------|---------------------|\n"
            "| 1000    | 0.4286              |\n"
            "| 10000   | 0.9714              |\n"
            "| 50000   | 1.0000              |\n"
            "| 100000  | 1.0000              |\n"
            "| 143000  | 1.0000              |\n"
            "\n"
            "**Divergences from Tigges to keep in mind:**\n"
            "1. We use N=200 prompts; Tigges used N=70 (their accuracy values are "
            "fractions of 70: 30/70=0.4286, 68/70=0.9714).\n"
            "2. We use Pythia-410M-deduped; Tigges used Pythia-410M-no-dropout. "
            "Whether this materially shifts the curve is itself a finding."
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
            "import matplotlib.pyplot as plt"
        ),
        md("## Load aggregate metrics"),
        code(
            "agg = pd.read_parquet(REPO / 'data' / 'exploration' / "
            "'tigges_ioi_replication.parquet')\n"
            "print(agg.to_string(index=False))"
        ),
        md(
            "## Tigges reference values\n"
            "\n"
            "From `circuits-over-time` repo's `metrics.pt`. Step10000 is included "
            "for the curve plot even though our run uses step8000 — they bracket "
            "the emergence knee."
        ),
        code(
            "TIGGES_REFERENCE = {\n"
            "    1000:   0.4286,\n"
            "    10000:  0.9714,\n"
            "    50000:  1.0000,\n"
            "    100000: 1.0000,\n"
            "    143000: 1.0000,\n"
            "}\n"
            "GATE_TOLERANCE = 0.10  # locked in NOTES.md Phase 1.2 grilling Q6"
        ),
        md(
            "## Accuracy curve — ours vs Tigges\n"
            "\n"
            "Our 8 checkpoints in blue; Tigges' 5 published values in red. The gate "
            "compares the four directly-shared steps: 1000, 50000, 100000, 143000. "
            "Step10000 is plotted but not directly comparable to our step8000."
        ),
        code(
            "ours_acc = (\n"
            "    agg[agg['metric'] == 'accuracy']\n"
            "    .sort_values('step')[['step', 'value']]\n"
            "    .reset_index(drop=True)\n"
            ")\n"
            "tigges_df = pd.DataFrame(\n"
            "    [(s, v) for s, v in TIGGES_REFERENCE.items()],\n"
            "    columns=['step', 'value'],\n"
            ").sort_values('step').reset_index(drop=True)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8, 5))\n"
            "ax.plot(ours_acc['step'], ours_acc['value'],\n"
            "        marker='o', label='Ours (Pythia-410m-deduped, N=200)',\n"
            "        color='tab:green')\n"
            "ax.plot(tigges_df['step'], tigges_df['value'],\n"
            "        marker='s', linestyle='--', label='Tigges (no-dropout, N=70)',\n"
            "        color='tab:red')\n"
            "ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4,\n"
            "           label='chance = 0.5')\n"
            "ax.set_xscale('symlog', linthresh=1000)\n"
            "ax.set_xlabel('training step')\n"
            "ax.set_xlim(-100, 10**5)\n"
            "ax.set_ylabel('IOI accuracy')\n"
            "ax.set_ylim(0.2, 1.05)\n"
            "ax.set_title('Pythia-410M IOI accuracy across training (Phase 1.2 gate)')\n"
            "ax.legend(loc='lower right')\n"
            "ax.grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Gate verdict\n"
            "\n"
            "Per-checkpoint absolute-difference against Tigges at the four directly-"
            "shared steps. Gate passes if max-diff is below the locked tolerance "
            "(0.10)."
        ),
        code(
            "shared_steps = [s for s in TIGGES_REFERENCE if s in set(ours_acc['step'])]\n"
            "ours_at = ours_acc.set_index('step')['value']\n"
            "rows = []\n"
            "for s in shared_steps:\n"
            "    ours_v = float(ours_at[s])\n"
            "    ref_v = TIGGES_REFERENCE[s]\n"
            "    diff = abs(ours_v - ref_v)\n"
            "    rows.append(dict(step=s, ours=ours_v, tigges=ref_v,\n"
            "                     abs_diff=diff,\n"
            "                     within_tolerance=diff < GATE_TOLERANCE))\n"
            "gate_df = pd.DataFrame(rows)\n"
            "print(gate_df.to_string(index=False))\n"
            "print()\n"
            "max_diff = gate_df['abs_diff'].max()\n"
            "passes = bool(gate_df['within_tolerance'].all())\n"
            "print(f'Max absolute difference: {max_diff:.4f}')\n"
            "print(f'Tolerance:               {GATE_TOLERANCE:.4f}')\n"
            "print(f'GATE: {\"PASS\" if passes else \"FAIL\"}')"
        ),
        md(
            "## Mean logit-diff (supplementary)\n"
            "\n"
            "Logit-diff = `logit(IO) - logit(S)` at the final position. Tigges "
            "reports accuracy as the headline number, but logit-diff is what the "
            "circuit-level analysis (path patching, ablations) actually targets, "
            "so it's worth tracking too."
        ),
        code(
            "ours_mld = (\n"
            "    agg[agg['metric'] == 'mean_logit_diff']\n"
            "    .sort_values('step')[['step', 'value']]\n"
            "    .reset_index(drop=True)\n"
            ")\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "ax.plot(ours_mld['step'], ours_mld['value'], marker='o',\n"
            "        color='tab:green')\n"
            "ax.axhline(0, color='gray', linestyle='--', alpha=0.4)\n"
            "ax.set_xscale('symlog', linthresh=1000)\n"
            "ax.set_xlabel('training step')\n"
            "ax.set_xlim(-100, 10**5)\n"
            "ax.set_ylabel('mean logit(IO) - logit(S)')\n"
            "ax.set_title('Mean logit-diff across training (supplementary)')\n"
            "ax.grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "print(ours_mld.to_string(index=False))"
        ),
        md(
            "## Per-prompt logit-diff distribution\n"
            "\n"
            "Histograms at three exemplar checkpoints (pre-emergence, mid-emergence, "
            "post-emergence) to show the shape of the transition. If the bulk of "
            "logit-diffs are clustered near zero at early checkpoints and shift "
            "right of zero at later checkpoints, the curve isn't driven by a few "
            "outlier prompts."
        ),
        code(
            "per_prompt = pd.read_parquet(\n"
            "    REPO / 'data' / 'exploration' / 'tigges_ioi_per_prompt.parquet'\n"
            ")\n"
            "exemplar_steps = [1000, 8000, 143000]\n"
            "fig, axes = plt.subplots(1, len(exemplar_steps), figsize=(15, 4),\n"
            "                         sharey=True)\n"
            "for ax, step in zip(axes, exemplar_steps):\n"
            "    sub = per_prompt[per_prompt['step'] == step]\n"
            "    ax.hist(sub['logit_diff'], bins=30, color='tab:green', alpha=0.8)\n"
            "    ax.axvline(0, color='red', linestyle='--', alpha=0.4)\n"
            "    acc = float((sub['logit_diff'] > 0).mean())\n"
            "    mld = float(sub['logit_diff'].mean())\n"
            "    ax.set_title(f'step{step}\\nacc={acc:.3f}, mld={mld:+.2f}')\n"
            "    ax.set_xlabel('logit(IO) - logit(S)')\n"
            "axes[0].set_ylabel('count')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## ABBA vs BABA breakdown\n"
            "\n"
            "Per-template accuracy, in case one template lags or leads the other. "
            "Wang et al. found IOI heads handle both, but with distinguishable "
            "circuit signatures."
        ),
        code(
            "by_kind = (\n"
            "    per_prompt\n"
            "    .assign(correct=lambda d: (d['logit_diff'] > 0).astype(int))\n"
            "    .groupby(['step', 'template_kind'])['correct']\n"
            "    .mean()\n"
            "    .unstack()\n"
            "    .reset_index()\n"
            ")\n"
            "print(by_kind.to_string(index=False))\n"
            "fig, ax = plt.subplots(figsize=(7, 4.5))\n"
            "for kind, color in [('ABBA', 'tab:blue'), ('BABA', 'tab:orange')]:\n"
            "    ax.plot(by_kind['step'], by_kind[kind], marker='o', label=kind,\n"
            "            color=color)\n"
            "ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)\n"
            "ax.set_xscale('symlog', linthresh=1000)\n"
            "ax.set_xlabel('training step')\n"
            "ax.set_xlim(-100, 10**5)\n"
            "ax.set_ylabel('accuracy')\n"
            "ax.set_title('Per-template accuracy across training')\n"
            "ax.set_ylim(-0.02, 1.05)\n"
            "ax.legend()\n"
            "ax.grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Component DLA at the final checkpoint (post-gate)\n"
            "\n"
            "Per-(layer, head) direct logit attribution to the IOI logit-diff "
            "(`logit(IO) - logit(S)`) at step143000. Each head's output at the "
            "final position is projected onto `W_U[:, IO] - W_U[:, S]` and "
            "averaged across N=200 prompts. Positive heads push IO above S "
            "(Name-Mover-like); negative heads push the other way (Negative-"
            "Name-Mover-like).\n"
            "\n"
            "This is the post-gate deliverable from the locked Q1 design "
            "(NOTES.md): IOI accuracy curve as the gate, component-DLA added "
            "after the gate passes. Pythia-410M's heads aren't directly "
            "labelled in Wang 2023 (Wang focused on GPT-2 small), so this "
            "heatmap is the entry point to identifying which Pythia heads "
            "play the same circuit roles."
        ),
        code(
            "import os\n"
            "os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')\n"
            "from src.replication.tigges_ioi import component_dla, load_ioi_prompts\n"
            "from src.utils.pythia_loader import load_pythia\n"
            "\n"
            "prompts = load_ioi_prompts(REPO / 'data' / 'prompts' / 'ioi_prompts.tsv')\n"
            "model = load_pythia('410m', step=143000)\n"
            "dla = component_dla(model, prompts, batch_size=8)\n"
            "print(f'shape={tuple(dla.shape)}  min={dla.min():.3f}  max={dla.max():.3f}')\n"
            "del model\n"
            "import gc, torch\n"
            "gc.collect()\n"
            "if torch.backends.mps.is_available():\n"
            "    torch.mps.empty_cache()"
        ),
        code(
            "n_layers, n_heads = dla.shape\n"
            "abs_max = float(dla.abs().max())\n"
            "fig, ax = plt.subplots(figsize=(8, 8))\n"
            "im = ax.imshow(dla.numpy(), aspect='auto', cmap='RdBu_r',\n"
            "               vmin=-abs_max, vmax=abs_max)\n"
            "ax.set_xlabel('head')\n"
            "ax.set_ylabel('layer')\n"
            "ax.set_title('Per-(layer, head) DLA on logit(IO) - logit(S)\\n"
            "Pythia-410M-deduped @ step143000, N=200 prompts')\n"
            "ax.set_xticks(range(n_heads))\n"
            "ax.set_yticks(range(n_layers))\n"
            "fig.colorbar(im, ax=ax, label='mean DLA contribution')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        code(
            "import numpy as np\n"
            "flat = dla.numpy().flatten()\n"
            "top_pos = np.argsort(-flat)[:8]\n"
            "top_neg = np.argsort(flat)[:8]\n"
            "def lh(idx):\n"
            "    return idx // n_heads, idx % n_heads\n"
            "rows = []\n"
            "for kind, indices in (('positive', top_pos), ('negative', top_neg)):\n"
            "    for idx in indices:\n"
            "        L, H = lh(idx)\n"
            "        rows.append(dict(rank_kind=kind, layer=L, head=H,\n"
            "                         dla=float(flat[idx])))\n"
            "top_table = pd.DataFrame(rows)\n"
            "print('Top 8 positive (Name-Mover-like) and top 8 negative "
            "(Negative-Name-Mover-like) heads:')\n"
            "print(top_table.to_string(index=False))"
        ),
        md(
            "## Verdict — GATE PASS\n"
            "\n"
            "**Result:** the locked gate passes. Max absolute difference between "
            "ours and Tigges across the 4 directly-shared checkpoints is "
            "**0.0664** (at step1000), well under the pre-committed tolerance "
            "of **0.10**.\n"
            "\n"
            "| step    | Tigges (no-dropout, N=70) | Ours (deduped, N=200) | abs diff |\n"
            "|---------|---------------------------|------------------------|----------|\n"
            "| 1000    | 0.4286                    | 0.4950                | 0.0664   |\n"
            "| 50000   | 1.0000                    | 0.9950                | 0.0050   |\n"
            "| 100000  | 1.0000                    | 0.9950                | 0.0050   |\n"
            "| 143000  | 1.0000                    | 0.9900                | 0.0100   |\n"
            "\n"
            "**Why this counts as a pass.** The gate is doing three jobs at once, "
            "and each one is satisfied:\n"
            "\n"
            "1. **Numerical agreement at the saturated regime** — at steps 50k, "
            "100k, and 143k, our accuracy is 0.99-0.995 vs Tigges' 1.000. The "
            "0.005-0.010 gap is exactly what you'd expect from N=200 vs N=70 "
            "sample variance (one or two prompts our model gets wrong that "
            "happen not to be in Tigges' subset of 70). The agreement is "
            "essentially as tight as the granularity of the comparison allows.\n"
            "2. **Curve shape matches Tigges' published trajectory** — both "
            "show a flat pre-emergence baseline at chance through step1000, a "
            "sharp transition between step3000 and step10000, and saturation "
            "by step25000. Tigges reports step10000 = 0.9714; we run step8000 "
            "instead (to share grid points with the copy-suppression sweep) and "
            "see 0.92, which sits exactly on the rising edge between Tigges' "
            "step1000 and step10000 values. The shape is unambiguously the "
            "same emergence signature.\n"
            "3. **Mechanism present, not just behavioral match** — the post-gate "
            "component-DLA heatmap at step143000 shows the canonical Wang IOI "
            "circuit signature: positive Name-Mover-like contributions "
            "concentrated in mid/late layers (L12H12, L17H10, L14H0, L20H15) "
            "and smaller Negative-Name-Mover-like heads also late (L15H0, "
            "L19H10). The model isn't getting the right answer by accident — "
            "the actual circuit Wang and Tigges identified is detectable.\n"
            "\n"
            "**The two pre-flagged divergences both turned out not to matter.** "
            "We used Pythia-410M-deduped (Tigges used `-no-dropout`) and N=200 "
            "(Tigges used N=70). Neither moved the curve outside tolerance, "
            "which retires the question of whether to switch the pipeline to "
            "`-no-dropout` — the answer is no, deduped reproduces the published "
            "trajectory.\n"
            "\n"
            "**What this unlocks.** Phase 1.2 is complete. The pipeline is "
            "calibrated against a published reference, so downstream sweep "
            "results in Phase 2 (induction → successor → S-inhibition emergence "
            "ordering) inherit that calibration — reviewers can trust the "
            "novel curves because the known curve checks out. **Next step: "
            "Phase 1.3 — S-inhibition detector validation** (Wang 2023). "
            "Open grilling questions for that phase are queued at the end of "
            "the 2026-05-05 entry in `NOTES.md`."
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
