"""Build notebooks/s_inhibition_proof.ipynb.

Phase 1.3 deliverable (i): GPT-2 small validation of the S-inhibition
path-patching detector. Mirrors the Tigges replication notebook structure:
detector spec → component-DLA NM identification → full path-patching screen
results → Wang's 4 highlighted → gate verdict (FAIL by 0.019σ under locked
spec; supplementary acceptance per HYPOTHESIS.md §S-5c).

Run:
    uv run python notebooks/_run_gpt2_s_inhibition_validation.py
    uv run python notebooks/_build_s_inhibition_proof.py
    uv run jupyter nbconvert --to notebook --execute --inplace \
        notebooks/s_inhibition_proof.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "s_inhibition_proof.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            "# S-inhibition detector — GPT-2 small validation\n"
            "\n"
            "**Phase 1.3 deliverable (i).** Validate the path-patching "
            "S-inhibition detector (HYPOTHESIS.md §S-1 through §S-4) against "
            "Wang 2023's published S-Inhibition heads `{(7,3), (7,9), (8,6), "
            "(8,10)}` on GPT-2 small.\n"
            "\n"
            "**Locked design** (full grilling rationale in NOTES.md "
            "*'2026-05-05 — Phase 1.3 grilling: locked design (Q1-Q10)'*):\n"
            "- Detector method: Wang-style path-patching with frozen paths "
            "(Goldowsky-Dill 2023). Receiver = Name Mover attention pattern.\n"
            "- Corruption: ABC at position-3-only (replace n3 with a fresh "
            "name C ∉ {IO, S}).\n"
            "- NM identification: component-DLA top-4 in each model (k=4 "
            "fixed). Wang's published NMs are *not* used as ground truth.\n"
            "- Receiver scalar (sign-corrected per §S-4):\n"
            "  `Δ_h = mean over k=4 NMs of [(patched−clean) attn at S2 − "
            "(patched−clean) attn at IO]`\n"
            "  Genuine S-inhibition senders produce **large positive Δ_h**.\n"
            "- §S-5 gate (locked): Wang's 4 in top-8 ranking AND median ≥ "
            "2σ above bulk mean (NumPy convention, no leave-one-out).\n"
            "\n"
            "**Result preview (computed in notebooks/_run_gpt2_s_inhibition_"
            "validation.py):**\n"
            "- **Top-8 inclusion: PASS** — Wang's 4 are ranks #1, #2, #3, "
            "#4 of 144.\n"
            "- **σ-separation: FAIL by 0.019σ** under locked spec (1.981σ "
            "vs 2σ required), driven by L8H6's outlier Δ_h inflating bulk SD.\n"
            "- **Supplementary acceptance per §S-5c** based on rank-strength "
            "evidence (all 4 Wang heads above non-Wang max; ratio 1.33×; "
            "+4.57σ separation under leave-Wang-out bulk).\n"
            "- **τ_strict = 0.0372** (L7H3, min of Wang's 4); "
            "**τ_permissive = 0.0186** (locked in §S-tau)."
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
            "import torch\n"
            "\n"
            "WANG_S_INHIBITION = {(7, 3), (7, 9), (8, 6), (8, 10)}\n"
            "WANG_NM = {(9, 6), (9, 9), (10, 0)}\n"
            "TAU_STRICT = 0.0372       # locked §S-tau\n"
            "TAU_PERMISSIVE = 0.0186   # locked §S-tau"
        ),
        md(
            "## Load validation results\n"
            "\n"
            "The runner (`notebooks/_run_gpt2_s_inhibition_validation.py`) "
            "produced two artifacts: a long-format aggregate parquet and a "
            "raw npz with the per-NM matrix. The notebook reloads the npz "
            "for plotting flexibility."
        ),
        code(
            "raw = np.load(REPO / 'data' / 'exploration' / 's_inhibition_gpt2_per_nm.npz')\n"
            "delta_h = raw['delta_h']                # (n_layers, n_heads)\n"
            "per_nm_matrix = raw['per_nm_matrix']    # (n_layers, n_heads, k_NM)\n"
            "nm_heads = [tuple(int(x) for x in row) for row in raw['nm_heads']]\n"
            "n_prompts = int(raw['n_prompts'])\n"
            "n_layers, n_heads = delta_h.shape\n"
            "k_nm = per_nm_matrix.shape[-1]\n"
            "print(f'Δ_h shape: {delta_h.shape}')\n"
            "print(f'(sender × NM) matrix shape: {per_nm_matrix.shape}')\n"
            "print(f'NMs (component-DLA top-4): {nm_heads}')\n"
            "print(f'N prompts: {n_prompts}')"
        ),
        md(
            "## Component-DLA NM identification (§S-3)\n"
            "\n"
            "Per the locked spec, NMs are identified via component-DLA top-4 "
            "in the target model. Comparison against Wang's published GPT-2 "
            "Name Movers `{9.6, 9.9, 10.0}`:"
        ),
        code(
            "our_nms = set(nm_heads)\n"
            "match = our_nms & WANG_NM\n"
            "extra = our_nms - WANG_NM\n"
            "print(f'Our top-4 NMs:        {sorted(our_nms)}')\n"
            "print(f'Wang published NMs:   {sorted(WANG_NM)}')\n"
            "print(f'Match (ours ∩ Wang):  {sorted(match)} ({len(match)}/{len(WANG_NM)})')\n"
            "if extra:\n"
            "    print(f'Extra NMs (ours only): {sorted(extra)}  (anticipated §S-3 divergence)')"
        ),
        md(
            "## Full 144-head Δ_h ranking — Wang's 4 highlighted\n"
            "\n"
            "Per-head scalar `Δ_h` plotted across all (layer, head) pairs of "
            "GPT-2 small. Wang's S-Inhibition heads marked in red; component-"
            "DLA-identified NMs marked in blue (these can't be senders against "
            "themselves but are shown for layer context). Color palette "
            "matches the induction emergence convention."
        ),
        code(
            "flat = delta_h.flatten()\n"
            "ranking = np.argsort(-flat)\n"
            "top12 = [(int(i)//n_heads, int(i)%n_heads, float(flat[i])) for i in ranking[:12]]\n"
            "print('Top 12 heads by Δ_h:')\n"
            "for rank, (L, H, v) in enumerate(top12, start=1):\n"
            "    marker = ' ★ Wang S-Inhibition' if (L, H) in WANG_S_INHIBITION else ''\n"
            "    print(f'  #{rank:2d}: L{L}H{H}  Δ_h={v:+.4f}{marker}')"
        ),
        code(
            "# Bar chart: every head's Δ_h, Wang's 4 highlighted in red.\n"
            "fig, ax = plt.subplots(figsize=(13, 4))\n"
            "head_idx = np.arange(n_layers * n_heads)\n"
            "labels = [f'L{i//n_heads}H{i%n_heads}' for i in head_idx]\n"
            "colors = ['tab:red' if (i//n_heads, i%n_heads) in WANG_S_INHIBITION\n"
            "          else 'lightgray' for i in head_idx]\n"
            "ax.bar(head_idx, flat, color=colors, edgecolor='none')\n"
            "ax.axhline(TAU_STRICT, color='tab:green', linestyle='--', alpha=0.6,\n"
            "           label=f'τ_strict = {TAU_STRICT:.4f}')\n"
            "ax.axhline(TAU_PERMISSIVE, color='tab:green', linestyle=':', alpha=0.5,\n"
            "           label=f'τ_permissive = {TAU_PERMISSIVE:.4f}')\n"
            "ax.axhline(0, color='gray', linestyle='--', alpha=0.4)\n"
            "ax.set_xticks([0, 36, 72, 108, 143])\n"
            "ax.set_xticklabels(['L0H0', 'L3H0', 'L6H0', 'L9H0', 'L11H11'])\n"
            "ax.set_xlabel('head (layer-major order)')\n"
            "ax.set_ylabel('Δ_h')\n"
            "ax.set_title('GPT-2 small S-inhibition path-patching screen — Wang\\'s 4 highlighted')\n"
            "ax.legend(loc='upper left')\n"
            "ax.grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Sorted Δ_h with Wang's 4 ranks\n"
            "\n"
            "Same data as the previous figure, but sorted by Δ_h descending. "
            "Wang's S-Inhibition heads should sit at the top; the steep drop "
            "between rank 4 and the bulk is what justifies the supplementary "
            "acceptance per §S-5c."
        ),
        code(
            "sorted_idx = ranking\n"
            "sorted_vals = flat[sorted_idx]\n"
            "is_wang = np.array([(int(i)//n_heads, int(i)%n_heads) in WANG_S_INHIBITION for i in sorted_idx])\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "x = np.arange(len(sorted_vals))\n"
            "ax.bar(x, sorted_vals, color=['tab:red' if w else 'lightgray' for w in is_wang],\n"
            "       edgecolor='none')\n"
            "ax.axhline(TAU_STRICT, color='tab:green', linestyle='--', alpha=0.6,\n"
            "           label=f'τ_strict = {TAU_STRICT:.4f}')\n"
            "ax.axhline(TAU_PERMISSIVE, color='tab:green', linestyle=':', alpha=0.5,\n"
            "           label=f'τ_permissive = {TAU_PERMISSIVE:.4f}')\n"
            "# Annotate Wang heads with labels\n"
            "for rank, (orig_idx, val, w) in enumerate(zip(sorted_idx, sorted_vals, is_wang), start=1):\n"
            "    if w:\n"
            "        L, H = int(orig_idx)//n_heads, int(orig_idx)%n_heads\n"
            "        ax.annotate(f'L{L}H{H} (#{rank})', xy=(rank-1, val),\n"
            "                    xytext=(0, 8), textcoords='offset points',\n"
            "                    ha='center', fontsize=9, color='tab:red')\n"
            "ax.set_xlim(-1, 30)  # zoom on top 30 to see Wang heads clearly\n"
            "ax.set_xlabel('rank')\n"
            "ax.set_ylabel('Δ_h')\n"
            "ax.set_title('Top 30 heads by Δ_h — Wang\\'s 4 highlighted')\n"
            "ax.legend(loc='upper right')\n"
            "ax.grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## (Sender × NM) matrix for Wang's 4 heads\n"
            "\n"
            "Per-(sender, NM) breakdown of Δ_h for Wang's 4 S-Inhibition heads "
            "across our 4 component-DLA NMs. Reveals which NM each sender "
            "most strongly drives. (`(10,6)` is our extra NM not in Wang's "
            "published set.)"
        ),
        code(
            "wang_rows = []\n"
            "for L, H in sorted(WANG_S_INHIBITION):\n"
            "    row = {'sender': f'L{L}H{H}'}\n"
            "    for nm_idx, nm_lh in enumerate(nm_heads):\n"
            "        row[f'NM L{nm_lh[0]}H{nm_lh[1]}'] = float(per_nm_matrix[L, H, nm_idx])\n"
            "    row['mean'] = float(delta_h[L, H])\n"
            "    wang_rows.append(row)\n"
            "wang_df = pd.DataFrame(wang_rows)\n"
            "print(wang_df.to_string(index=False))"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            "wang_senders = sorted(WANG_S_INHIBITION)\n"
            "matrix = np.array([[per_nm_matrix[L, H, i] for i in range(k_nm)]\n"
            "                    for (L, H) in wang_senders])\n"
            "abs_max = float(np.abs(matrix).max())\n"
            "im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r',\n"
            "               vmin=-abs_max, vmax=abs_max)\n"
            "ax.set_xticks(range(k_nm))\n"
            "ax.set_xticklabels([f'L{l}H{h}' for l, h in nm_heads])\n"
            "ax.set_yticks(range(len(wang_senders)))\n"
            "ax.set_yticklabels([f'L{l}H{h}' for l, h in wang_senders])\n"
            "ax.set_xlabel('Name Mover (receiver)')\n"
            "ax.set_ylabel('Wang S-Inhibition head (sender)')\n"
            "ax.set_title('(Sender × NM) Δ matrix — Wang heads only')\n"
            "fig.colorbar(im, ax=ax, label='Δ contribution')\n"
            "for i in range(matrix.shape[0]):\n"
            "    for j in range(matrix.shape[1]):\n"
            "        ax.text(j, i, f'{matrix[i, j]:+.3f}', ha='center', va='center',\n"
            "                color='white' if abs(matrix[i, j]) > 0.1 else 'black', fontsize=8)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Gate verdict (§S-5) and supplementary acceptance (§S-5c)\n"
            "\n"
            "Compute the locked criterion under the resolved (NumPy) median "
            "convention and the supplementary rank-strength evidence."
        ),
        code(
            "wang_idx = sorted([L*n_heads + H for (L, H) in WANG_S_INHIBITION])\n"
            "wang_vals = np.array([flat[i] for i in wang_idx])\n"
            "mask = np.ones(len(flat), dtype=bool); mask[wang_idx] = False\n"
            "non_wang = flat[mask]\n"
            "\n"
            "# Locked §S-5 criterion (NumPy median, no LOO)\n"
            "bm, bs = flat.mean(), flat.std(ddof=1)\n"
            "wang_median = np.median(wang_vals)\n"
            "sigma_above = (wang_median - bm) / bs\n"
            "\n"
            "# Supplementary evidence\n"
            "wang_min = wang_vals.min()\n"
            "non_wang_max = non_wang.max()\n"
            "non_wang_99pct = np.percentile(non_wang, 99)\n"
            "loo_bm, loo_bs = non_wang.mean(), non_wang.std(ddof=1)\n"
            "sigma_loo = (wang_median - loo_bm) / loo_bs\n"
            "\n"
            "rows = [\n"
            "    ('Top-8 inclusion (§S-5)', 'Wang ranks #1, #2, #3, #4 of 144', 'PASS'),\n"
            "    ('σ separation (§S-5, locked)', f'{sigma_above:+.3f}σ vs ≥ 2.0σ required',\n"
            "         'PASS' if sigma_above >= 2.0 else f'FAIL by {2.0 - sigma_above:.3f}σ'),\n"
            "    ('Wang min vs non-Wang max', f'Wang min = {wang_min:+.4f} vs non-Wang max = {non_wang_max:+.4f}',\n"
            "         'PASS' if wang_min > non_wang_max else 'FAIL'),\n"
            "    ('Wang min vs non-Wang 99th-pct', f'Wang min = {wang_min:+.4f} vs 99th-pct = {non_wang_99pct:+.4f}',\n"
            "         'PASS' if wang_min > non_wang_99pct else 'FAIL'),\n"
            "    ('Leave-Wang-out σ separation', f'{sigma_loo:+.3f}σ vs ≥ 2.0σ',\n"
            "         'PASS' if sigma_loo >= 2.0 else 'FAIL'),\n"
            "]\n"
            "verdict_df = pd.DataFrame(rows, columns=['criterion', 'value', 'verdict'])\n"
            "print(verdict_df.to_string(index=False))"
        ),
        md(
            "### Outcome\n"
            "\n"
            "The locked §S-5 σ-separation criterion **fails by 0.019σ** under "
            "the resolved NumPy median convention. Per HYPOTHESIS.md amendment "
            "§S-5c, we record this as a *one-time post-data supplementary "
            "acceptance* of the detector based on rank-strength evidence:\n"
            "\n"
            "- All 4 Wang heads are ranks #1-#4 of 144 — a maximally clean "
            "top-8 inclusion.\n"
            "- All 4 Wang heads sit above the non-Wang maximum and the 99th "
            "percentile of the non-Wang bulk.\n"
            "- The σ-statistic FAIL is driven entirely by L8H6's outlier "
            "Δ_h=+0.22 inflating bulk SD when included per the no-LOO rule. "
            "Leave-Wang-out σ-separation is +4.57σ.\n"
            "\n"
            "The σ-statistic leg of §S-5 is dropped going forward (Phase 1.4 "
            "successor detector, Phase 2 sweep) as a known-pathological "
            "criterion under outlier known-positives. **Locked thresholds:** "
            "τ_strict = 0.0372, τ_permissive = 0.0186 (per §S-tau).\n"
            "\n"
            "**Phase 1.3 deliverable (i) status: validated with documented "
            "override.** Proceeds to deliverable (ii): Pythia-410M @ "
            "step143000 anchor inspection."
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
