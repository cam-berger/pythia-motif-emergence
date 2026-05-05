"""Build notebooks/copy_suppression_pythia_proof.ipynb.

Pythia copy-suppression proof notebook — analog of induction_heads_proof.ipynb
but for the copy-suppression search on Pythia-410M-deduped @ step143000 (the
pre-registered pilot anchor).

Structure (per Q6 of Day 3 design grilling):
  Part A — detection: load anchor parquet, scatter plot, top-K rankings, branching
    select_proof_target() -> (TOP_HEADS, CONTROL, frame).
  Part B — proof: data-driven worked example from per-position OV cache,
    bird's-eye and focused attention, single-position + corpus-wide ablation
    (Q8 matrix: delta-logit x NLL x single x corpus).
  Part C — contrast with GPT-2 L10H7: side-by-side ablation showing L2H8's
    functional effect is opposite to a real copy-suppression head's, locking
    Path C as the registered pilot outcome.

Pre-conditions:
  - data/pilot/copy_suppression_pythia_410m_step143000.parquet exists (run
    notebooks/_run_pythia_anchor.py first).
  - data/pilot/copy_suppression_pythia_410m_step143000_per_position.npz exists.
  - data/corpora/copy_suppression_corpus.txt exists.

Run:
    uv run python notebooks/_build_copy_suppression_pythia_proof.py
    uv run jupyter nbconvert --to notebook --execute --inplace \
        notebooks/copy_suppression_pythia_proof.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "copy_suppression_pythia_proof.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            "# Copy-suppression search in Pythia-410M-deduped @ step143000 — Path C registered\n"
            "\n"
            "**Pre-registered pilot anchor.** Per `HYPOTHESIS.md` §\"Pilot decision rule\", "
            "we apply the McDougall two-criterion detector (QK > 0.3 AND OV < 0) to all "
            "384 heads of Pythia-410M-deduped at the final checkpoint, on the canonical "
            "Wikipedia-derived corpus. Decision rule:\n"
            "\n"
            "1. ≥3 heads pass + qualitatively confirmed → Path A.\n"
            "2. 1-2 heads pass + qualitatively confirmed → Path A with caveat.\n"
            "3. 0 heads pass, OR numerically-passing heads fail qualitative inspection → Path C.\n"
            "4. Tie / ambiguous → Path C.\n"
            "\n"
            "**Result (locked in `PILOT_RESULTS.md`).** 1 head numerically passes strict (L2H8: "
            "QK=0.328, OV=-0.023) — the *Weak positive* numerical state. Day 4 qualitative "
            "inspection: L2H8's attention pattern is textbook duplicate-attending (0.91 "
            "attention back to prior occurrence at the rank-1 worked-example position), but "
            "**corpus-wide ablation of L2H8 lowers duplicate-token logits by 0.009 — the "
            "opposite direction of suppression**. L2H8 is a previous-token / induction-precursor "
            "head, not a copy-suppression head. By decision-rule path 3, the project pivots to "
            "Path C: **the third motif of the H1 ordering claim becomes S-inhibition, not "
            "copy-suppression** (per `HYPOTHESIS.md` §\"Pivot hypothesis (H1-C)\").\n"
            "\n"
            "This notebook documents that finding end-to-end:\n"
            "\n"
            "**Part A — detection.** Load anchor parquet; visualize the (QK, OV) plane for "
            "all 384 heads; identify L2H8 as the only strict-criterion-passing head; rank "
            "supplementary candidates by most-negative OV among heads with QK ≥ 0.05.\n"
            "\n"
            "**Part B — proof of L2H8's behavior.** Data-driven worked example from the "
            "per-position OV cache; bird's-eye and focused attention plots; "
            "single-position + corpus-wide ablation matrix (d-logit × NLL × single × "
            "corpus per Q8 of design grilling).\n"
            "\n"
            "**Part C — contrast with GPT-2 L10H7.** Side-by-side ablation: L10H7 "
            "(known-positive) corpus-wide d-logit = +0.032 (suppression direction); L2H8 "
            "corpus-wide d-logit = -0.009 (opposite). Different mechanisms; Pythia has "
            "no McDougall-style copy-suppression heads."
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
            "from torch.nn.functional import log_softmax\n"
            "from transformer_lens import HookedTransformer\n"
            "\n"
            "from notebooks._lib.sweep_io import read_long, to_wide\n"
            "from src.detectors.copy_suppression import _build_dup_index\n"
            "from src.utils.corpus_io import load_corpus\n"
            "from src.utils.pythia_loader import load_pythia\n"
            "\n"
            "torch.set_grad_enabled(False)\n"
            "\n"
            "QK_STRICT = 0.3\n"
            "OV_THRESHOLD = 0.0\n"
            "\n"
            "ANCHOR_PARQUET = REPO / 'data' / 'pilot' / 'copy_suppression_pythia_410m_step143000.parquet'\n"
            "ANCHOR_NPZ = REPO / 'data' / 'pilot' / 'copy_suppression_pythia_410m_step143000_per_position.npz'"
        ),
        code(
            "def upper_tri_mask(n: int) -> np.ndarray:\n"
            "    return torch.triu(torch.ones(n, n), diagonal=1).bool().numpy()\n"
            "\n"
            "def make_ablate_hook(layer: int, head: int):\n"
            "    name = f'blocks.{layer}.attn.hook_z'\n"
            "    def fn(z, hook):\n"
            "        z[:, :, head, :] = 0\n"
            "        return z\n"
            "    return (name, fn)"
        ),
        md("## Part A — detection"),
        md(
            "### Load anchor scores and per-position OV cache"
        ),
        code(
            "df = read_long(ANCHOR_PARQUET)\n"
            "wide = to_wide(df, index_cols=('size','step','layer','head'))\n"
            "wide['passes_strict'] = (wide['copy_suppression_qk'] > QK_STRICT) & (wide['copy_suppression_ov'] < OV_THRESHOLD)\n"
            "n_layers = int(wide['layer'].max() + 1)\n"
            "n_heads = int(wide['head'].max() + 1)\n"
            "print(f'anchor shape: n_layers={n_layers}, n_heads={n_heads}')\n"
            "print(f'eligible duplicate positions: see PILOT_RESULTS.md (2769)')\n"
            "print(f'heads passing strict criterion: {wide[\"passes_strict\"].sum()} of {len(wide)}')\n"
            "\n"
            "z = np.load(ANCHOR_NPZ)\n"
            "per_position_ov = z['per_position_ov']  # (n_eligible, n_layers, n_heads)\n"
            "per_position_meta = z['per_position_meta']  # (n_eligible, 2)\n"
            "print(f'per-position OV cache: shape={per_position_ov.shape}, n_eligible={len(per_position_meta)}')"
        ),
        md(
            "### (QK, OV) scatter for all 384 heads"
        ),
        code(
            "qk = wide['copy_suppression_qk'].values\n"
            "ov = wide['copy_suppression_ov'].values\n"
            "layers = wide['layer'].values\n"
            "heads = wide['head'].values\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(11, 7))\n"
            "scatter = ax.scatter(qk, ov, c=layers, cmap='viridis', s=30, alpha=0.7,\n"
            "                     edgecolor='k', linewidth=0.3)\n"
            "ax.axhline(OV_THRESHOLD, color='red', linestyle='--', alpha=0.4, label='OV threshold (0)')\n"
            "ax.axvline(QK_STRICT, color='red', linestyle='--', alpha=0.4, label=f'QK strict ({QK_STRICT})')\n"
            "\n"
            "# Highlight strict-passing head(s) and the qualitative-fail outcome.\n"
            "strict_mask = wide['passes_strict'].values\n"
            "if strict_mask.any():\n"
            "    ax.scatter(qk[strict_mask], ov[strict_mask], s=180, marker='*',\n"
            "               color='red', edgecolor='black', linewidth=1.5, zorder=10,\n"
            "               label=f'Strict-pass ({strict_mask.sum()} head(s)): qualitatively FAIL Day-4 inspection')\n"
            "    for li, hi in zip(layers[strict_mask], heads[strict_mask]):\n"
            "        ix = np.where((layers == li) & (heads == hi))[0][0]\n"
            "        ax.annotate(f'L{int(li)}H{int(hi)}', xy=(qk[ix], ov[ix]),\n"
            "                    xytext=(10, -10), textcoords='offset points', color='red', fontsize=10)\n"
            "\n"
            "ax.set_xlabel('QK score (mean attention to prior-duplicate tokens)')\n"
            "ax.set_ylabel('OV score (mean DLA on current token)')\n"
            "ax.set_title('Pythia-410M-deduped @ step143000: every head on the (QK, OV) plane\\n'\n"
            "             'point color = layer (early purple, late yellow); strict-passing head starred')\n"
            "ax.grid(alpha=0.3)\n"
            "ax.legend(loc='lower left', fontsize=9)\n"
            "plt.colorbar(scatter, ax=ax, label='layer')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "**Reading the scatter.** The lower-right quadrant (high QK, very negative OV) "
            "is where copy-suppression heads should live. On Pythia-410M @ step143000, "
            "this quadrant is essentially empty:\n"
            "\n"
            "- Heads with high QK (right side) cluster around OV ≈ 0 — duplicate-attending "
            "but not suppressing.\n"
            "- Heads with very negative OV (bottom) are mid-to-late layers (yellow) but "
            "have low QK (< 0.05) — not attending strongly to prior duplicates.\n"
            "- Only L2H8 squeaks past both thresholds, with marginal OV (-0.023). The "
            "qualitative inspection in `PILOT_RESULTS.md` § Manual qualitative inspection "
            "shows this is functionally a previous-token head, not a copy-suppression "
            "head.\n"
            "\n"
            "Compare to GPT-2 small on the same canonical corpus (`copy_suppression_proof.ipynb`): "
            "L10H7 sits at OV ≈ -0.8, with QK ≈ 0.02 — strong-OV-but-low-QK by the same "
            "raw-text dilution. Pythia's analog of L10H7's *position* (low-QK, "
            "very-negative-OV) is L18H15 (QK=0.026, OV=-0.593) but it doesn't pass the "
            "strict criterion either."
        ),
        md(
            "### Top-10 rankings"
        ),
        code(
            "wide_sorted_ov = wide.sort_values('copy_suppression_ov').reset_index(drop=True)\n"
            "wide_sorted_qk = wide.sort_values('copy_suppression_qk', ascending=False).reset_index(drop=True)\n"
            "\n"
            "print('Top-10 heads by most-negative OV (strongest suppression direction, head-level mean):')\n"
            "for _, r in wide_sorted_ov.head(10).iterrows():\n"
            "    qk_v = r['copy_suppression_qk']; ov_v = r['copy_suppression_ov']\n"
            "    star = '  <-- passes strict' if (qk_v > QK_STRICT and ov_v < 0) else ''\n"
            "    print(f\"  L{int(r['layer']):2d}H{int(r['head']):2d}: QK={qk_v:.3f} OV={ov_v:+.3f}{star}\")\n"
            "\n"
            "print('\\nTop-10 heads by QK (strongest attention to prior duplicates):')\n"
            "for _, r in wide_sorted_qk.head(10).iterrows():\n"
            "    qk_v = r['copy_suppression_qk']; ov_v = r['copy_suppression_ov']\n"
            "    star = '  <-- passes strict' if (qk_v > QK_STRICT and ov_v < 0) else ''\n"
            "    print(f\"  L{int(r['layer']):2d}H{int(r['head']):2d}: QK={qk_v:.3f} OV={ov_v:+.3f}{star}\")"
        ),
        md(
            "### select_proof_target — branching helper per Q6\n"
            "\n"
            "The Day 3 design grilling pre-committed to a deterministic branching rule "
            "for the proof target:\n"
            "- **Outcome 1** (≥1 strict candidate): top-3 strict candidates by most-negative OV.\n"
            "- **Outcome 2** (calibrated candidates only): dropped — calibrated scheme failed validation.\n"
            "- **Outcome 3** (empty): GPT-2 L10H7 reference + framing as 'we looked, here's the absence'.\n"
            "\n"
            "In the registered Path C universe, we have outcome 1 numerically (L2H8 passes strict) "
            "with a qualitative-fail verdict. Branching rule still selects L2H8 as the proof target, "
            "and Part C contrasts its behavior against GPT-2 L10H7 to make the Path-C "
            "registration cleanly visible."
        ),
        code(
            "def select_proof_target(wide_df):\n"
            "    strict = wide_df[wide_df['passes_strict']].sort_values('copy_suppression_ov')\n"
            "    if len(strict) > 0:\n"
            "        top_heads = [(int(r['layer']), int(r['head'])) for _, r in strict.head(3).iterrows()]\n"
            "        # Control: lowest |QK|+|OV| across all heads\n"
            "        s = (wide_df['copy_suppression_qk'].abs() + wide_df['copy_suppression_ov'].abs())\n"
            "        ctl_idx = s.idxmin()\n"
            "        ctl = (int(wide_df.loc[ctl_idx,'layer']), int(wide_df.loc[ctl_idx,'head']))\n"
            "        frame = ('Outcome 1 — strict candidate(s) found numerically; '\n"
            "                 'qualitative inspection per PILOT_RESULTS.md determines whether they are '\n"
            "                 'real copy-suppression heads')\n"
            "        return top_heads, ctl, frame\n"
            "    # Outcome 3 fallback: GPT-2 reference\n"
            "    return [(10, 7)], (0, 0), 'Outcome 3 — empty in Pythia; using GPT-2 L10H7 reference'\n"
            "\n"
            "TOP_HEADS, CONTROL, FRAME = select_proof_target(wide)\n"
            "print(f'Frame: {FRAME}')\n"
            "print(f'TOP_HEADS = {TOP_HEADS}')\n"
            "print(f'CONTROL = {CONTROL}')"
        ),
        md("## Part B — proof of L2H8's behavior"),
        md(
            "### Load Pythia-410M-deduped + canonical corpus\n"
            "\n"
            "Need the model for ablation forward passes and attention-pattern caching. "
            "Same corpus the anchor detector ran on, so worked-example positions index "
            "into the same passages."
        ),
        code(
            "model = load_pythia('410m', step=143000)\n"
            "device = next(model.parameters()).device\n"
            "n_layers, n_heads = int(model.cfg.n_layers), int(model.cfg.n_heads)\n"
            "passages = load_corpus()\n"
            "sequences = [model.to_tokens(p.text)[0].cpu() for p in passages]\n"
            "print(f'model loaded: device={device}, n_layers={n_layers}, n_heads={n_heads}')\n"
            "print(f'corpus: {len(sequences)} passages')"
        ),
        md(
            "### Data-driven worked example — top-3 positions where L2H8 fires hardest\n"
            "\n"
            "Q7 design rule: sort all eligible corpus positions by per-position OV "
            "ascending; the rank-1 position is the one where the head's per-position OV "
            "contribution is most-negative — i.e., where it would suppress most strongly "
            "*if* the OV criterion captured the real mechanism."
        ),
        code(
            "L_T, H_T = TOP_HEADS[0]  # rank-1 strict candidate\n"
            "ov_th = per_position_ov[:, L_T, H_T]\n"
            "rank_order = np.argsort(ov_th)\n"
            "\n"
            "print(f'top-3 corpus positions for L{L_T}H{H_T} (most-negative per-position OV):')\n"
            "for r in range(3):\n"
            "    k = int(rank_order[r])\n"
            "    p_idx, pos = int(per_position_meta[k][0]), int(per_position_meta[k][1])\n"
            "    seq = sequences[p_idx]\n"
            "    tok_id = int(seq[pos].item())\n"
            "    tok_str = model.to_string([tok_id])\n"
            "    lo = max(0, pos - 5); hi = min(len(seq), pos + 3)\n"
            "    surrounding = model.to_string(seq[lo:hi].tolist())\n"
            "    print(f'  rank {r+1}: passage {p_idx} ({passages[p_idx].title!r}), pos {pos}')\n"
            "    print(f'           token = {tok_str!r}, OV = {ov_th[k]:+.3f}')\n"
            "    print(f'           context = ...{surrounding!r}...')\n"
            "\n"
            "# Lock the rank-1 worked example for visual analysis below.\n"
            "K = int(rank_order[0])\n"
            "EX_P_IDX = int(per_position_meta[K][0])\n"
            "EX_POS = int(per_position_meta[K][1])\n"
            "EX_TOKENS = sequences[EX_P_IDX].clone()\n"
            "EX_TOK_ID = int(EX_TOKENS[EX_POS].item())\n"
            "EX_TOK_STR = model.to_string([EX_TOK_ID])\n"
            "EX_TITLE = passages[EX_P_IDX].title\n"
            "EX_STR_TOKENS = model.to_str_tokens(passages[EX_P_IDX].text)\n"
            "print(f'\\nworked example locked: passage={EX_TITLE!r}, pos={EX_POS}, '\n"
            "      f'token={EX_TOK_STR!r}, per-position OV={ov_th[K]:+.3f}')"
        ),
        md(
            "### Bird's-eye attention patterns on the worked-example passage\n"
            "\n"
            "All 384 heads (24 × 16) plotted with upper-triangular masking. Watch for "
            "off-diagonal attention from later positions back to earlier occurrences "
            "of the same token — the QK signature shared by induction *and* "
            "copy-suppression. L2H8 is highlighted in red."
        ),
        code(
            "tokens_2d = EX_TOKENS.unsqueeze(0).to(device)\n"
            "_, attn_cache = model.run_with_cache(\n"
            "    tokens_2d, names_filter=lambda n: n.endswith('hook_pattern'), return_type=None\n"
            ")\n"
            "n_tok = len(EX_STR_TOKENS)\n"
            "tri_mask = upper_tri_mask(n_tok)\n"
            "\n"
            "fig = plt.figure(figsize=(20, 30))\n"
            "for layer in range(n_layers):\n"
            "    pat_layer = attn_cache[f'blocks.{layer}.attn.hook_pattern']\n"
            "    for h in range(n_heads):\n"
            "        ax = fig.add_subplot(n_layers, n_heads, n_heads * layer + h + 1)\n"
            "        a = pat_layer[0, h, :n_tok, :n_tok].detach().cpu().float().numpy()\n"
            "        a_masked = np.ma.masked_array(a, tri_mask)\n"
            "        ax.imshow(a_masked)\n"
            "        ax.axis('off')\n"
            "        if (layer, h) == (L_T, H_T):\n"
            "            for spine in ax.spines.values():\n"
            "                spine.set_edgecolor('red')\n"
            "                spine.set_linewidth(2)\n"
            "            ax.axis('on')\n"
            "            ax.set_xticks([]); ax.set_yticks([])\n"
            "fig.suptitle(f'Pythia-410M-deduped @ step143000 attention on {EX_TITLE!r}; '\n"
            "             f'L{L_T}H{H_T} outlined in red', y=1.0)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "### Focused — L2H8 vs control with token labels"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(16, 7))\n"
            "for ax, (layer, head) in zip(axes, [(L_T, H_T), CONTROL]):\n"
            "    pat = attn_cache[f'blocks.{layer}.attn.hook_pattern'][0, head, :n_tok, :n_tok]\n"
            "    a = pat.detach().cpu().float().numpy()\n"
            "    a_masked = np.ma.masked_array(a, tri_mask)\n"
            "    label_extra = ' (rank-1 strict)' if (layer, head) == (L_T, H_T) else ' (control)'\n"
            "    im = ax.imshow(a_masked)\n"
            "    ax.set_xticks(range(n_tok))\n"
            "    ax.set_yticks(range(n_tok))\n"
            "    ax.set_xticklabels(EX_STR_TOKENS, rotation=90, fontsize=6)\n"
            "    ax.set_yticklabels(EX_STR_TOKENS, fontsize=6)\n"
            "    ax.axhline(EX_POS - 0.5, color='red', linewidth=0.8, alpha=0.6)\n"
            "    ax.axhline(EX_POS + 0.5, color='red', linewidth=0.8, alpha=0.6)\n"
            "    qk_v = float(wide[(wide['layer']==layer) & (wide['head']==head)]['copy_suppression_qk'].iloc[0])\n"
            "    ov_v = float(wide[(wide['layer']==layer) & (wide['head']==head)]['copy_suppression_ov'].iloc[0])\n"
            "    ax.set_title(f'L{layer}H{head}{label_extra}\\n'\n"
            "                 f'QK={qk_v:.3f}, OV={ov_v:+.3f}; worked-ex pos {EX_POS} highlighted')\n"
            "    plt.colorbar(im, ax=ax, fraction=0.046)\n"
            "fig.suptitle(f'Attention patterns on {EX_TITLE!r}: L{L_T}H{H_T} vs control', y=1.02)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "**The QK signature is unambiguous.** L2H8 attends from the worked-example "
            "query position back to the *prior* occurrence of the duplicate token — the "
            "textbook QK pattern for both induction and copy-suppression. The control "
            "head shows generic local-attention patterns with no duplicate-back focus.\n"
            "\n"
            "The next cells test the *OV* side of the mechanism: ablation. If L2H8 is "
            "doing copy-suppression, ablating it should *raise* the duplicate-token logit "
            "(suppression removed). If it's doing copying / promotion (induction-precursor "
            "behavior), ablation should *lower* the duplicate-token logit."
        ),
        md(
            "### Single-position ablation at the worked-example position"
        ),
        code(
            "baseline_logits = model(tokens_2d, return_type='logits')[0]\n"
            "with model.hooks(fwd_hooks=[make_ablate_hook(L_T, H_T)]):\n"
            "    target_ablated = model(tokens_2d, return_type='logits')[0]\n"
            "with model.hooks(fwd_hooks=[make_ablate_hook(*CONTROL)]):\n"
            "    control_ablated = model(tokens_2d, return_type='logits')[0]\n"
            "\n"
            "logit_pred_pos = EX_POS - 1\n"
            "tok_i = EX_TOK_ID\n"
            "base_logit = float(baseline_logits[logit_pred_pos, tok_i].item())\n"
            "tgt_logit = float(target_ablated[logit_pred_pos, tok_i].item())\n"
            "ctl_logit = float(control_ablated[logit_pred_pos, tok_i].item())\n"
            "base_nll = float(-log_softmax(baseline_logits[logit_pred_pos], dim=-1)[tok_i].item())\n"
            "tgt_nll = float(-log_softmax(target_ablated[logit_pred_pos], dim=-1)[tok_i].item())\n"
            "ctl_nll = float(-log_softmax(control_ablated[logit_pred_pos], dim=-1)[tok_i].item())\n"
            "\n"
            "print(f'Worked-example position: passage {EX_P_IDX} ({EX_TITLE!r}), pos {EX_POS}, '\n"
            "      f'token={EX_TOK_STR!r}\\n')\n"
            "print(f'{\"head\":<22} {\"logit\":>10} {\"d-logit\":>10} {\"NLL\":>10} {\"d-NLL\":>10}')\n"
            "print(f'{\"baseline\":<22} {base_logit:>+10.3f} {0.0:>+10.3f} {base_nll:>10.3f} {0.0:>+10.3f}')\n"
            "print(f'{f\"L{L_T}H{H_T} ablated (target)\":<22} {tgt_logit:>+10.3f} '\n"
            "      f'{tgt_logit-base_logit:>+10.3f} {tgt_nll:>10.3f} {tgt_nll-base_nll:>+10.3f}')\n"
            "print(f'{f\"L{CONTROL[0]}H{CONTROL[1]} ablated (control)\":<22} {ctl_logit:>+10.3f} '\n"
            "      f'{ctl_logit-base_logit:>+10.3f} {ctl_nll:>10.3f} {ctl_nll-base_nll:>+10.3f}')\n"
            "print()\n"
            "print('For copy-suppression: d-logit on duplicate token should be POSITIVE')\n"
            "print('(ablating the suppressor lets the duplicate logit rise).')\n"
            "\n"
            "# Visual: top-10 logits at logit_pred_pos with vs without target head\n"
            "top_k = 10\n"
            "base_top = torch.topk(baseline_logits[logit_pred_pos], top_k)\n"
            "tgt_top_at_base_idx = target_ablated[logit_pred_pos, base_top.indices]\n"
            "labels = [model.to_string([int(idx)]).strip() or '\\\\n' for idx in base_top.indices.tolist()]\n"
            "x = np.arange(top_k)\n"
            "fig, ax = plt.subplots(figsize=(11, 4))\n"
            "ax.bar(x - 0.2, base_top.values.cpu().float().numpy(), width=0.4, label='baseline', color='C0')\n"
            "ax.bar(x + 0.2, tgt_top_at_base_idx.cpu().float().numpy(), width=0.4,\n"
            "       label=f'L{L_T}H{H_T} ablated', color='C3')\n"
            "ax.set_xticks(x)\n"
            "ax.set_xticklabels([f'{l!r}' for l in labels], rotation=40, ha='right', fontsize=9)\n"
            "ax.set_ylabel('logit')\n"
            "ax.set_title(f'Top-10 baseline predictions at pos {logit_pred_pos}->'\n"
            "             f'{EX_POS} ({EX_TITLE!r}); duplicate token = {EX_TOK_STR!r}')\n"
            "ax.legend()\n"
            "ax.grid(axis='y', alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "### Corpus-wide ablation — Q8 matrix\n"
            "\n"
            "Single-position numbers are noisy; the corpus-wide aggregate over all 2769 "
            "eligible duplicate positions is the robust functional test. Both d-logit and "
            "d-NLL reported."
        ),
        code(
            "def measure_corpus_change(model, sequences, fwd_hooks=None):\n"
            "    dlogit_total = 0.0; dnll_total = 0.0; n = 0\n"
            "    for tokens_1d in sequences:\n"
            "        tokens = tokens_1d.to(device).unsqueeze(0)\n"
            "        tok_list = tokens_1d.tolist()\n"
            "        dup_map = _build_dup_index(tok_list)\n"
            "        if not dup_map: continue\n"
            "        baseline = model(tokens, return_type='logits')[0]\n"
            "        if fwd_hooks is None:\n"
            "            modified = baseline\n"
            "        else:\n"
            "            with model.hooks(fwd_hooks=fwd_hooks):\n"
            "                modified = model(tokens, return_type='logits')[0]\n"
            "        for i in dup_map:\n"
            "            if i == 0: continue\n"
            "            t = tok_list[i]\n"
            "            dlogit_total += (modified[i-1, t].item() - baseline[i-1, t].item())\n"
            "            base_nll = float(-log_softmax(baseline[i-1], dim=-1)[t].item())\n"
            "            mod_nll = float(-log_softmax(modified[i-1], dim=-1)[t].item())\n"
            "            dnll_total += (mod_nll - base_nll)\n"
            "            n += 1\n"
            "    return dlogit_total / max(n, 1), dnll_total / max(n, 1), n\n"
            "\n"
            "tgt_dlogit, tgt_dnll, n_corpus = measure_corpus_change(model, sequences,\n"
            "    fwd_hooks=[make_ablate_hook(L_T, H_T)])\n"
            "ctl_dlogit, ctl_dnll, _ = measure_corpus_change(model, sequences,\n"
            "    fwd_hooks=[make_ablate_hook(*CONTROL)])\n"
            "\n"
            "print(f'Q8 ablation matrix — N={n_corpus} eligible corpus positions:\\n')\n"
            "print(f'{\"head\":<28} {\"d-logit (single)\":>16} {\"d-NLL (single)\":>16} '\n"
            "      f'{\"d-logit (corpus)\":>16} {\"d-NLL (corpus)\":>16}')\n"
            "print(f'{f\"L{L_T}H{H_T} (rank-1 strict)\":<28} {tgt_logit-base_logit:>+16.3f} '\n"
            "      f'{tgt_nll-base_nll:>+16.3f} {tgt_dlogit:>+16.4f} {tgt_dnll:>+16.4f}')\n"
            "print(f'{f\"L{CONTROL[0]}H{CONTROL[1]} (control)\":<28} {ctl_logit-base_logit:>+16.3f} '\n"
            "      f'{ctl_nll-base_nll:>+16.3f} {ctl_dlogit:>+16.4f} {ctl_dnll:>+16.4f}')\n"
            "print()\n"
            "print('For a copy-suppression head:')\n"
            "print('  d-logit (corpus) > 0  -> ablation raises duplicate-token logits')\n"
            "print('  d-NLL  (corpus) > 0  -> ablation hurts next-token prediction overall')\n"
            "print()\n"
            "print(f'Observed: L{L_T}H{H_T} d-logit (corpus) = {tgt_dlogit:+.4f}')\n"
            "if tgt_dlogit > 0:\n"
            "    print('  -> consistent with suppression direction')\n"
            "else:\n"
            "    print(f'  -> OPPOSITE of suppression direction. L{L_T}H{H_T} is')\n"
            "    print(f'     functionally promoting duplicate tokens, not suppressing them.')"
        ),
        md("## Part C — contrast with GPT-2 L10H7"),
        md(
            "If L2H8 were a real copy-suppression head, its corpus-wide d-logit on duplicates "
            "should match the sign of GPT-2 L10H7's on the same canonical corpus. Run "
            "the equivalent ablation on GPT-2 to make the contrast visible."
        ),
        code(
            "# Free Pythia-410M to load GPT-2.\n"
            "del attn_cache, baseline_logits, target_ablated, control_ablated\n"
            "del model\n"
            "if torch.backends.mps.is_available():\n"
            "    torch.mps.empty_cache()\n"
            "\n"
            "gpt2 = HookedTransformer.from_pretrained('gpt2', device=device)\n"
            "gpt2_sequences = [gpt2.to_tokens(p.text)[0].cpu() for p in passages]\n"
            "L_GPT, H_GPT = (10, 7)\n"
            "\n"
            "gpt_dlogit, gpt_dnll, n_gpt = measure_corpus_change(gpt2, gpt2_sequences,\n"
            "    fwd_hooks=[make_ablate_hook(L_GPT, H_GPT)])\n"
            "\n"
            "print('Corpus-wide ablation on canonical corpus (same passages, different tokenizer):')\n"
            "print()\n"
            "print(f'{\"model / head\":<32} {\"d-logit (corpus)\":>18} {\"d-NLL (corpus)\":>18} {\"N\":>8}')\n"
            "print(f'{f\"GPT-2 small L10H7 (reference)\":<32} {gpt_dlogit:>+18.4f} {gpt_dnll:>+18.4f} {n_gpt:>8d}')\n"
            "print(f'{f\"Pythia-410M-dedup L{L_T}H{H_T}\":<32} {tgt_dlogit:>+18.4f} {tgt_dnll:>+18.4f} {n_corpus:>8d}')\n"
            "print()\n"
            "if gpt_dlogit > 0 and tgt_dlogit < 0:\n"
            "    print('Sign contrast: GPT-2 L10H7 raises duplicate logits when ablated (suppression');\n"
            "    print('signature). Pythia L2H8 lowers them (promotion signature). Different mechanisms;')\n"
            "    print('Path C registered — Pythia has no McDougall-style copy-suppression heads.')"
        ),
        md(
            "## Conclusion\n"
            "\n"
            "**Part A.** The McDougall two-criterion detector applied to all 384 heads of "
            "Pythia-410M-deduped @ step143000 on the canonical 7.5k-token corpus identifies "
            "**1 head** (L2H8) passing both strict thresholds (QK > 0.3 AND OV < 0). The "
            "(QK, OV) plane shows the lower-right quadrant — where copy-suppression heads "
            "should live — is essentially empty.\n"
            "\n"
            "**Part B.** L2H8's QK signature is textbook: 0.91 attention from the rank-1 "
            "worked-example query position back to the prior occurrence of the duplicate "
            "token. But its OV behavior is functionally opposite of suppression: corpus-wide "
            "ablation across 2769 eligible positions *lowers* duplicate-token logits by a "
            "small amount (-0.009). L2H8 is a *previous-token / induction-precursor head* "
            "(per Singh 2024), not a copy-suppression head.\n"
            "\n"
            "**Part C.** GPT-2 small L10H7 on the *same canonical corpus* shows positive "
            "corpus-wide d-logit (+0.032) — the suppression direction. The two heads sit on "
            "opposite sides of the d-logit zero line; the mechanism is genuinely different.\n"
            "\n"
            "**Pilot decision: Path C registered.** Per `HYPOTHESIS.md` decision rule "
            "path 3, the numerically-passing head failing qualitative inspection registers "
            "Path C. The project pivots: the third motif of the H1 ordering claim becomes "
            "**S-inhibition** (the suppression component of the IOI circuit, Wang et al. "
            "2023), not copy-suppression. Phase 2 detector targets are induction, "
            "successor, and S-inhibition.\n"
            "\n"
            "**What's still publishable.** This negative-result finding is itself a "
            "contribution to the BlackboxNLP audience: McDougall et al. 2024 explicitly "
            "named *\"do copy-suppression heads exist in Pythia and Llama?\"* as future "
            "work. We've answered for Pythia (deduped, scaled to 410M, final checkpoint): "
            "**not in the McDougall-published-mechanism sense**, even though some heads "
            "numerically clear the published thresholds. The threshold-transfer issue "
            "(strict QK > 0.3 not reaching mean ≥ 0.3 on raw text even for L10H7 itself) "
            "compounds with the missing-mechanism finding to give a clean negative result.\n"
            "\n"
            "Day 5 of the pilot week locks the path commit and updates `HYPOTHESIS.md` to "
            "make H1-C the registered hypothesis going forward."
        ),
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
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
