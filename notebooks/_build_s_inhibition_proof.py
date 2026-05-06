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
            "## Geometric intuition: the S-inhibition fingerprint in attention space\n"
            "\n"
            "Path-patching is the *causal* test for S-inhibition: it asks "
            "*does corrupting this head's input change Name Mover behaviour "
            "in the predicted way?* But behind the causal evidence there's "
            "an attention-pattern fingerprint that's worth seeing directly, "
            "the same way the induction proof showed the *shifted diagonal* "
            "for prefix-matching.\n"
            "\n"
            "**The IOI prompt has four positions of interest:**\n"
            "\n"
            "- **n1**: the first name in the prompt (e.g., 'When **Mary** and John...')\n"
            "- **n2**: the second name ('When Mary and **John** went...')\n"
            "- **n3** (= S2): the second-clause subject, repeating one of the earlier names ('...the store, **John** gave a present to')\n"
            "- **END**: the last token, where the model emits the IO prediction (' Mary' or ' John')\n"
            "\n"
            "In ABBA prompts: n1 is the IO (unique name), n2 = n3 = S (the duplicate). In BABA prompts: n2 is the IO, n1 = n3 = S. **In both templates the model must predict the IO at END.**\n"
            "\n"
            "**The two competing attention signatures at the END query:**\n"
            "\n"
            "- **Name Mover (NM) signature**: high attention from END to **IO**. Reads the IO name and writes it strongly to the next-token prediction.\n"
            "- **S-inhibition (SI) signature**: high attention from END to **S2** (the duplicated subject). Doesn't write the IO directly; instead, its output redirects downstream NMs' queries away from S2 and toward IO. *That's the mechanism — and the fingerprint is the END→S2 attention itself.*\n"
            "\n"
            "**Where the path-patching scalar comes from.** When we ABC-corrupt the n3 position (replacing the duplicate with a fresh name C), the SI head loses its target — there's no S2 in the corrupt prompt for the head to attend to. Its output changes, and downstream NMs no longer get the redirection signal. Path-patching measures exactly this: the *change* in NM attention pattern under sender corruption. A head with no END→S2 attention can't produce a path-patching signal; a head with strong END→S2 attention is the canonical S-inhibition mechanism.\n"
            "\n"
            "Below: load GPT-2, pick a clean IOI prompt, and look at the attention pattern of Wang's strongest S-Inhibition head (L8H6) vs a Name Mover (L9H6) vs a control head."
        ),
        md("### Setup: load GPT-2 small and pick a worked example"),
        code(
            "from transformer_lens import HookedTransformer\n"
            "from src.replication.tigges_ioi import build_ioi_prompts\n"
            "\n"
            "model = HookedTransformer.from_pretrained('gpt2')\n"
            "tokenizer = model.tokenizer\n"
            "\n"
            "# Pick a single ABBA prompt for clear visualization. Names chosen for short total length.\n"
            "ex_prompt = 'When Mary and John went to the store, John gave a present to'\n"
            "ex_io = 'Mary'\n"
            "ex_s = 'John'\n"
            "ex_template = 'ABBA'\n"
            "\n"
            "ex_tokens = model.to_tokens(ex_prompt, prepend_bos=True)\n"
            "ex_str_tokens = model.to_str_tokens(ex_prompt, prepend_bos=True)\n"
            "io_tok_id = tokenizer.encode(' ' + ex_io, add_special_tokens=False)[0]\n"
            "s_tok_id = tokenizer.encode(' ' + ex_s, add_special_tokens=False)[0]\n"
            "\n"
            "tok_list = ex_tokens[0].cpu().tolist()\n"
            "io_positions = [i for i, t in enumerate(tok_list) if t == io_tok_id]\n"
            "s_positions = [i for i, t in enumerate(tok_list) if t == s_tok_id]\n"
            "io_pos = io_positions[0]                        # IO appears once (n1 in ABBA)\n"
            "s2_pos = s_positions[-1]                        # S2 = last occurrence of S (n3)\n"
            "s1_pos = s_positions[0]                         # S1 = first S occurrence (n2)\n"
            "end_pos = ex_tokens.shape[1] - 1\n"
            "\n"
            "print(f'tokens (with BOS): {list(enumerate(ex_str_tokens))}')\n"
            "print()\n"
            "print(f'  IO  = {ex_io!r:8s} at position {io_pos}')\n"
            "print(f'  S1  = {ex_s!r:8s} at position {s1_pos}  (n2)')\n"
            "print(f'  S2  = {ex_s!r:8s} at position {s2_pos}  (n3, the duplicate the model must suppress)')\n"
            "print(f'  END                  at position {end_pos}  (where IO is predicted)')"
        ),
        md(
            "### Run the prompt and cache attention patterns\n"
            "\n"
            "Cache attention patterns at every layer; we'll pull out the heads of interest below."
        ),
        code(
            "_, ex_cache = model.run_with_cache(\n"
            "    ex_tokens, names_filter=lambda name: name.endswith('hook_pattern'),\n"
            "    return_type=None,\n"
            ")\n"
            "print('cached layers:', sorted({k for k in ex_cache.keys()})[:3], '... etc')"
        ),
        md(
            "### Three heads, three signatures\n"
            "\n"
            "Plot the attention matrix of three heads on this prompt:\n"
            "\n"
            "- **L8H6**: Wang's strongest S-Inhibition head (our top path-patching ranker). Predicted signature: high attention from END to S2.\n"
            "- **L9H6**: a Wang-published Name Mover. Predicted signature: high attention from END to IO.\n"
            "- **L0H5**: a control head from layer 0 (no IOI role expected). Predicted signature: diffuse / attend-to-self.\n"
            "\n"
            "Stars mark the (END query, IO key) and (END query, S2 key) cells. The S-inhibition fingerprint is *bright at the S2 star and dim at the IO star*; the Name Mover fingerprint is the opposite."
        ),
        code(
            "import matplotlib.patches as mpatches\n"
            "\n"
            "TARGETS = [\n"
            "    ((8, 6),  'L8H6 — Wang S-Inhibition'),\n"
            "    ((9, 6),  'L9H6 — Wang Name Mover'),\n"
            "    ((0, 5),  'L0H5 — control'),\n"
            "]\n"
            "\n"
            "n_pos = ex_tokens.shape[1]\n"
            "tri_mask = np.triu(np.ones((n_pos, n_pos), dtype=bool), k=1)\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))\n"
            "for ax, ((L, H), title) in zip(axes, TARGETS):\n"
            "    pat = ex_cache[f'blocks.{L}.attn.hook_pattern'][0, H].detach().cpu().float().numpy()\n"
            "    pat_masked = np.ma.masked_array(pat, tri_mask)\n"
            "    im = ax.imshow(pat_masked, cmap='viridis', vmin=0, vmax=max(0.05, pat.max()))\n"
            "    ax.set_xticks(range(n_pos))\n"
            "    ax.set_yticks(range(n_pos))\n"
            "    ax.set_xticklabels(ex_str_tokens, rotation=90, fontsize=8)\n"
            "    ax.set_yticklabels(ex_str_tokens, fontsize=8)\n"
            "    ax.set_xlabel('key (attended-to position)')\n"
            "    if ax is axes[0]:\n"
            "        ax.set_ylabel('query (current position)')\n"
            "    ax.set_title(title)\n"
            "    # Mark the (END query, IO key) and (END query, S2 key) cells\n"
            "    ax.scatter([io_pos], [end_pos], marker='*', s=240, color='tab:red',\n"
            "               edgecolor='white', linewidth=1.2, label=f'END → IO ({ex_io})', zorder=5)\n"
            "    ax.scatter([s2_pos], [end_pos], marker='*', s=240, color='tab:cyan',\n"
            "               edgecolor='white', linewidth=1.2, label=f'END → S2 ({ex_s})', zorder=5)\n"
            "    ax.legend(loc='upper right', fontsize=8)\n"
            "    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)\n"
            "fig.suptitle(f'Attention patterns on: {ex_prompt!r}', y=1.02)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "### Quantitative read of the same figure\n"
            "\n"
            "For each of the three heads (plus the rest of Wang's S-Inhibition heads), what fraction of attention does the END token spend on S2 vs on IO? S-Inhibition heads should have `attn_END→S2 ≫ attn_END→IO`; Name Movers should have the opposite."
        ),
        code(
            "rows = []\n"
            "shown_heads = [(8, 6), (7, 3), (7, 9), (8, 10), (9, 6), (9, 9), (10, 0), (0, 5)]\n"
            "for (L, H) in shown_heads:\n"
            "    pat = ex_cache[f'blocks.{L}.attn.hook_pattern'][0, H].detach().cpu().float().numpy()\n"
            "    a_io = float(pat[end_pos, io_pos])\n"
            "    a_s2 = float(pat[end_pos, s2_pos])\n"
            "    role = ('S-Inh' if (L, H) in WANG_S_INHIBITION\n"
            "            else ('NM' if (L, H) in WANG_NM\n"
            "                  else 'control'))\n"
            "    rows.append(dict(\n"
            "        head=f'L{L}H{H}', role=role,\n"
            "        attn_END_to_S2=a_s2, attn_END_to_IO=a_io,\n"
            "        S2_minus_IO=a_s2 - a_io,\n"
            "    ))\n"
            "tbl = pd.DataFrame(rows)\n"
            "print(tbl.to_string(index=False, float_format=lambda v: f'{v:.4f}'))"
        ),
        md(
            "### Reading the table\n"
            "\n"
            "All four Wang S-Inhibition heads should show **`attn_END→S2 > attn_END→IO`** (positive `S2_minus_IO`). Name Movers should show the opposite — they read the IO and ignore the duplicate. The control head should be diffuse with low values on both targets.\n"
            "\n"
            "**Connecting back to the path-patching scalar Δ_h.** Recall that Δ_h (HYPOTHESIS.md §S-4) is the change in NM attention shift under ABC corruption — `(patched − clean) attn at S2  −  (patched − clean) attn at IO`, averaged across the k=4 NMs. The mechanism: an SI head's clean output (whose attention pattern is what we're visualizing here) writes into the residual stream at END, and Name Movers' query computation reads this. Without the SI head's contribution (= what ABC corruption removes), NMs no longer get the redirection signal and attend more to S2, less to IO. Path-patching detects exactly that downstream consequence; the attention figure above is the mechanistic *upstream* picture that produces it."
        ),
        md(
            "### One more verification: the END row across all 144 heads\n"
            "\n"
            "Bird's-eye check: at the END query, plot the (attn_S2 − attn_IO) value for every (layer, head). Wang's S-Inhibition heads should be the most positive entries; Name Movers the most negative."
        ),
        code(
            "diff_grid = np.zeros((model.cfg.n_layers, model.cfg.n_heads))\n"
            "for L in range(model.cfg.n_layers):\n"
            "    layer_pat = ex_cache[f'blocks.{L}.attn.hook_pattern'][0].detach().cpu().float().numpy()\n"
            "    for H in range(model.cfg.n_heads):\n"
            "        diff_grid[L, H] = layer_pat[H, end_pos, s2_pos] - layer_pat[H, end_pos, io_pos]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8, 6.5))\n"
            "abs_max = float(np.abs(diff_grid).max())\n"
            "im = ax.imshow(diff_grid, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max, aspect='auto')\n"
            "ax.set_xticks(range(model.cfg.n_heads))\n"
            "ax.set_yticks(range(model.cfg.n_layers))\n"
            "ax.set_xlabel('head')\n"
            "ax.set_ylabel('layer')\n"
            "ax.set_title('attn_END→S2  −  attn_END→IO  per (layer, head) on the ABBA example')\n"
            "for L, H in WANG_S_INHIBITION:\n"
            "    ax.add_patch(plt.Rectangle((H - 0.45, L - 0.45), 0.9, 0.9,\n"
            "                                fill=False, edgecolor='red', linewidth=1.5))\n"
            "for L, H in WANG_NM:\n"
            "    ax.add_patch(plt.Rectangle((H - 0.45, L - 0.45), 0.9, 0.9,\n"
            "                                fill=False, edgecolor='blue', linewidth=1.5))\n"
            "fig.colorbar(im, ax=ax, label='attn_S2 − attn_IO')\n"
            "ax.text(0.02, 0.98, 'red box = Wang S-Inhibition\\nblue box = Wang Name Mover',\n"
            "        transform=ax.transAxes, va='top', ha='left',\n"
            "        bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85),\n"
            "        fontsize=9)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "Reading the heatmap: positive (red) cells are heads that, on this prompt, attend to S2 more than IO at the END query — the S-inhibition direction. Negative (blue) cells attend to IO more — the Name Mover direction. Wang's S-Inhibition heads (red boxes) should sit in the red regions; Wang's Name Movers (blue boxes) in the blue regions.\n"
            "\n"
            "*This is a single-prompt snapshot.* The path-patching screen above averaged across N=200 prompts and used the causal `(patched − clean)` shift, which is more robust. But the per-prompt attention picture is the *mechanism* picture, and it lines up with the path-patching ranking: heads that produce strong END→S2 attention on IOI prompts are the same heads whose path-patching scalars top the screen.\n"
            "\n"
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
