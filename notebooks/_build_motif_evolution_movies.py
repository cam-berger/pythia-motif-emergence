"""Build notebooks/motif_evolution_movies.ipynb.

Per-size 3-motif co-evolution animations. One animation per Pythia size,
each with side-by-side panels for induction / successor / S-inhibition,
animating across all 40 §H2-1 checkpoints.

For Pythia-2.8B, S-inhibition is omitted (sweep halted at 8/40 cells
under §H4-7-supersede); the 2.8B animation has 2 panels (ind + suc).

Outputs:
- notebooks/figures/motif_evolution_{size}.gif (5 standalone GIFs)
- Inline HTML5 players embedded in the notebook output.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "motif_evolution_movies.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            "# Per-size motif co-evolution movies\n"
            "\n"
            "Side-by-side animations of the three attention-head motifs "
            "(induction, successor, S-inhibition) evolving across the §H2-1 "
            "40-checkpoint grid, one animation per Pythia size. Each panel "
            "shows the (layer, head) score grid for that motif, with a "
            "shared frame index across the 3 panels so motif co-emergence "
            "is visually comparable.\n"
            "\n"
            "**Per-motif color scales (separate colorbars per panel):**\n"
            "- Induction: prefix-matching score, range [0, 1]\n"
            "- Successor: §SU-1b lift score (real_DLA − null_DLA), range varies by size\n"
            "- S-inhibition: §S-1 path-patching Δ_h scalar, range varies by size\n"
            "\n"
            "**2.8B caveat (§H4-7-supersede):** the 2.8B S-inhibition sweep "
            "was halted at 8/40 cells. The 2.8B animation therefore has only "
            "2 panels (induction + successor); the S-inhibition panel is "
            "omitted. The 8 partial S-inhibition cells at 2.8B are preserved "
            "as `data/exploration/phase4_2_8b_s_inhibition_per_prompt/` but "
            "are not animated here."
        ),
        md("## Setup"),
        code(
            "import os\n"
            "os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')\n"
            "os.environ.setdefault('HF_HUB_OFFLINE', '1')\n"
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
            "import matplotlib.animation as animation\n"
            "from IPython.display import HTML, display\n"
            "from notebooks._lib.sweep_io import read_long\n"
            "\n"
            "INDUCTION_THRESHOLD = 0.3\n"
            "TAU_LIFT = 0.13496\n"
            "TAU_STRICT = 0.0372\n"
            "\n"
            "def sweep_path(motif: str, size: str) -> Path:\n"
            "    if size == '1b':\n"
            "        return REPO / 'data' / 'exploration' / f'phase3_1b_{motif}_sweep.parquet'\n"
            "    if size == '2.8b':\n"
            "        return REPO / 'data' / 'exploration' / f'phase4_2_8b_{motif}_sweep.parquet'\n"
            "    return REPO / 'data' / 'exploration' / f'phase2_{motif}_sweep.parquet'\n"
            "\n"
            "STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,\n"
            "         1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,\n"
            "         10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,\n"
            "         20000, 24000, 29000, 35000, 41000, 49000, 59000, 70000,\n"
            "         84000, 100000, 120000, 143000]\n"
            "\n"
            "FIG_DIR = REPO / 'notebooks' / 'figures'\n"
            "FIG_DIR.mkdir(parents=True, exist_ok=True)"
        ),
        md(
            "## Load all motif sweep parquets (5 sizes × 3 motifs, dedup phase2)"
        ),
        code(
            "# phase2_*_sweep.parquet contains all 3 phase2 sizes; load once per motif then filter.\n"
            "df_ind = pd.concat([\n"
            "    read_long(REPO / 'data' / 'exploration' / 'phase2_induction_sweep.parquet'),\n"
            "    read_long(REPO / 'data' / 'exploration' / 'phase3_1b_induction_sweep.parquet'),\n"
            "    read_long(REPO / 'data' / 'exploration' / 'phase4_2_8b_induction_sweep.parquet'),\n"
            "]).reset_index(drop=True)\n"
            "df_suc = pd.concat([\n"
            "    read_long(REPO / 'data' / 'exploration' / 'phase2_successor_sweep.parquet'),\n"
            "    read_long(REPO / 'data' / 'exploration' / 'phase3_1b_successor_sweep.parquet'),\n"
            "    read_long(REPO / 'data' / 'exploration' / 'phase4_2_8b_successor_sweep.parquet'),\n"
            "]).reset_index(drop=True)\n"
            "df_si = pd.concat([\n"
            "    read_long(REPO / 'data' / 'exploration' / 'phase2_s_inhibition_sweep.parquet'),\n"
            "    read_long(REPO / 'data' / 'exploration' / 'phase3_1b_s_inhibition_sweep.parquet'),\n"
            "]).reset_index(drop=True)\n"
            "# 2.8b s_inhibition omitted (§H4-7-supersede).\n"
            "MOTIF_DF = {'induction': df_ind, 'successor': df_suc, 's_inhibition': df_si}\n"
            "MOTIF_LABEL = {'induction': 'Induction (prefix-match)',\n"
            "               'successor': 'Successor (lift_dla)',\n"
            "               's_inhibition': 'S-inhibition (Δ_h)'}\n"
            "MOTIF_THRESHOLD = {'induction': INDUCTION_THRESHOLD,\n"
            "                   'successor': TAU_LIFT,\n"
            "                   's_inhibition': TAU_STRICT}\n"
            "for motif, df in MOTIF_DF.items():\n"
            "    print(f'{motif}: {len(df):,} rows, sizes={sorted(df[\"size\"].unique().tolist())}')"
        ),
        md(
            "## Helper: build heatmap-frames for one (size, motif)\n"
            "\n"
            "Returns a list of (n_layers, n_heads) arrays — one per step in "
            "STEPS — and the (vmin, vmax) for that panel's color scale."
        ),
        code(
            "def build_frames(size: str, motif: str):\n"
            "    df = MOTIF_DF[motif]\n"
            "    sub = df[df['size'] == size]\n"
            "    if sub.empty:\n"
            "        return None, None, None  # signals 'skip this panel'\n"
            "    n_layers = int(sub['layer'].max() + 1)\n"
            "    n_heads = int(sub['head'].max() + 1)\n"
            "    frames = []\n"
            "    for step in STEPS:\n"
            "        cell = sub[sub['step'] == step]\n"
            "        grid = np.zeros((n_layers, n_heads))\n"
            "        for _, r in cell.iterrows():\n"
            "            grid[int(r['layer']), int(r['head'])] = r['score']\n"
            "        frames.append(grid)\n"
            "    all_vals = np.concatenate([f.flatten() for f in frames])\n"
            "    return frames, float(all_vals.min()), float(all_vals.max())"
        ),
        md(
            "## Render per-size animations\n"
            "\n"
            "5 sizes × 1 animation each. Each animation is built with "
            "`matplotlib.animation.FuncAnimation`, saved as a GIF at 3 fps "
            "(13 s playback for 40 frames), and embedded inline as an "
            "HTML5 player."
        ),
        code(
            "SIZES = ['70m', '160m', '410m', '1b', '2.8b']\n"
            "MOTIFS = ['induction', 'successor', 's_inhibition']\n"
            "\n"
            "for size in SIZES:\n"
            "    # Determine which motifs have data for this size.\n"
            "    motifs_for_size = []\n"
            "    motif_data = {}\n"
            "    for motif in MOTIFS:\n"
            "        frames, vmin, vmax = build_frames(size, motif)\n"
            "        if frames is None:\n"
            "            continue\n"
            "        motifs_for_size.append(motif)\n"
            "        motif_data[motif] = (frames, vmin, vmax)\n"
            "    n_panels = len(motifs_for_size)\n"
            "    if n_panels == 0:\n"
            "        print(f'{size}: no motif data — skipping')\n"
            "        continue\n"
            "    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5))\n"
            "    if n_panels == 1:\n"
            "        axes = [axes]\n"
            "    ims = []\n"
            "    for ax, motif in zip(axes, motifs_for_size):\n"
            "        frames, vmin, vmax = motif_data[motif]\n"
            "        cmap = 'viridis' if vmin >= 0 else 'RdBu_r'\n"
            "        if cmap == 'RdBu_r':\n"
            "            limit = max(abs(vmin), abs(vmax))\n"
            "            im = ax.imshow(frames[0], vmin=-limit, vmax=limit, cmap=cmap, aspect='auto')\n"
            "        else:\n"
            "            im = ax.imshow(frames[0], vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')\n"
            "        thr = MOTIF_THRESHOLD[motif]\n"
            "        ax.set_title(f'{MOTIF_LABEL[motif]}\\nthreshold = {thr}', fontsize=10)\n"
            "        ax.set_xlabel('head')\n"
            "        ax.set_ylabel('layer')\n"
            "        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)\n"
            "        ims.append(im)\n"
            "    title = fig.suptitle(\n"
            "        f'Pythia-{size} — motif co-evolution at step{STEPS[0]} (frame 1/{len(STEPS)})',\n"
            "        y=1.0,\n"
            "    )\n"
            "    if size == '2.8b' and 's_inhibition' not in motifs_for_size:\n"
            "        # Annotate the omission per §H4-7-supersede\n"
            "        fig.text(0.99, 0.01,\n"
            "                 'S-inhibition omitted: sweep halted at 8/40 cells (§H4-7-supersede)',\n"
            "                 ha='right', va='bottom', fontsize=8, style='italic', color='gray')\n"
            "\n"
            "    def make_update(ims_local, motifs_local, motif_data_local, title_local, size_local):\n"
            "        def _update(frame_idx):\n"
            "            for im, motif in zip(ims_local, motifs_local):\n"
            "                im.set_array(motif_data_local[motif][0][frame_idx])\n"
            "            title_local.set_text(\n"
            "                f'Pythia-{size_local} — motif co-evolution at step{STEPS[frame_idx]} '\n"
            "                f'(frame {frame_idx+1}/{len(STEPS)})'\n"
            "            )\n"
            "            return ims_local + [title_local]\n"
            "        return _update\n"
            "\n"
            "    update = make_update(ims, motifs_for_size, motif_data, title, size)\n"
            "    ani = animation.FuncAnimation(fig, update, frames=len(STEPS), interval=333, blit=False)\n"
            "    out_gif = FIG_DIR / f'motif_evolution_{size}.gif'\n"
            "    ani.save(out_gif, writer=animation.PillowWriter(fps=3))\n"
            "    print(f'  Pythia-{size}: {n_panels} panels, wrote {out_gif.relative_to(REPO)}')\n"
            "    plt.close(fig)\n"
            "    display(HTML(f'<h3>Pythia-{size}</h3>'))\n"
            "    display(HTML(ani.to_jshtml()))"
        ),
        md(
            "## Files written\n"
            "\n"
            "- `notebooks/figures/motif_evolution_70m.gif` (3 panels: ind / suc / si)\n"
            "- `notebooks/figures/motif_evolution_160m.gif` (3 panels)\n"
            "- `notebooks/figures/motif_evolution_410m.gif` (3 panels)\n"
            "- `notebooks/figures/motif_evolution_1b.gif` (3 panels)\n"
            "- `notebooks/figures/motif_evolution_2.8b.gif` (2 panels: ind / suc; "
            "  S-inhibition omitted per §H4-7-supersede)\n"
            "\n"
            "Inline HTML5 players above are interactive (play / pause / scrub) "
            "in JupyterLab and GitHub-rendered notebooks. The standalone GIFs "
            "are for sharing or paper-figure use."
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
