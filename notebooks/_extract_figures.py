"""Extract every embedded PNG figure from the four executed notebooks.

Output: data/figures/<notebook_stem>__cell<NN>__fig<MM>.png
Also prints a short manifest with the index, source notebook, and the markdown
cell title that immediately precedes each figure (useful caption guess).
"""

from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NOTEBOOKS = [
    "h1c_ordering_test.ipynb",
    "induction_full_sweep.ipynb",
    "successor_full_sweep.ipynb",
    "s_inhibition_full_sweep.ipynb",
    "causal_dependence.ipynb",
]


def first_heading(md_text: str) -> str | None:
    for line in md_text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return re.sub(r"^#+\s*", "", line).strip()
    # otherwise first non-empty line
    for line in md_text.splitlines():
        if line.strip():
            return line.strip()[:120]
    return None


def main() -> None:
    manifest = []
    for nb_name in NOTEBOOKS:
        nb_path = REPO / "notebooks" / nb_name
        nb = json.loads(nb_path.read_text())
        last_md = None
        fig_count_for_nb = 0
        for ci, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "markdown":
                last_md = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
                continue
            if cell["cell_type"] != "code":
                continue
            for oi, out in enumerate(cell.get("outputs", [])):
                data = out.get("data", {})
                png = data.get("image/png")
                if not png:
                    continue
                stem = nb_path.stem
                fig_count_for_nb += 1
                name = f"{stem}__cell{ci:02d}__fig{fig_count_for_nb:02d}.png"
                out_path = OUT / name
                out_path.write_bytes(base64.b64decode(png))
                cap = first_heading(last_md) if last_md else None
                manifest.append({
                    "notebook": nb_name,
                    "cell_index": ci,
                    "fig_index_in_notebook": fig_count_for_nb,
                    "preceding_heading": cap,
                    "filename": name,
                    "size_kb": round(out_path.stat().st_size / 1024, 1),
                })
    (OUT / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest)} figures to {OUT}")
    for m in manifest:
        print(f"  {m['filename']:<70s}  ({m['size_kb']:>5.1f} KB)  {m['preceding_heading']}")


if __name__ == "__main__":
    main()
