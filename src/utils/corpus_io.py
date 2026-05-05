"""Load the canonical corpus built by notebooks/_build_canonical_corpus.py.

The corpus is plain UTF-8 text with passage-delimiter header lines:

    # ... (file header comment lines)

    === <title> | revid=<int> | rev_ts=<ISO8601> ===
    <passage text, possibly multi-line>

    === <next title> | ... ===
    ...

`load_corpus()` returns a list of `Passage` objects (title, revid, rev_ts,
text). Tokenization is the caller's responsibility — different models in this
project use different tokenizers (GPT-2 BPE for GPT-2 small calibration;
GPT-NeoX BPE for Pythia 70M/160M/410M), and the corpus is defined as text so
the same passages can drive every detector run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "corpora"
    / "copy_suppression_corpus.txt"
)

_HEADER_RE = re.compile(
    r"^=== (?P<title>.+?) \| revid=(?P<revid>\d+) \| rev_ts=(?P<rev_ts>[^=]+?) ===$"
)


@dataclass(frozen=True)
class Passage:
    title: str
    revid: int
    rev_ts: str
    text: str


def load_corpus(path: Path | str | None = None) -> list[Passage]:
    p = Path(path) if path is not None else DEFAULT_PATH
    raw = p.read_text(encoding="utf-8")
    lines = raw.splitlines()

    passages: list[Passage] = []
    cur_meta: tuple[str, int, str] | None = None
    cur_lines: list[str] = []

    def flush() -> None:
        nonlocal cur_meta, cur_lines
        if cur_meta is None:
            return
        text = "\n".join(cur_lines).strip()
        if text:
            title, revid, rev_ts = cur_meta
            passages.append(Passage(title=title, revid=revid, rev_ts=rev_ts, text=text))
        cur_meta = None
        cur_lines = []

    for line in lines:
        if line.startswith("#"):
            continue
        m = _HEADER_RE.match(line.strip())
        if m:
            flush()
            cur_meta = (m["title"], int(m["revid"]), m["rev_ts"].strip())
            cur_lines = []
            continue
        if cur_meta is not None:
            cur_lines.append(line)

    flush()
    if not passages:
        raise ValueError(f"No passages parsed from {p}")
    return passages
