"""Build data/corpora/copy_suppression_corpus.txt.

The canonical corpus for copy-suppression detector runs across the project
(GPT-2 calibration, Pythia 410M anchor, Pythia emergence sweep, proof
notebooks). Locked in by Q3 of the Day 3 design grilling: ~30-50 Wikipedia
featured-article opening passages, ~6k tokens total, re-tokenized per model.

Provenance: each passage's revision ID and fetch timestamp are recorded in the
output file header so the corpus is reproducible — anyone re-running this
script can fetch the same revisions from Wikipedia's MediaWiki API. The
*committed* corpus file is the canonical snapshot; the script is just how it's
built or rebuilt.

Run:
    uv run python notebooks/_build_canonical_corpus.py
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Per-passage truncation target. Wikipedia lead-intros average ~600 tokens
# (GPT-NeoX), too long for 30-50-passage corpus targeting ~6-8k tokens total.
# 150 words at sentence boundary lands in the 180-220 token range.
TARGET_WORDS_PER_PASSAGE = 150

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "corpora" / "copy_suppression_corpus.txt"

# Curated for diversity + high duplicate-token density (named entities,
# pronouns, content words that repeat across an encyclopedic intro paragraph).
# Avoid topics where the lead is heavy on lists, code, dialogue.
ARTICLES: list[str] = [
    # People
    "Albert Einstein",
    "Marie Curie",
    "Charles Darwin",
    "Leonardo da Vinci",
    "Mahatma Gandhi",
    "Jane Austen",
    "Ada Lovelace",
    "William Shakespeare",
    "Ludwig van Beethoven",
    "Alexander the Great",
    # Places
    "Tokyo",
    "Mount Everest",
    "Pacific Ocean",
    "Iceland",
    "Stonehenge",
    "Eiffel Tower",
    "Nile",
    "Library of Alexandria",
    "Antarctica",
    "Sahara",
    # Science / concepts
    "Photosynthesis",
    "Quantum mechanics",
    "DNA",
    "Black hole",
    "Solar System",
    "Theory of relativity",
    "Antibiotic",
    "Climate change",
    "Penicillin",
    "Volcano",
    # Culture / history
    "World War II",
    "The Beatles",
    "Roman Empire",
    "Internet",
    "Coffee",
    "Industrial Revolution",
    "Buddhism",
    "Mona Lisa",
    "Photography",
    "Octopus",
]

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "pythia-motif-emergence-research/0.1 "
    "(https://github.com/cam-berger/pythia-motif-emergence; "
    "evike.cb@gmail.com) python-urllib/3.12"
)


def fetch_extract(title: str) -> tuple[str, int, str]:
    """Return (plain-text lead extract, revision ID, ISO timestamp)."""
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "extracts|revisions",
        "titles": title,
        "explaintext": "1",
        "exintro": "1",
        "rvprop": "ids|timestamp",
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    pages = data["query"]["pages"]
    if not pages:
        raise RuntimeError(f"No pages returned for title={title!r}")
    page = pages[0]
    if page.get("missing"):
        raise RuntimeError(f"Wikipedia page missing for title={title!r}")
    extract = page["extract"].strip()
    rev = page["revisions"][0]
    return extract, int(rev["revid"]), rev["timestamp"]


def normalize(text: str) -> str:
    # Strip excessive blank lines; keep paragraph structure but compact.
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[str] = []
    blank = False
    for ln in lines:
        if not ln:
            if not blank and out:
                out.append("")
            blank = True
        else:
            out.append(ln)
            blank = False
    return "\n".join(out).strip()


# Sentence-boundary regex: split after . ! ? when followed by whitespace +
# capital letter or end of string. Conservative; over-keeps boundaries (e.g.,
# "Mr. Smith") which is fine because we walk sentence-by-sentence accumulating
# word counts — we don't index into specific sentences.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def truncate_at_sentence(text: str, target_words: int) -> str:
    """Return the longest prefix of `text` ending at a sentence boundary
    such that the prefix contains <= target_words + first-overshoot words."""
    sentences = _SENT_SPLIT.split(text)
    out: list[str] = []
    word_count = 0
    for sent in sentences:
        sw = len(sent.split())
        if out and word_count + sw > target_words:
            # Already past the line on the next sentence; stop here.
            break
        out.append(sent)
        word_count += sw
        if word_count >= target_words:
            break
    return " ".join(out).strip()


def build() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fetched: list[tuple[str, int, str, str]] = []  # (title, revid, ts, text)
    for i, title in enumerate(ARTICLES, 1):
        print(f"[{i:2d}/{len(ARTICLES)}] {title}", flush=True)
        try:
            extract, revid, ts = fetch_extract(title)
        except Exception as e:
            print(f"  WARNING: failed ({e!r}); skipping", file=sys.stderr)
            continue
        text = normalize(extract)
        if not text:
            print(f"  WARNING: empty extract for {title!r}; skipping", file=sys.stderr)
            continue
        text = truncate_at_sentence(text, TARGET_WORDS_PER_PASSAGE)
        fetched.append((title, revid, ts, text))
        time.sleep(0.1)  # courtesy delay; Wikipedia is generous but not infinite

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = [
        "# Canonical corpus for copy-suppression detector runs.",
        "#",
        "# Built by notebooks/_build_canonical_corpus.py",
        f"# Build timestamp (UTC): {now_iso}",
        f"# Source: English Wikipedia, MediaWiki API ({API})",
        f"# Articles fetched: {len(fetched)} of {len(ARTICLES)} attempted",
        f"# Per-passage truncation: first <= {TARGET_WORDS_PER_PASSAGE} words "
        "rounded up to next sentence boundary",
        "#",
        "# Format: each passage delimited by a header line of the form",
        "#   === <title> | revid=<int> | rev_ts=<ISO8601> ===",
        "# followed by the plain-text lead intro on subsequent lines and a blank",
        "# line separator. Use src.utils.corpus_io.load_corpus() to parse.",
        "#",
        "# Reproducibility: each passage's revid is recorded; passages can be",
        "# refetched verbatim via",
        "#   curl 'https://en.wikipedia.org/w/api.php?action=query&format=json"
        "&prop=extracts&explaintext=1&exintro=1&revids=<revid>'",
        "",
    ]
    body: list[str] = []
    for title, revid, ts, text in fetched:
        body.append(f"=== {title} | revid={revid} | rev_ts={ts} ===")
        body.append(text)
        body.append("")

    OUT.write_text("\n".join(header + body), encoding="utf-8")
    n_chars = sum(len(t) for _, _, _, t in fetched)
    print(
        f"\nWrote {OUT.relative_to(ROOT)}: {len(fetched)} passages, "
        f"{n_chars} chars (~{n_chars // 4} tokens by 4-chars/token rule of thumb)"
    )


if __name__ == "__main__":
    build()
