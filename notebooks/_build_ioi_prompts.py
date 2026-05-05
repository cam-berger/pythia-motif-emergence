"""Generate the canonical Tigges IOI prompt set.

Writes `data/prompts/ioi_prompts.tsv` (committed). Re-running with the same
seed reproduces the same file byte-for-byte except for the timestamp header.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.replication.tigges_ioi import (  # noqa: E402
    DEFAULT_TOKENIZER_NAME,
    WANG_NAMES,
    build_ioi_prompts,
    filter_single_token_names,
    get_default_tokenizer,
    save_ioi_prompts,
)

OUT_PATH = REPO_ROOT / "data" / "prompts" / "ioi_prompts.tsv"
SEED = 0
N = 200


def main() -> None:
    tokenizer = get_default_tokenizer()
    name_pool = filter_single_token_names(WANG_NAMES, tokenizer)
    print(
        f"Filtered Wang names: kept {len(name_pool)}/{len(WANG_NAMES)} as "
        f"single-token under {DEFAULT_TOKENIZER_NAME}"
    )
    print(f"  kept:    {', '.join(name_pool)}")
    dropped = tuple(n for n in WANG_NAMES if n not in name_pool)
    if dropped:
        print(f"  dropped: {', '.join(dropped)}")

    prompts = build_ioi_prompts(seed=SEED, n=N, tokenizer=tokenizer)
    abba = sum(1 for p in prompts if p.template_kind == "ABBA")
    baba = sum(1 for p in prompts if p.template_kind == "BABA")
    print(f"Generated {len(prompts)} prompts (ABBA={abba}, BABA={baba}).")

    save_ioi_prompts(
        prompts,
        OUT_PATH,
        seed=SEED,
        tokenizer_name=DEFAULT_TOKENIZER_NAME,
        name_pool=name_pool,
    )
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")

    print()
    print("First 3 prompts:")
    for p in prompts[:3]:
        print(f"  [{p.template_kind}] {p.text!r}")
        print(
            f"    IO={p.io_name!r} (id={p.io_token_id})  "
            f"S={p.s_name!r} (id={p.s_token_id})"
        )


if __name__ == "__main__":
    main()
