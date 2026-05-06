"""Successor-head detector via cross-category direct logit attribution.

Implements Phase 1.4's locked detector specification (HYPOTHESIS.md §SU-1
through §SU-3). For each candidate head h:

  1. Score the mean cross-category DLA at the END token across 3-context
     comma-separated ordinal prompts in 4 categories — days, months,
     numerals (mixing 1-20 digit form and one-twenty word form within a
     single category), letters.
  2. First-token DLA: target the first token of `f' {item}'` under the
     active tokenizer regardless of total token length (§SU-2).
  3. Score on shuffled prompts (within-category prefix permutation, one
     fixed seed-pinned permutation per (category, base prompt) pair) gives
     the head's per-head shuffled mean cross-category DLA.
  4. Pool the per-head shuffled means across all heads → 95th percentile
     is the threshold τ (§SU-3).

A head's mean cross-category DLA on real prompts that exceeds τ is a
candidate successor head (§SU-3). Validation gate at GPT-2 small (§SU-4):
L9H1 must rank in top-3 AND L9H1 must clear τ.

Validation target: GPT-2 small L9H1, per L (2023) "Mechanistically
interpreting time in GPT-2 small" (LessWrong); cited by Gould et al. 2024
§5 but not their identification. See §SU-0 for the brief-misattribution
correction.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

DAYS: tuple[str, ...] = (
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
)
MONTHS: tuple[str, ...] = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
NUMERALS_DIGITS: tuple[str, ...] = tuple(str(i) for i in range(1, 21))
NUMERALS_WORDS: tuple[str, ...] = (
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
)
LETTERS: tuple[str, ...] = tuple(chr(ord("A") + i) for i in range(26))

CATEGORIES: tuple[str, ...] = ("days", "months", "numerals", "letters")

# Validation targets locked in HYPOTHESIS.md §SU-0.
GPT2_SMALL_L9H1: tuple[int, int] = (9, 1)


@dataclass(frozen=True)
class SuccessorPrompt:
    """One ordinal prompt with its clean and shuffled forms.

    `clean_text` is the verbatim prompt fed to the model; `shuffled_text`
    has the three context items reordered by a deterministic permutation
    that differs from the identity (so the shuffled prompt is never byte-
    identical to the clean prompt). The target is `c4` (the next ordinal
    item after the three contexts), and `target_first_token_id` is the
    first token of `f' {target_word}'` under the tokenizer used at
    construction time.
    """

    clean_text: str
    shuffled_text: str
    target_word: str
    target_first_token_id: int
    target_first_token_str: str
    category: str


def _category_items(category: str) -> tuple[tuple[str, ...], ...]:
    """Return the per-sub-sequence item lists for a category.

    Most categories have one sub-sequence; numerals has two (digits and
    words), each independently traversed for 3-context windows. The cross-
    category aggregation treats both numerals sub-sequences as the same
    category.
    """
    if category == "days":
        return (DAYS,)
    if category == "months":
        return (MONTHS,)
    if category == "numerals":
        return (NUMERALS_DIGITS, NUMERALS_WORDS)
    if category == "letters":
        return (LETTERS,)
    raise ValueError(f"unknown category: {category!r}")


def _shuffle_prefix(
    prefix: tuple[str, str, str], rng: random.Random
) -> tuple[str, str, str]:
    """Return a deterministic permutation of `prefix` that differs from it.

    Three context items have 6 permutations (5 if we exclude identity).
    The local rng is seeded per (category, prompt index) pair so the shuffle
    is reproducible at construction time.
    """
    perms = [p for p in (
        (prefix[0], prefix[1], prefix[2]),
        (prefix[0], prefix[2], prefix[1]),
        (prefix[1], prefix[0], prefix[2]),
        (prefix[1], prefix[2], prefix[0]),
        (prefix[2], prefix[0], prefix[1]),
        (prefix[2], prefix[1], prefix[0]),
    ) if p != prefix]
    return rng.choice(perms)


def build_successor_prompts(
    tokenizer: PreTrainedTokenizerBase, *, seed: int = 0
) -> list[SuccessorPrompt]:
    """Build the locked 4-category prompt set with clean and shuffled forms.

    Per §SU-1: 3-context comma-separated, predict-4th. Per §SU-2: first-
    token DLA target encoded as `tokenizer.encode(f' {target}')[0]`. Per
    §SU-3: within-category prefix permutation, one fixed seed-pinned
    permutation per (category, base prompt) pair, differing from identity.
    """
    prompts: list[SuccessorPrompt] = []
    counter = 0
    for category in CATEGORIES:
        for items in _category_items(category):
            for i in range(len(items) - 3):
                c1, c2, c3, target = items[i], items[i + 1], items[i + 2], items[i + 3]
                clean = f"{c1}, {c2}, {c3}, "
                local_rng = random.Random(seed * 1_000_003 + counter)
                shuffled_prefix = _shuffle_prefix((c1, c2, c3), local_rng)
                shuffled = f"{shuffled_prefix[0]}, {shuffled_prefix[1]}, {shuffled_prefix[2]}, "
                ids = tokenizer.encode(f" {target}", add_special_tokens=False)
                if not ids:
                    raise RuntimeError(
                        f"target word {target!r} encoded to empty token list"
                    )
                first_id = int(ids[0])
                first_str = tokenizer.decode([first_id])
                prompts.append(
                    SuccessorPrompt(
                        clean_text=clean,
                        shuffled_text=shuffled,
                        target_word=target,
                        target_first_token_id=first_id,
                        target_first_token_str=first_str,
                        category=category,
                    )
                )
                counter += 1
    return prompts


_TSV_COLUMNS: tuple[str, ...] = (
    "category",
    "clean_text",
    "shuffled_text",
    "target_word",
    "target_first_token_id",
    "target_first_token_str",
)


def save_successor_prompts(
    prompts: list[SuccessorPrompt],
    path: Path | str,
    *,
    seed: int,
    tokenizer_name: str,
) -> Path:
    """Persist prompts as a TSV with a `#`-prefixed provenance header."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    header = [
        "# Phase 1.4 successor prompt set (HYPOTHESIS.md §SU-1 through §SU-3)",
        f"# Generated: {timestamp}",
        f"# Seed: {seed}",
        f"# Tokenizer: {tokenizer_name}",
        f"# Total prompts: {len(prompts)}",
        f"# Categories: {', '.join(CATEGORIES)}",
        "# Format: 3-context comma-separated, predict 4th. First-token DLA target.",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        for line in header:
            f.write(line + "\n")
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(_TSV_COLUMNS)
        for p in prompts:
            writer.writerow([
                p.category,
                p.clean_text,
                p.shuffled_text,
                p.target_word,
                p.target_first_token_id,
                p.target_first_token_str,
            ])
    return out


@dataclass(frozen=True)
class SuccessorResult:
    """Output of the cross-category DLA screen.

    Per HYPOTHESIS.md §SU-1b (lift-form supersede of §SU-1's raw-DLA score):

    `real_dla[layer, head]` is the per-head mean cross-category DLA on
    clean prompts; `null_dla[layer, head]` is the same statistic on
    within-category-prefix-shuffled prompts. The **gate-relevant scalar**
    is `lift = real_dla − null_dla` per head, exposed as `lift_dla` for
    convenience. Heads with positive `lift` are successor candidates;
    heads with `real_dla` high but `lift` ≈ 0 are category-token-boosters
    that the §SU-1b lift form correctly excludes.

    `per_category_real[cat][layer, head]` and `per_category_null[cat]`
    expose the per-category breakdown — useful for the qualitative cross-
    category-breadth gate (§SU-5 Pythia anchor) and for the proof
    notebook's per-category visualization.

    `lift_threshold` is the 95th percentile of the pooled `lift_dla`
    distribution (one value per head, pooled across all heads), per the
    §SU-1b-3 specification.
    """

    real_dla: torch.Tensor
    null_dla: torch.Tensor
    per_category_real: dict[str, torch.Tensor]
    per_category_null: dict[str, torch.Tensor]
    lift_dla: torch.Tensor
    lift_threshold: float
    n_prompts: int


def _compute_per_prompt_dla(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Per-(layer, head, prompt) DLA at the final token toward target_ids.

    Returns a `(n_layers, n_heads, B)` tensor on CPU. Implementation mirrors
    `tigges_ioi.component_dla` — uses `use_attn_result=True` to expose per-
    head residual contributions, projects each head's last-position output
    onto the unembedding direction for the target token, and accumulates.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    B = int(tokens.shape[0])
    device = next(model.parameters()).device

    W_U = model.W_U
    directions = W_U[:, target_ids.to(W_U.device)].T.to(torch.float32)

    out = torch.zeros(n_layers, n_heads, B, dtype=torch.float32)

    prev_setting = model.cfg.use_attn_result
    model.set_use_attn_result(True)
    model.eval()
    try:
        with torch.no_grad():
            for chunk_start in range(0, B, batch_size):
                idxs = slice(chunk_start, chunk_start + batch_size)
                batch_tokens = tokens[idxs].to(device)
                batch_dirs = directions[idxs].to(device)
                _, cache = model.run_with_cache(
                    batch_tokens,
                    names_filter=lambda name: "attn.hook_result" in name,
                )
                b = int(batch_tokens.shape[0])
                for layer in range(n_layers):
                    result = cache[f"blocks.{layer}.attn.hook_result"][:, -1, :, :]
                    contrib = (result * batch_dirs[:, None, :]).sum(dim=-1)
                    out[layer, :, chunk_start : chunk_start + b] = (
                        contrib.T.to(torch.float32).cpu()
                    )
                del cache
    finally:
        model.set_use_attn_result(prev_setting)

    return out


def successor_screen(
    model: HookedTransformer,
    prompts: list[SuccessorPrompt],
    *,
    batch_size: int = 8,
) -> SuccessorResult:
    """Run the cross-category DLA screen on real and shuffled prompts.

    Tokenizes each prompt's clean and shuffled forms with the model's
    tokenizer (BOS prepended), groups by sequence length so we can batch
    cleanly, and computes per-(layer, head, prompt) DLA at the END token
    toward the target's first token id. Aggregates first per-category
    (mean across prompts in that category), then mean across the four
    categories to give the per-head scalar `real_dla[layer, head]`. Same
    aggregation applied to the shuffled prompts gives `null_dla`.

    The threshold τ is the 95th percentile of the pooled per-head null
    means — pooled at the per-head-mean level (one value per head), not
    per-prompt — per §SU-3.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    clean_token_rows = [
        model.to_tokens(p.clean_text, prepend_bos=True)[0] for p in prompts
    ]
    shuffled_token_rows = [
        model.to_tokens(p.shuffled_text, prepend_bos=True)[0] for p in prompts
    ]

    # BPE is context-sensitive; the same items in different orders can
    # tokenize to different total lengths. Process clean and shuffled
    # variants as independent length-groupings — we only need the END-token
    # DLA from each, not a per-prompt clean-vs-shuffled correspondence.
    real_dla_per_prompt = torch.zeros(n_layers, n_heads, len(prompts), dtype=torch.float32)
    null_dla_per_prompt = torch.zeros(n_layers, n_heads, len(prompts), dtype=torch.float32)

    for variant, token_rows, dest in (
        ("clean", clean_token_rows, real_dla_per_prompt),
        ("shuffled", shuffled_token_rows, null_dla_per_prompt),
    ):
        by_len: dict[int, list[int]] = defaultdict(list)
        for i, t in enumerate(token_rows):
            by_len[int(t.shape[0])].append(i)
        for length, indices in sorted(by_len.items()):
            batch_tokens = torch.stack([token_rows[i] for i in indices])
            target_ids = torch.tensor(
                [prompts[i].target_first_token_id for i in indices], dtype=torch.long
            )
            dla = _compute_per_prompt_dla(
                model, batch_tokens, target_ids, batch_size=batch_size
            )
            for k, i in enumerate(indices):
                dest[:, :, i] = dla[:, :, k]

    per_category_real: dict[str, torch.Tensor] = {}
    per_category_null: dict[str, torch.Tensor] = {}
    for cat in CATEGORIES:
        cat_indices = [i for i, p in enumerate(prompts) if p.category == cat]
        if not cat_indices:
            continue
        per_category_real[cat] = real_dla_per_prompt[:, :, cat_indices].mean(dim=-1)
        per_category_null[cat] = null_dla_per_prompt[:, :, cat_indices].mean(dim=-1)

    real_stack = torch.stack([per_category_real[c] for c in per_category_real])
    null_stack = torch.stack([per_category_null[c] for c in per_category_null])
    real_dla = real_stack.mean(dim=0)
    null_dla = null_stack.mean(dim=0)
    lift_dla = real_dla - null_dla

    lift_threshold = float(torch.quantile(lift_dla.flatten(), 0.95).item())

    return SuccessorResult(
        real_dla=real_dla,
        null_dla=null_dla,
        per_category_real=per_category_real,
        per_category_null=per_category_null,
        lift_dla=lift_dla,
        lift_threshold=lift_threshold,
        n_prompts=len(prompts),
    )


def gate_verdict(
    result: SuccessorResult,
    target: tuple[int, int] = GPT2_SMALL_L9H1,
    *,
    top_k: int = 3,
) -> dict:
    """Apply the §SU-1b-4 conjunctive gate: target in top-k by lift AND lift > τ.

    Returns a dict suitable for printing or pandas serialization. The
    `passes` key is `True` only if both legs hold.
    """
    n_layers, n_heads = result.lift_dla.shape
    flat = result.lift_dla.flatten()
    ranking = torch.argsort(flat, descending=True)
    target_idx = target[0] * n_heads + target[1]
    rank = int((ranking == target_idx).nonzero().item()) + 1
    target_lift = float(flat[target_idx].item())
    target_real = float(result.real_dla[target[0], target[1]].item())
    target_null = float(result.null_dla[target[0], target[1]].item())
    in_top_k = rank <= top_k
    clears_threshold = target_lift > result.lift_threshold
    return dict(
        target_layer=target[0],
        target_head=target[1],
        target_rank_by_lift=rank,
        target_lift=target_lift,
        target_real_dla=target_real,
        target_null_dla=target_null,
        lift_threshold=result.lift_threshold,
        in_top_k=in_top_k,
        clears_threshold=clears_threshold,
        top_k=top_k,
        passes=in_top_k and clears_threshold,
    )
