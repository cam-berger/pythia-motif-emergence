"""Tigges 2024 IOI replication for Pythia-410M.

Reference: Tigges et al., "LLMs and the Abstraction and Reasoning Corpus" /
"Investigating Circuits in Transformer LMs Trained for the IOI Task"
(circuits-over-time, NeurIPS 2024). We replicate the IOI accuracy curve across
Pythia-410M training checkpoints as a methodological gate for our own
emergence-curve work — if our pipeline reproduces the published accuracy
trajectory, downstream sweeps inherit that calibration.

Prompt design (Wang et al. 2023 single-clause template):

    "When {n1} and {n2} went to the {place}, {n3} gave a {obj} to"

ABBA: n1=A, n2=B, n3=B  (positions A B B; answer A; IO=A, S=B)
BABA: n1=B, n2=A, n3=B  (positions B A B; answer A; IO=A, S=B)

In both, A is the unique "indirect object" name (mentioned once before the
answer position) and B is the repeated "subject" name. The model's task is to
predict the IO at the final position. Logit-difference is logit(IO) - logit(S);
accuracy is the fraction of prompts where logit(IO) > logit(S).

Tokenizer filter: every name appears in-context with a leading space (e.g.,
" Mary" after "When", " John" after " and", " John" after "store, "). We keep
only names whose ` Name` form encodes to exactly one GPT-NeoX BPE token, so
single-position scoring at the final logit row is unambiguous.

Divergences from Tigges' published replication (documented for the runner):
  - We use N=200 prompts (Tigges used N=70). Larger N reduces variance in the
    accuracy estimate; should not bias the curve.
  - Our pipeline loads `EleutherAI/pythia-410m-deduped`. Tigges used
    `pythia-410m-no-dropout`. Whether deduped vs no-dropout meaningfully
    changes the IOI emergence curve is an open question — flagged in NOTES.md.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_TOKENIZER_NAME = "EleutherAI/pythia-410m-deduped"

WANG_NAMES: tuple[str, ...] = (
    "Mary", "John", "James", "Robert", "Michael", "William",
    "David", "Joseph", "Thomas", "Charles", "Christopher", "Daniel",
    "Matthew", "Anthony", "Mark", "Donald", "Steven", "Andrew",
    "Kenneth", "George", "Joshua", "Kevin", "Brian", "Edward",
)

WANG_PLACES: tuple[str, ...] = (
    "store", "restaurant", "school", "hospital",
    "office", "house", "church", "airport",
)

WANG_OBJECTS: tuple[str, ...] = (
    "drink", "kiss", "present", "note",
    "bone", "ring", "snack", "book",
)

TemplateKind = Literal["ABBA", "BABA"]

PROMPT_TEMPLATE = "When {n1} and {n2} went to the {place}, {n3} gave a {obj} to"


@dataclass(frozen=True)
class IOIPrompt:
    """One IOI prompt and the token ids needed to score it.

    `text` is fed to the model verbatim and ends without a trailing space; the
    model's next-token distribution at the last position is read against
    `io_token_id` (correct answer) and `s_token_id` (the lure).
    """

    text: str
    io_name: str
    s_name: str
    place: str
    obj: str
    template_kind: TemplateKind
    io_token_id: int
    s_token_id: int


def get_default_tokenizer() -> PreTrainedTokenizerBase:
    """Return Pythia's GPT-NeoX tokenizer."""
    return AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_NAME)


def filter_single_token_names(
    names: tuple[str, ...],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[str, ...]:
    """Keep names whose ` Name` form encodes to exactly one token.

    The IOI template never places a name at sentence-initial position, so the
    leading-space form is the form the model actually sees in every slot.
    """
    kept: list[str] = []
    for name in names:
        ids = tokenizer.encode(f" {name}", add_special_tokens=False)
        if len(ids) == 1:
            kept.append(name)
    return tuple(kept)


def _build_one_prompt(
    a: str,
    b: str,
    place: str,
    obj: str,
    kind: TemplateKind,
    tokenizer: PreTrainedTokenizerBase,
) -> IOIPrompt:
    if kind == "ABBA":
        n1, n2, n3 = a, b, b
    elif kind == "BABA":
        n1, n2, n3 = b, a, b
    else:
        raise ValueError(f"unknown template kind: {kind!r}")
    text = PROMPT_TEMPLATE.format(n1=n1, n2=n2, place=place, n3=n3, obj=obj)
    io_id = tokenizer.encode(f" {a}", add_special_tokens=False)[0]
    s_id = tokenizer.encode(f" {b}", add_special_tokens=False)[0]
    return IOIPrompt(
        text=text,
        io_name=a,
        s_name=b,
        place=place,
        obj=obj,
        template_kind=kind,
        io_token_id=io_id,
        s_token_id=s_id,
    )


def build_ioi_prompts(
    *,
    seed: int = 0,
    n: int = 200,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> list[IOIPrompt]:
    """Generate `n` IOI prompts (50/50 ABBA/BABA), reproducible from seed.

    For each prompt: sample two distinct names, one place, one object, then
    fill the chosen template. The template-kind schedule is half-and-half and
    shuffled. All names are checked single-token at construction time.
    """
    if tokenizer is None:
        tokenizer = get_default_tokenizer()
    if n % 2 != 0:
        raise ValueError(f"n={n} must be even (50/50 ABBA/BABA split).")

    names = filter_single_token_names(WANG_NAMES, tokenizer)
    if len(names) < 2:
        raise RuntimeError(
            f"Need >=2 single-token names; got {len(names)}: {names}"
        )

    rng = random.Random(seed)
    half = n // 2
    schedule: list[TemplateKind] = ["ABBA"] * half + ["BABA"] * half
    rng.shuffle(schedule)

    prompts: list[IOIPrompt] = []
    for kind in schedule:
        a, b = rng.sample(names, 2)
        place = rng.choice(WANG_PLACES)
        obj = rng.choice(WANG_OBJECTS)
        prompts.append(_build_one_prompt(a, b, place, obj, kind, tokenizer))

    for p in prompts:
        for label, name in (("IO", p.io_name), ("S", p.s_name)):
            ids = tokenizer.encode(f" {name}", add_special_tokens=False)
            if len(ids) != 1:
                raise AssertionError(
                    f"{label} name {name!r} not single-token (got {ids})"
                )
    return prompts


_TSV_COLUMNS: tuple[str, ...] = (
    "text",
    "io_name",
    "s_name",
    "place",
    "obj",
    "template_kind",
    "io_token_id",
    "s_token_id",
)


def save_ioi_prompts(
    prompts: list[IOIPrompt],
    path: Path | str,
    *,
    seed: int,
    tokenizer_name: str,
    name_pool: tuple[str, ...],
) -> Path:
    """Write prompts as TSV with `#`-prefixed provenance header."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    header_lines = [
        "# Tigges IOI replication prompt set",
        f"# Generated: {timestamp}",
        f"# Seed: {seed}",
        f"# N: {len(prompts)}",
        f"# Tokenizer: {tokenizer_name}",
        f"# Single-token name pool ({len(name_pool)}): {', '.join(name_pool)}",
        f"# Template: {PROMPT_TEMPLATE!r}",
        "# ABBA: n1=A, n2=B, n3=B (positions A B B; answer=A; IO=A, S=B)",
        "# BABA: n1=B, n2=A, n3=B (positions B A B; answer=A; IO=A, S=B)",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        for line in header_lines:
            f.write(line + "\n")
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(_TSV_COLUMNS)
        for p in prompts:
            writer.writerow([
                p.text,
                p.io_name,
                p.s_name,
                p.place,
                p.obj,
                p.template_kind,
                p.io_token_id,
                p.s_token_id,
            ])
    return out


def load_ioi_prompts(path: Path | str) -> list[IOIPrompt]:
    """Load prompts from a TSV produced by `save_ioi_prompts`."""
    p = Path(path)
    body_lines: list[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                body_lines.append(line)
    reader = csv.DictReader(body_lines, delimiter="\t")
    out: list[IOIPrompt] = []
    for row in reader:
        out.append(
            IOIPrompt(
                text=row["text"],
                io_name=row["io_name"],
                s_name=row["s_name"],
                place=row["place"],
                obj=row["obj"],
                template_kind=row["template_kind"],
                io_token_id=int(row["io_token_id"]),
                s_token_id=int(row["s_token_id"]),
            )
        )
    return out


@dataclass(frozen=True)
class ScoreResult:
    """Per-prompt and aggregate IOI scores from a single forward-pass run.

    `logit_diff[i] = logit_io[i] - logit_s[i]` at the final token position.
    `accuracy` is the gate metric: fraction of prompts where IO outscores S.
    `mean_logit_diff` is the supplementary metric reported alongside.
    """

    logit_io: torch.Tensor
    logit_s: torch.Tensor
    logit_diff: torch.Tensor
    accuracy: float
    mean_logit_diff: float


def score_prompts(
    model: HookedTransformer,
    prompts: list[IOIPrompt],
    *,
    batch_size: int = 16,
) -> ScoreResult:
    """Score a list of IOI prompts on `model` and return per-prompt + aggregate.

    Strategy: tokenize each prompt with the model's tokenizer (BOS prepended),
    group prompts by sequence length so each batch is a clean rectangular
    tensor (no padding), and read the final-position logits at `io_token_id`
    and `s_token_id`. Pythia-410M with N=200 prompts is fast enough on MPS
    that batch_size=16 is comfortable; smaller sizes also work.
    """
    device = next(model.parameters()).device
    n = len(prompts)
    logit_io = torch.empty(n, dtype=torch.float32)
    logit_s = torch.empty(n, dtype=torch.float32)

    tokens_per_prompt: list[torch.Tensor] = [
        model.to_tokens(p.text, prepend_bos=True)[0] for p in prompts
    ]
    by_len: dict[int, list[int]] = {}
    for i, toks in enumerate(tokens_per_prompt):
        by_len.setdefault(int(toks.shape[0]), []).append(i)

    model.eval()
    with torch.no_grad():
        for length, indices in by_len.items():
            for chunk_start in range(0, len(indices), batch_size):
                idxs = indices[chunk_start : chunk_start + batch_size]
                batch = torch.stack([tokens_per_prompt[i] for i in idxs]).to(device)
                logits = model(batch)
                last = logits[:, -1, :].to(torch.float32).cpu()
                for j, i in enumerate(idxs):
                    p = prompts[i]
                    logit_io[i] = last[j, p.io_token_id]
                    logit_s[i] = last[j, p.s_token_id]

    logit_diff = logit_io - logit_s
    accuracy = float((logit_diff > 0).to(torch.float32).mean().item())
    mean_logit_diff = float(logit_diff.mean().item())
    return ScoreResult(
        logit_io=logit_io,
        logit_s=logit_s,
        logit_diff=logit_diff,
        accuracy=accuracy,
        mean_logit_diff=mean_logit_diff,
    )


def component_dla(
    model: HookedTransformer,
    prompts: list[IOIPrompt],
    *,
    batch_size: int = 8,
) -> torch.Tensor:
    """Per-(layer, head) DLA contribution to the logit-diff at the final position.

    For each prompt, the head's output at the final position is projected onto
    the unembedding direction `W_U[:, IO] - W_U[:, S]`. Averaging across prompts
    gives a per-head map of who pushes logit(IO) above logit(S) — Wang's Name
    Mover heads should be strongly positive, S-Inhibition heads should be
    positive (they suppress S), Negative Name Movers should be strongly
    negative.

    Returns:
        Tensor of shape `(n_layers, n_heads)` with the mean DLA contribution
        to the IOI logit-diff. Positive = pushes IO over S.

    Implementation note: enables `use_attn_result=True` for the duration of
    the call (and restores the previous setting). Memory cost on Pythia-410M
    is modest (~250MB at batch_size=8); reduce `batch_size` if MPS OOMs.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    device = next(model.parameters()).device
    W_U = model.W_U

    tokens_per_prompt: list[torch.Tensor] = [
        model.to_tokens(p.text, prepend_bos=True)[0] for p in prompts
    ]
    by_len: dict[int, list[int]] = {}
    for i, toks in enumerate(tokens_per_prompt):
        by_len.setdefault(int(toks.shape[0]), []).append(i)

    accumulator = torch.zeros(n_layers, n_heads, dtype=torch.float32)

    prev_use_attn_result = model.cfg.use_attn_result
    model.set_use_attn_result(True)
    model.eval()
    try:
        with torch.no_grad():
            for length, indices in by_len.items():
                for chunk_start in range(0, len(indices), batch_size):
                    idxs = indices[chunk_start : chunk_start + batch_size]
                    batch = torch.stack([tokens_per_prompt[i] for i in idxs]).to(device)
                    directions = torch.stack(
                        [
                            W_U[:, prompts[i].io_token_id]
                            - W_U[:, prompts[i].s_token_id]
                            for i in idxs
                        ]
                    )  # (B, d_model)
                    _, cache = model.run_with_cache(
                        batch,
                        names_filter=lambda name: "attn.hook_result" in name,
                    )
                    for layer in range(n_layers):
                        result = cache[f"blocks.{layer}.attn.hook_result"][
                            :, -1, :, :
                        ]  # (B, n_heads, d_model)
                        contrib = (result * directions[:, None, :]).sum(dim=-1)
                        accumulator[layer] += (
                            contrib.sum(dim=0).to(torch.float32).cpu()
                        )
                    del cache
    finally:
        model.set_use_attn_result(prev_use_attn_result)

    accumulator /= len(prompts)
    return accumulator
