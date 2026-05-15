"""Delimiter-attention head detector for the atlas-v1 head-family inventory.

Detects heads that concentrate attention mass on delimiter tokens
(comma, period, newline) in natural-prose context. Unlike the other 4
atlas detectors, this one needs natural text (random tokens contain no
delimiters) — it loads a small synthetic-prose corpus from
``data/atlas/delimiter_corpus.json``.

Per src.atlas.thresholds.DELIMITER (atlas-v1 exploratory): a head
"passes" when ≥20% of its attention mass lands on delimiter positions.
Mass-fraction normalization (not mean-per-delimiter) makes the score
invariant to how many delimiters happen to appear in any given sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.atlas.thresholds import DELIMITER
from src.detectors.protocol import PerHeadScores


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_PATH = REPO_ROOT / "data" / "atlas" / "delimiter_corpus.json"

DELIMITER_CHARS = (",", ".", "\n")


def _collect_delimiter_token_ids(model) -> set[int]:
    """Resolve {',', '.', '\\n'} plus their leading-space variants to token IDs.

    Uses ``tokenizer.encode(s, add_special_tokens=False)`` directly rather than
    HookedTransformer's ``to_single_token``: the latter is unreliable across
    Pythia size variants because the tokenizer's ``add_special_tokens`` default
    is not consistent across checkpoints (410m encode prepends BOS, 2.8b does
    not — and HookedTransformer's ``to_tokens`` over-strips on 2.8b, returning
    an empty sequence for single-char inputs).
    """
    candidates = list(DELIMITER_CHARS) + [" ,", " ."]
    ids: set[int] = set()
    for s in candidates:
        toks = model.tokenizer.encode(s, add_special_tokens=False)
        if len(toks) == 1:
            ids.add(int(toks[0]))
    return ids


def _build_token_windows(model, corpus: list[str], n_sequences: int, seq_len: int, seed: int) -> torch.Tensor:
    """Tokenize each corpus string with BOS, then slice/pad to (n_sequences, seq_len) deterministically.

    Tokenization goes through ``model.tokenizer.encode(text, add_special_tokens=False)``
    + explicit BOS prepend, sidestepping HookedTransformer's ``to_tokens`` whose
    BOS-stripping is inconsistent across Pythia size variants (see
    ``_collect_delimiter_token_ids``).
    """
    rng = torch.Generator(device="cpu").manual_seed(seed)
    bos_id = model.tokenizer.bos_token_id if model.tokenizer.bos_token_id is not None else 0
    pad_id = model.tokenizer.eos_token_id if model.tokenizer.eos_token_id is not None else 0

    all_tokens: list[torch.Tensor] = []
    for text in corpus:
        body_ids = model.tokenizer.encode(text, add_special_tokens=False)
        # Explicit BOS prepend so we control the prefix; position 0 is always BOS.
        toks = torch.tensor([bos_id] + body_ids, dtype=torch.long)
        all_tokens.append(toks)

    windows = torch.zeros(n_sequences, seq_len, dtype=torch.long)
    order = torch.randperm(len(corpus), generator=rng).tolist()
    for i in range(n_sequences):
        toks = all_tokens[order[i % len(corpus)]]
        if toks.shape[0] >= seq_len:
            start = int(torch.randint(0, toks.shape[0] - seq_len + 1, (1,), generator=rng).item())
            window = toks[start:start + seq_len].clone()
            window[0] = bos_id  # ensure BOS at pos 0
        else:
            window = torch.full((seq_len,), pad_id, dtype=torch.long)
            window[0] = bos_id
            body = toks[1:]  # skip the BOS we prepended; cycle the content
            for j in range(1, seq_len):
                window[j] = body[(j - 1) % body.shape[0]]
        windows[i] = window
    return windows


def delimiter_attention_score(
    model,
    *,
    corpus_path: Path | None = None,
    n_sequences: int = 50,
    seq_len: int = 256,
    seed: int = 0,
    batch_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (scores, n_delimiter_positions_per_sample).

    scores: (n_layers, n_heads) — mean over sequences of (attention-mass-on-delimiter /
            attention-mass-on-non-self).
    n_delim: (n_sequences,) — count of delimiter positions per sample (diagnostic).
    """
    path = corpus_path or DEFAULT_CORPUS_PATH
    corpus = json.loads(path.read_text())
    if not corpus:
        raise ValueError(f"delimiter corpus at {path} is empty")

    device = next(model.parameters()).device
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)

    tokens = _build_token_windows(model, corpus, n_sequences, seq_len, seed)
    delim_ids = _collect_delimiter_token_ids(model)

    # Per-sample delimiter masks (which key positions are delimiters), excluding pos 0 (BOS).
    delim_mask = torch.zeros(n_sequences, seq_len, dtype=torch.bool)
    for i in range(n_sequences):
        for tid in delim_ids:
            delim_mask[i] |= (tokens[i] == tid)
        delim_mask[i, 0] = False  # never count BOS as a delimiter
    n_delim = delim_mask.sum(dim=1).to(torch.int64)

    # Accumulators.
    score_sum = torch.zeros(n_layers, n_heads, dtype=torch.float32)
    n_terms_used = 0

    model.eval()
    with torch.no_grad():
        for b0 in range(0, n_sequences, batch_size):
            b1 = min(b0 + batch_size, n_sequences)
            batch_tokens = tokens[b0:b1].to(device)
            batch_delim_mask = delim_mask[b0:b1].to(device)  # (B, K)
            pattern_names = [f"blocks.{L}.attn.hook_pattern" for L in range(n_layers)]
            _, cache = model.run_with_cache(
                batch_tokens, names_filter=lambda n: n in pattern_names
            )
            for L in range(n_layers):
                pat = cache[pattern_names[L]]  # (B, H, Q, K)
                # mass on delimiter keys per query — exclude self by zeroing diagonal.
                B, H, Q, K = pat.shape
                pat_no_self = pat.clone()
                diag_idx = torch.arange(min(Q, K), device=device)
                pat_no_self[:, :, diag_idx, diag_idx] = 0.0

                # Skip query position 0 (BOS) since it has no context to attend to.
                pat_q = pat_no_self[:, :, 1:, :]                  # (B, H, Q-1, K)
                mass_total = pat_q.sum(dim=-1)                    # (B, H, Q-1)

                key_mask = batch_delim_mask[:, None, None, :].expand(-1, H, Q - 1, -1)
                mass_delim = (pat_q * key_mask.to(pat_q.dtype)).sum(dim=-1)  # (B, H, Q-1)

                # frac per (sample, head, query). Set frac=0 where mass_total=0 (shouldn't happen).
                eps = 1e-12
                frac = mass_delim / mass_total.clamp(min=eps)
                # Skip samples with zero delimiter positions to avoid degenerate 0 entries.
                sample_valid = batch_delim_mask.any(dim=-1)        # (B,)
                if sample_valid.any():
                    per_sample_per_head = frac.mean(dim=-1)        # (B, H) — mean over queries
                    score_sum[L] += per_sample_per_head[sample_valid].sum(dim=0).to(torch.float32).cpu()
            n_terms_used += int(sample_valid.sum().item())
            del cache

    scores = score_sum / max(n_terms_used, 1)
    return scores, n_delim


class DelimiterDetector:
    """Detector-protocol adapter for delimiter / punctuation attention heads."""

    motif = "delimiter"
    name = "delimiter_attention"

    def __init__(
        self,
        *,
        corpus_path: Path | None = None,
        n_sequences: int = 50,
        seq_len: int = 256,
        seed: int = 0,
        batch_size: int = 4,
    ):
        self.threshold = DELIMITER
        self.corpus_path = corpus_path
        self.n_sequences = n_sequences
        self.seq_len = seq_len
        self.seed = seed
        self.batch_size = batch_size

    def score(self, model):
        scores, n_delim = delimiter_attention_score(
            model,
            corpus_path=self.corpus_path,
            n_sequences=self.n_sequences,
            seq_len=self.seq_len,
            seed=self.seed,
            batch_size=self.batch_size,
        )
        return PerHeadScores(
            scores=scores,
            motif=self.motif,
            detector_name=self.name,
            aux=dict(n_delimiter_positions=n_delim),
        )
