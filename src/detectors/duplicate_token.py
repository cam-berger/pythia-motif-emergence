"""Duplicate-token attention-head detector (atlas-v1 head-family inventory).

Reference: Olsson et al. 2022 — duplicate-token heads attend, at query
position p, to a previous occurrence of the same token. We reuse the
Olsson repetition-sequence construction from `src.detectors.induction`:
tokens 0..(half-1) are random; tokens half..(seq_len-1) are a copy. By
construction, for any p in the second half, tokens[p] == tokens[p - half],
so attention from p to (p - half) is attention to the previous occurrence
of the same token — a fixed-offset slice with offset = seq_len/2.

Threshold: ``DUPLICATE_TOKEN`` (atlas-v1 exploratory), see
``src.atlas.thresholds``.
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from src.atlas.thresholds import DUPLICATE_TOKEN
from src.detectors.induction import build_repetition_sequences
from src.detectors.protocol import PerHeadScores


def duplicate_token_score(
    model: HookedTransformer,
    *,
    n_sequences: int = 50,
    seq_len: int = 100,
    seed: int = 0,
    batch_size: int = 8,
) -> torch.Tensor:
    """Compute per-(layer, head) duplicate-token attention scores.

    For each query position p in [half, seq_len-1] (where half = seq_len/2),
    the duplicate target key index is (p - half) — the previous occurrence
    of the same token by Olsson-repetition construction. Score for a head
    is the mean of attention[batch, head, p, p - half] over (sequence,
    second-half position p).

    Args:
        model: A loaded HookedTransformer.
        n_sequences: number of random repetition sequences (default 50).
        seq_len: total sequence length; must be even (default 100). The
            duplicate-by-construction offset is seq_len/2.
        seed: RNG seed for sequence generation.
        batch_size: forward-pass batch size.

    Returns:
        torch.Tensor of shape (n_layers, n_heads) on CPU, dtype float32.
    """
    if seq_len % 2 != 0:
        raise ValueError(
            f"seq_len must be even, got {seq_len} — needs duplicate by construction"
        )
    half = seq_len // 2
    vocab_size = int(model.cfg.d_vocab)
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)

    rng = torch.Generator(device="cpu").manual_seed(seed)
    tokens = build_repetition_sequences(n_sequences, seq_len, vocab_size, rng)

    # BOS prepending — consistent with prefix_matching_score. HookedTransformer
    # honours ``cfg.default_prepend_bos`` inside ``run_with_cache`` when the
    # input is a string; for integer tokens we pass through as-is. The
    # duplicate-by-construction relation tokens[p] == tokens[p - half] holds
    # for p in [half, seq_len-1] with offset half = seq_len/2.
    second_half_positions = torch.arange(half, seq_len, dtype=torch.long)
    target_positions = second_half_positions - half  # fixed offset = seq_len/2

    score_sum = torch.zeros((n_layers, n_heads), dtype=torch.float32)
    n_terms = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_start in range(0, n_sequences, batch_size):
            batch = tokens[batch_start : batch_start + batch_size].to(device)
            _, cache = model.run_with_cache(
                batch,
                names_filter=lambda name: "hook_pattern" in name,
                return_type=None,
            )
            q_idx = second_half_positions.to(device)
            k_idx = target_positions.to(device)
            for layer in range(n_layers):
                pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
                # pattern: (batch, n_heads, seq_len+1, seq_len+1)
                # Fixed-offset slice: attention from q_idx[i] to k_idx[i].
                # Shape: (batch, n_heads, half).
                gathered = pattern[:, :, q_idx, k_idx]
                per_batch_seq = gathered.mean(dim=-1).to(torch.float32).cpu()
                # Sum over sequences in this batch.
                score_sum[layer] += per_batch_seq.sum(dim=0)
            n_terms += batch.shape[0]

    return score_sum / max(n_terms, 1)


class DuplicateTokenDetector:
    """Detector-protocol adapter for the duplicate-token attention motif."""

    motif = "duplicate_token"
    name = "duplicate_token_attention"

    def __init__(
        self,
        *,
        n_sequences: int = 50,
        seq_len: int = 100,
        seed: int = 0,
        batch_size: int = 8,
    ):
        self.threshold = DUPLICATE_TOKEN
        self.n_sequences = n_sequences
        self.seq_len = seq_len
        self.seed = seed
        self.batch_size = batch_size

    def score(self, model) -> PerHeadScores:
        scores = duplicate_token_score(
            model,
            n_sequences=self.n_sequences,
            seq_len=self.seq_len,
            seed=self.seed,
            batch_size=self.batch_size,
        )
        return PerHeadScores(
            scores=scores,
            motif=self.motif,
            detector_name=self.name,
            aux=None,
        )
