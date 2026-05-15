"""Previous-token attention head detector (atlas-v1 head-family inventory).

Reference: Anthropic, "A Mathematical Framework for Transformer Circuits"
(2021). A previous-token head attends from query position p to key position
p-1 — i.e., the diagonal one below the main diagonal of the attention
pattern.

Operational definition (PROJECT_BRIEF.md, atlas-v1):
    For each (layer, head), compute the mean over (sequence, query
    position p ∈ [1, seq_len-1]) of attention[batch, head, p, p-1] on
    Olsson random-token-repetition sequences with BOS prepended. The
    sequence construction matches the induction detector
    (``build_repetition_sequences``) so atlas detectors share an
    evaluation surface.

The exploratory threshold (0.20) lives in ``src.atlas.thresholds`` —
"this head puts 20%+ of its attention mass on the immediately preceding
token, averaged across positions and sequences."
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from src.atlas.thresholds import PREVIOUS_TOKEN
from src.detectors.induction import build_repetition_sequences
from src.detectors.protocol import PerHeadScores


def previous_token_score(
    model: HookedTransformer,
    *,
    n_sequences: int = 50,
    seq_len: int = 100,
    seed: int = 0,
    batch_size: int = 8,
) -> torch.Tensor:
    """Compute the per-(layer, head) previous-token attention score.

    Builds ``n_sequences`` Olsson repetition sequences of length
    ``seq_len`` (matching the induction detector's evaluation surface),
    prepends BOS, runs the model with attention-pattern caching, and
    returns the mean of ``pattern[..., p, p-1]`` over
    ``p ∈ [1, seq_len-1]`` and over sequences.

    Args:
        model: A loaded HookedTransformer.
        n_sequences: number of random sequences to average over.
        seq_len: length of each sequence *before* BOS prepending.
        seed: RNG seed for sequence generation.
        batch_size: forward-pass batch size.

    Returns:
        Float32 CPU tensor of shape ``(n_layers, n_heads)`` with values in
        ``[0, 1]``.
    """
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2, got {seq_len}")

    vocab_size = int(model.cfg.d_vocab)
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    bos_id = int(model.tokenizer.bos_token_id)

    rng = torch.Generator(device="cpu").manual_seed(seed)
    raw_tokens = build_repetition_sequences(n_sequences, seq_len, vocab_size, rng)
    # BOS prepend — matches the induction detector's run-time prompt shape.
    bos_col = torch.full((n_sequences, 1), bos_id, dtype=torch.long)
    tokens = torch.cat([bos_col, raw_tokens], dim=1)
    full_len = tokens.shape[1]  # seq_len + 1

    device = next(model.parameters()).device

    score_sum = torch.zeros((n_layers, n_heads), dtype=torch.float32)
    n_seqs_seen = 0

    with torch.no_grad():
        for batch_start in range(0, n_sequences, batch_size):
            batch = tokens[batch_start : batch_start + batch_size].to(device)
            _, cache = model.run_with_cache(
                batch,
                names_filter=lambda name: "hook_pattern" in name,
                return_type=None,
            )
            for layer in range(n_layers):
                pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
                # pattern: (batch, n_heads, full_len, full_len).
                # pattern[..., p, p-1] for p in [1, full_len-1] is the
                # sub-diagonal — slice it directly.
                sub_diag = pattern.diagonal(offset=-1, dim1=-2, dim2=-1)
                # sub_diag: (batch, n_heads, full_len - 1), indexed by p-1
                # for p ∈ [1, full_len-1].
                per_batch_seq = sub_diag.mean(dim=-1).to(torch.float32).cpu()
                # per_batch_seq: (batch, n_heads).
                score_sum[layer] += per_batch_seq.sum(dim=0)
            n_seqs_seen += batch.shape[0]
            del cache

    scores = score_sum / max(n_seqs_seen, 1)
    return scores


class PreviousTokenDetector:
    """Detector-protocol adapter for the previous-token attention motif."""

    motif = "previous_token"
    name = "previous_token_attention"

    def __init__(
        self,
        *,
        n_sequences: int = 50,
        seq_len: int = 100,
        seed: int = 0,
        batch_size: int = 8,
    ):
        self.threshold = PREVIOUS_TOKEN
        self.n_sequences = n_sequences
        self.seq_len = seq_len
        self.seed = seed
        self.batch_size = batch_size

    def score(self, model) -> PerHeadScores:
        scores = previous_token_score(
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
