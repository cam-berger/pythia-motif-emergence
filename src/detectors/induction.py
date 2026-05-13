"""Induction-head detector via Olsson's prefix-matching score.

Reference: Olsson et al. 2022, "In-context Learning and Induction Heads."

Construction (PROJECT_BRIEF.md §4):
    Build random-token-repetition sequences of length 100 with a single
    repeat: tokens 0..49 are random, then tokens 50..99 = tokens 0..49.
    For each (layer, head), compute the mean attention from positions 51..99
    to the position immediately following the previous occurrence of the
    current token. Average over many random sequences.

A head with prefix-matching score > 0.3 is a candidate induction head; > 0.5
is a strong induction head. Threshold for the Day 1 validation gate: ≥1 head
with score > 0.5 on Pythia-410M @ step143000.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer


@dataclass(frozen=True)
class PrefixMatchingResult:
    """Per-(layer, head) prefix-matching scores.

    `scores` shape: (n_layers, n_heads). Values in [0, 1].

    When `prefix_matching_score` is called with `return_per_sequence=True`,
    `per_sequence_scores` is populated with shape
    `(n_sequences, n_layers, n_heads)` — the per-sequence per-head score
    before averaging across sequences. Phase 2 bootstrap uses this to
    resample sequences with replacement and recompute the aggregate.
    """

    scores: torch.Tensor
    n_sequences: int
    seq_len: int
    per_sequence_scores: torch.Tensor | None = None

    def top_k(self, k: int = 5) -> list[tuple[int, int, float]]:
        """Return the top-k (layer, head, score) triples by score."""
        flat = self.scores.flatten()
        n_layers, n_heads = self.scores.shape
        top_vals, top_idx = torch.topk(flat, k=min(k, flat.numel()))
        return [
            (int(idx.item() // n_heads), int(idx.item() % n_heads), float(val.item()))
            for val, idx in zip(top_vals, top_idx, strict=True)
        ]


def build_repetition_sequences(
    n_sequences: int,
    seq_len: int,
    vocab_size: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """Generate random-token sequences with a single repetition.

    For each sequence: tokens 0..(half-1) are random; tokens half..(seq_len-1)
    are a copy of tokens 0..(half-1). seq_len must be even.

    Returns:
        Long tensor of shape (n_sequences, seq_len) on CPU.
    """
    if seq_len % 2 != 0:
        raise ValueError(f"seq_len must be even, got {seq_len}")
    half = seq_len // 2
    first = torch.randint(0, vocab_size, (n_sequences, half), generator=rng, dtype=torch.long)
    return torch.cat([first, first], dim=1)


def prefix_matching_score(
    model: HookedTransformer,
    n_sequences: int = 50,
    seq_len: int = 100,
    seed: int = 0,
    batch_size: int = 8,
    return_per_sequence: bool = False,
) -> PrefixMatchingResult:
    """Compute Olsson prefix-matching score for every (layer, head) in `model`.

    For each token at position p in the second half (p >= half), the prefix-
    matching target is position (p - half + 1): the token *after* the previous
    occurrence of the current token. Score for a head is the mean attention
    weight from p to that target, averaged over positions and sequences.

    Args:
        model: A loaded HookedTransformer.
        n_sequences: number of random sequences to average over (paper: 50).
        seq_len: total sequence length (paper: 100; first half random, second
            half copy).
        seed: RNG seed for sequence generation. Pythia has only one training
            seed; this seed governs *evaluation* sequences and is part of
            detector reproducibility, not pre-registration.
        batch_size: forward-pass batch size. Smaller is safer on MPS.

    Returns:
        PrefixMatchingResult with per-(layer, head) scores in [0, 1].
    """
    if seq_len < 4:
        raise ValueError(f"seq_len must be >= 4, got {seq_len}")
    half = seq_len // 2
    vocab_size = int(model.cfg.d_vocab)
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)

    rng = torch.Generator(device="cpu").manual_seed(seed)
    tokens = build_repetition_sequences(n_sequences, seq_len, vocab_size, rng)

    # Score accumulator: sum of per-position attention to target across all
    # sequences and second-half positions; divide at the end.
    score_sum = torch.zeros((n_layers, n_heads), dtype=torch.float32)
    n_terms = 0
    per_seq: torch.Tensor | None = (
        torch.zeros((n_sequences, n_layers, n_heads), dtype=torch.float32)
        if return_per_sequence
        else None
    )

    # Targets: for each second-half position p (in [half, seq_len-1]), the
    # target column is (p - half + 1). Position p == half maps to target 1.
    # Position p == seq_len - 1 maps to target half. Position p == seq_len-1
    # corresponds to current token = first[half-1], whose previous occurrence
    # is at position half-1, so the "next" position is half. That's covered.
    # We skip p == half because the previous occurrence of tokens[0] is at
    # position 0, and "the position right after" is 1 — fine, included.
    second_half_positions = torch.arange(half, seq_len, dtype=torch.long)
    target_positions = second_half_positions - half + 1  # shape: (half,)

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_start in range(0, n_sequences, batch_size):
            batch = tokens[batch_start : batch_start + batch_size].to(device)
            # `pattern` keys give attention shape (batch, head, q, k).
            _, cache = model.run_with_cache(
                batch,
                names_filter=lambda name: name.endswith("pattern"),
                return_type=None,
            )
            for layer in range(n_layers):
                pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
                # pattern: (batch, n_heads, seq_len, seq_len)
                # gather attention from second-half query positions to their
                # respective target key positions.
                # Resulting shape: (batch, n_heads, half).
                q_idx = second_half_positions.to(device)
                k_idx = target_positions.to(device)
                gathered = pattern[:, :, q_idx, k_idx]
                # mean over second-half positions → (batch, n_heads).
                per_batch_seq = gathered.mean(dim=-1).to(torch.float32).cpu()
                score_sum[layer] += per_batch_seq.sum(dim=0)
                if per_seq is not None:
                    per_seq[batch_start : batch_start + batch.shape[0], layer] = per_batch_seq
            n_terms += batch.shape[0]

    scores = score_sum / max(n_terms, 1)
    return PrefixMatchingResult(
        scores=scores,
        n_sequences=n_sequences,
        seq_len=seq_len,
        per_sequence_scores=per_seq,
    )


class InductionDetector:
    """Detector-protocol adapter for the Olsson QK prefix-matching detector."""

    motif = "induction"
    name = "induction_qk"

    def __init__(
        self,
        *,
        n_sequences: int = 50,
        seq_len: int = 100,
        seed: int = 0,
        batch_size: int = 8,
    ):
        from src.locked_thresholds import INDUCTION_QK
        self.threshold = INDUCTION_QK
        self.n_sequences = n_sequences
        self.seq_len = seq_len
        self.seed = seed
        self.batch_size = batch_size

    def score(self, model):
        from src.detectors.protocol import PerHeadScores
        result = prefix_matching_score(
            model,
            n_sequences=self.n_sequences,
            seq_len=self.seq_len,
            seed=self.seed,
            batch_size=self.batch_size,
        )
        return PerHeadScores(
            scores=result.scores,
            motif=self.motif,
            detector_name=self.name,
            aux=result,
        )
