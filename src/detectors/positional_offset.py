"""Positional-offset attention-head detector (atlas-v1 head-family inventory).

Generalises the previous-token detector to a small set of nonzero relative
offsets. For each (layer, head), we measure the mean attention along each of
the diagonals corresponding to ``k ∈ K_SET = (-3, -2, +1, +2, +3)`` and take
the maximum.

The detector deliberately EXCLUDES ``k = -1`` because that offset is the
``PreviousTokenDetector``'s territory; the atlas head-family typology is
designed to be mutually exclusive at the motif level, so previous-token
heads should not also classify as positional-offset heads.

Operational definition (PROJECT_BRIEF.md, atlas-v1):
    Build ``n_sequences`` Olsson random-token-repetition sequences of
    length ``seq_len`` (matching the induction / previous-token detectors'
    evaluation surface) and prepend BOS. For every (layer, head) and every
    ``k ∈ K_SET``, compute the mean of ``pattern[..., p, p+k]`` over the
    valid query positions (those where ``p+k`` is in range) and over
    sequences. The head's score is the max across the five k values; the
    head's ``dominant_k`` is the k that achieves that max.

The exploratory threshold (0.20) lives in ``src.atlas.thresholds`` —
"this head puts 20%+ of its attention mass on a single nonzero positional
offset (other than ``k=-1``), averaged across positions and sequences."
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from src.atlas.thresholds import POSITIONAL_OFFSET
from src.detectors.induction import build_repetition_sequences
from src.detectors.protocol import PerHeadScores


# Nonzero offsets to test, with k=-1 deliberately excluded (that diagonal is
# the PreviousTokenDetector's territory in the atlas-v1 typology).
K_SET: tuple[int, ...] = (-3, -2, 1, 2, 3)


def positional_offset_score(
    model: HookedTransformer,
    *,
    n_sequences: int = 50,
    seq_len: int = 100,
    seed: int = 0,
    batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-(layer, head) positional-offset attention score.

    For each ``k ∈ K_SET``, the mean of ``pattern[..., p, p+k]`` over valid
    query positions ``p`` (those where ``p+k`` is in range) is computed and
    averaged across sequences. The head's score is the maximum of those five
    means; the head's ``dominant_k`` is the k attaining that maximum.

    Args:
        model: A loaded HookedTransformer.
        n_sequences: number of random sequences to average over.
        seq_len: length of each sequence *before* BOS prepending.
        seed: RNG seed for sequence generation.
        batch_size: forward-pass batch size.

    Returns:
        Tuple ``(scores, dominant_k)``:
          - ``scores``: float32 CPU tensor of shape ``(n_layers, n_heads)``
            with values in ``[0, 1]``.
          - ``dominant_k``: int64 CPU tensor of shape ``(n_layers, n_heads)``
            whose entries are drawn from ``K_SET``.
    """
    # Need at least one k to be realizable: max(|k|) = 3 ⇒ require seq_len >= 4
    # (so e.g. k=+3 has p=0 valid). After BOS prepend full_len = seq_len + 1.
    if seq_len < 4:
        raise ValueError(f"seq_len must be >= 4, got {seq_len}")

    vocab_size = int(model.cfg.d_vocab)
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    bos_id = int(model.tokenizer.bos_token_id)

    rng = torch.Generator(device="cpu").manual_seed(seed)
    raw_tokens = build_repetition_sequences(n_sequences, seq_len, vocab_size, rng)
    # BOS prepend — matches the induction / previous-token detectors' shape.
    bos_col = torch.full((n_sequences, 1), bos_id, dtype=torch.long)
    tokens = torch.cat([bos_col, raw_tokens], dim=1)

    device = next(model.parameters()).device

    # per_k_score_sum[i, layer, head] accumulates the per-sequence mean
    # attention along diagonal K_SET[i], summed across sequences.
    per_k_score_sum = torch.zeros(
        (len(K_SET), n_layers, n_heads), dtype=torch.float32
    )
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
                # torch.diagonal with offset=k along the last two dims returns
                # the k-diagonal of every (q, k_pos) matrix. The convention is
                # that ``offset=+k`` selects entries ``M[r, r+k]`` and
                # ``offset=-k`` selects entries ``M[r, r-k] = M[r-k+k, r-k]``;
                # in attention semantics, offset=+k corresponds to attention
                # from query position p to key position p+k.
                for i, k in enumerate(K_SET):
                    diag = pattern.diagonal(offset=k, dim1=-2, dim2=-1)
                    # diag: (batch, n_heads, full_len - |k|)
                    per_batch_seq = diag.mean(dim=-1).to(torch.float32).cpu()
                    # per_batch_seq: (batch, n_heads)
                    per_k_score_sum[i, layer] += per_batch_seq.sum(dim=0)
            n_seqs_seen += batch.shape[0]
            del cache

    # Mean across sequences → (len(K_SET), n_layers, n_heads).
    per_k_means = per_k_score_sum / max(n_seqs_seen, 1)

    # Max across the K dimension; argmax indexes into K_SET to recover k.
    max_vals, max_idx = per_k_means.max(dim=0)
    # Map argmax indices in [0, len(K_SET)) back to the actual k values.
    k_tensor = torch.tensor(K_SET, dtype=torch.long)
    dominant_k = k_tensor[max_idx]

    return max_vals, dominant_k


class PositionalOffsetDetector:
    """Detector-protocol adapter for the positional-offset attention motif.

    The aux payload carries the per-head ``dominant_k`` tensor (int64, same
    shape as ``scores``) so downstream consumers can split positional-offset
    heads by which offset they specialise on without re-running the detector.
    """

    motif = "positional_offset"
    name = "positional_offset_attention"

    def __init__(
        self,
        *,
        n_sequences: int = 50,
        seq_len: int = 100,
        seed: int = 0,
        batch_size: int = 8,
    ):
        self.threshold = POSITIONAL_OFFSET
        self.n_sequences = n_sequences
        self.seq_len = seq_len
        self.seed = seed
        self.batch_size = batch_size

    def score(self, model) -> PerHeadScores:
        scores, dominant_k = positional_offset_score(
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
            aux=dict(dominant_k=dominant_k),
        )
