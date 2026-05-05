"""Copy-suppression head detector (McDougall two-criterion).

Reference: McDougall et al. 2024, "Copy Suppression: Comprehensively Understanding
an Attention Head" (BlackboxNLP).

Behavior of a copy-suppression head: at a query position whose token has appeared
earlier in context, the head attends back to the prior occurrence(s) and writes a
*negative* contribution to the current token's logit, suppressing the prediction
that other components are pushing up.

Two criteria (PROJECT_BRIEF.md §4):
    QK criterion: attention from query position i to prior key positions j where
        token_j == token_i, summed across all such j, exceeds 0.3 on average.
    OV criterion: the head's contribution to the logit of token_i at position i
        is negative on average (DLA < 0).

DLA approximation: per-head residual contribution projected onto the unembedding
direction for token_i. With TransformerLens' default fold_ln=True, the final
LayerNorm is folded into W_U, so head_out @ W_U is the linearized DLA — accurate
up to per-position LN scaling, which we ignore (standard mech-interp practice).

Validation target: GPT-2 small layer 10 head 7.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer


@dataclass(frozen=True)
class CopySuppressionResult:
    """Per-(layer, head) QK and OV scores for copy-suppression detection.

    Both tensors have shape (n_layers, n_heads). QK is in [0, 1] (it's a sum of
    attention probabilities, capped by 1 since attention rows sum to 1). OV is
    unbounded but should be negative for copy-suppression candidates.

    When `collect_per_position=True` is passed to `copy_suppression_score`, the
    `per_position_ov` and `per_position_meta` fields are populated. Their
    rows are aligned: `per_position_meta[k] == (passage_idx, position)` is the
    (passage, query position) for the k-th row of `per_position_ov`, which has
    shape `(n_total_eligible, n_layers, n_heads)`. This is the side-cache
    needed for data-driven worked-example selection (Q7 rule from the Day 3
    design): for a target head, sort by `per_position_ov[:, L, H]` ascending
    and the top-K rows are the corpus positions where head (L, H) most strongly
    suppresses.
    """

    qk_scores: torch.Tensor
    ov_scores: torch.Tensor
    n_positions: int
    per_position_ov: torch.Tensor | None = None
    per_position_meta: list[tuple[int, int]] | None = None

    def candidates(
        self, qk_threshold: float = 0.3, ov_threshold: float = 0.0
    ) -> list[tuple[int, int]]:
        """Return (layer, head) pairs passing both criteria."""
        mask = (self.qk_scores > qk_threshold) & (self.ov_scores < ov_threshold)
        return [
            (int(layer), int(head))
            for layer in range(mask.shape[0])
            for head in range(mask.shape[1])
            if bool(mask[layer, head].item())
        ]

    def top_k_qk(self, k: int = 5) -> list[tuple[int, int, float]]:
        """Top-k (layer, head, qk_score) by QK score."""
        return _top_k(self.qk_scores, k, descending=True)

    def top_k_ov_negative(self, k: int = 5) -> list[tuple[int, int, float]]:
        """Top-k (layer, head, ov_score) by most-negative OV (i.e., strongest suppression)."""
        return _top_k(self.ov_scores, k, descending=False)


def _top_k(scores: torch.Tensor, k: int, descending: bool) -> list[tuple[int, int, float]]:
    flat = scores.flatten()
    n_heads = scores.shape[1]
    top_vals, top_idx = torch.topk(flat, k=min(k, flat.numel()), largest=descending)
    return [
        (int(idx.item() // n_heads), int(idx.item() % n_heads), float(val.item()))
        for val, idx in zip(top_vals, top_idx, strict=True)
    ]


def _build_dup_index(token_ids: list[int]) -> dict[int, list[int]]:
    """For each query position i, return list of prior positions j (j < i) with
    token_j == token_i. Positions with no prior duplicate are absent from the dict.
    """
    seen: dict[int, list[int]] = {}
    dup_map: dict[int, list[int]] = {}
    for i, tok in enumerate(token_ids):
        prior = seen.get(tok)
        if prior is not None:
            dup_map[i] = list(prior)
        seen.setdefault(tok, []).append(i)
    return dup_map


def copy_suppression_score(
    model: HookedTransformer,
    sequences: torch.Tensor | list[torch.Tensor],
    *,
    collect_per_position: bool = False,
) -> CopySuppressionResult:
    """Compute McDougall QK and OV scores for every (layer, head) in `model`.

    Args:
        model: A loaded HookedTransformer.
        sequences: Either a 2D tensor of shape (n_seq, seq_len) — uniform length —
            or a list of 1D tensors with token IDs (variable length). Positions
            with at least one prior duplicate token contribute to the scores;
            positions without prior duplicates are skipped.
        collect_per_position: if True, also collect per-(passage, position) OV
            contributions to all heads. Required for data-driven worked-example
            selection (Q7 rule). Adds memory cost
            ~ n_eligible_total * n_layers * n_heads * 4 bytes.

    Returns:
        CopySuppressionResult with per-(layer, head) QK and OV scores averaged
        across all eligible positions in all sequences. If `collect_per_position`
        is True, the result also has `per_position_ov` and `per_position_meta`
        populated.
    """
    if isinstance(sequences, torch.Tensor):
        if sequences.dim() != 2:
            raise ValueError(f"sequences tensor must be 2D, got shape {tuple(sequences.shape)}")
        seq_iter: list[torch.Tensor] = [sequences[i] for i in range(sequences.shape[0])]
    else:
        seq_iter = list(sequences)
        for s in seq_iter:
            if s.dim() != 1:
                raise ValueError(
                    f"each sequence in list must be 1D, got shape {tuple(s.shape)}"
                )

    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)

    qk_sum = torch.zeros((n_layers, n_heads), dtype=torch.float32)
    ov_sum = torch.zeros((n_layers, n_heads), dtype=torch.float32)
    n_positions = 0

    per_position_ov_chunks: list[torch.Tensor] = []
    per_position_meta: list[tuple[int, int]] = []

    device = next(model.parameters()).device
    W_U = model.W_U  # (d_model, d_vocab)

    with torch.no_grad():
        for passage_idx, tokens_1d in enumerate(seq_iter):
            tokens = tokens_1d.to(device)
            tok_list = tokens.tolist()
            dup_map = _build_dup_index(tok_list)
            if not dup_map:
                continue

            _, cache = model.run_with_cache(
                tokens.unsqueeze(0),
                names_filter=lambda name: name.endswith("hook_pattern")
                or name.endswith("hook_z"),
                return_type=None,
            )

            eligible = list(dup_map.keys())

            if collect_per_position:
                # Pre-allocate (n_eligible, n_layers, n_heads) for this passage.
                passage_ov = torch.zeros(
                    (len(eligible), n_layers, n_heads), dtype=torch.float32
                )
                pos_index_map = {pos: idx for idx, pos in enumerate(eligible)}

            for layer in range(n_layers):
                pattern = cache[f"blocks.{layer}.attn.hook_pattern"]  # (1, h, q, k)
                z = cache[f"blocks.{layer}.attn.hook_z"]  # (1, q, h, d_head)
                W_O = model.blocks[layer].attn.W_O  # (h, d_head, d_model)
                head_out = torch.einsum("bqhd, hdm -> bqhm", z, W_O)  # (1, q, h, d_model)

                for i in eligible:
                    tok_i = tok_list[i]
                    js = torch.tensor(dup_map[i], device=device, dtype=torch.long)
                    qk_per_head = pattern[0, :, i, js].sum(dim=-1).float().cpu()
                    ov_per_head = (head_out[0, i] @ W_U[:, tok_i]).float().cpu()
                    qk_sum[layer] += qk_per_head
                    ov_sum[layer] += ov_per_head
                    if collect_per_position:
                        passage_ov[pos_index_map[i], layer] = ov_per_head

            if collect_per_position:
                per_position_ov_chunks.append(passage_ov)
                per_position_meta.extend((passage_idx, pos) for pos in eligible)

            n_positions += len(eligible)

    if n_positions == 0:
        raise ValueError(
            "No eligible positions found in `sequences`. Provide sequences "
            "containing at least one repeated token."
        )

    qk_scores = qk_sum / n_positions
    ov_scores = ov_sum / n_positions

    per_position_ov: torch.Tensor | None = None
    per_position_meta_out: list[tuple[int, int]] | None = None
    if collect_per_position and per_position_ov_chunks:
        per_position_ov = torch.cat(per_position_ov_chunks, dim=0)
        per_position_meta_out = per_position_meta

    return CopySuppressionResult(
        qk_scores=qk_scores,
        ov_scores=ov_scores,
        n_positions=n_positions,
        per_position_ov=per_position_ov,
        per_position_meta=per_position_meta_out,
    )
