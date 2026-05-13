"""Alternative detectors for §H1-C-altdetectors cross-readout consistency.

Implements the three locked alt-detectors from HYPOTHESIS.md §H1-C-altdetectors-1:

  - induction_ov_score: Olsson 2022 OV-circuit verification (per-head DLA on
    prior-occurrence-next-token direction in random-token-repetition prompts).
  - successor_argmax_k_of_7: L (2023) graded argmax-K-of-7 on day-of-week
    transitions (per-head count of correctly argmax'd targets).
  - s_inhibition_compdla_at_s2: Wang 2023 §3 Component-DLA on (IO-S) at the
    S2 token position (per-head DLA scalar).

Thresholds (§H1-C-altdetectors-2-r-supersede, 2026-05-12):
  TAU_IND_OV   = +13.592629  (95th-pct of OV-score across 144 GPT-2 small heads)
  K_MIN        =  2          (ceil-95th-pct of K-score across 144 GPT-2 small heads)
  TAU_SI_DLA   = +0.247095   (95th-pct of CompDLA-S2 across 144 GPT-2 small heads)

Each function takes a loaded `HookedTransformer` (any Pythia size or GPT-2)
and returns per-(layer, head) alt-scores. Locked detector thresholds and
reference-set logic live in `_run_pythia_anchor_altdetectors_validation.py`.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from transformer_lens import HookedTransformer

from src.detectors.induction import build_repetition_sequences
from src.detectors.s_inhibition import _locate_positions
from src.detectors.successor import DAYS
from src.replication.tigges_ioi import IOIPrompt

# §H1-C-altdetectors-2-r-supersede locked thresholds
# (imported from src.locked_thresholds for audit-trail)
from src.locked_thresholds import (
    ALT_K_MIN as _ALT_K_MIN,
    ALT_TAU_IND_OV as _ALT_TAU_IND_OV,
    ALT_TAU_SI_DLA as _ALT_TAU_SI_DLA,
)

TAU_IND_OV: float = _ALT_TAU_IND_OV.value
K_MIN: int = int(_ALT_K_MIN.value)
TAU_SI_DLA: float = _ALT_TAU_SI_DLA.value


def induction_ov_score(
    model: HookedTransformer,
    *,
    n_sequences: int = 50,
    seq_len: int = 100,
    seed: int = 0,
    batch_size: int = 8,
) -> torch.Tensor:
    """Per-head OV-circuit verification score (Olsson 2022).

    For each position p in the second half of a repetition sequence, the
    "prior-occurrence-next-token" target is tokens[p - half + 1]. Per-head
    score = mean over (sequences × second-half positions) of the head's DLA
    at position p toward W_U[:, target].

    Mechanically distinct from the QK locked detector: tests OV-side rather
    than QK-side. Same prompts (n_sequences=50, seq_len=100, seed=0).
    """
    if seq_len % 2 != 0:
        raise ValueError(f"seq_len must be even, got {seq_len}")
    half = seq_len // 2
    vocab_size = int(model.cfg.d_vocab)
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    device = next(model.parameters()).device

    rng = torch.Generator(device="cpu").manual_seed(seed)
    tokens = build_repetition_sequences(n_sequences, seq_len, vocab_size, rng)

    W_U = model.W_U

    score_sum = torch.zeros(n_layers, n_heads, dtype=torch.float32)
    n_terms = 0

    second_half_positions = torch.arange(half, seq_len, dtype=torch.long)
    target_offsets = second_half_positions - half + 1

    prev_use_attn_result = model.cfg.use_attn_result
    model.set_use_attn_result(True)
    model.eval()
    try:
        with torch.no_grad():
            for batch_start in range(0, n_sequences, batch_size):
                batch_tokens = tokens[batch_start : batch_start + batch_size].to(device)
                B = int(batch_tokens.shape[0])
                _, cache = model.run_with_cache(
                    batch_tokens,
                    names_filter=lambda n: "attn.hook_result" in n,
                    return_type=None,
                )
                target_ids = batch_tokens[:, target_offsets.to(device)]  # (B, half)
                directions = W_U[:, target_ids].permute(1, 2, 0).to(torch.float32)

                for layer in range(n_layers):
                    result = cache[f"blocks.{layer}.attn.hook_result"]
                    second_half_result = result[:, half:, :, :].to(torch.float32)
                    contrib = (second_half_result * directions[:, :, None, :]).sum(dim=-1)
                    per_seq_per_head = contrib.mean(dim=1).cpu()
                    score_sum[layer] += per_seq_per_head.sum(dim=0)
                del cache
                n_terms += B
    finally:
        model.set_use_attn_result(prev_use_attn_result)

    return score_sum / max(n_terms, 1)


def successor_argmax_k_of_7(
    model: HookedTransformer,
    *,
    batch_size: int = 16,
) -> torch.Tensor:
    """Per-head argmax-K-of-7 score (L 2023 graded protocol).

    For each of 7 cyclic day-of-week transitions (Mon→Tue, ..., Sun→Mon),
    build a 3-context prompt and at the END position compute each head's
    DLA toward each of the 7 day-target-tokens. Per-head argmax over the
    7 day tokens is its prediction; K_score = count of correct argmaxes
    over 7 transitions.

    Mechanically distinct from lift_dla cross-category: per-head argmax-
    correctness on a fixed 7-class set rather than continuous-lift across
    4 ordinal categories.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    device = next(model.parameters()).device
    tokenizer = model.tokenizer
    W_U = model.W_U

    day_token_ids: list[int] = []
    for day in DAYS:
        ids = tokenizer.encode(f" {day}", add_special_tokens=False)
        if not ids:
            raise RuntimeError(f"day {day!r} encoded to empty token list")
        day_token_ids.append(int(ids[0]))
    day_token_ids_t = torch.tensor(day_token_ids, dtype=torch.long)

    prompts: list[tuple[str, int]] = []
    for i in range(7):
        c1 = DAYS[i]
        c2 = DAYS[(i + 1) % 7]
        c3 = DAYS[(i + 2) % 7]
        target_idx = (i + 3) % 7
        text = f"{c1}, {c2}, {c3}, "
        prompts.append((text, target_idx))

    token_rows = [model.to_tokens(t, prepend_bos=True)[0] for t, _ in prompts]
    targets = [target_idx for _, target_idx in prompts]
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(token_rows):
        by_len[int(t.shape[0])].append(i)

    per_prompt_per_day = torch.zeros(n_layers, n_heads, 7, 7, dtype=torch.float32)
    day_directions = W_U[:, day_token_ids_t.to(W_U.device)].T.to(torch.float32)

    prev_use_attn_result = model.cfg.use_attn_result
    model.set_use_attn_result(True)
    model.eval()
    try:
        with torch.no_grad():
            for length, indices in sorted(by_len.items()):
                for chunk_start in range(0, len(indices), batch_size):
                    idxs = indices[chunk_start : chunk_start + batch_size]
                    batch = torch.stack([token_rows[i] for i in idxs]).to(device)
                    _, cache = model.run_with_cache(
                        batch,
                        names_filter=lambda n: "attn.hook_result" in n,
                        return_type=None,
                    )
                    for layer in range(n_layers):
                        head_out = cache[f"blocks.{layer}.attn.hook_result"][:, -1, :, :].to(torch.float32)
                        contrib = torch.einsum(
                            "bhd,kd->bhk", head_out, day_directions.to(head_out.device)
                        ).cpu()
                        for k, i in enumerate(idxs):
                            per_prompt_per_day[layer, :, i, :] = contrib[k]
                    del cache
    finally:
        model.set_use_attn_result(prev_use_attn_result)

    argmax_day = per_prompt_per_day.argmax(dim=-1)
    targets_t = torch.tensor(targets, dtype=torch.long)
    matches = (argmax_day == targets_t[None, None, :]).to(torch.int32)
    k_score = matches.sum(dim=-1).to(torch.float32)
    return k_score


def s_inhibition_compdla_at_s2(
    model: HookedTransformer,
    prompts: list[IOIPrompt],
    *,
    batch_size: int = 8,
) -> torch.Tensor:
    """Per-head Component-DLA on (IO-S) at the S2 token position (Wang 2023 §3).

    Mirrors `tigges_ioi.component_dla` but evaluates at S2 (the n3 occurrence
    of the duplicate name in 'B gave a obj to') rather than END. Direction is
    W_U[:, io_token_id] - W_U[:, s_token_id], averaged over prompts.

    Mechanically distinct from §S-1 path-patching Δ_h: a direct DLA readout
    at S2, not a frozen-path perturbation.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    device = next(model.parameters()).device
    W_U = model.W_U

    token_rows = [model.to_tokens(p.text, prepend_bos=True)[0] for p in prompts]
    s2_positions: list[int] = []
    for p, toks in zip(prompts, token_rows, strict=True):
        positions = _locate_positions(toks, p.s_token_id, p.io_token_id)
        s2_positions.append(positions.s2_pos)

    by_len: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(token_rows):
        by_len[int(t.shape[0])].append(i)

    accumulator = torch.zeros(n_layers, n_heads, dtype=torch.float32)

    prev_use_attn_result = model.cfg.use_attn_result
    model.set_use_attn_result(True)
    model.eval()
    try:
        with torch.no_grad():
            for length, indices in by_len.items():
                for chunk_start in range(0, len(indices), batch_size):
                    idxs = indices[chunk_start : chunk_start + batch_size]
                    batch = torch.stack([token_rows[i] for i in idxs]).to(device)
                    s2_idx = torch.tensor(
                        [s2_positions[i] for i in idxs], dtype=torch.long, device=device
                    )
                    directions = torch.stack(
                        [
                            W_U[:, prompts[i].io_token_id]
                            - W_U[:, prompts[i].s_token_id]
                            for i in idxs
                        ]
                    ).to(torch.float32)
                    _, cache = model.run_with_cache(
                        batch,
                        names_filter=lambda n: "attn.hook_result" in n,
                        return_type=None,
                    )
                    B = int(batch.shape[0])
                    batch_idx = torch.arange(B, device=device)
                    for layer in range(n_layers):
                        full_result = cache[f"blocks.{layer}.attn.hook_result"]
                        result_at_s2 = full_result[batch_idx, s2_idx, :, :].to(torch.float32)
                        contrib = (result_at_s2 * directions[:, None, :]).sum(dim=-1)
                        accumulator[layer] += contrib.sum(dim=0).cpu()
                    del cache
    finally:
        model.set_use_attn_result(prev_use_attn_result)

    accumulator /= len(prompts)
    return accumulator


# ---- Detector-protocol adapters (§H1-C-altdetectors) ----------------------


class InductionOvDetector:
    """Detector-protocol adapter for the OV-circuit induction alt detector."""

    motif = "induction"
    name = "induction_ov"

    def __init__(
        self,
        *,
        n_sequences: int = 50,
        seq_len: int = 100,
        seed: int = 0,
        batch_size: int = 8,
    ):
        from src.locked_thresholds import ALT_TAU_IND_OV
        self.threshold = ALT_TAU_IND_OV
        self.n_sequences = n_sequences
        self.seq_len = seq_len
        self.seed = seed
        self.batch_size = batch_size

    def score(self, model):
        from src.detectors.protocol import PerHeadScores
        scores = induction_ov_score(
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


class SuccessorArgmaxKDetector:
    """Detector-protocol adapter for the successor argmax-K-of-7 alt detector."""

    motif = "successor"
    name = "successor_argmax_k_of_7"

    def __init__(self, *, batch_size: int = 16):
        from src.locked_thresholds import ALT_K_MIN
        self.threshold = ALT_K_MIN
        self.batch_size = batch_size

    def score(self, model):
        from src.detectors.protocol import PerHeadScores
        scores = successor_argmax_k_of_7(model, batch_size=self.batch_size)
        return PerHeadScores(
            scores=scores,
            motif=self.motif,
            detector_name=self.name,
            aux=None,
        )


class SInhibitionCompdlaDetector:
    """Detector-protocol adapter for the S-inhibition CompDLA-at-S2 alt detector."""

    motif = "s_inhibition"
    name = "s_inhibition_compdla"

    def __init__(self, prompts, *, batch_size: int = 8):
        from src.locked_thresholds import ALT_TAU_SI_DLA
        self.threshold = ALT_TAU_SI_DLA
        self.prompts = prompts
        self.batch_size = batch_size

    def score(self, model):
        from src.detectors.protocol import PerHeadScores
        scores = s_inhibition_compdla_at_s2(
            model, self.prompts, batch_size=self.batch_size
        )
        return PerHeadScores(
            scores=scores,
            motif=self.motif,
            detector_name=self.name,
            aux=None,
        )
