"""S-inhibition head detector via path-patching.

Implements Phase 1.3's locked detector specification (HYPOTHESIS.md §S-1
through §S-4). For each candidate sender head h:

  1. Build an ABC-corrupted twin of each clean IOI prompt (n3 = a fresh
     name C ∉ {IO, S}; n1 and n2 unchanged). §S-2.
  2. Path-patch h's z at the END position with its corrupt-prompt value;
     freeze all attention/MLP outputs at intermediate layers to clean. §S-1.
  3. Read each receiver Name Mover's attention pattern at the END query.
  4. Compute the receiver scalar

         Δ_h = mean over k=4 NMs of [(patched_attn_S2 - clean_attn_S2)
                                     - (patched_attn_IO - clean_attn_IO)]

     averaged across N=200 prompts. Sign convention §S-4: positive Δ_h for
     genuine S-inhibition senders.

Name Movers are identified per §S-3: top-k=4 by component-DLA on the
(IO − S) logit difference, computed on the clean prompt set in the target
model. The k=4 receiver set is *not* hardcoded to Wang's published GPT-2
NMs; if our component-DLA top-4 disagrees with Wang's labelling, the
disagreement is recorded as a finding and validation proceeds with our top-4.

Validation target on GPT-2 small: Wang 2023's S-Inhibition heads
{(7, 3), (7, 9), (8, 6), (8, 10)} should appear in the top-8 of Δ_h with
the median ≥ 2σ above the bulk mean (HYPOTHESIS.md §S-5).
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from src.replication.path_patching import path_patch_one_sender
from src.replication.tigges_ioi import (
    IOIPrompt,
    WANG_NAMES,
    component_dla,
    filter_single_token_names,
)

WANG_S_INHIBITION_HEADS: tuple[tuple[int, int], ...] = (
    (7, 3),
    (7, 9),
    (8, 6),
    (8, 10),
)


@dataclass(frozen=True)
class CorruptedIOIPrompt:
    """A clean IOI prompt paired with its ABC-corrupted twin.

    The corrupt text replaces the n3 (second-clause subject) name with a
    fresh `c_name` ∉ {IO, S} drawn from the single-token name pool. Tokenizer
    consistency between clean and corrupt is enforced at construction time.
    """

    clean_text: str
    corrupt_text: str
    io_name: str
    s_name: str
    c_name: str
    template_kind: str
    io_token_id: int
    s_token_id: int
    c_token_id: int


def build_abc_corrupted_prompts(
    prompts: list[IOIPrompt],
    tokenizer: PreTrainedTokenizerBase,
    *,
    seed: int = 0,
    name_pool: tuple[str, ...] | None = None,
) -> list[CorruptedIOIPrompt]:
    """For each clean IOI prompt, construct the ABC-corrupted variant.

    Per-prompt determinism: each prompt's C selection is seeded by
    (seed + prompt_index), so the corruption is reproducible regardless of
    iteration order.
    """
    if name_pool is None:
        name_pool = filter_single_token_names(WANG_NAMES, tokenizer)
    if len(name_pool) < 3:
        raise RuntimeError(
            f"name pool must have at least 3 single-token names; got {len(name_pool)}"
        )

    out: list[CorruptedIOIPrompt] = []
    for i, p in enumerate(prompts):
        rng = random.Random(seed * 1_000_003 + i)
        candidates = [n for n in name_pool if n != p.io_name and n != p.s_name]
        c = rng.choice(candidates)

        # n3 in both BABA and ABBA equals the s_name (the duplicate). Replace
        # the LAST occurrence of " {s_name} gave" — this is unambiguously n3
        # because the template only has one " gave" verb.
        marker = f" {p.s_name} gave"
        if marker not in p.text:
            raise RuntimeError(
                f"prompt {i} missing expected n3 marker {marker!r}: {p.text!r}"
            )
        corrupt_text = p.text.replace(marker, f" {c} gave", 1)

        c_ids = tokenizer.encode(f" {c}", add_special_tokens=False)
        if len(c_ids) != 1:
            raise RuntimeError(
                f"C={c!r} not single-token under tokenizer (got {c_ids})"
            )

        out.append(
            CorruptedIOIPrompt(
                clean_text=p.text,
                corrupt_text=corrupt_text,
                io_name=p.io_name,
                s_name=p.s_name,
                c_name=c,
                template_kind=p.template_kind,
                io_token_id=p.io_token_id,
                s_token_id=p.s_token_id,
                c_token_id=c_ids[0],
            )
        )
    return out


@dataclass(frozen=True)
class _PromptPositions:
    seq_len: int
    s2_pos: int  # last (= n3) occurrence of s_token_id in clean tokens
    io_pos: int  # unique occurrence of io_token_id in clean tokens
    end_pos: int  # last token position = seq_len - 1


def _locate_positions(
    clean_token_row: torch.Tensor,
    s_token_id: int,
    io_token_id: int,
) -> _PromptPositions:
    """For a single tokenized clean prompt, find S2, IO, and END positions."""
    s_hits = (clean_token_row == s_token_id).nonzero(as_tuple=True)[0]
    io_hits = (clean_token_row == io_token_id).nonzero(as_tuple=True)[0]
    if len(s_hits) < 2:
        raise RuntimeError(
            f"expected ≥2 occurrences of s_token_id={s_token_id} in clean prompt; "
            f"got {len(s_hits)}. Tokens: {clean_token_row.tolist()}"
        )
    if len(io_hits) < 1:
        raise RuntimeError(
            f"expected ≥1 occurrence of io_token_id={io_token_id} in clean prompt; "
            f"got {len(io_hits)}. Tokens: {clean_token_row.tolist()}"
        )
    return _PromptPositions(
        seq_len=int(clean_token_row.shape[0]),
        s2_pos=int(s_hits[-1].item()),
        io_pos=int(io_hits[0].item()),
        end_pos=int(clean_token_row.shape[0] - 1),
    )


@dataclass(frozen=True)
class SInhibitionResult:
    """Output of the S-inhibition path-patching screen.

    `delta_h[layer, head]` is the per-head scalar averaged across the k=4 NMs
    and across N prompts (HYPOTHESIS.md §S-4). Wang's S-Inhibition heads on
    GPT-2 should produce large positive entries.

    `per_nm_matrix[layer, head, nm_idx]` is the same scalar before averaging
    across NMs, with `nm_heads[nm_idx]` giving the NM identity. Used for the
    (sender × NM) figure in the proof notebook.
    """

    delta_h: torch.Tensor
    per_nm_matrix: torch.Tensor
    nm_heads: list[tuple[int, int]]
    n_prompts: int
    per_prompt_delta: torch.Tensor | None = None

    def candidates(self, threshold: float) -> list[tuple[int, int]]:
        """Return (layer, head) pairs whose Δ_h ≥ threshold, sorted descending."""
        n_layers, n_heads = self.delta_h.shape
        items: list[tuple[float, tuple[int, int]]] = []
        for L in range(n_layers):
            for H in range(n_heads):
                v = float(self.delta_h[L, H].item())
                if v >= threshold:
                    items.append((v, (L, H)))
        items.sort(reverse=True)
        return [lh for _, lh in items]


def _compute_delta_for_group(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    positions: list[_PromptPositions],
    nm_heads: list[tuple[int, int]],
    senders: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For one length-grouped batch, run the screen and compute Δ_h_i_p.

    Per-sender NM filtering: a sender at layer L can only causally affect
    receivers at strictly higher layers via the residual stream. Senders at
    layers >= L_r cannot reach receiver L_r and contribute 0 to that NM's
    column in the per_nm_sum matrix; if no receivers are downstream of a
    sender, the sender is skipped entirely (its row is all zeros).

    Returns:
        per_nm_sum: (n_senders, k_NM) sum-across-prompts of Δ_h_i_p, with
            zero entries for sender→NM pairs where the sender is not
            strictly upstream of the NM.
        valid_count: (n_senders, k_NM) integer matrix counting how many of
            the prompts were "valid" (= NM strictly downstream of sender);
            this is constant within a row and equals B for upstream pairs,
            0 otherwise. Caller uses this to compute proper per-(sender, NM)
            means across length-groups.
        n_prompts_in_group: scalar = clean_tokens.shape[0].
    """
    B = clean_tokens.shape[0]
    end_pos = positions[0].end_pos
    assert all(p.end_pos == end_pos for p in positions)

    s2_idx = torch.tensor([p.s2_pos for p in positions], dtype=torch.long)
    io_idx = torch.tensor([p.io_pos for p in positions], dtype=torch.long)
    batch_idx = torch.arange(B, dtype=torch.long)

    n_senders = len(senders)
    k_nm = len(nm_heads)
    per_nm_sum = torch.zeros(n_senders, k_nm, dtype=torch.float32)
    valid = torch.zeros(n_senders, k_nm, dtype=torch.long)
    per_prompt_delta = torch.zeros(n_senders, k_nm, B, dtype=torch.float32)

    receiver_layers_all = sorted({nl for nl, _ in nm_heads})
    sender_layers_all = {sl for sl, _ in senders}
    cache_names = _required_cache_names_for_screen(
        model, sender_layers_all, set(receiver_layers_all)
    )
    name_filter = lambda name: name in cache_names  # noqa: E731

    with torch.no_grad():
        _, clean_cache = model.run_with_cache(
            clean_tokens, names_filter=name_filter, return_type=None
        )
        _, corrupt_cache = model.run_with_cache(
            corrupt_tokens, names_filter=name_filter, return_type=None
        )

    clean_at_s2_per_nm: list[torch.Tensor] = []
    clean_at_io_per_nm: list[torch.Tensor] = []
    for nm_layer, nm_head in nm_heads:
        clean_pat = clean_cache[f"blocks.{nm_layer}.attn.hook_pattern"].cpu()
        clean_at_s2_per_nm.append(
            clean_pat[batch_idx, nm_head, end_pos, s2_idx].to(torch.float32)
        )
        clean_at_io_per_nm.append(
            clean_pat[batch_idx, nm_head, end_pos, io_idx].to(torch.float32)
        )

    for s_idx, (sl, sh) in enumerate(senders):
        downstream_nm_idx = [i for i, (nl, _) in enumerate(nm_heads) if nl > sl]
        if not downstream_nm_idx:
            continue
        downstream_nm_layers = sorted({nm_heads[i][0] for i in downstream_nm_idx})
        captured = path_patch_one_sender(
            model,
            clean_tokens,
            clean_cache=clean_cache,
            corrupt_cache=corrupt_cache,
            sender_layer=sl,
            sender_head=sh,
            sender_position=end_pos,
            receiver_layers=downstream_nm_layers,
        )
        for nm_idx in downstream_nm_idx:
            nm_layer, nm_head = nm_heads[nm_idx]
            patched_pat = captured[nm_layer].cpu()
            patched_at_s2 = patched_pat[batch_idx, nm_head, end_pos, s2_idx].to(torch.float32)
            patched_at_io = patched_pat[batch_idx, nm_head, end_pos, io_idx].to(torch.float32)
            delta = (patched_at_s2 - clean_at_s2_per_nm[nm_idx]) - (
                patched_at_io - clean_at_io_per_nm[nm_idx]
            )
            per_nm_sum[s_idx, nm_idx] = delta.sum()
            per_prompt_delta[s_idx, nm_idx, :] = delta
            valid[s_idx, nm_idx] = B

    return per_nm_sum, valid, torch.tensor(B), per_prompt_delta


def _required_cache_names_for_screen(
    model: HookedTransformer,
    sender_layers: set[int],
    receiver_layers: set[int],
) -> set[str]:
    """Names we need from clean and corrupt caches for the full screen."""
    if not sender_layers or not receiver_layers:
        return set()
    earliest = min(sender_layers)
    latest = max(receiver_layers)
    names: set[str] = set()
    for layer in range(earliest, latest):
        names.add(f"blocks.{layer}.hook_attn_out")
        names.add(f"blocks.{layer}.hook_mlp_out")
    for layer in receiver_layers:
        names.add(f"blocks.{layer}.attn.hook_pattern")
    for layer in sender_layers:
        names.add(f"blocks.{layer}.attn.hook_z")
    return names


def s_inhibition_screen(
    model: HookedTransformer,
    clean_prompts: list[IOIPrompt],
    corrupt_prompts: list[CorruptedIOIPrompt],
    *,
    senders: list[tuple[int, int]] | None = None,
    k_nm: int = 4,
    batch_size: int = 50,
    nm_dla_batch_size: int = 8,
    return_per_prompt: bool = False,
    nm_heads_override: list[tuple[int, int]] | None = None,
) -> SInhibitionResult:
    """Run the full S-inhibition detector screen.

    Identifies NMs as the top-k_nm heads by component-DLA on (IO − S), then
    path-patches every sender (or the supplied subset) with each prompt's
    ABC-corrupted twin. Returns per-head Δ_h averaged over N prompts and
    k_nm NMs, plus the per-(sender, NM) breakdown matrix.

    Length-grouping: prompts of differing tokenized length are processed in
    separate sub-batches; within each sub-batch all prompts have identical T
    so we can batch the path-patching efficiently.

    Args:
        model: TransformerLens model.
        clean_prompts: N IOI prompts (`tigges_ioi.IOIPrompt`).
        corrupt_prompts: corresponding ABC-corrupted prompts (length N, same
            order; index `i` of corrupt corresponds to index `i` of clean).
        senders: list of (layer, head) pairs to screen. If None, screens all
            heads at all layers.
        k_nm: number of Name Movers to use as receivers (default 4 per §S-3).
        batch_size: max prompts per length-group batch.
        nm_dla_batch_size: batch size for the component-DLA NM identification
            step (forwarded to `tigges_ioi.component_dla`).
    """
    if len(clean_prompts) != len(corrupt_prompts):
        raise ValueError(
            f"clean_prompts ({len(clean_prompts)}) and corrupt_prompts "
            f"({len(corrupt_prompts)}) must be the same length"
        )
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    if senders is None:
        senders = [(L, H) for L in range(n_layers) for H in range(n_heads)]

    # 1. Identify NMs via component-DLA top-k on clean prompts, OR use
    # caller-supplied pinned NMs (HYPOTHESIS.md §H5-6: pin NMs across ablation
    # conditions so the receiver geometry is fixed and any Δ_h change is
    # attributable to the path through the pinned receivers, not to a shift
    # in which heads are receivers).
    if nm_heads_override is not None:
        if len(nm_heads_override) != k_nm:
            raise ValueError(
                f"nm_heads_override length {len(nm_heads_override)} does not "
                f"match k_nm={k_nm}"
            )
        nm_heads: list[tuple[int, int]] = [(int(L), int(H)) for L, H in nm_heads_override]
    else:
        dla = component_dla(model, clean_prompts, batch_size=nm_dla_batch_size)
        flat = dla.flatten()
        top_idx = torch.argsort(flat, descending=True)[:k_nm]
        nm_heads = [
            (int(i.item()) // n_heads, int(i.item()) % n_heads) for i in top_idx
        ]

    # 2. Tokenize clean and corrupt; group by sequence length.
    clean_token_rows = [model.to_tokens(p.text, prepend_bos=True)[0] for p in clean_prompts]
    corrupt_token_rows = [
        model.to_tokens(p.corrupt_text, prepend_bos=True)[0] for p in corrupt_prompts
    ]
    if any(c.shape != cc.shape for c, cc in zip(clean_token_rows, corrupt_token_rows)):
        raise RuntimeError(
            "clean and corrupt token lengths differ for some prompt; ABC "
            "corruption must preserve length."
        )

    by_len: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(clean_token_rows):
        by_len[int(t.shape[0])].append(i)

    # 3. For each length group, compute per-(sender, NM) sum across prompts.
    n_senders = len(senders)
    n_total = len(clean_prompts)
    per_nm_sum_total = torch.zeros(n_senders, k_nm, dtype=torch.float32)
    valid_total = torch.zeros(n_senders, k_nm, dtype=torch.long)
    per_prompt_delta_total: torch.Tensor | None = (
        torch.zeros(n_senders, k_nm, n_total, dtype=torch.float32)
        if return_per_prompt
        else None
    )
    total_prompts = 0
    device = next(model.parameters()).device

    for length, indices in sorted(by_len.items()):
        for chunk_start in range(0, len(indices), batch_size):
            chunk_idx = indices[chunk_start : chunk_start + batch_size]
            clean_batch = torch.stack([clean_token_rows[i] for i in chunk_idx]).to(device)
            corrupt_batch = torch.stack(
                [corrupt_token_rows[i] for i in chunk_idx]
            ).to(device)
            positions = [
                _locate_positions(
                    clean_token_rows[i],
                    clean_prompts[i].s_token_id,
                    clean_prompts[i].io_token_id,
                )
                for i in chunk_idx
            ]
            chunk_sum, chunk_valid, chunk_n, chunk_per_prompt = _compute_delta_for_group(
                model,
                clean_batch,
                corrupt_batch,
                positions=positions,
                nm_heads=nm_heads,
                senders=senders,
            )
            per_nm_sum_total += chunk_sum
            valid_total += chunk_valid
            total_prompts += int(chunk_n.item())
            if per_prompt_delta_total is not None:
                for k_in_chunk, prompt_i in enumerate(chunk_idx):
                    per_prompt_delta_total[:, :, prompt_i] = chunk_per_prompt[:, :, k_in_chunk]

    # 4. Average per (sender, NM) by valid prompt count; aggregate Δ_h per sender
    # over its downstream NMs.
    per_nm_matrix = torch.zeros(n_layers, n_heads, k_nm, dtype=torch.float32)
    delta_h = torch.zeros(n_layers, n_heads, dtype=torch.float32)
    for s_idx, (L, H) in enumerate(senders):
        valid_row = valid_total[s_idx]
        downstream_mask = valid_row > 0
        if not downstream_mask.any():
            continue
        means = torch.zeros(k_nm, dtype=torch.float32)
        means[downstream_mask] = (
            per_nm_sum_total[s_idx, downstream_mask]
            / valid_row[downstream_mask].to(torch.float32)
        )
        per_nm_matrix[L, H] = means
        delta_h[L, H] = means[downstream_mask].mean()

    return SInhibitionResult(
        delta_h=delta_h,
        per_nm_matrix=per_nm_matrix,
        nm_heads=nm_heads,
        n_prompts=total_prompts,
        per_prompt_delta=per_prompt_delta_total,
    )
