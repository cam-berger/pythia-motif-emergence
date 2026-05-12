"""GPT-2 small validation of §H1-C-altdetectors three alternative detectors.

Implements HYPOTHESIS.md §H1-C-altdetectors-1 + §H1-C-altdetectors-2 (the
(c1-uniform) threshold-locking rule). Derives the three TBD thresholds:

  - τ_ind_OV  = min OV-score across (heads with QK-prefix-match > 0.30)
  - K_min     = min argmax-K-of-7 score across (heads with lift_dla ≥ 0.13496)
  - τ_si_DLA  = min Component-DLA-at-S2 across (heads with §S-1 Δ_h ≥ 0.0372)

Alternative detectors (§H1-C-altdetectors-1):
  - Induction OV: per-head DLA at second-half positions toward the
    prior-occurrence-next-token in random-token-repetition sequences.
  - Successor argmax-K-of-7: per-head count of day-of-week transitions
    {Mon→Tue, …, Sun→Mon} where head's DLA correctly argmaxes the target
    day among the 7 day tokens.
  - S-inhibition Component-DLA at S2: per-head DLA on (IO−S) at the S2
    token position in the 200-prompt IOI set (BABA + ABBA, seed=0).

Outputs:
  - data/exploration/gpt2_small_altdetector_validation.parquet
      long-format per-head per-detector scores + per-motif reference-set
      membership flags + derived thresholds
  - data/exploration/gpt2_small_altdetector_per_head.npz
      raw tensors (ov_score, k_score, compdla_s2)

Locked-detector reference sets are recomputed in this script for self-
containment (GPT-2 small is fast; ~15 min total).
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402

from src.detectors.induction import (  # noqa: E402
    build_repetition_sequences,
    prefix_matching_score,
)
from src.detectors.s_inhibition import (  # noqa: E402
    _locate_positions,
    build_abc_corrupted_prompts,
    s_inhibition_screen,
)
from src.detectors.successor import (  # noqa: E402
    DAYS,
    build_successor_prompts,
    successor_screen,
)
from src.replication.tigges_ioi import (  # noqa: E402
    WANG_NAMES,
    build_ioi_prompts,
    filter_single_token_names,
)
from src.utils.mps_compat import assert_mps_fallback_enabled  # noqa: E402

OUT_AGG = REPO_ROOT / "data" / "exploration" / "gpt2_small_altdetector_validation.parquet"
OUT_RAW = REPO_ROOT / "data" / "exploration" / "gpt2_small_altdetector_per_head.npz"

SEED = 0
N_SEQUENCES = 50
SEQ_LEN = 100
N_IOI_PROMPTS = 200
IOI_BATCH_SIZE = 50
SUCC_BATCH_SIZE = 16
IND_BATCH_SIZE = 8

# Locked detector thresholds (§H1-C + §SU-tau + §S-tau)
TAU_IND_QK = 0.30
TAU_SUC_LIFT = 0.13496
TAU_SI_PP = 0.0372


# ----------------------------------------------------------------------------
# Alt-detector: Induction OV-circuit score
# ----------------------------------------------------------------------------


def induction_ov_score(
    model: HookedTransformer,
    *,
    n_sequences: int = N_SEQUENCES,
    seq_len: int = SEQ_LEN,
    seed: int = SEED,
    batch_size: int = IND_BATCH_SIZE,
) -> torch.Tensor:
    """Per-head OV-circuit verification score per Olsson 2022 (§H1-C-altdetectors-1).

    For each second-half position p in a repetition sequence, the "prior-
    occurrence-next-token" direction is W_U[:, target] where target =
    tokens[p - half + 1] (the token immediately following the prior occurrence
    of tokens[p]). Per-head OV-score = mean over (sequences × second-half
    positions) of the head's direct-logit-attribution at position p toward
    that target direction.

    Uses the same random-token-repetition prompts as the locked QK detector
    (seed=0, n=50, len=100). Mechanically distinct: tests OV-side rather
    than QK-side of the induction circuit.
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

    W_U = model.W_U  # (d_model, d_vocab)

    score_sum = torch.zeros(n_layers, n_heads, dtype=torch.float32)
    n_terms = 0

    second_half_positions = torch.arange(half, seq_len, dtype=torch.long)
    target_offsets = second_half_positions - half + 1  # (half,)

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
                # For each second-half position, target_id = batch_tokens[b, target_offsets[p_idx]]
                # We want DLA = sum over d_model of (result[b, p, h, :] * W_U[:, target_id])
                # Vectorize: gather target token IDs once, build per-prompt directions.
                target_ids = batch_tokens[:, target_offsets.to(device)]  # (B, half)
                # directions[b, t, :] = W_U[:, target_ids[b, t]]
                directions = W_U[:, target_ids].permute(1, 2, 0).to(torch.float32)  # (B, half, d_model)

                for layer in range(n_layers):
                    result = cache[f"blocks.{layer}.attn.hook_result"]
                    # result shape: (B, seq_len, n_heads, d_model)
                    # We need positions in second_half: result[:, half:, :, :]
                    second_half_result = result[:, half:, :, :].to(torch.float32)
                    # (B, half, n_heads, d_model) × (B, half, 1, d_model) → sum over d_model
                    contrib = (second_half_result * directions[:, :, None, :]).sum(dim=-1)
                    # contrib shape: (B, half, n_heads); mean over half positions, sum over batch
                    per_seq_per_head = contrib.mean(dim=1).cpu()  # (B, n_heads)
                    score_sum[layer] += per_seq_per_head.sum(dim=0)
                del cache
                n_terms += B
    finally:
        model.set_use_attn_result(prev_use_attn_result)

    return score_sum / max(n_terms, 1)


# ----------------------------------------------------------------------------
# Alt-detector: Successor argmax-K-of-7 (L 2023 day-of-week protocol)
# ----------------------------------------------------------------------------


def successor_argmax_k_of_7(
    model: HookedTransformer,
    *,
    batch_size: int = SUCC_BATCH_SIZE,
) -> torch.Tensor:
    """Per-head K-score (integer in 0..7) on the 7 day-of-week transitions.

    For each transition {(Sun,Mon,Tue→Wed), (Mon,Tue,Wed→Thu), …, (Fri,Sat,Sun→Mon)},
    build the 3-context prompt 'c1, c2, c3, ' and at the END position compute
    each head's DLA toward each of the 7 day-target-tokens. The head's argmax
    over the 7 day tokens is its prediction; K_score counts how many of the 7
    transitions the head correctly argmaxes the target day.

    Mechanically distinct from the locked successor detector (lift_dla cross-
    category): a per-head argmax-correctness test on a fixed 7-class set,
    not a continuous-lift cross-category aggregate. Per §H1-C-altdetectors-1.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    device = next(model.parameters()).device
    tokenizer = model.tokenizer
    W_U = model.W_U  # (d_model, d_vocab)

    # 7 transitions: starting indices 0..6, target = (i+3) % 7
    day_token_ids: list[int] = []
    for day in DAYS:
        ids = tokenizer.encode(f" {day}", add_special_tokens=False)
        if not ids:
            raise RuntimeError(f"day {day!r} encoded to empty token list")
        day_token_ids.append(int(ids[0]))
    day_token_ids_t = torch.tensor(day_token_ids, dtype=torch.long)  # (7,)

    # Build 7 prompts using cyclic day-of-week sliding window:
    # i=0: (Mon, Tue, Wed) → Thu; i=1: (Tue, Wed, Thu) → Fri; ...; i=6: (Sun, Mon, Tue) → Wed
    prompts: list[tuple[str, int]] = []
    for i in range(7):
        c1 = DAYS[i]
        c2 = DAYS[(i + 1) % 7]
        c3 = DAYS[(i + 2) % 7]
        target_idx = (i + 3) % 7
        text = f"{c1}, {c2}, {c3}, "
        prompts.append((text, target_idx))

    # Tokenize and group by length.
    token_rows = [model.to_tokens(t, prepend_bos=True)[0] for t, _ in prompts]
    targets = [target_idx for _, target_idx in prompts]
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(token_rows):
        by_len[int(t.shape[0])].append(i)

    # Per-head per-prompt DLA toward each of the 7 day tokens.
    # Shape: (n_layers, n_heads, 7, 7) where last two = (prompt_idx, day_idx)
    # We'll first compute (n_layers, n_heads, 7_prompts, 7_days), then argmax.
    per_prompt_per_day = torch.zeros(n_layers, n_heads, 7, 7, dtype=torch.float32)

    # Direction tensor for the 7 day tokens, shape (7, d_model)
    day_directions = W_U[:, day_token_ids_t.to(W_U.device)].T.to(torch.float32)  # (7, d_model)

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
                    B = int(batch.shape[0])
                    for layer in range(n_layers):
                        # result[B, -1, n_heads, d_model] → head's last-position output
                        head_out = cache[f"blocks.{layer}.attn.hook_result"][:, -1, :, :].to(torch.float32)
                        # contrib[b, h, d_idx] = sum_d head_out[b, h, d] * day_directions[d_idx, d]
                        # shape: (B, n_heads, 7)
                        contrib = torch.einsum("bhd,kd->bhk", head_out, day_directions.to(head_out.device)).cpu()
                        for k, i in enumerate(idxs):
                            per_prompt_per_day[layer, :, i, :] = contrib[k]
                    del cache
    finally:
        model.set_use_attn_result(prev_use_attn_result)

    # For each (layer, head, prompt): argmax over 7 days
    argmax_day = per_prompt_per_day.argmax(dim=-1)  # (n_layers, n_heads, 7)
    targets_t = torch.tensor(targets, dtype=torch.long)
    # K_score[L, H] = sum over prompts of (argmax_day[L, H, p] == targets_t[p])
    matches = (argmax_day == targets_t[None, None, :]).to(torch.int32)
    k_score = matches.sum(dim=-1)  # (n_layers, n_heads), values in {0..7}
    return k_score.to(torch.float32), per_prompt_per_day, day_token_ids_t


# ----------------------------------------------------------------------------
# Alt-detector: S-inhibition Component-DLA at S2
# ----------------------------------------------------------------------------


def s_inhibition_compdla_at_s2(
    model: HookedTransformer,
    prompts: list,
    *,
    batch_size: int = 8,
) -> torch.Tensor:
    """Per-head Component-DLA on (IO−S) direction at the S2 token position.

    Mirrors `src.replication.tigges_ioi.component_dla` but evaluates the
    head's residual contribution at the S2 position (the n3 occurrence
    of the duplicate name in 'B gave a obj to') rather than at END.
    The direction is W_U[:, io_token_id] - W_U[:, s_token_id].

    Per §H1-C-altdetectors-1: a direct DLA readout at S2, not a frozen-
    path patching perturbation. Mechanically distinct from §S-1 Δ_h.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    device = next(model.parameters()).device
    W_U = model.W_U  # (d_model, d_vocab)

    # Locate S2 position for every prompt.
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
                    s2_idx = torch.tensor([s2_positions[i] for i in idxs], dtype=torch.long, device=device)
                    directions = torch.stack(
                        [
                            W_U[:, prompts[i].io_token_id]
                            - W_U[:, prompts[i].s_token_id]
                            for i in idxs
                        ]
                    ).to(torch.float32)  # (B, d_model)
                    _, cache = model.run_with_cache(
                        batch,
                        names_filter=lambda n: "attn.hook_result" in n,
                        return_type=None,
                    )
                    B = int(batch.shape[0])
                    batch_idx = torch.arange(B, device=device)
                    for layer in range(n_layers):
                        full_result = cache[f"blocks.{layer}.attn.hook_result"]
                        # full_result: (B, seq_len, n_heads, d_model)
                        result_at_s2 = full_result[batch_idx, s2_idx, :, :].to(torch.float32)
                        # (B, n_heads, d_model) × (B, 1, d_model)
                        contrib = (result_at_s2 * directions[:, None, :]).sum(dim=-1)
                        accumulator[layer] += contrib.sum(dim=0).cpu()
                    del cache
    finally:
        model.set_use_attn_result(prev_use_attn_result)

    accumulator /= len(prompts)
    return accumulator


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    assert_mps_fallback_enabled()
    OUT_AGG.parent.mkdir(parents=True, exist_ok=True)

    print("Loading GPT-2 small...", flush=True)
    t0 = time.time()
    model = HookedTransformer.from_pretrained("gpt2")
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"  loaded in {time.time()-t0:.1f}s; n_layers={n_layers}, n_heads={n_heads}")

    # ------------------------------------------------------------------
    # 1. Locked detectors (recompute on GPT-2 small for reference sets)
    # ------------------------------------------------------------------

    print("\n=== [1/3] Locked QK induction detector (n_seq=50, len=100, seed=0) ===", flush=True)
    t0 = time.time()
    qk_result = prefix_matching_score(
        model, n_sequences=N_SEQUENCES, seq_len=SEQ_LEN, seed=SEED,
        batch_size=IND_BATCH_SIZE,
    )
    qk_scores = qk_result.scores  # (n_layers, n_heads)
    print(f"  done in {time.time()-t0:.1f}s; top-5: {qk_result.top_k(5)}")

    print("\n=== [2/3] Locked lift_dla successor detector (4-category, seed=0) ===", flush=True)
    t0 = time.time()
    succ_prompts = build_successor_prompts(model.tokenizer, seed=SEED)
    succ_result = successor_screen(model, succ_prompts, batch_size=SUCC_BATCH_SIZE)
    lift_dla = succ_result.lift_dla
    print(f"  done in {time.time()-t0:.1f}s; lift_dla L9H1={float(lift_dla[9,1]):+.4f}")

    print("\n=== [3/3] Locked §S-1 path-patching S-inhibition (200 IOI prompts) ===", flush=True)
    t0 = time.time()
    ioi_clean = build_ioi_prompts(seed=SEED, n=N_IOI_PROMPTS, tokenizer=model.tokenizer)
    ioi_corrupt = build_abc_corrupted_prompts(ioi_clean, model.tokenizer, seed=SEED)
    si_result = s_inhibition_screen(
        model, ioi_clean, ioi_corrupt, batch_size=IOI_BATCH_SIZE,
    )
    si_delta = si_result.delta_h  # (n_layers, n_heads)
    print(f"  done in {time.time()-t0:.1f}s; NMs={si_result.nm_heads}")

    # ------------------------------------------------------------------
    # 2. Reference sets via locked thresholds
    # ------------------------------------------------------------------

    ind_ref_mask = qk_scores > TAU_IND_QK
    suc_ref_mask = lift_dla >= TAU_SUC_LIFT
    si_ref_mask = si_delta >= TAU_SI_PP

    print(f"\n=== Reference sets (GPT-2 small) ===")
    print(f"  induction (QK > {TAU_IND_QK}):  {int(ind_ref_mask.sum())} heads")
    print(f"  successor (lift ≥ {TAU_SUC_LIFT}):  {int(suc_ref_mask.sum())} heads")
    print(f"  S-inhibition (Δ_h ≥ {TAU_SI_PP}):  {int(si_ref_mask.sum())} heads")

    def _list(mask: torch.Tensor) -> list[tuple[int, int]]:
        return [(int(L), int(H)) for L, H in mask.nonzero().tolist()]

    ind_ref = _list(ind_ref_mask)
    suc_ref = _list(suc_ref_mask)
    si_ref = _list(si_ref_mask)
    print(f"  ind ref: {ind_ref}")
    print(f"  suc ref: {suc_ref}")
    print(f"  si  ref: {si_ref}")

    # ------------------------------------------------------------------
    # 3. Alt-detectors
    # ------------------------------------------------------------------

    print("\n=== Alt-detector [1/3]: Induction OV-circuit verification ===", flush=True)
    t0 = time.time()
    ov_score = induction_ov_score(model)
    print(f"  done in {time.time()-t0:.1f}s")

    print("\n=== Alt-detector [2/3]: Successor argmax-K-of-7 ===", flush=True)
    t0 = time.time()
    k_score, k_per_prompt, day_ids = successor_argmax_k_of_7(model)
    print(f"  done in {time.time()-t0:.1f}s; day token ids: {day_ids.tolist()}")

    print("\n=== Alt-detector [3/3]: S-inhibition Component-DLA at S2 ===", flush=True)
    t0 = time.time()
    compdla_s2 = s_inhibition_compdla_at_s2(model, ioi_clean, batch_size=8)
    print(f"  done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 4. Derive thresholds via (c1-uniform) rule: min alt-score across ref set
    # ------------------------------------------------------------------

    def _min_over_ref(alt_scores: torch.Tensor, ref_mask: torch.Tensor) -> float:
        if int(ref_mask.sum()) == 0:
            return float("nan")
        return float(alt_scores[ref_mask].min().item())

    tau_ind_OV = _min_over_ref(ov_score, ind_ref_mask)
    K_min_raw = _min_over_ref(k_score, suc_ref_mask)
    K_min = int(K_min_raw) if not (K_min_raw != K_min_raw) else None  # NaN check
    tau_si_DLA = _min_over_ref(compdla_s2, si_ref_mask)

    print(f"\n=== Locked alt-thresholds (§H1-C-altdetectors-2-locked) ===")
    print(f"  τ_ind_OV   = {tau_ind_OV:+.6f}  (min OV-score across {len(ind_ref)} ind-ref heads)")
    print(f"  K_min      = {K_min}            (min K-score across {len(suc_ref)} suc-ref heads; raw={K_min_raw})")
    print(f"  τ_si_DLA   = {tau_si_DLA:+.6f}  (min CompDLA-S2 across {len(si_ref)} si-ref heads)")

    # ------------------------------------------------------------------
    # 5. Persist
    # ------------------------------------------------------------------

    rows: list[dict] = []
    for L in range(n_layers):
        for H in range(n_heads):
            rows.append(dict(
                model="gpt2-small", layer=L, head=H,
                qk_score=float(qk_scores[L, H]),
                lift_dla=float(lift_dla[L, H]),
                si_delta=float(si_delta[L, H]),
                ov_score=float(ov_score[L, H]),
                k_score=int(k_score[L, H].item()),
                compdla_s2=float(compdla_s2[L, H]),
                in_ind_ref=bool(ind_ref_mask[L, H]),
                in_suc_ref=bool(suc_ref_mask[L, H]),
                in_si_ref=bool(si_ref_mask[L, H]),
            ))
    df = pd.DataFrame(rows)
    df.attrs["tau_ind_OV"] = tau_ind_OV
    df.attrs["K_min"] = K_min
    df.attrs["tau_si_DLA"] = tau_si_DLA
    df.to_parquet(OUT_AGG, index=False)
    print(f"\nWrote {OUT_AGG.relative_to(REPO_ROOT)}  ({len(df):,} rows)")

    np.savez_compressed(
        OUT_RAW,
        qk_scores=qk_scores.numpy(),
        lift_dla=lift_dla.numpy(),
        si_delta=si_delta.numpy(),
        ov_score=ov_score.numpy(),
        k_score=k_score.numpy(),
        compdla_s2=compdla_s2.numpy(),
        ind_ref_mask=ind_ref_mask.numpy(),
        suc_ref_mask=suc_ref_mask.numpy(),
        si_ref_mask=si_ref_mask.numpy(),
        tau_ind_OV=np.array(tau_ind_OV, dtype=np.float64),
        K_min=np.array(K_min if K_min is not None else -1, dtype=np.int64),
        tau_si_DLA=np.array(tau_si_DLA, dtype=np.float64),
        day_token_ids=day_ids.numpy(),
    )
    print(f"Wrote {OUT_RAW.relative_to(REPO_ROOT)}")

    print("\n=== NEXT: write §H1-C-altdetectors-2-locked amendment with these three values ===")
    print(f"  τ_ind_OV  = {tau_ind_OV:+.6f}")
    print(f"  K_min     = {K_min}")
    print(f"  τ_si_DLA  = {tau_si_DLA:+.6f}")


if __name__ == "__main__":
    main()
