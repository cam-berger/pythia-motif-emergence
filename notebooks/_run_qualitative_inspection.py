"""Day 4 qualitative inspection of Pythia-410M anchor candidates.

For each candidate from the top-5 in PILOT_RESULTS.md (L2H8 strict, plus
L19H3, L1H4, L0H15, L1H6 as supplementary OV-strong heads with QK >= 0.05),
inspect:

  1. The rank-1 worked-example position from the per-position OV cache —
     where this head's per-position OV contribution is most-negative.
  2. Attention pattern at that position: does the head attend from the query
     position back to the prior occurrence(s) of token_i?
  3. Ablation effect: does ablating this head at this position raise the
     logit on token_i (the duplicate)?
  4. Top-3 worked-example positions for sanity (does the same head fire on
     similar position-types?).

Outputs printed summaries; the human (Day 4 inspector) writes verdicts into
PILOT_RESULTS.md based on the printed evidence.

Run:
    uv run python notebooks/_run_qualitative_inspection.py
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from pathlib import Path
from time import time

import numpy as np
import torch
from torch.nn.functional import log_softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.corpus_io import load_corpus
from src.utils.pythia_loader import load_pythia

ANCHOR_NPZ = ROOT / "data" / "pilot" / "copy_suppression_pythia_410m_step143000_per_position.npz"

CANDIDATES = [
    (2, 8, "rank 1 (strict-passing)"),
    (19, 3, "rank 2 (OV-strong, QK ≥ 0.05)"),
    (1, 4, "rank 3 (high-QK, weak-OV)"),
    (0, 15, "rank 4 (mid-QK, mid-OV)"),
    (1, 6, "rank 5 (mid-QK, mid-OV)"),
]


def ablate_head_hook(layer: int, head: int):
    name = f"blocks.{layer}.attn.hook_z"

    def hook(z, hook):
        z[:, :, head, :] = 0
        return z

    return (name, hook)


def inspect_head(
    model, sequences, passages, per_position_ov, per_position_meta, layer, head, label, device
):
    print()
    print("=" * 78)
    print(f"L{layer}H{head} — {label}")
    print("=" * 78)

    # Per-position OV ranking for this head
    ov_lh = per_position_ov[:, layer, head]  # (n_eligible,)
    rank_order = np.argsort(ov_lh)  # ascending — most-negative first

    print(f"\n  Per-position OV stats: min={ov_lh.min():+.3f}, "
          f"median={np.median(ov_lh):+.3f}, max={ov_lh.max():+.3f}")
    print(f"  Mean (head-level OV score): {ov_lh.mean():+.4f}")

    print(f"\n  Top-3 worked-example positions (most-negative per-position OV):")
    for r in range(3):
        k = rank_order[r]
        passage_idx, pos = per_position_meta[k]
        seq = sequences[passage_idx]
        tok_id = int(seq[pos].item())
        tok_str = model.to_string([tok_id])
        lo = max(0, pos - 5)
        hi = min(len(seq), pos + 3)
        surrounding = model.to_string(seq[lo:hi].tolist())
        print(f"    rank {r+1}: passage {passage_idx} ({passages[passage_idx].title!r}), "
              f"pos {pos}, token={tok_str!r}, OV={ov_lh[k]:+.3f}")
        print(f"             context = ...{surrounding!r}...")

    # Detailed inspection at rank-1 position
    k = int(rank_order[0])
    passage_idx, pos = int(per_position_meta[k][0]), int(per_position_meta[k][1])
    seq = sequences[passage_idx]
    tok_id = int(seq[pos].item())
    tok_str = model.to_string([tok_id])

    if pos == 0:
        print(f"\n  WARNING: rank-1 pos is 0; cannot inspect. Skipping ablation.")
        return

    tokens_2d = seq.unsqueeze(0).to(device)

    # Attention pattern at the query position
    _, attn_cache = model.run_with_cache(
        tokens_2d, names_filter=lambda n: n.endswith("hook_pattern"), return_type=None
    )
    pat = attn_cache[f"blocks.{layer}.attn.hook_pattern"][0, head]  # (q, k)

    # Find prior occurrences of tok_i in seq[:pos]
    prior_positions = [i for i in range(pos) if int(seq[i].item()) == tok_id]
    print(f"\n  Rank-1 query position: pos={pos}, token={tok_str!r}")
    print(f"  Prior occurrences of {tok_str!r}: positions {prior_positions}")

    if not prior_positions:
        print("  WARNING: no prior occurrences found — should not happen for an eligible position.")
        return

    # Attention from query (pos) back to all positions
    attn_row = pat[pos].cpu().float().numpy()
    print(f"  Attention from pos {pos} to prior duplicates: ")
    total_to_priors = 0.0
    for j in prior_positions:
        a_ij = float(attn_row[j])
        total_to_priors += a_ij
        # Get token in surrounding context for j
        lo = max(0, j - 2)
        hi = min(len(seq), j + 3)
        ctx_j = model.to_string(seq[lo:hi].tolist())
        print(f"    -> pos {j} (...{ctx_j!r}...): attn = {a_ij:.3f}")
    print(f"  Total attention to prior-duplicate positions: {total_to_priors:.3f}")
    # Top-5 attention destinations
    top5_idx = np.argsort(attn_row)[::-1][:5]
    print(f"  Top-5 attention destinations from pos {pos}:")
    for j in top5_idx:
        ctx_j = model.to_string(seq[max(0, j-2):min(len(seq), j+3)].tolist())
        is_dup = j in prior_positions
        marker = "  <-- prior duplicate" if is_dup else ""
        print(f"    pos {j} (...{ctx_j!r}...): attn = {attn_row[j]:.3f}{marker}")

    # Ablation effect
    baseline_logits = model(tokens_2d, return_type="logits")[0]
    with model.hooks(fwd_hooks=[ablate_head_hook(layer, head)]):
        ablated_logits = model(tokens_2d, return_type="logits")[0]

    logit_pred_pos = pos - 1
    base_logit = float(baseline_logits[logit_pred_pos, tok_id].item())
    abl_logit = float(ablated_logits[logit_pred_pos, tok_id].item())
    base_nll = float(-log_softmax(baseline_logits[logit_pred_pos], dim=-1)[tok_id].item())
    abl_nll = float(-log_softmax(ablated_logits[logit_pred_pos], dim=-1)[tok_id].item())

    print(f"\n  Ablation at rank-1 worked-example position:")
    print(f"    {'logit on':<25} {'baseline':>10} {'ablated':>10} {'delta':>10}")
    print(f"    {tok_str!r:<25} {base_logit:>+10.3f} {abl_logit:>+10.3f} "
          f"{abl_logit-base_logit:>+10.3f}")
    print(f"    NLL                       {base_nll:>10.3f} {abl_nll:>10.3f} "
          f"{abl_nll-base_nll:>+10.3f}")

    # Top-5 baseline predictions vs top-5 ablated predictions
    base_top = torch.topk(baseline_logits[logit_pred_pos], 5)
    abl_top = torch.topk(ablated_logits[logit_pred_pos], 5)
    print(f"\n  Top-5 baseline predictions at pos {pos-1}:")
    for v, idx in zip(base_top.values.tolist(), base_top.indices.tolist()):
        tok = model.to_string([idx])
        marker = "  <-- duplicate token" if idx == tok_id else ""
        print(f"    {tok!r:>20}: logit {v:+.3f}{marker}")
    print(f"  Top-5 with L{layer}H{head} ablated:")
    for v, idx in zip(abl_top.values.tolist(), abl_top.indices.tolist()):
        tok = model.to_string([idx])
        marker = "  <-- duplicate token" if idx == tok_id else ""
        print(f"    {tok!r:>20}: logit {v:+.3f}{marker}")

    # Heuristic verdict signals
    qk_qual = total_to_priors > 0.10  # head sends >10% attention to prior duplicates
    ov_qual = (abl_logit - base_logit) > 0.20  # ablation raised duplicate logit clearly
    print(f"\n  Heuristic signals:")
    print(f"    QK qualitative (attn to priors > 0.10):     {qk_qual} ({total_to_priors:.3f})")
    print(f"    OV qualitative (delta-logit on dup > 0.20): {ov_qual} ({abl_logit-base_logit:+.3f})")
    if qk_qual and ov_qual:
        verdict = "PASS"
    elif qk_qual or ov_qual:
        verdict = "WEAK"
    else:
        verdict = "FAIL"
    print(f"  Heuristic verdict: {verdict}")
    print(f"  (Final verdict in PILOT_RESULTS.md is human-confirmed; this is a guideline.)")


def main() -> None:
    torch.set_grad_enabled(False)

    print("Loading per-position OV cache ...")
    z = np.load(ANCHOR_NPZ)
    per_position_ov = z["per_position_ov"]  # (n_eligible, n_layers, n_heads)
    per_position_meta = z["per_position_meta"]  # (n_eligible, 2)
    print(f"  shape={per_position_ov.shape}, n_eligible={len(per_position_meta)}")

    print("\nLoading Pythia-410M-deduped @ step143000 ...")
    t0 = time()
    model = load_pythia("410m", step=143000)
    device = next(model.parameters()).device
    print(f"  loaded in {time() - t0:.1f}s")

    print("\nLoading canonical corpus ...")
    passages = load_corpus()
    sequences = [model.to_tokens(p.text)[0].cpu() for p in passages]

    for layer, head, label in CANDIDATES:
        inspect_head(
            model, sequences, passages, per_position_ov, per_position_meta,
            layer, head, label, device
        )


if __name__ == "__main__":
    main()
