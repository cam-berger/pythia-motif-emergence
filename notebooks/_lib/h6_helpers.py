"""§H6-causal shared helpers (pilot at 70M + 410M).

Per HYPOTHESIS.md §H6-causal-2 through §H6-causal-7-agg (locked 2026-05-11).
Houses the head-selection procedures (top-K induction with the 3-way NM /
SI-sender / suc-receiver exclusion, score-bracketed control), the per-prompt
successor-lift readout against pinned receivers, and the per-readout +
cross-readout classifiers (5-pattern aggregate, OR semantics for B/C).

Numerical / procedural locks (all inherited verbatim from §H5-causal-family
unless noted):

- INDUCTION_THRESHOLD = 0.30                §IND-1 / Phase 1.2
- HARD_K_MIN          = 4                   §H6-causal-2 final-pass halt-rule
- NULL band (Readouts A, C): [0.8, 1.2]     §H5-causal-2-6 / §SU-1b
- DEP threshold (Readouts A, C): < 0.5      §H5-causal-2-6
- GENERIC threshold (Readouts A, C): < 0.7  §H5-causal-2-6
- Aggregate priority: GENERIC > DEP-multi > DEP-single > NULL³ > MIXED
- OR semantics for the B/C aggregate (DEP_BC_only fires on B=DEP OR C=DEP)

This module does NOT instantiate a model or touch the parquet store; it
takes already-loaded DataFrames / arrays and returns deterministic
selections. The model-dependent piece — `compute_successor_lift_per_prompt`
— takes a model with whatever hooks are currently installed and returns
per-(layer, head, prompt) lift_dla restricted to the pinned receiver set.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from src.detectors.successor import (
    CATEGORIES,
    SuccessorPrompt,
    _compute_per_prompt_dla,
)
from src.locked_thresholds import (
    DEP_THRESHOLD as _DEP_THRESHOLD,
    GENERIC_THRESHOLD as _GENERIC_THRESHOLD,
    HARD_K_MIN as _HARD_K_MIN,
    INDUCTION_QK as _INDUCTION_QK,
    NULL_HIGH as _NULL_HIGH,
    NULL_LOW as _NULL_LOW,
)

# ---- locks (imported from src.locked_thresholds for audit-trail) ---------

INDUCTION_THRESHOLD: float = _INDUCTION_QK.value
HARD_K_MIN: int = int(_HARD_K_MIN.value)

NULL_LO: float = _NULL_LOW.value
NULL_HI: float = _NULL_HIGH.value
DEP_THRESHOLD: float = _DEP_THRESHOLD.value
GENERIC_THRESHOLD: float = _GENERIC_THRESHOLD.value

CTRL_BRACKET_WIDTH_INIT: float = 0.05
CTRL_BRACKET_WIDTH_STEP: float = 0.025


# ---- selection procedures -------------------------------------------------


def _excluded_label(
    lh: tuple[int, int],
    nm_set: set[tuple[int, int]],
    si_set: set[tuple[int, int]],
    suc_receivers_set: set[tuple[int, int]],
) -> str | None:
    """Return the §H6 exclusion-clause label for `lh`, or None if it survives.

    Order: NM > SI > SUC_RECV (arbitrary but stable for audit logging).
    A head can be in multiple sets; the returned label is just the first
    hit — `n_excluded_*` counts in the verdict parquet are tallied
    separately by the runner.
    """
    if lh in nm_set:
        return "NM"
    if lh in si_set:
        return "SI"
    if lh in suc_receivers_set:
        return "SUC_RECV"
    return None


def select_top_induction(
    df_ind: pd.DataFrame,
    size: str,
    step: int,
    k_size: int,
    nm_set: Iterable[tuple[int, int]],
    si_set: Iterable[tuple[int, int]],
    suc_receivers_set: Iterable[tuple[int, int]],
    *,
    induction_threshold: float = INDUCTION_THRESHOLD,
    hard_k_min: int = HARD_K_MIN,
) -> tuple[list[tuple[int, int, float]], int, list[tuple[tuple[int, int], str, float]]]:
    """§H6-causal-2 + §H6-causal-2-bis: top-K_size induction heads with 3-way
    exclusion (NM ∪ SI senders ∪ suc receivers), §H5-3 tie-break.

    Procedure:
      1. Sort all heads at (size, step) by Olsson prefix-match `score`
         desc, then layer asc, head asc.
      2. Walk down the ranked list; admit a head if it is NOT in any of
         the three exclusion sets AND `score >= induction_threshold`.
         Record skipped heads with their exclusion label.
      3. Stop once `k_size` heads are admitted, OR the candidate list is
         exhausted.
      4. If `len(selected) < hard_k_min`, raise — this is the
         §H6-causal-2 HALT-COEXTENSIVE trigger.
      5. Otherwise return `(selected, widen_depth, excluded_log)`. The
         runner inspects `len(selected)` against `k_size` (K_locked) and
         sets the `structural_caveat_k_exhausted` flag if shorter.

    `widen_depth` is the 0-indexed rank of the last admitted head minus
    `k_size - 1` (i.e. how many positions beyond the initial top-K we had
    to walk because of exclusions); 0 means no widening was required.
    """
    nm = {(int(l), int(h)) for l, h in nm_set}
    si = {(int(l), int(h)) for l, h in si_set}
    sr = {(int(l), int(h)) for l, h in suc_receivers_set}

    sub = df_ind[(df_ind["size"] == size) & (df_ind["step"] == step)].copy()
    sub = sub.sort_values(by=["score", "layer", "head"], ascending=[False, True, True])
    sub = sub.reset_index(drop=True)

    selected: list[tuple[int, int, float]] = []
    excluded_log: list[tuple[tuple[int, int], str, float]] = []
    last_rank_admitted = -1
    for rank, r in sub.iterrows():
        l, h, s = int(r["layer"]), int(r["head"]), float(r["score"])
        if s < induction_threshold:
            break
        lbl = _excluded_label((l, h), nm, si, sr)
        if lbl is not None:
            excluded_log.append(((l, h), lbl, s))
            continue
        selected.append((l, h, s))
        last_rank_admitted = int(rank)
        if len(selected) >= k_size:
            break

    if len(selected) < hard_k_min:
        raise RuntimeError(
            f"§H6-causal-2 HALT-COEXTENSIVE: only {len(selected)} induction "
            f"heads survived the 3-way (NM + SI + suc-receiver) exclusion at "
            f"({size}, step={step}); hard minimum is {hard_k_min}. "
            f"Excluded: {excluded_log}. Pre-data: this triggers the "
            f"HALT-COEXTENSIVE pattern; sub-amendment required before "
            f"re-attempting §H6 at this anchor."
        )

    # widen_depth = (rank-of-last-admitted) - (K - 1). 0 if no widening.
    widen_depth = max(0, last_rank_admitted - (len(selected) - 1))
    return selected, widen_depth, excluded_log


def select_ctrl_induction(
    df_ind: pd.DataFrame,
    size: str,
    step: int,
    ind_set: Iterable[tuple[int, int]],
    nm_set: Iterable[tuple[int, int]],
    si_set: Iterable[tuple[int, int]],
    suc_receivers_set: Iterable[tuple[int, int]],
    k: int,
    *,
    bw_init: float = CTRL_BRACKET_WIDTH_INIT,
    bw_step: float = CTRL_BRACKET_WIDTH_STEP,
    induction_threshold: float = INDUCTION_THRESHOLD,
    seed: int = 0,
) -> tuple[list[tuple[int, int, float]], float]:
    """§H6-causal-3: score-bracket-matched random control, with the same
    3-way exclusion as the ablation set.

    The bracket is `[induction_threshold - bw, induction_threshold)` on
    Olsson prefix-match score. Excludes `ind_set ∪ nm_set ∪ si_set ∪
    suc_receivers_set`. Widens by `bw_step` (default 0.025) until at least
    `k` non-conflicting candidates are present. `rng = np.random.default_rng(seed)`
    (seed = 0 per §H6-causal-3, inherited from §H5-4).

    Returns `(chosen, final_bw)`. Raises with §H6-causal-3 HALT-CTRL-EMPTY
    if the bracket can't be filled even at `bw > 1.0`.
    """
    excl = (
        {(int(l), int(h)) for l, h in ind_set}
        | {(int(l), int(h)) for l, h in nm_set}
        | {(int(l), int(h)) for l, h in si_set}
        | {(int(l), int(h)) for l, h in suc_receivers_set}
    )

    sub = df_ind[(df_ind["size"] == size) & (df_ind["step"] == step)].copy()
    bw = bw_init
    while True:
        lo, hi = induction_threshold - bw, induction_threshold
        mask = (sub["score"] >= lo) & (sub["score"] < hi)
        cand = sub[mask].copy()
        cand = cand[
            ~cand.apply(lambda r: (int(r["layer"]), int(r["head"])) in excl, axis=1)
        ]
        if len(cand) >= k:
            break
        bw += bw_step
        if bw > 1.0:
            raise RuntimeError(
                f"§H6-causal-3 HALT-CTRL-EMPTY: could not assemble {k} ctrl "
                f"candidates in [{lo:.3f}, {hi:.3f}) at ({size}, step={step}) "
                f"even with bracket_width={bw:.3f}. Sub-amendment required."
            )
    cand = cand.sort_values(by=["layer", "head"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(cand), size=k, replace=False)
    chosen = cand.iloc[sorted(chosen_idx.tolist())]
    return (
        [(int(r["layer"]), int(r["head"]), float(r["score"])) for _, r in chosen.iterrows()],
        float(bw),
    )


# ---- Readout A: successor lift_dla per (receiver, prompt) -----------------


def compute_successor_lift_per_prompt(
    model,
    prompts: list[SuccessorPrompt],
    suc_receiver_heads: list[tuple[int, int]],
    *,
    batch_size: int = 8,
) -> dict[tuple[int, int], np.ndarray]:
    """§SU-1b per-prompt lift_dla, restricted to the pinned receiver set.

    Runs the model (under whatever forward hooks are currently installed
    — `model.reset_hooks()` is NOT called inside) over both clean and
    shuffled variants of every prompt, computes per-(layer, head, prompt)
    DLA at the END token toward the target's first-token id (the §SU-2
    target), then for each prompt subtracts the *category-mean* of the
    shuffled DLA to give the per-prompt lift.

    Aggregation per §SU-1b: `lift_per_prompt[i] = real_dla[i] -
    mean_null_for_category(prompt_categories[i])`. The category-null
    baseline is the within-category mean across shuffled prompts; this
    matches the offline-screen aggregation (mean within category before
    mean across categories) at the per-prompt grain.

    Returns `dict[(layer, head), np.ndarray]` of shape `(len(prompts),)`,
    keyed by the heads in `suc_receiver_heads` only — we compute the full
    per-head matrix internally (the path-patching forward pass cost is
    independent of the slice) and slice at aggregation time.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    clean_token_rows = [
        model.to_tokens(p.clean_text, prepend_bos=True)[0] for p in prompts
    ]
    shuffled_token_rows = [
        model.to_tokens(p.shuffled_text, prepend_bos=True)[0] for p in prompts
    ]

    real_per_prompt = torch.zeros(n_layers, n_heads, len(prompts), dtype=torch.float32)
    null_per_prompt = torch.zeros(n_layers, n_heads, len(prompts), dtype=torch.float32)
    for token_rows, dest in (
        (clean_token_rows, real_per_prompt),
        (shuffled_token_rows, null_per_prompt),
    ):
        by_len: dict[int, list[int]] = defaultdict(list)
        for i, t in enumerate(token_rows):
            by_len[int(t.shape[0])].append(i)
        for length, indices in sorted(by_len.items()):
            batch_tokens = torch.stack([token_rows[i] for i in indices])
            target_ids = torch.tensor(
                [prompts[i].target_first_token_id for i in indices], dtype=torch.long
            )
            dla = _compute_per_prompt_dla(
                model, batch_tokens, target_ids, batch_size=batch_size
            )
            for k, i in enumerate(indices):
                dest[:, :, i] = dla[:, :, k]

    # Per-category mean of the shuffled (null) DLA, broadcast back per prompt.
    prompt_cat: list[str] = [p.category for p in prompts]
    cat_mean: dict[str, torch.Tensor] = {}
    for cat in CATEGORIES:
        cat_indices = [i for i, c in enumerate(prompt_cat) if c == cat]
        if not cat_indices:
            continue
        cat_mean[cat] = null_per_prompt[:, :, cat_indices].mean(dim=-1)  # (L, H)

    lift_per_prompt = torch.zeros_like(real_per_prompt)
    for i, cat in enumerate(prompt_cat):
        lift_per_prompt[:, :, i] = real_per_prompt[:, :, i] - cat_mean[cat]

    out: dict[tuple[int, int], np.ndarray] = {}
    for l, h in suc_receiver_heads:
        out[(int(l), int(h))] = lift_per_prompt[int(l), int(h), :].cpu().numpy().astype(np.float32)
    return out


# ---- per-readout classifier (A and C share the band; B has its own) -------


def classify_lift(
    ratio_suc_lift: float,
    suc_lo: float,
    suc_hi: float,
    ratio_ctrl_lift: float,
    ctrl_lo: float,
    ctrl_hi: float,
) -> str:
    """§H6-causal-7 Readout-A / Readout-C verdict band classifier.

    Used for Readout A (aggregate suc lift_dla ratio) and, with identical
    thresholds, Readout C (IO−S logit-diff ratio). NULL band [0.8, 1.2]
    inclusive on both bounds; DEP requires `ratio_suc < 0.5` with
    `suc_hi < 0.5` (the CI excluding 0.5) AND ctrl inside the NULL band;
    GENERIC if both ratios are below 0.7; else MIXED.
    """
    suc_in_null = (NULL_LO <= ratio_suc_lift <= NULL_HI) and (NULL_LO <= suc_lo) and (suc_hi <= NULL_HI)
    ctrl_in_null = (NULL_LO <= ratio_ctrl_lift <= NULL_HI) and (NULL_LO <= ctrl_lo) and (ctrl_hi <= NULL_HI)
    suc_dep = (ratio_suc_lift < DEP_THRESHOLD) and (suc_hi < DEP_THRESHOLD)
    ctrl_in_null_loose = NULL_LO <= ratio_ctrl_lift <= NULL_HI
    both_below_generic = (
        (ratio_suc_lift < GENERIC_THRESHOLD) and (ratio_ctrl_lift < GENERIC_THRESHOLD)
    )

    if suc_in_null and ctrl_in_null:
        return "NULL"
    if suc_dep and ctrl_in_null_loose:
        return "DEP"
    if both_below_generic:
        return "GENERIC"
    return "MIXED"


# ---- cross-readout aggregate ---------------------------------------------


def aggregate_cross_readout(
    verdict_A: str, verdict_B: str, verdict_C: str
) -> str:
    """§H6-causal-7-agg cross-readout aggregate.

    Five patterns, OR semantics for B/C (DEP_BC_only fires when B OR C
    returns DEP, not require both). Priority order, applied top-down:

      1. GENERIC_any   — any of the three readouts = GENERIC.
      2. NULL_all      — all three = NULL.
      3. DEP_A_and_BC  — A = DEP AND (B = DEP OR C = DEP).
      4. DEP_A_only    — A = DEP AND B != DEP AND C != DEP.
      5. DEP_BC_only   — A != DEP AND (B = DEP OR C = DEP).
      6. MIXED         — the catch-all.

    The five named patterns (1-5; "GENERIC_any" maps to the GENERIC bucket
    in §H6-causal-7-agg's table) are pre-committed; see `PAPER_HEADLINES`.
    """
    if "GENERIC" in (verdict_A, verdict_B, verdict_C):
        return "GENERIC_any"
    if verdict_A == "NULL" and verdict_B == "NULL" and verdict_C == "NULL":
        return "NULL_all"

    a_dep = verdict_A == "DEP"
    bc_dep = (verdict_B == "DEP") or (verdict_C == "DEP")
    if a_dep and bc_dep:
        return "DEP_A_and_BC"
    if a_dep and not bc_dep:
        return "DEP_A_only"
    if (not a_dep) and bc_dep:
        return "DEP_BC_only"
    return "MIXED"


# ---- pre-committed paper headlines (per §H6-causal-7-agg table) -----------

PAPER_HEADLINES: dict[str, str] = {
    "NULL_all": (
        "Induction heads are NOT a root for either successor or S-inhibition at "
        "inference time in Pythia-{size} step143000. Combined with the §H5-causal-"
        "family NULL × NULL on the suc → si branch, the ordered temporal "
        "emergence (μ_ind < μ_suc < μ_si) is fully decoupled from inference-time "
        "architectural causal structure at this anchor. Strong direct-mechanistic "
        "refutation of the §H1-C compositional reading; the registered emergence "
        "pattern is consistent with convergent training dynamics, not with a "
        "forward-pass causal chain."
    ),
    "DEP_A_only": (
        "Induction → successor is a real causal chain at inference time in "
        "Pythia-{size} step143000; induction → S-inhibition is not. Combined "
        "with the §H5-causal-family suc → si NULL × NULL, the tree story is "
        "partially confirmed on the suc branch but cleanly falsified on the si "
        "branch. Successor inherits its inference-time computation from "
        "induction; S-inhibition does not."
    ),
    "DEP_BC_only": (
        "Induction → S-inhibition is a real causal chain at inference time in "
        "Pythia-{size} step143000; induction → successor is not. The §H1-C "
        "compositional reading is partially confirmed on the si branch. "
        "Surprising given the structural-reuse data (ind ∩ suc nearly empty), "
        "and supportive of the A12 deep-dive observation that ind ∩ si is "
        "non-empty at every size larger than 70M."
    ),
    "DEP_A_and_BC": (
        "Induction sits at the root of the full successor / S-inhibition tree at "
        "inference time in Pythia-{size} step143000. The temporal emergence "
        "ordering ind → {{suc, si}} reflects a true architectural ordering "
        "rooted in induction. The §H5-causal-family suc → si NULL stands — the "
        "two branches are parallel descendants of induction, not sequential."
    ),
    "GENERIC_any": (
        "Per-readout heterogeneous dependence at Pythia-{size} step143000; no "
        "global verdict on §H6-causal at this anchor. Reported per-readout with "
        "CIs. Readout-sensitivity caveat applies — particularly relevant if "
        "Readout C drops generically as observed in §H5-causal-3-record at 1B."
    ),
    "MIXED": (
        "Per-readout heterogeneous dependence at Pythia-{size} step143000; no "
        "global verdict on §H6-causal at this anchor. Reported per-readout with "
        "CIs. Readout-sensitivity caveat applies — particularly relevant if "
        "Readout C drops generically as observed in §H5-causal-3-record at 1B."
    ),
}


def build_paper_headline(
    pattern: str,
    size: str,
    *,
    structural_caveat_k_exhausted: bool,
    k_effective: int,
    k_locked: int,
) -> str:
    """Format a pre-committed paper headline for `pattern` at `size`, with
    the §H6-causal-7-agg K-exhaustion prefix prepended when applicable.

    The prefix template (locked):
      "[K_effective = {ke} < K_locked = {kl}; structural exhaustion of
       induction-detected population by §H6-2 + §H6-2-bis exclusions] "
    """
    template = PAPER_HEADLINES[pattern]
    body = template.format(size=size)
    if structural_caveat_k_exhausted:
        prefix = (
            f"[K_effective = {k_effective} < K_locked = {k_locked}; "
            f"structural exhaustion of induction-detected population by "
            f"§H6-2 + §H6-2-bis exclusions] "
        )
        return prefix + body
    return body
