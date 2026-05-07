"""§H5-causal anchor experiment: causal-dependence ablation at Pythia-410m step143000.

Pre-registered in HYPOTHESIS.md §H5-causal (commit `5fda282`, 2026-05-07).
Tests whether the §S-1 path-patching Δ_h of the registered top-3 S-inhibition
senders depends on the top-5 successor heads being live, vs a score-bracket-
matched random control set drawn from heads near-but-below τ_lift.

Locked decisions (§H5-3, §H5-4, §H5-5, §H5-6, §H5-7, §H5-8):
- Suc set: top-5 by score_suc at this checkpoint, ties broken by layer asc
  then head asc.
- Ctrl set: random sample from {(l,h) : score_suc(l,h) ∈ [τ_lift - 0.05, τ_lift)},
  bracket widened symmetrically by 0.025 if <5 candidates, seed=0.
- Si senders: top-3 by clean Δ_h at anchor (computed live in this run).
- NMs: pinned to clean component-DLA top-4, frozen across all conditions.
- Mean-ablation: replace `hook_z[:, :, head, :]` with batch-mean per length
  group, broadcast across batch.
- DEP gate: 0.5 × clean Δ_h. NULL band: ±20%. GENERIC overrides DEP.
- Bootstrap: B=200 paired per-prompt, seed=1. CI on drop ratio.

Outputs (data/exploration/, all gitignored per *.parquet rule):
- phase4_causal_410m_anchor.parquet           per-prompt Δ_h
- phase4_causal_410m_anchor_summary.parquet   per-(condition, sender) means + CI
- phase4_causal_410m_anchor_verdict.parquet   single-row gate verdict
- phase4_causal_410m_anchor.log               captured stdout (via tee)
"""

from __future__ import annotations

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

from notebooks._lib.sweep_io import read_long  # noqa: E402
from src.detectors.s_inhibition import (  # noqa: E402
    build_abc_corrupted_prompts,
    s_inhibition_screen,
)
from src.replication.tigges_ioi import load_ioi_prompts  # noqa: E402
from src.utils.mps_compat import assert_mps_fallback_enabled  # noqa: E402
from src.utils.pythia_loader import load_pythia  # noqa: E402

EXPL = REPO_ROOT / "data" / "exploration"
OUT_PER_PROMPT = EXPL / "phase4_causal_410m_anchor.parquet"
OUT_SUMMARY = EXPL / "phase4_causal_410m_anchor_summary.parquet"
OUT_VERDICT = EXPL / "phase4_causal_410m_anchor_verdict.parquet"

PROMPTS_PATH = REPO_ROOT / "data" / "prompts" / "ioi_prompts.tsv"
SIZE = "410m"
STEP = 143000

# §SU-tau / §S-tau locks
TAU_LIFT = 0.13496
TAU_STRICT = 0.0372

# §H5-3 / §H5-4
ABLATE_K = 5
BRACKET_WIDTH_INIT = 0.05
BRACKET_WIDTH_STEP = 0.025

# §H5-5
SI_SENDER_K = 3

# §H5-7 gate thresholds
DEP_THRESHOLD = 0.5
NULL_BAND = 0.20

# §H5-8
B_BOOTSTRAP = 200


def select_top_suc(df_suc: pd.DataFrame, k: int) -> list[tuple[int, int, float]]:
    """§H5-3: top-k by score_suc; ties broken by layer asc then head asc."""
    sub = df_suc[(df_suc["size"] == SIZE) & (df_suc["step"] == STEP)].copy()
    sub = sub.sort_values(by=["score", "layer", "head"], ascending=[False, True, True])
    head_rows = sub.head(k)
    return [(int(r["layer"]), int(r["head"]), float(r["score"])) for _, r in head_rows.iterrows()]


def select_ctrl_set(
    df_suc: pd.DataFrame,
    suc_set: list[tuple[int, int, float]],
    k: int,
    bracket_width_init: float,
    bracket_step: float,
) -> tuple[list[tuple[int, int, float]], float]:
    """§H5-4: random from score-bracket near-but-below τ_lift; widen if <k.

    Returns (ctrl_set, final_bracket_width).
    """
    sub = df_suc[(df_suc["size"] == SIZE) & (df_suc["step"] == STEP)].copy()
    suc_layer_head = {(l, h) for l, h, _ in suc_set}
    bw = bracket_width_init
    while True:
        lo, hi = TAU_LIFT - bw, TAU_LIFT
        mask = (sub["score"] >= lo) & (sub["score"] < hi)
        candidates = sub[mask].copy()
        candidates = candidates[
            ~candidates.apply(lambda r: (int(r["layer"]), int(r["head"])) in suc_layer_head, axis=1)
        ]
        if len(candidates) >= k:
            break
        bw += bracket_step
        if bw > 1.0:
            raise RuntimeError(
                f"could not assemble {k} ctrl candidates even with bracket_width={bw}"
            )
    candidates = candidates.sort_values(by=["layer", "head"]).reset_index(drop=True)
    rng = np.random.default_rng(0)
    chosen_idx = rng.choice(len(candidates), size=k, replace=False)
    chosen = candidates.iloc[sorted(chosen_idx.tolist())]
    return (
        [(int(r["layer"]), int(r["head"]), float(r["score"])) for _, r in chosen.iterrows()],
        float(bw),
    )


def select_top_si(df_si: pd.DataFrame, k: int) -> list[tuple[int, int, float]]:
    """§H5-5: top-k S-inhibition senders by Δ_h at this checkpoint."""
    sub = df_si[(df_si["size"] == SIZE) & (df_si["step"] == STEP)].copy()
    sub = sub.sort_values(by=["score", "layer", "head"], ascending=[False, True, True])
    rows = sub.head(k)
    return [(int(r["layer"]), int(r["head"]), float(r["score"])) for _, r in rows.iterrows()]


def precompute_mean_z_by_length(
    model,
    clean_prompts,
    ablate_set: list[tuple[int, int]],
) -> dict[tuple[int, int, int], torch.Tensor]:
    """Precompute mean of hook_z[:, :, head, :] across batch, per length group.

    Returns dict keyed by (layer, head, seq_len) → tensor shape (seq_len, d_head),
    on CPU.
    """
    device = next(model.parameters()).device
    by_len: dict[int, list[int]] = defaultdict(list)
    token_rows = [model.to_tokens(p.text, prepend_bos=True)[0] for p in clean_prompts]
    for i, t in enumerate(token_rows):
        by_len[int(t.shape[0])].append(i)

    layers_to_cache = sorted({l for l, _ in ablate_set})
    name_filter = lambda n: any(n == f"blocks.{L}.attn.hook_z" for L in layers_to_cache)  # noqa: E731

    out: dict[tuple[int, int, int], torch.Tensor] = {}
    with torch.no_grad():
        for length, idxs in sorted(by_len.items()):
            tokens = torch.stack([token_rows[i] for i in idxs]).to(device)
            _, cache = model.run_with_cache(tokens, names_filter=name_filter, return_type=None)
            for layer, head in ablate_set:
                z = cache[f"blocks.{layer}.attn.hook_z"].cpu()
                # z shape (B, T, n_heads, d_head); mean over B
                mean_z = z[:, :, head, :].mean(dim=0)  # (T, d_head)
                out[(layer, head, length)] = mean_z.to(torch.float32)
    return out


def install_mean_ablation_hooks(
    model,
    ablate_set: list[tuple[int, int]],
    mean_z_by_length: dict[tuple[int, int, int], torch.Tensor],
) -> None:
    """§H5-2: install permanent forward hooks that mean-ablate hook_z[..., head, :]."""
    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, head in ablate_set:
        by_layer[layer].append(head)
    device = next(model.parameters()).device

    for layer, heads in by_layer.items():
        layer_local = layer
        heads_local = list(heads)

        def hook_fn(z, hook, _layer=layer_local, _heads=heads_local):
            T = int(z.shape[1])
            for h in _heads:
                key = (_layer, h, T)
                if key not in mean_z_by_length:
                    raise RuntimeError(
                        f"no mean_z precomputed for (layer={_layer}, head={h}, T={T})"
                    )
                mean = mean_z_by_length[key].to(device=z.device, dtype=z.dtype)
                z[:, :, h, :] = mean[None, :, :]
            return z

        model.add_perma_hook(f"blocks.{layer_local}.attn.hook_z", hook_fn, dir="fwd")


def run_condition(
    model,
    clean_prompts,
    corrupt_prompts,
    si_senders: list[tuple[int, int]],
    nm_heads_pinned: list[tuple[int, int]],
    label: str,
    log,
) -> dict[tuple[int, int], np.ndarray]:
    """Run §S-1 detector with pinned NMs and return per-prompt Δ_h per si sender.

    Per-prompt Δ_h is mean across downstream NMs (from result.per_prompt_delta).
    Returns dict {(layer, head): np.ndarray of shape (n_prompts,)}.
    """
    log(f"  [{label}] running s_inhibition_screen with {len(si_senders)} senders, "
        f"NMs pinned to {nm_heads_pinned}")
    t0 = time.time()
    result = s_inhibition_screen(
        model,
        clean_prompts,
        corrupt_prompts,
        senders=si_senders,
        return_per_prompt=True,
        nm_heads_override=nm_heads_pinned,
    )
    log(f"  [{label}] screen complete in {time.time()-t0:.1f}s")
    n_prompts = result.n_prompts
    per_prompt = result.per_prompt_delta  # (n_senders, k_nm, n_prompts)
    out: dict[tuple[int, int], np.ndarray] = {}
    for s_idx, (sl, sh) in enumerate(si_senders):
        downstream_nm_idx = [i for i, (nl, _) in enumerate(nm_heads_pinned) if nl > sl]
        if not downstream_nm_idx:
            out[(sl, sh)] = np.zeros(n_prompts, dtype=np.float32)
            continue
        deltas = per_prompt[s_idx, downstream_nm_idx, :].numpy()  # (k_down, n_prompts)
        out[(sl, sh)] = deltas.mean(axis=0).astype(np.float32)
    return out


def bootstrap_drop_ratio(
    clean_per_prompt: np.ndarray,
    ablated_per_prompt: np.ndarray,
    B: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Paired per-prompt bootstrap of (mean_ablated / mean_clean).

    Returns (point, ci_low_95, ci_high_95).
    """
    n = clean_per_prompt.shape[0]
    point = float(ablated_per_prompt.mean()) / max(float(clean_per_prompt.mean()), 1e-10)
    ratios = np.zeros(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        c = float(clean_per_prompt[idx].mean())
        a = float(ablated_per_prompt[idx].mean())
        ratios[b] = a / max(c, 1e-10)
    lo, hi = float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))
    return point, lo, hi


def classify_per_sender(ratio_suc: float, ratio_ctrl: float) -> str:
    """§H5-7 per-sender classifier."""
    suc_drop = ratio_suc <= DEP_THRESHOLD
    ctrl_drop = ratio_ctrl <= DEP_THRESHOLD
    null_low, null_high = 1.0 - NULL_BAND, 1.0 + NULL_BAND
    suc_in_null = null_low <= ratio_suc <= null_high
    ctrl_in_null = null_low <= ratio_ctrl <= null_high

    if suc_drop and ctrl_drop:
        return "GENERIC"
    if suc_drop and ctrl_in_null:
        return "DEP"
    if suc_in_null and ctrl_in_null:
        return "NULL"
    return "MIXED"


def aggregate_verdict(per_sender: list[str]) -> str:
    """§H5-7 aggregation rule: NULL > GENERIC > DEP > MIXED in priority,
    where 'priority' here means: GENERIC if any sender shows it; NULL if all
    senders show NULL; DEP if all 3 show DEP, or 2 of 3 show DEP and the third
    is in NULL band (handled at the per-sender level — if a sender doesn't
    fit DEP/NULL/GENERIC it's MIXED). Otherwise MIXED.
    """
    if "GENERIC" in per_sender:
        return "GENERIC"
    if all(p == "NULL" for p in per_sender):
        return "NULL"
    if all(p == "DEP" for p in per_sender):
        return "DEP"
    if per_sender.count("DEP") == 2 and per_sender.count("NULL") == 1:
        return "DEP"  # 2-of-3 with the third in NULL band
    return "MIXED"


PAPER_HEADLINES = {
    "NULL": (
        "S-inhibition Δ_h is independent of successor heads at convergence in "
        "Pythia-410m. Direct mechanistic refutation of the forward-pass "
        "compositional reading of §H1-C."
    ),
    "DEP": (
        "S-inhibition causally depends on successor at Pythia-410m convergence; "
        "the §H1-C ordering reflects an inference-time chain. Surprising given "
        "the layer-depth asymmetry; warrants follow-up on the indirect routing."
    ),
    "GENERIC": (
        "Methodological note: Δ_h is not robust to ablation of any near-threshold "
        "head at this checkpoint; metric sensitivity is insufficient for the "
        "test. Verdict deferred pending re-tooling."
    ),
    "MIXED": (
        "Per-sender heterogeneous dependence; no global verdict on §H1-C "
        "compositional reading. Reported by sender."
    ),
}


def main() -> None:
    log_path = EXPL / "phase4_causal_410m_anchor.log"
    log_lines: list[str] = []

    def log(*args, **kwargs):  # tee to stdout and to log buffer
        msg = " ".join(str(a) for a in args)
        print(msg, **kwargs, flush=True)
        log_lines.append(msg)

    assert_mps_fallback_enabled()
    log(f"=== §H5-causal anchor experiment, Pythia-{SIZE}-deduped step{STEP} ===")
    log(f"Pre-registration: HYPOTHESIS.md §H5-causal (commit before this run).")
    t_start = time.time()

    # --- Read parquets to derive ablation sets per §H5-3 / §H5-4 ---
    df_suc = read_long(EXPL / "phase2_successor_sweep.parquet")
    df_si = read_long(EXPL / "phase2_s_inhibition_sweep.parquet")

    suc_set = select_top_suc(df_suc, ABLATE_K)
    log(f"\n§H5-3 suc set (top-{ABLATE_K} by score_suc):")
    for l, h, s in suc_set:
        clears = "≥τ_lift" if s >= TAU_LIFT else "below τ_lift"
        log(f"  L{l}H{h}  score={s:.4f}  ({clears})")

    ctrl_set, widened_bw = select_ctrl_set(
        df_suc, suc_set, ABLATE_K, BRACKET_WIDTH_INIT, BRACKET_WIDTH_STEP
    )
    log(f"\n§H5-4 ctrl set (bracket [{TAU_LIFT - widened_bw:.4f}, {TAU_LIFT:.4f}), "
        f"bracket_width={widened_bw:.3f}, seed=0):")
    for l, h, s in ctrl_set:
        log(f"  L{l}H{h}  score={s:.4f}")

    si_set = select_top_si(df_si, SI_SENDER_K)
    log(f"\n§H5-5 si senders (top-{SI_SENDER_K} by Δ_h at clean step{STEP}):")
    for l, h, s in si_set:
        log(f"  L{l}H{h}  Δ_h_clean_pre={s:.4f}")

    # --- Read pinned NMs from §S-8 anchor inspection ---
    nm_npz = np.load(EXPL / "s_inhibition_pythia_410m_anchor_per_nm.npz")
    nm_heads_pinned: list[tuple[int, int]] = [(int(L), int(H)) for L, H in nm_npz["nm_heads"]]
    log(f"\n§H5-6 pinned NMs (component-DLA top-4 from clean §S-8 anchor):")
    for nl, nh in nm_heads_pinned:
        log(f"  L{nl}H{nh}")

    # --- Load model ---
    log(f"\nLoading Pythia-{SIZE}-deduped @ step{STEP}...")
    t0 = time.time()
    model = load_pythia(SIZE, step=STEP)
    log(f"  loaded in {time.time()-t0:.1f}s; n_layers={model.cfg.n_layers}, "
        f"n_heads={model.cfg.n_heads}")

    clean = load_ioi_prompts(PROMPTS_PATH)
    corrupt = build_abc_corrupted_prompts(clean, model.tokenizer, seed=0)
    log(f"  N={len(clean)} clean+corrupt IOI prompts (GPT-NeoX BPE)")

    si_senders = [(l, h) for l, h, _ in si_set]
    suc_layer_heads = [(l, h) for l, h, _ in suc_set]
    ctrl_layer_heads = [(l, h) for l, h, _ in ctrl_set]

    # --- Condition 1: clean ---
    model.reset_hooks()
    log("\n=== Condition 1: clean (no ablation) ===")
    clean_per_prompt = run_condition(
        model, clean, corrupt, si_senders, nm_heads_pinned, "clean", log
    )

    # --- Precompute mean_z for suc and ctrl sets (using clean model) ---
    log("\nPrecomputing mean_z for suc set...")
    t0 = time.time()
    mean_z_suc = precompute_mean_z_by_length(model, clean, suc_layer_heads)
    log(f"  done in {time.time()-t0:.1f}s; {len(mean_z_suc)} (layer, head, T) entries")

    log("Precomputing mean_z for ctrl set...")
    t0 = time.time()
    mean_z_ctrl = precompute_mean_z_by_length(model, clean, ctrl_layer_heads)
    log(f"  done in {time.time()-t0:.1f}s; {len(mean_z_ctrl)} (layer, head, T) entries")

    # --- Condition 2: suc_ablated ---
    model.reset_hooks()
    install_mean_ablation_hooks(model, suc_layer_heads, mean_z_suc)
    log("\n=== Condition 2: suc_ablated ===")
    log(f"  perma hooks installed on {len(suc_layer_heads)} suc heads")
    suc_per_prompt = run_condition(
        model, clean, corrupt, si_senders, nm_heads_pinned, "suc_ablated", log
    )

    # --- Condition 3: ctrl_ablated ---
    model.reset_hooks()
    install_mean_ablation_hooks(model, ctrl_layer_heads, mean_z_ctrl)
    log("\n=== Condition 3: ctrl_ablated ===")
    log(f"  perma hooks installed on {len(ctrl_layer_heads)} ctrl heads")
    ctrl_per_prompt = run_condition(
        model, clean, corrupt, si_senders, nm_heads_pinned, "ctrl_ablated", log
    )
    model.reset_hooks()

    # --- Bootstrap and verdict ---
    log("\n=== Bootstrap (B=200 paired, seed=1) and per-sender classification ===")
    rng = np.random.default_rng(1)
    summary_rows: list[dict] = []
    per_prompt_rows: list[dict] = []
    per_sender_pattern: list[str] = []
    for sl, sh in si_senders:
        c = clean_per_prompt[(sl, sh)]
        a_suc = suc_per_prompt[(sl, sh)]
        a_ctrl = ctrl_per_prompt[(sl, sh)]
        for cond_label, arr in [("clean", c), ("suc_ablated", a_suc), ("ctrl_ablated", a_ctrl)]:
            for pi, val in enumerate(arr):
                per_prompt_rows.append(dict(
                    condition=cond_label, si_sender_layer=sl, si_sender_head=sh,
                    prompt_idx=pi, delta_h=float(val),
                ))
        ratio_suc, suc_lo, suc_hi = bootstrap_drop_ratio(c, a_suc, B_BOOTSTRAP, rng)
        ratio_ctrl, ctrl_lo, ctrl_hi = bootstrap_drop_ratio(c, a_ctrl, B_BOOTSTRAP, rng)
        pattern = classify_per_sender(ratio_suc, ratio_ctrl)
        per_sender_pattern.append(pattern)
        log(f"  L{sl}H{sh}: clean Δ̄={c.mean():+.4f}  "
            f"suc_ablated Δ̄={a_suc.mean():+.4f} (ratio={ratio_suc:.3f}, CI=[{suc_lo:.3f},{suc_hi:.3f}])  "
            f"ctrl_ablated Δ̄={a_ctrl.mean():+.4f} (ratio={ratio_ctrl:.3f}, CI=[{ctrl_lo:.3f},{ctrl_hi:.3f}])  "
            f"→ {pattern}")
        for cond_label, mean_val, ratio, lo, hi in [
            ("clean", float(c.mean()), 1.0, 1.0, 1.0),
            ("suc_ablated", float(a_suc.mean()), ratio_suc, suc_lo, suc_hi),
            ("ctrl_ablated", float(a_ctrl.mean()), ratio_ctrl, ctrl_lo, ctrl_hi),
        ]:
            summary_rows.append(dict(
                condition=cond_label,
                si_sender_layer=sl, si_sender_head=sh,
                delta_h_mean=mean_val,
                drop_ratio_mean=ratio, ratio_ci_low=lo, ratio_ci_high=hi,
                ablate_set_repr=(
                    "" if cond_label == "clean"
                    else (",".join(f"L{l}H{h}" for l, h in suc_layer_heads) if cond_label == "suc_ablated"
                          else ",".join(f"L{l}H{h}" for l, h in ctrl_layer_heads))
                ),
                widened_bracket_width=widened_bw if cond_label == "ctrl_ablated" else float("nan"),
                n_prompts=int(c.shape[0]),
            ))

    # --- Aggregate verdict ---
    aggregate = aggregate_verdict(per_sender_pattern)
    headline = PAPER_HEADLINES[aggregate]
    log(f"\n=== §H5-7 aggregate verdict ===")
    log(f"per-sender patterns: {per_sender_pattern}")
    log(f"aggregate: {aggregate}")
    log(f"matched paper headline:")
    log(f"  {headline}")

    # --- Write parquets ---
    EXPL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_prompt_rows).to_parquet(OUT_PER_PROMPT, index=False)
    pd.DataFrame(summary_rows).to_parquet(OUT_SUMMARY, index=False)
    verdict_row = pd.DataFrame([dict(
        pattern=aggregate,
        n_senders=len(per_sender_pattern),
        n_dep=per_sender_pattern.count("DEP"),
        n_null=per_sender_pattern.count("NULL"),
        n_generic=per_sender_pattern.count("GENERIC"),
        n_mixed=per_sender_pattern.count("MIXED"),
        per_sender_patterns=";".join(per_sender_pattern),
        widened_bracket_width=widened_bw,
        suc_set=",".join(f"L{l}H{h}" for l, h in suc_layer_heads),
        ctrl_set=",".join(f"L{l}H{h}" for l, h in ctrl_layer_heads),
        si_senders=",".join(f"L{l}H{h}" for l, h in si_senders),
        nm_heads=",".join(f"L{l}H{h}" for l, h in nm_heads_pinned),
        paper_headline=headline,
    )])
    verdict_row.to_parquet(OUT_VERDICT, index=False)
    log(f"\nWrote {OUT_PER_PROMPT.relative_to(REPO_ROOT)}")
    log(f"Wrote {OUT_SUMMARY.relative_to(REPO_ROOT)}")
    log(f"Wrote {OUT_VERDICT.relative_to(REPO_ROOT)}")

    # --- Write log ---
    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"\nWall: {(time.time()-t_start)/60:.1f} min")
    print(f"Log: {log_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
