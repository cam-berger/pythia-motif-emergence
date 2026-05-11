# pythia-motif-emergence

Mechanistic interpretability research project characterizing the emergence dynamics of three attention-head motifs — induction, successor, and S-inhibition (Path C, locked at Week 1 pilot) — across the Pythia training-checkpoint suite. Three-track structure, all PASSing their registered targets as of 2026-05-11: (1) pre-registered emergence at 3 small Pythia sizes (§H1-C / §H2-5 PASS at p = 0.00463 with §H2-9-R scale-dependence reframe); (2) pre-registered inference-time causal-dependence ablation at 410m + 1B + 2.8B (§H5-causal / §H5-causal-2 / §H5-causal-3 — NULL × NULL at 410m and 2.8B, scale-stable disjointness across head-count tiers 384 + 1024); (3) pre-registered head-count-axis scaling at Pythia-2.8B (§H4-scaling → §H4-7-supersede DEFERRED → §H4-supersede PASS: reversal_rate = 1.000, max_count = 5).

**Canonical pre-reg record:** [`HYPOTHESIS.md`](./HYPOTHESIS.md) and its amendment chain. **Original project plan:** [`PROJECT_BRIEF.md`](./PROJECT_BRIEF.md). **Paper-narrative re-organization:** [`WRITEUP.md`](./WRITEUP.md). This README is a thin status surface.

**Pre-registration:** [`HYPOTHESIS.md`](./HYPOTHESIS.md) and [`PILOT_RESULTS.md`](./PILOT_RESULTS.md) are committed before any pilot code runs. Their commit timestamp is the pre-registration anchor.

## Why this project exists

Tigges et al. (NeurIPS 2024) characterized IOI and greater-than circuit emergence across Pythia checkpoints but did not study successor heads or copy-suppression. McDougall et al. (BlackboxNLP 2024) explicitly named *"do copy-suppression heads exist in Pythia and Llama?"* as future work. The infrastructure (`curt-tigges/circuits-over-time`, `hannamw/EAP-IG`) exists; the experimental design — three tracks: a pre-registered emergence-ordering test at three Pythia sizes (§H1-C / §H2-5, gate passed at p = 0.00463), a pre-registered head-count-axis scaling test at Pythia-2.8B (§H4-scaling, awaiting compute), and a pre-registered inference-time causal-dependence ablation at the 410m anchor (§H5-causal / §H5-causal-2, both NULL — successor and S-inhibition causally disjoint at inference time) — is the contribution.

If the ordering predicted by the hypothesis (induction → successor → suppression) holds across all three sizes, that is consistent with a compositional account in which corrective mechanisms emerge after the copying behaviors they correct. If it doesn't, the inconsistency is itself an interesting null result.

## What's planned

One paper + arXiv preprint + LessWrong post + clean GitHub repo with one notebook per main figure, submitted simultaneously. Target venue: BlackboxNLP at EMNLP 2026 (fallback ICLR MI Workshop 2027). Solo researcher, M5 Pro hardware, 16-week timeline.

The full plan — weekly milestones, decision gates, pre-committed limitations, failure-mode contingencies — is in `PROJECT_BRIEF.md`.

## Status

| Phase | Weeks | Status |
|---|---|---|
| 0 — Reading | parallel with 1–4 | not started |
| 1 — Tooling & replication | 1–4 | in progress |
| 1.0 — Pilot (Path A vs Path C decision) | week 1 | **✓ complete — Path C registered** (`PILOT_RESULTS.md`) |
| 1.1 — TransformerLens fluency on Pythia | week 2 | substantially complete (de facto from Day 1–4 work) |
| 1.2 — Tigges IOI replication | week 3 | **✓ complete — gate PASS** (max abs-diff 0.066 < 0.10 vs Tigges 2024 at the 4 directly-shared steps; `notebooks/tigges_ioi_replication.ipynb`) |
| 1.3 — Detector validation | week 4 | **✓ complete — S-inhibition validated** (GPT-2 Wang's 4 = ranks #1-#4 of 144; σ-criterion FAIL by 0.019σ overridden on rank-strength per §S-5c; Pythia anchor passes both Q8 gates; 18-cell sweep shows emergence in all 3 sizes by step143000; `notebooks/s_inhibition_proof.ipynb` + `s_inhibition_emergence_exploration.ipynb`) |
| 1.4 — Successor detector | week 4 | **✓ complete — successor validated** (GPT-2 small L9H1 rank #1 of 144 by lift = +0.39 under §SU-1b lift-form supersede; Pythia-410M anchor passes both gates with L22H6 cross-category positive in all 4 categories; **H1-C ordering HOLDS in all 3 Pythia sizes** in the preview grid; `notebooks/successor_proof.ipynb` + `successor_emergence_exploration.ipynb`) |
| 2 — Novel sweep | 5–10 | **✓ complete — registered gate passes; reframed to scale-dependent emergence** (joint sign-test p = 0.00463 < 0.005 locked gate, but 4/9 cells right-censored or marginal at the upper sentinel; only Pythia-410m is a robust per-size confirmation; 160m S-inhibition leg has overlapping bootstrap CI with successor; 70m S-inhibition does not emerge during training. Phase 2 headline is now **scale-dependent emergence of S-inhibition** + **temporal-vs-architectural depth asymmetry**, not "H1-C HOLDS jointly". 40 cells × 3 sizes × 3 motifs; bootstrap CIs on μ; B=1000 reversal-rate per (size, pair) replaces the original prior-under-exchangeability descriptive p-values. Reframe registered in `HYPOTHESIS.md` §H2-9-R; full audit trail in the headline notebook `notebooks/h1c_ordering_test.ipynb`.) |
| 3a — 1B head-count probe (§H3-scale) | 11 | **✓ complete — REGR pattern; sealed historical record** (pre-registered §H3-scale 5-leg conjunctive gate FAILS at A.i + A.iii + B.i + B.ii; A.ii passes at the maximum: `P(μ_si^1B < μ_si^410m) = 1.000` over 1000 paired bootstraps. Empirical signature beyond the matched REGR headline: count saturates at max=3 (matches 410m); timing accelerates monotonically; H1-C ordering breaks at 1B (S-inhibition before successor). Reframed under §H4-scaling as a head-count regression — 1B has 128 heads, narrower than 410m's 384 — not a valid scale-up. Anchor inspections passed; no TOOLING-pattern at d_model=2048. Outputs: `data/exploration/phase3_1b_*` parquets, `h1c_ordering_test.ipynb` §H3-scale verdict section. The §H3-scale-8-vis amendment integrated 1B as a 4th size in the existing per-motif `*_full_sweep.ipynb` notebooks; the 1B-only sweep notebooks were superseded and removed.) |
| 3b — Head-count-axis scaling (§H4-scaling → §H4-7-supersede → §H4-supersede) | 12–13 | **✓ complete — §H4-supersede PASS** (§H4-scaling pre-registered before any 2.8B compute. Initial full-grid sweep halted at 8/40 cells under §H4-7 escape hatch (~57 min/cell vs ~6 min/cell projected); §H4-7-supersede committed 2026-05-08 as DEFERRED. §H4-supersede committed 2026-05-10 registered a reduced 10-cell grid `[5000, 7000, 10000, 14000, 20000, 29000, 41000, 49000, 59000, 70000]` against the verbatim §H4-2 gate; sweep ran ~5h 55min overnight 2026-05-10/11. **Verdict: PASS on both legs** — (A.timing) reversal_rate = 1.000 over B=1000 paired bootstrap replicates (gate ≥ 0.95); (A.count) max_count_si^2.8B = 5 at step 29000 (gate ≥ 5). μ_si^2.8B ≈ 9021 vs μ_si^410m ≈ 24088 — 2.8B is ~2.7× faster on S-inhibition emergence. Outputs: `phase4_2_8b_s_inhibition_supersede_sweep.parquet`, `phase4_2_8b_h4supersede_verdict.parquet`, `phase4_2_8b_h4supersede_bootstrap_mu.parquet`; `notebooks/h1c_ordering_test.ipynb` §H4-supersede section.) |
| 3c — Inference-time causal dependence (§H5-causal + §H5-causal-2 + §H5-causal-3) | 12 | **✓ complete — NULL × NULL at 410m and 2.8B; scale-stable causal-disjointness** (pre-registered §H5-causal mean-ablates top-5 successor heads on hook_z and reads back §S-1 path-patching Δ_h on the top-3 S-inhibition senders vs a score-bracket-matched control. **410m: Metric A NULL (3/3 senders), Metric B NULL (ratio_suc = 0.986, ratio_ctrl = 0.979)** → converging-evidence claim at head-count tier 384. **2.8B (§H5-causal-3-2.8b, pre-data 2026-05-10): Metric A NULL (3/3 senders, ratios ≈ 1.001 ± 0.005), Metric B NULL (ratio_suc = 0.984, ratio_ctrl = 0.984)** → claim extends to head-count tier 1024 on BOTH metrics. **1B (§H5-causal-3-record, post-data 2026-05-07): Metric A NULL replicates, Metric B MIXED (ratio_suc = 0.790, ratio_ctrl = 0.797 — both ablations drop ~21% similarly)** → reframed as narrow-architecture readout-specificity artifact (1B's 128-head head-count regression loses readout specificity), NOT a falsification. Combined: causal-disjointness of successor and S-inhibition at inference time is scale-stable across head-count tiers 384 + 1024. `phase4_causal_{410m,1b,2_8b}_anchor*` parquets; `notebooks/causal_dependence.ipynb` per-size + per-metric + cross-size verdict tables.) |
| 4 — Writeup | 14–16 | **WRITEUP.md three-track narrative complete; paper draft pending** — Tracks 1 + 2 + 3 all PASS or NULL-as-target. Paper-narrative re-organization in `WRITEUP.md`; HYPOTHESIS.md remains canonical pre-reg record. |

Update this table as gates pass. Do not add findings to the README before the corresponding gate.

## Setup

```bash
uv sync
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Python 3.12 (uv-managed). MPS-only — no CUDA path supported.

## Repository layout

See `PROJECT_BRIEF.md` §6.

## License

[MIT](./LICENSE).
