# Context — terms and conventions

This is a working glossary for cross-cutting terms used throughout the project. The four motifs (induction, successor, copy-suppression, S-inhibition) are defined operationally in `PROJECT_BRIEF.md` §4 — that section is the source of truth for motif behavior and detector definitions; this file is for terms the brief silently assumes.

A future contributor (or future Claude Code session) landing in this repo cold should be able to resolve any unfamiliar mech-interp term in under a minute by reading this file plus the brief.

## Cross-cutting terms

**Attention head.** A single (layer, head) pair in a transformer's attention module. Pythia-410M has 24 layers × 16 heads = 384 attention heads. Each head has independent QK and OV circuits.

**QK circuit.** The attention-pattern-producing pathway: $W_Q W_K^\top$ acts on query/key residual streams to produce attention weights. Determines *where* a head attends.

**OV circuit.** The output-producing pathway: $W_O W_V$ acts on the attended values to produce the head's contribution to the residual stream. Determines *what* a head writes once it has decided where to attend.

**Residual stream.** The cumulative sum of all layer contributions at a given token position. Each attention head and MLP reads from it and writes back to it. Most mech-interp work treats the residual stream as the canonical decomposable object.

**DLA — Direct Logit Attribution.** A head's contribution to the final logit of a specific token, computed by projecting the head's residual-stream output onto the unembedding direction for that token. Positive DLA = head pushes the token's logit *up*. Negative DLA = head pushes it *down*. Used to detect copy-suppression (negative DLA on the attended-to token) and S-inhibition (negative DLA on the duplicate-name token).

**Prefix-matching score (Olsson 2022).** Operational detector for induction heads. Construct random-token sequences of length 100 with a repeat at position 50. Score = mean attention from positions 51–100 to the position immediately following the previous occurrence of the current token, averaged over 50 random sequences. A head with score > 0.3 is a candidate induction head.

**Emergence step ($\mu_{s,m}$).** For model size $s$ and motif $m$, the training step at which the per-(layer, head) pass-count trajectory reaches a milestone. Four proxies coexist in the project, all unified through `src/analysis/emergence_step.py`:
- `logistic_mu` — μ from the logistic fit (PROJECT_BRIEF.md §3). Used by the §H1-C 3-size joint sign-test at p ≈ 0.00463.
- `half_max` — first step where pass-count ≥ ⌈0.50 × max⌉. Used by the §H1-C-2.8b-extension under within-motif normalization (locked 2026-05-12).
- `half_final` — first step where pass-count ≥ ⌈0.50 × final-step count⌉. Used by §H1-C-altdetectors-2-rr-4.
- `first_geq_k` — first step where pass-count ≥ a fixed k. Used by proxy-sensitivity sweeps.

The hypothesis is a claim about the *ordering* of $\mu$ across motifs within each size.

**Pilot.** Week 1 protocol that decides between Path A (copy-suppression) and Path C (S-inhibition) by testing whether copy-suppression heads exist in Pythia-410M. See `HYPOTHESIS.md` § *Pilot decision rule* and `PROJECT_BRIEF.md` §3.

**Path A / Path C.** The two pre-registered project paths. Path A: study induction + successor + copy-suppression. Path C: study induction + successor + S-inhibition. Path B exists in earlier project drafts and is not pursued.

**Activation patching / EAP-IG.** Causal-attribution methods used in Tigges-style circuit analysis. Activation patching swaps an activation from one input into another input's forward pass and measures the downstream effect; EAP-IG (Hanna 2024) is the integrated-gradients variant of edge-attribution patching that we use for the Tigges replication in Week 3.

## Conventions

- **Pythia checkpoints.** Steps `0, 1, 2, 4, 8, ..., 512` (early, powers of 2) and then every 1000 from `step1000` to `step143000` (final). 154 total per size. Our sweep schedule (40 per size) is in `checkpoints.yaml`.
- **Model sizes used.** Pythia-70M, Pythia-160M, Pythia-410M (pre-registered triple); Pythia-1B (post-hoc head-count regression check, §H4-1); Pythia-2.8B (post-hoc extension under §H1-C-2.8b-extension and §H4-supersede / §H4-fullgrid).
- **Locked numeric thresholds.** All pre-registered numeric pass/fail gates live in `src/locked_thresholds.py` as the programmatic mirror of HYPOTHESIS.md. New thresholds are introduced via a HYPOTHESIS.md amendment + a matching entry in the registry + a matching entry in `tests/test_locked_thresholds.py::EXPECTED`.
- **Numerical precision on MPS.** fp32 default; fp16 acceptable. Avoid bf16 paths — MPS support is unreliable. See `PROJECT_BRIEF.md` §7 for the full set of MPS gotchas.
- **Reference heads (validation targets).** Induction → known GPT-2 small induction heads. Successor → GPT-2 medium L9H1 (Gould 2024). Copy-suppression → GPT-2 small L10H7 (McDougall 2024). S-inhibition → GPT-2 small heads 7.3, 7.9, 8.6, 8.10 (Wang 2023).
