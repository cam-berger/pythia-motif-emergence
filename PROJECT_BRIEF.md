# Pythia Motif Emergence: Project Brief for Claude Code

> **Audience.** This document is the source of truth for an autonomous coding agent (Claude Code) executing a mechanistic interpretability research project on a local M5 Pro machine. It captures intent, hypothesis, pilot gate, weekly plan, and constraints in one place. Update this file as the canonical project state; agent decisions should reference it.

## 1. Intent

Produce a single workshop-quality artifact (BlackboxNLP at EMNLP 2026, fallback ICLR MI Workshop 2027) characterizing the **emergence dynamics of three attention-head motifs** — induction, successor, and a suppressive motif (copy-suppression OR S-inhibition, gated by Week 1 pilot) — across the Pythia training-checkpoint suite.

**Why this project.** Tigges et al. (NeurIPS 2024) characterized IOI and greater-than circuit emergence across Pythia checkpoints but did not study successor heads or copy-suppression. McDougall et al. (BlackboxNLP 2024) explicitly named "do copy-suppression heads exist in Pythia and Llama?" as future work. The infrastructure (`curt-tigges/circuits-over-time`, `hannamw/EAP-IG`) exists; the experimental design is the contribution.

**Single deliverable.** One paper + arXiv preprint + LessWrong post + clean GitHub repo with one notebook per main figure. Submitted simultaneously.

**Hard constraints.**
- Solo researcher, no mentor lined up. Project must be executable without human research advisor.
- Hardware: Apple M5 Pro, 64 GB unified memory, MPS only (no CUDA). 1.6 TB free SSD.
- Timeline: 16 weeks from project start to submission.
- Bar to clear: one specific falsifiable claim, one intervention experiment, one comparison to a sensible baseline, clean reproducibility, honest limitations, 8 pages + appendix.

## 2. Primary and pivot hypotheses

### H1-A (Path A — primary, if pilot succeeds)

In Pythia models of varying scale (70M, 160M, 410M), three attention-head motifs — induction, successor, copy-suppression — emerge during training in a consistent ordering: **induction first, successor second, copy-suppression third**. This ordering reflects a compositional structure in which corrective mechanisms emerge after the copying behaviors they correct.

### H1-C (Path C — pivot, if pilot fails)

In Pythia models of varying scale, three attention-head motifs — induction, successor, **S-inhibition** (the suppression component of the IOI circuit, Wang et al. 2023) — emerge in a consistent ordering: induction first, successor second, S-inhibition third. Same compositional principle, generalized beyond the GPT-2-specific copy-suppression motif.

### Operational definition (both paths)

For each model size $s \in \{70M, 160M, 410M\}$ and motif $m$, define the **emergence step** $\mu_{s,m}$ as the training step at which the count of detected heads reaches half its final-checkpoint value, computed via logistic fit to count-vs-log-step.

**Hypothesis predicts:** $\mu_{s,\text{induction}} < \mu_{s,\text{successor}} < \mu_{s,\text{suppression}}$ for all three sizes.

**Falsified if:** any size shows reversed order with statistically distinguishable gaps (permutation test, $p < 0.005$ threshold for the joint claim across sizes).

### What is *not* claimed

- No claim of universality across architectures (Pythia/GPT-NeoX only, unless extended).
- No claim about absolute emergence times being reproducible (single-seed limitation).
- No causal claim about compositional dependency without Extension A.
- Path A only proceeds if copy-suppression is *demonstrated* to exist in Pythia during pilot.

## 3. Week 1 pilot — the gating decision

**Purpose.** Decide between Path A and Path C before sinking weeks into either.

### Pilot protocol

| Day | Action | Validation |
|---|---|---|
| 1 | Environment setup (Python 3.12, TransformerLens v3.x, MPS fallback, Pythia-410M load) | Reproduce known induction-head detection on Pythia-410M (prefix-matching > 0.5 on at least one head) |
| 2 | Implement McDougall two-criterion copy-suppression detector | Detector fires on GPT-2 small layer 10 head 7 (published reference) |
| 3 | Apply detector to Pythia-410M final checkpoint (`step143000`) | Record full score distribution across all 384 heads |
| 4 | Manual qualitative validation of top-5 candidate heads | Inspect attention patterns + direct logit attribution on test prompts |
| 5 | Apply decision rule, commit `PILOT_RESULTS.md`, lock path | Path A or Path C selected and documented |

### Decision rule (apply in order)

1. **Strong positive → Path A.** ≥ 3 heads pass both McDougall criteria (QK + negative DLA) with attention patterns and DLA signs that pass manual inspection.
2. **Weak positive → Path A with caveat.** 1–2 heads pass both criteria, qualitatively confirmed. Proceed but flag in paper that copy-suppression in Pythia is sparser than in GPT-2.
3. **Negative → Path C.** 0 heads pass both criteria, OR numerically-passing heads fail qualitative inspection.
4. **Tie / ambiguous → Path C.** Default to the cleaner motif. Reviewers probe weak findings hardest.

### Pre-registration

Before any pilot code runs, commit `HYPOTHESIS.md` and `PILOT_RESULTS.md` (template; populate by Day 5) to the repo. This is small-scale pre-registration to protect against post-hoc rationalization concerns.

## 4. The three motifs (definitions and detectors)

### Induction heads (always part of the project)
- **Behavior:** given `A B ... A`, predict `B`.
- **Mechanism:** two-head circuit, previous-token head (early layer) + induction head (later layer).
- **Detector:** Olsson's prefix-matching score on random-token-repetition sequences (length 100, repeat at position 50). Score = mean attention from positions 51–100 to the position right after the previous occurrence of current token, averaged over 50 random sequences.
- **Validation target:** known GPT-2 small induction heads.
- **Threshold:** prefix-matching score > 0.3 (validate empirically against GPT-2 reference heads).

### Successor heads (always part of the project)
- **Behavior:** given an ordinal sequence (Mon, Tue, Wed, ... | Jan, Feb, ... | 1, 2, 3, ... | one, two, three, ...), predict the next ordinal element.
- **Mechanism:** OV circuit implements abstract "+1 in ordinal space" using MLP-produced ordinal direction.
- **Detector:** cross-category direct logit attribution. For each head, compute DLA to the next ordinal element across days, months, numerals (1–20, both digit and word forms), and letters. Score = mean DLA across categories. Cross-category requirement distinguishes true successor heads from sequence-memorizers.
- **Validation target:** GPT-2 medium layer 9 head 1 (Gould et al. 2024).
- **Threshold:** mean cross-category DLA exceeds the 95th percentile of a null distribution computed on category-shuffled prompts.

### Copy-suppression heads (Path A only)
- **Behavior:** sees current token, attends to prior occurrence of same token, writes negative residual contribution to that token's logit.
- **Mechanism:** corrective — pushes down on tokens other components are pushing up on.
- **Detector (McDougall two-criterion):**
  - QK criterion: attention from position $i$ to position $j$ where $token_j == token_i$, attention weight > 0.3.
  - OV criterion: head's contribution to logit of $token_i$ at position $i$ is negative (DLA < 0).
- **Validation target:** GPT-2 small layer 10 head 7.
- **Architectural risk:** McDougall studied GPT-2 only. May not exist in Pythia. The pilot tests this.

### S-inhibition heads (Path C only, replaces copy-suppression)
- **Behavior:** in IOI prompts, suppresses prediction of the duplicate (subject) name, allowing name-mover heads to promote the indirect-object name.
- **Mechanism:** writes negative residual contribution to the duplicate-name token via an inhibition signal.
- **Detector:** for IOI prompts (Wang et al. 100-prompt set), measure each head's DLA contribution to the *subject* name at the final position. S-inhibition heads have consistently negative DLA on the subject name.
- **Validation target:** GPT-2 small heads 7.3, 7.9, 8.6, 8.10 (Wang et al. 2023).
- **Threshold:** mean DLA on subject name across 100 IOI prompts < threshold calibrated against GPT-2 small reference heads.

## 5. Weekly plan

### Phase 0 — Reading (parallel with weeks 1–4)

| Tranche | Timing | Papers (in execution order) |
|---|---|---|
| 1 — Foundations | Days 1–7, week 1 | Elhage 2021 *Mathematical Framework*; Olsson 2022 *In-context Learning and Induction Heads*; Wang 2023 *IOI*; Conmy 2023 *ACDC*; Hanna 2023 *Greater-than* |
| 2 — Three motifs | Days 8–14, week 2 | Olsson 2022 (re-read §4 with code); Singh 2024 *What needs to go right for an induction head?*; Gould 2024 *Successor Heads*; McDougall 2024 *Copy Suppression* |
| 3 — Direct ancestors | Before week 5 | Biderman 2023 *Pythia*; Tigges 2024 *Circuit Analyses Are Consistent*; Edelman 2024 *Evolution of Statistical Induction Heads* |
| 4 — Methods (just-in-time) | Weeks 3–4 | Syed/Rager/Conmy 2023 *Attribution Patching*; Hanna 2024 *EAP-IG*; Heimersheim & Nanda 2024 *How to use and interpret activation patching* |

Reading is parallel with coding. Don't gate execution on completing reading.

### Phase 1 — Tooling and replication (Weeks 1–4)

**Week 1: Pilot (gating decision) + foundational reading.**
- Days 1–5: pilot protocol above.
- Days 6–7: by end of week, path is locked, environment is validated, Tranche 1 reading complete.

**Week 2: TransformerLens fluency on Pythia.**
- Hook the residual stream of Pythia-410M, extract attention patterns for one head on one prompt, verify by hand.
- Run a clean ablation experiment: zero-ablate one head, measure logit change.
- Compute direct logit attribution for one head on one token. This is the workhorse computation; if it's not clean here, it won't be clean anywhere.
- Tranche 2 reading complete.

**Week 3: Replicate Tigges et al. IOI emergence on Pythia-410M.**
- Make-or-break tooling validation milestone.
- Clone `curt-tigges/circuits-over-time`. Build IOI prompt set (100 prompts, Wang et al. canonical).
- Run EAP-IG at 5 widely-spaced checkpoints (`step1000`, `step10000`, `step50000`, `step100000`, `step143000`).
- Compute IOI-task accuracy at each checkpoint. Plot emergence curve.
- **Gate:** curve must match Tigges Figure 2 within ~10% in absolute terms. If not, debug before proceeding. Common culprits: wrong tokenizer version, batch-size dependence, MPS numerical drift.

**Week 4: Implement and validate the three detectors.**
- For each of the three motifs (path-determined): implement detector, validate on GPT-2 small reference heads, then apply to Pythia-410M final checkpoint.
- Each detector validated → can be trusted across the checkpoint sweep.
- Tranche 4 reading complete.

### Phase 2 — Novel sweep (Weeks 5–10)

**Week 5: Final-checkpoint motif characterization across all three sizes.**
- Apply all three detectors to Pythia-70M, 160M, 410M at `step143000`.
- First publishable observation: do all three motifs exist in all three sizes? Expect successor heads to be marginal in 70M.

**Weeks 6–8: Checkpoint sweep.**
- 30–40 log-spaced checkpoints × 3 sizes × 3 detectors ≈ 300+ detector runs.
- Total compute: 50–80 hours M5 Pro. Run overnight in batches.
- Output: single canonical dataset `motif_sweep.parquet` with schema `(size, step, layer, head, motif, score)`. Save raw scores, not just thresholded counts — allows post-hoc threshold robustness.
- Spot-check at every 5th checkpoint by hand to catch detector drift.

**Weeks 9–10: Statistical analysis.**
- Per (size, motif): fit logistic emergence curve $count(step) \approx L / (1 + \exp(-k(\log(step) - \mu)))$.
- Extract emergence step $\mu_{s,m}$ per cell.
- **Primary test:** permutation test on ordering. Under $H_0$ (exchangeable order), probability of observed order in all 3 sizes by chance is $(1/6)^3 \approx 0.0046$.
- **Robustness:** bootstrap CIs on $\mu$ via detector-threshold variation (not seed variation — Pythia has only one seed).
- **Stretch:** structural-reuse analysis (track head identity across motifs over training).

### Phase 3 — Push past replication (Weeks 11–13)

**Recommended: Extension B — Structural reuse.**
- Path-independent (works for both Path A and Path C).
- Uses already-collected sweep data → no new compute.
- Directly tests the mechanistic interpretation of the ordering.
- For each pair of motifs (induction → suppression, induction → successor, successor → suppression), compute fraction of late-motif-positive heads that were earlier-motif-positive at any prior checkpoint.
- Strong finding: ≥ 50% reuse → compositional buildup story strengthened.
- Null finding: motifs use disjoint head sets → ordering is real but compositional interpretation is weakened, paper pivots to "ordering without structural reuse" framing.

**Alternative: Extension A — Causal ablation.** Reserved for if Phase 2 finishes ahead of schedule. Ablate induction heads at a checkpoint right after their emergence, fine-tune briefly, ask whether successor heads still form. Higher reward, much higher cost. Default: skip unless ahead of schedule by week 10.

### Phase 4 — Writeup (Weeks 14–16)

**Week 14: Figures.**
- Headline figure: 3×1 panels (one per Pythia size), x-axis log(training step), y-axis count of detected heads above threshold per motif, three lines per panel.
- Supporting figures: motif emergence-step comparison bar chart; representative attention patterns at emergence; structural-reuse Sankey if Extension B done.

**Week 15: Draft.**
- 8 pages + appendix. Sections: intro (1), background (1, including QK/OV decomposition for non-mech-interp readers), methods (2), results (3), limitations (0.5, single-seed issue stated honestly), discussion (0.5).
- Appendix: detector validation against GPT-2 reference heads, threshold-robustness sweeps, per-checkpoint raw counts.

**Week 16: Polish + ship.**
- Three simultaneous deliverables: arXiv submission, GitHub repo polish (one notebook per figure: `figure_1.ipynb` ... `figure_5.ipynb`), LessWrong post (~2,000 words: paper intro + headline figure + "what this means").
- Submit to BlackboxNLP at EMNLP 2026 (typical ARR commit deadline late Aug / early Sept 2026).

## 6. Repository layout

```
pythia-motif-emergence/
├── HYPOTHESIS.md               # pre-registered hypothesis + decision rule
├── PILOT_RESULTS.md            # populated by Day 5 of week 1
├── PROJECT_BRIEF.md            # this file
├── README.md                   # public-facing summary
├── pyproject.toml              # uv-managed; pin TransformerLens >=3.x
├── checkpoints.yaml            # 30-40 log-spaced steps per Pythia size
├── src/
│   ├── detectors/
│   │   ├── induction.py        # Olsson prefix-matching score
│   │   ├── successor.py        # cross-category ordinal DLA
│   │   ├── copy_suppression.py # McDougall two-criterion (Path A)
│   │   └── s_inhibition.py     # IOI subject-name DLA (Path C)
│   ├── replication/
│   │   └── tigges_ioi.py       # week 3 replication target
│   ├── sweep/
│   │   ├── runner.py           # batched sweep over (size, step, detector)
│   │   └── schema.py           # motif_sweep.parquet schema
│   ├── analysis/
│   │   ├── emergence.py        # logistic fit, μ extraction
│   │   ├── permutation.py      # ordering permutation test
│   │   └── structural_reuse.py # Extension B
│   └── utils/
│       ├── pythia_loader.py    # checkpoint-aware loading
│       └── mps_compat.py       # MPS workarounds (fp32, fused=False, etc.)
├── notebooks/
│   ├── pilot_validation.ipynb
│   ├── figure_1.ipynb          # ... through figure_5.ipynb
│   └── manual_inspection.ipynb # qualitative spot-checks
├── data/
│   ├── prompts/                # IOI prompts, ordinal probes, repetition seqs
│   ├── activations/            # cached residual stream activations (gitignored)
│   └── motif_sweep.parquet     # canonical sweep output
└── tests/
    └── test_detectors.py       # validate against GPT-2 reference heads
```

## 7. Hardware and tooling constraints

### MPS-specific gotchas
- `PYTORCH_ENABLE_MPS_FALLBACK=1` must be set in shell rc.
- Avoid bf16 paths; prefer fp16 or fp32 on MPS.
- `fused=False` in Adam (MPS in-place bug as of mid-2026).
- SAELens issue #392: MPS kernel deaths on default tutorial. Workaround: pre-cache activations, fp32, smaller batches.
- Pin TransformerLens >=3.x (post PR-#1068 MPS attention fix) and Python <=3.12.

### Compute budget
- Forward pass on Pythia-410M with all 154 checkpoints fits in 1.6 TB SSD.
- Detector runs are forward-pass-only → tractable on MPS.
- EAP-IG circuit recovery on Pythia 70M–410M: tractable, MPS-friendly.
- Activation caching for sweep: pre-compute residual streams once per (size, checkpoint) pair, reuse across detectors.

### Fallback compute
- NDIF API key (free for US residents — apply early) for any analysis needing Llama-70B-scale evidence (unlikely, but available).
- $50 budget on Modal/RunPod for emergency final-figure runs if MPS times out during writeup.

## 8. Decision points and gates

| Gate | Week | Pass criterion | Fail action |
|---|---|---|---|
| Pilot decision | 1 | Decision rule applied, path locked | N/A — gate produces decision, not failure |
| Tigges replication | 3 | IOI emergence curve within ~10% of published | Debug tooling; do NOT proceed to week 4 |
| Detector validation | 4 | All three detectors fire on GPT-2 reference heads | Fix detector before applying to Pythia |
| Sweep sanity check | 6 | Spot-check passes at every 5th checkpoint | Re-run failing detector on full sweep |
| Hypothesis test | 10 | Permutation test result | Either H1 or H1' becomes paper's primary claim |
| Phase 3 trigger | 11 | Phase 2 complete on time | If behind, skip Extension B and ship narrower paper |

## 9. Pre-committed limitations

These get written into the paper's limitations section verbatim. No retrospective backsliding.

1. **Single-seed per Pythia size.** No within-size variance estimate. "Consistency" claim is across-size, not across-seed.
2. **No cross-architecture universality claim.** Pythia (GPT-NeoX) only. OLMo-2 / Llama would be follow-up work.
3. **Detector-threshold sensitivity.** Reported with bootstrap CIs across thresholds, not pretended away.
4. **No causal claim** unless Extension A is completed. Default framing: "consistent with a compositional account."
5. **Path A is conditional.** If pilot pivots to Path C, the paper studies S-inhibition, not copy-suppression. Pivot decision is documented.

## 10. Failure modes and what each becomes

| Failure mode | What happens | Paper still publishable? |
|---|---|---|
| Copy-suppression doesn't exist in Pythia | Pivot to Path C in week 1 | Yes — was in plan |
| Tigges IOI replication fails | Debug; if unfixable, project blocked | No — must resolve before proceeding |
| Successor heads marginal in 70M | Drop 70M from successor analysis, report only 160M+410M | Yes — narrower scope |
| Ordering is reversed in one size | Paper reports inconsistent ordering finding | Yes — interesting null result |
| All three motifs emerge simultaneously (no resolution) | Paper reports emergence times below resolution of checkpoint spacing | Yes — methodological contribution about Pythia checkpoint density |
| MPS too slow for full sweep | Reduce to 2 sizes (160M, 410M); use Modal credits for 410M | Yes — narrower scope |
| Phase 3 Extension B finds no structural reuse | Paper reports ordering-without-reuse → revises mechanistic interpretation | Yes — sharper finding |

## 11. Public artifact strategy

Final week ships three things on the same day:
1. **arXiv preprint** with full appendix.
2. **GitHub repo** with one notebook per figure, reproducible from `pyproject.toml`. Main figure notebooks must run end-to-end on a fresh M5 Pro install.
3. **LessWrong / Alignment Forum post** with paper intro, headline figure, "what this means" section. Cross-link to arXiv and GitHub.

Stretch goal during Phase 2: share intermediate results in EleutherAI Discord and reach out to Curt Tigges or Stefan Heimersheim for informal feedback. Feedback that turns into co-authorship multiplies grad-school value substantially.

## 12. What "done" looks like

- `HYPOTHESIS.md` and `PILOT_RESULTS.md` committed before any analysis.
- Tigges IOI replication notebook reproduces published curve.
- Three validated detectors with passing tests against GPT-2 reference heads.
- `motif_sweep.parquet` with 300+ rows of (size, step, layer, head, motif, score).
- Logistic fits + permutation test result documented in `analysis/emergence.py`.
- Headline figure rendered from canonical data.
- 8-page draft + appendix.
- BlackboxNLP submission filed.
- arXiv + GitHub + LessWrong shipped within 24 hours of submission.
