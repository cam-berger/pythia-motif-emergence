# Pre-registered Hypothesis

**This document is committed before any pilot code runs.** Its commit timestamp is the pre-registration anchor. The full project context is in `PROJECT_BRIEF.md`; this file extracts the falsifiable claim and the decision rule that gates Path A versus Path C.

## Primary hypothesis (H1-A — if Week 1 pilot succeeds)

In Pythia models of varying scale (70M, 160M, 410M), three attention-head motifs — induction, successor, copy-suppression — emerge during training in a consistent ordering: **induction first, successor second, copy-suppression third.** This ordering reflects a compositional structure in which corrective mechanisms emerge after the copying behaviors they correct.

## Pivot hypothesis (H1-C — if Week 1 pilot fails)

In Pythia models of varying scale, three attention-head motifs — induction, successor, **S-inhibition** (the suppression component of the IOI circuit, Wang et al. 2023) — emerge in a consistent ordering: induction first, successor second, S-inhibition third. Same compositional principle, generalized beyond the GPT-2-specific copy-suppression motif.

## Operational definition (both paths)

For each model size $s \in \{70M, 160M, 410M\}$ and motif $m$, define the **emergence step** $\mu_{s,m}$ as the training step at which the count of detected heads reaches half its final-checkpoint value, computed via logistic fit to count-vs-log-step.

$$\text{count}(\text{step}) \approx \frac{L}{1 + \exp(-k(\log(\text{step}) - \mu))}$$

The hypothesis predicts:

$$\mu_{s,\text{induction}} < \mu_{s,\text{successor}} < \mu_{s,\text{suppression}} \quad \text{for all } s.$$

## Falsification criterion

The hypothesis is **falsified** if any of the three sizes shows a reversed order with statistically distinguishable gaps.

- **Test:** permutation test on ordering. Under $H_0$ (exchangeable order), the probability of observing the predicted order in all three sizes by chance is $(1/6)^3 \approx 0.0046$.
- **Threshold:** $p < 0.005$ for the joint claim across sizes.

## What is *not* claimed

- No claim of universality across architectures (Pythia / GPT-NeoX only, unless extended).
- No claim about absolute emergence times being reproducible (single-seed limitation per Pythia size).
- No causal claim about compositional dependency without Extension A (causal ablation).
- Path A only proceeds if copy-suppression is *demonstrated* to exist in Pythia during pilot. If the pilot finds no copy-suppression heads, the project pivots to Path C and the registered hypothesis becomes H1-C, with the pivot decision documented in `PILOT_RESULTS.md`.

## Pilot decision rule (Week 1)

The pilot applies the McDougall two-criterion copy-suppression detector to Pythia-410M at `step143000` (final checkpoint) and inspects the top-5 candidate heads qualitatively. The decision rule is applied **in order**:

1. **Strong positive → Path A.** ≥ 3 heads pass both McDougall criteria (QK > 0.3 attention from position $i$ to prior $j$ where $\text{token}_j = \text{token}_i$, AND DLA < 0 to $\text{token}_i$ at position $i$) with attention patterns and DLA signs that pass manual inspection.
2. **Weak positive → Path A with caveat.** 1–2 heads pass both criteria, qualitatively confirmed. Proceed but flag in the paper that copy-suppression in Pythia is sparser than in GPT-2.
3. **Negative → Path C.** 0 heads pass both criteria, OR numerically-passing heads fail qualitative inspection.
4. **Tie / ambiguous → Path C.** Default to the cleaner motif. Reviewers probe weak findings hardest.

The path selected by this rule is recorded in `PILOT_RESULTS.md` by Day 5 of Week 1, and is binding.

## Pre-committed limitations

These will appear in the paper's limitations section verbatim. No retrospective backsliding.

1. **Single-seed per Pythia size.** No within-size variance estimate. The "consistency" claim is across-size, not across-seed.
2. **No cross-architecture universality claim.** Pythia (GPT-NeoX) only. OLMo-2 / Llama would be follow-up work.
3. **Detector-threshold sensitivity.** Reported with bootstrap CIs across thresholds, not pretended away.
4. **No causal claim** unless Extension A is completed. Default framing: *"consistent with a compositional account."*
5. **Path A is conditional.** If the pilot pivots to Path C, the paper studies S-inhibition, not copy-suppression. Pivot decision is documented in `PILOT_RESULTS.md`.

## Amendment policy

This document is the pre-registered hypothesis. Any change to the hypothesis, operational definition, falsification criterion, or decision rule made *after* the first pilot result is recorded must be made in a new commit that explicitly acknowledges it post-dates pilot results. The original commit history is the source of truth for what was registered.

## Amendment 2026-05-05 — validation reframing and calibrated supplementary scheme

**Posted after partial pilot results.** Day 2's GPT-2 calibration of the McDougall two-criterion detector revealed a threshold-transfer issue: the published QK > 0.3 threshold was inherited from McDougall 2024, where QK was computed on filtered data (positions where the model strongly predicts a prior token). On raw prose corpora — including the canonical corpus this project uses for Pythia application (`data/corpora/copy_suppression_corpus.txt`, ~7.5k tokens) — even the published reference head **GPT-2 small L10H7** plateaus at mean QK ≈ 0.10, well below 0.3. Under a strict reading of the registered detector, L10H7 fails to validate against itself.

This amendment makes three changes, all transparent and dated. None relax the path-decision rule.

### 1. Validation reframing (gating change)

**Original (pre-pilot):** *"Detector fires on GPT-2 small layer 10 head 7."* Implicitly assumed McDougall's filtered-data threshold transferred to raw text.

**Reframed (this amendment):** validation is reported in two parallel forms on the canonical corpus:

- **Strict validation (McDougall threshold):** L10H7 mean QK is recorded; PASS only if QK > 0.3 *and* OV < 0. Under raw-text application this is expected to FAIL, and that failure is interpreted as *the strict threshold under-detects on raw text*, not as *the motif is absent*.
- **Calibrated validation (locked here):** L10H7 must (a) rank in the top 5 of all 144 GPT-2 small heads by most-negative OV, and (b) have mean QK > 0.05 on the canonical corpus.

If calibrated validation fails, the supplementary analysis below is dropped and the project reverts to strict-only on Pythia.

### 2. Calibrated supplementary scheme (non-gating)

A second detector criterion is added strictly for *threshold-sensitivity reporting* per limitation §54 (which already pre-committed to bootstrap CIs across thresholds). It is reported alongside the strict criterion but does **not** gate the Path A vs Path C decision:

- **Calibrated criterion:** OV < 0 AND QK > 0.05 on raw-prose canonical corpus.
- **Reported in:** `PILOT_RESULTS.md` § Supplementary analyses (separate from the gating section).
- **Selection rule for the proof notebook's worked example** (when strict candidates are absent): top-3 by most-negative OV among heads passing the calibrated criterion, with deterministic fallback to GPT-2 L10H7 if the calibrated set is empty everywhere across the Pythia sweep.

### 3. Pilot decision rule (unchanged)

The Path A vs Path C decision rule in §"Pilot decision rule (Week 1)" continues to key on the **strict** criterion exclusively. The calibrated scheme is descriptive, not gating. If strict returns 0 passing heads on Pythia-410M @ `step143000`, Path C is registered regardless of what the calibrated scheme finds.

The amendment exists to document threshold transfer transparently — not to relax detection requirements.

### Rationale

The strict threshold likely under-detects copy-suppression on raw text in *all* models, including GPT-2. Hiding that fact would require either:
- silently switching what "validation" means (bad pre-reg hygiene), or
- reporting a known-positive-fails-its-own-detector finding without context (uninterpretable).

Reframing validation as dual-reported (strict + calibrated) and adding the calibrated supplementary as a separate non-gating analysis preserves the original conservative gating while making the threshold-transfer issue itself a finding to be reported, per §54.
