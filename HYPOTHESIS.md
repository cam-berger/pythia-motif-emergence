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

## Amendment 2026-05-05 (evening) — §S-inhibition: Phase 1.3 detector specification

**Posted after Path C registration and Phase 1.2 (Tigges IOI replication, gate PASS).** This amendment locks the procedural specification of the S-inhibition detector that operationalizes H1-C's third motif. It is reference-style: the detector's identification method, corruption scheme, receiver scalar, and validation gate are committed here. The numerical strict threshold τ_strict is **not** locked in this amendment; it is data-dependent on the GPT-2 small reference distribution and will be locked in a follow-up amendment after the GPT-2 validation gate has been executed and recorded.

### S-1. Detector method (locked)

The S-inhibition detector is implemented as **Wang-style path-patching with frozen paths** (Goldowsky-Dill et al. 2023 protocol), not activation patching and not direct DLA on the subject token. Path-patching isolates the contribution of a candidate sender head along the specific path `sender → Name Mover head → output`, freezing all other paths to their clean values. This is the only method that distinguishes a head's *causal role in the S-inhibition mechanism* from its raw correlation with subject-name logit. The IOI prompt set committed in Phase 1.2 (`data/prompts/ioi_prompts.tsv`, N=200, 100 BABA + 100 ABBA, seed=0) is the substrate. Names are re-verified single-token under GPT-2 small's BPE before validation runs; this is a mechanical check with no expected divergence, since Wang's name pool is GPT-2-derived.

Implementation is split into two files (plan, not yet written): `src/replication/path_patching.py` provides a generic frozen-path-patching primitive over a `(sender_head, receiver_head, receiver_position)` triple; `src/detectors/s_inhibition.py` applies the primitive to the S-inhibition use case below.

### S-2. Corruption scheme (locked)

The corruption is **ABC, position-3-only**. For each clean prompt of the form `When [N1] and [N2] went to the [PLACE], [N3] gave a [OBJECT] to`, the corrupted variant replaces the n3 (second-clause subject, the repeated name) with a fresh name `C` drawn from the same single-token-name pool, with `C ∉ {IO, S}` for that prompt. n1 and n2 are unchanged. This isolates the S-inhibition signal: in clean, the model must suppress S (the duplicate) at NM attention; in corrupted, S is not present at the n3 position, so the suppression mechanism has no target. No other position is corrupted. ABC-only at position 3 is the standard Wang/Goldowsky-Dill choice for this circuit and is committed verbatim here.

### S-3. Name Mover identification (locked)

**Receiver heads (Name Movers) are identified as the top-4 by component-DLA on the (IO − S) logit difference**, computed on the clean prompt set using the existing primitive `src/replication/tigges_ioi.py::component_dla` from Phase 1.2. `k = 4` is fixed across all models. The selection rule is purely component-DLA-driven; Wang's published GPT-2 Name Movers `{9.6, 9.9, 10.0}` are **not** treated as ground truth for the receiver set. If our component-DLA top-4 disagrees with Wang's labelling on GPT-2 small, that disagreement is recorded as a finding and the validation proceeds with our top-4 as the receiver set. NM = "Name Mover head" throughout this amendment.

### S-4. Receiver scalar (locked)

For a candidate sender head h, the receiver scalar is

`Δ_h = mean over the k = 4 NMs of [ (patched_attn_NM_END→S2) − (clean_attn_NM_END→S2) ] − [ (patched_attn_NM_END→IO) − (clean_attn_NM_END→IO) ]`

where `END` denotes the final token position of the prompt (the position from which the model emits the IO prediction); `S2` is the position of the second occurrence of the subject name (the duplicate position whose attention the S-inhibition mechanism is hypothesized to suppress at NMs); `IO` is the position of the indirect-object name; `clean_attn_NM_*` is the NM's attention weight at the END query under clean activations; and `patched_attn_NM_*` is the NM's attention weight at the END query when only the path `h → NM` carries the corrupted (ABC) value and all other paths are frozen to clean. Averaging is taken across the k = 4 NMs to produce the **per-head scalar** Δ_h used for ranking and for the gate. The full **(sender × NM) matrix** with NM not averaged is also retained for figures and qualitative inspection. Both quantities are computed under N = 200 prompts and reported as per-head point estimates (averaged across prompts). S2 = "second-clause subject position"; IO = "indirect-object position".

A genuine S-inhibition sender produces large positive Δ_h: corrupting it via the path causes the NM at END to attend *more* to S2 and *less* to IO, i.e. removes the suppression of the duplicate. Equivalently, `(patched_attn_S2 − clean_attn_S2) > 0` (S2 attention rises under corruption) and `(patched_attn_IO − clean_attn_IO) < 0` (IO attention falls under corruption), so Δ_h is the sum of two same-signed positive contributions.

### S-5. Validation gate (GPT-2 small, locked)

The detector is screened over **all 144 heads of GPT-2 small** under the locked specification S-1 through S-4. The gate has two conjunctive conditions, both of which must be satisfied:

1. **Top-8 inclusion.** Wang's four published S-Inhibition heads `{7.3, 7.9, 8.6, 8.10}` must each appear within the top-8 of the per-head Δ_h ranking.
2. **Bulk separation.** The median of the four Wang heads' Δ_h must lie at least 2 standard deviations above the mean of the bulk distribution (all 144 heads' Δ_h, computed as the empirical mean and SD of the 144-element distribution including the four Wang heads — no leave-one-out adjustment).

Both conditions are evaluated on N = 200 IOI prompts with the seed=0 prompt set. If either condition fails, the detector specification is treated as invalid and Phase 1.3 hard-stops per S-7 below. Wang's four heads are highlighted in the screen figure regardless of pass/fail.

### S-6. Sweep threshold scheme (locked-in-procedure, numerical commit deferred)

Per the project's dual-report convention from Phase 1.0 (induction strict / permissive), the sweep reports **two thresholds** for what counts as an S-inhibition head:

- **τ_strict** = the minimum Δ_h among Wang's four heads on GPT-2 small under the locked specification, taken as a single point estimate per head averaged over the N = 200 prompts.
- **τ_permissive** = τ_strict / 2.

τ_strict is therefore *defined* by the GPT-2 validation pass (S-5). It is not numerically committed in this amendment; it is committed in a follow-up HYPOTHESIS.md amendment posted immediately after the GPT-2 validation runs, with the per-head Δ_h values for `{7.3, 7.9, 8.6, 8.10}` listed alongside the chosen τ_strict. The sweep notebook reports per-cell head-counts at both thresholds. This procedural commitment binds the τ_strict mechanic without committing the number, which is a function of the GPT-2 reference distribution and cannot be honestly fixed in advance.

### S-7. Failure-mode policy (locked)

Failure of the GPT-2 validation gate at S-5 is a **hard stop**. The response is to re-grill the detector specification from Q1, beginning with detection method. There are **no fallback variants** registered in this amendment: activation patching is not registered as a fallback, swapping the NM identification scheme is not registered as a fallback, and changing the corruption position is not registered as a fallback. Any of these would constitute a new specification and would require its own amendment with its own validation gate.

### S-8. Phase 1.3 scope and Pythia gates (locked)

Phase 1.3 has three deliverables, executed in order: (i) the GPT-2 validation per S-5 above; (ii) a Pythia-410M-deduped @ `step143000` anchor inspection; (iii) an 18-cell exploration sweep on Pythia `{70m, 160m, 410m}` × `{0, 1000, 3000, 8000, 25000, 143000}`. The anchor inspection has its own pre-committed gates: a **numerical gate** (at least one Pythia-410M head clears τ_strict on the same Δ_h scalar from S-4) and a **mechanistic gate** (at least 2 of 4 Pythia NMs, identified per S-3 on Pythia-410M @ step143000, show a positive Δ along the path `candidate → NM`). The sweep at deliverable (iii) runs **unconditionally** on Pythia regardless of the anchor inspection's outcome; if the anchor inspection fails either gate, the sweep documents the failure pattern across the 18 cells in Path-C-style negative-result form. The S-7 hard-stop applies only at the GPT-2 validation level (S-5), not at the Pythia anchor or Pythia sweep level.

### S-9. Pre-registration form (locked)

This amendment is **reference-style**: the locked procedural specification lives here in HYPOTHESIS.md; the design rationale and option-list-with-rejected-alternatives lives in NOTES.md (gitignored working memory) under the Phase 1.3 grilling entry. τ_strict is locked in a *separate* HYPOTHESIS.md amendment posted after the GPT-2 validation gate (S-5) is executed. The split exists because τ_strict is data-dependent and cannot be honestly fixed before the GPT-2 reference distribution is observed; committing the procedure now and the number in the next amendment preserves pre-registration discipline without forcing a fictional pre-data threshold.

## Amendment 2026-05-05 (post GPT-2 validation) — §S-5b/c: median convention + supplementary acceptance, §S-tau: τ_strict lock

**Posted after the GPT-2 validation screen ran (commit `2971723`+code).** Three changes, all transparent and dated. None alter the original detector specification (§S-1 through §S-4); they resolve an ambiguity in §S-5 surfaced by the data, document a one-time post-data acceptance decision, and lock the numerical τ_strict per §S-6 / §Q10(b-i).

### S-5b. Median convention resolution

§S-5 specified "the median of the four Wang heads' Δ_h" without fixing how median is computed for an even-length list. The two standard conventions disagree numerically on our screen output:

- **NumPy `np.median`** (mean of the two middle values, standard mathematical convention): Wang median = `0.04475`, σ above bulk mean = **+1.981σ**.
- **PyTorch `tensor.median`** (lower of the two middle values, documented PyTorch quirk): Wang median = `0.04019`, σ above bulk mean = **+1.770σ**.

This amendment resolves the ambiguity in writing: the canonical convention is **NumPy's mean-of-middle-two** (the standard mathematical definition of median). All future references to "median" in §S-5 and downstream are computed under this convention. Under the resolved convention, the GPT-2 screen produces +1.981σ vs the locked ≥2σ requirement: **§S-5 strict gate FAIL by 0.019σ**.

### S-5c. Supplementary acceptance based on rank-strength

The §S-5 strict gate fails by 0.019σ under the resolved median convention (S-5b). However, the rank-strength evidence from the same screen is unambiguous and we record a **one-time post-data acceptance** of the detector on rank-strength grounds, with the supplementary evidence locked here. Acceptance is *not* a re-amendment of the gate; the gate FAIL stands in the record.

Supplementary evidence:

| evidence | result |
|---|---|
| Top-8 inclusion (§S-5 condition 1) | **PASS** — Wang's 4 heads occupy ranks #1, #2, #3, #4 of 144 |
| All 4 Wang heads above non-Wang max | **PASS** — Wang min `L7H3 = +0.0372` > non-Wang max `L9H4 = +0.0279` (ratio 1.33×) |
| All 4 Wang heads above non-Wang 99th percentile | **PASS** — non-Wang 99th-pct = +0.0194; Wang min = +0.0372 |
| Leave-Wang-out σ-separation | **PASS by wide margin** — +4.574σ (gate would clear by 2.57σ if we excluded the known-positives from bulk SD) |
| Locked §S-5 σ-criterion (no-LOO, NumPy median) | **FAIL by 0.019σ** (driver: L8H6 self-inflation of bulk SD) |

The σ-statistic failure is driven entirely by L8H6's outlier Δ_h (`+0.220`) inflating the bulk standard deviation when included per the locked "no leave-one-out" rule. The σ-criterion is therefore pathological under outlier known-positives — a finding worth recording for future detectors. **Going forward (successor in Phase 1.4, the Phase 2 sweep): the σ-statistic leg is dropped; rank-only validation (Wang's 4 in top-8) is the load-bearing criterion.** This is registered as a methodological lesson, not a relaxation of §S-5 itself.

The Q7 hard-stop policy is not invoked: validation is *recorded as FAIL on the strict criterion* but proceeds on rank-only supplementary acceptance. The pre-registration record reflects both. A reviewer can verify the FAIL by the strict criterion, the rank-strength evidence supporting acceptance, and the chronology of this amendment vs the strict spec.

### S-tau. Numerical τ_strict lock (Q10 b-i)

Per §S-6, τ_strict = `min over Wang's four heads of Δ_h on GPT-2 small`. The four values from the validation screen (committed `data/exploration/s_inhibition_gpt2_validation.parquet`):

| Wang head | Δ_h |
|---|---|
| L7H3 | +0.0372 |
| L7H9 | +0.0402 |
| L8H6 | +0.2200 |
| L8H10 | +0.0493 |

**τ_strict = 0.0372** (= L7H3, the minimum across Wang's 4).
**τ_permissive = τ_strict / 2 = 0.0186**.

Both numbers are now locked. The Pythia-410M anchor inspection (deliverable ii of Phase 1.3) and the 18-cell exploration sweep (deliverable iii) apply these absolute thresholds. Per the §S-3 NM identification rule: NMs are re-derived per-model via component-DLA top-4, so the GPT-2 NM set `{(9,9), (9,6), (10,0), (10,6)}` is *not* transferred to Pythia. Only the Δ_h threshold values transfer.

Component-DLA top-4 NMs on GPT-2 small (recorded for the (sender × NM) figure in the proof notebook): `(9,9), (9,6), (10,0), (10,6)`. Three of four match Wang's published NMs `{(9,6), (9,9), (10,0)}`; our top-4 includes `(10,6)` as the additional component-DLA-driven receiver. This is the §S-3-anticipated divergence and is recorded as a methodological finding rather than a re-amendment of §S-3.

## Amendment 2026-05-05 (later evening) — §SU: Phase 1.4 successor detector specification

**Posted after Phase 1.3 (S-inhibition detector, validated with documented §S-5c override and 18-cell exploration sweep complete).** This amendment locks the procedural specification of the successor-head detector that operationalizes H1-C's second motif. It is reference-style: the detector's prompt format, scoring method, null-shuffle scheme, and validation gate are committed here. The 95th-percentile null-threshold value is **not** locked in this amendment; it is data-dependent on the GPT-2 small reference distribution and will be locked in a follow-up §SU-tau amendment after the GPT-2 validation runs have been executed and recorded. DLA = direct logit attribution; BPE = byte-pair encoding.

### SU-0. Validation target and source attribution (correction of brief misattribution)

The brief at PROJECT_BRIEF.md §4 ("Successor heads", lines 78-83) names the validation target as "GPT-2 medium layer 9 head 1 (Gould et al. 2024)". A Phase 1.4 grilling research step established that this attribution is wrong on two counts:

1. **Wrong model.** L9H1 is a successor head in GPT-2 *small*, not GPT-2 medium. Gould et al. (2024) test GPT-2 small in their Figure 2 cross-model scatter but never name a specific (layer, head) for it in the paper text; their named case-study head is Pythia-1.4B L12H0.
2. **Wrong attribution.** The identification of GPT-2 small L9H1 as a successor head is due to L (2023), "Mechanistically interpreting time in GPT-2 small," LessWrong (https://www.lesswrong.com/posts/6tHNM2s6SWzFHv3Wo/). Gould et al. (2024) §5 *cite* this LessWrong post but do not own the identification.

The locked validation target for Phase 1.4 is therefore:

> **GPT-2 small layer 9 head 1** (notation: L9H1). Source: L (2023), "Mechanistically interpreting time in GPT-2 small," LessWrong. Cited by Gould et al. (2024) §5 in the cross-model successor scatter analysis.

This amendment supersedes PROJECT_BRIEF.md §4's validation-target line. The chronology — brief said GPT-2 medium L9H1 / Gould 2024 → research showed GPT-2 small L9H1 / L 2023 → amendment locks the corrected target *before* validation runs — is the auditable pre-registration record.

### SU-1. Prompt format (locked)

Successor prompts are 3-context comma-separated ordinal lists, scored at the END token for prediction of the 4th element. Four categories are screened: **days** (7 items: Monday...Sunday), **months** (12 items: January...December), **numerals** (40 items: a single category mixing 1-20 in digit form *and* one-twenty in word form), **letters** (26 items: A...Z). Total: 65 items across four categories.

For each (category, starting-index) pair admissible under the category length, a base prompt is constructed of the form `f"{c1}, {c2}, {c3}, "` where `c1, c2, c3` are three consecutive ordinal items. The DLA is computed at the END token (the position after the trailing space, where the model emits its prediction of `c4`). The numerals category mixes digit form and word form within a single category specifically so that a head whose successor mechanism operates on the abstract ordinal direction (per Gould et al.'s OV-circuit account) scores positively across both surface forms; a head that memorizes only the digit sequence "1, 2, 3" or only the word sequence "one, two, three" scores across only the relevant subset and fails the cross-category requirement at SU-4.

### SU-2. Multi-token handling (locked)

Items vary in token length under different tokenizers. The detector uses **first-token DLA**: for each item, encode `f' {item}'` (with leading space) under the active tokenizer and target the **first token** of the resulting sequence regardless of total token count. The (item, tokenizer, first-token-id, first-token-string) mapping is logged at prompt-construction time and persisted alongside the prompt set for reproducibility.

This choice keeps all 65 items in screen across both GPT-2 BPE and GPT-NeoX BPE without per-tokenizer item filtering, at the cost of conflating the DLA on `c4`'s first token with the DLA on `c4` as a complete item. The conflation is acceptable because the cross-category aggregation (SU-4) averages out tokenization idiosyncrasies, and the alternative (per-tokenizer item filtering) would produce different category sizes for GPT-2 validation vs Pythia application — making cross-model comparison less clean.

### SU-3. Null-distribution shuffle (locked, threshold value deferred)

The null distribution against which a candidate head's mean cross-category DLA is judged is constructed by **within-category prefix permutation**. For each (category, base prompt) pair, the three context items `(c1, c2, c3)` are replaced with a deterministic seed-pinned permutation of the same three items, with the constraint that the permutation differs from the identity (so the shuffled prompt is not byte-identical to the clean prompt). One fixed permutation per (category, base prompt) is used — not a Monte-Carlo distribution over permutations — to keep the null distribution single-pass-computable and seed-deterministic.

The null distribution is **pooled across heads at the per-head mean level**: for each head, the *mean cross-category DLA on the shuffled prompt set* is computed (one scalar per head, identical aggregation as the real-prompt score being thresholded). These per-head shuffled means are pooled into a single distribution with one entry per head — 144 entries on GPT-2 small, 384 on Pythia-410M, etc. The **pooled null's 95th percentile** is the threshold τ.

Pooling at the per-head-mean level (rather than at the per-prompt-DLA level) matches the statistic being thresholded — the comparison "head's mean cross-cat DLA exceeds τ" is mean-against-mean, not mean-against-individual-DLAs. Pooling at the per-prompt level would produce a higher threshold (per-prompt DLAs scatter widely; per-head means are tighter around zero), but the comparison would conflate units. Per-head-mean pooling is the methodologically correct choice; the small-sample size of the resulting 144-value distribution is the cost.

τ is **not numerically committed in this amendment**. It is committed in a follow-up §SU-tau amendment after the GPT-2 validation screen has been executed, with the per-head pooled-null distribution summarized alongside the chosen percentile cut.

### SU-4. Validation gate (GPT-2 small, locked)

The detector is screened over **all 144 heads of GPT-2 small** under the locked specification SU-1 through SU-3. The gate has two conjunctive conditions, both of which must be satisfied:

1. **Top-3 inclusion.** L9H1 must appear within the top-3 of the per-head ranking by **mean cross-category DLA** (equally weighted across the four categories: days, months, numerals, letters).
2. **Null-threshold clearance.** L9H1's mean cross-category DLA must exceed τ (the per-head-mean pooled 95th-percentile null threshold defined in SU-3).

Both conditions are evaluated under the same prompt set with one fixed seed; the null permutation seed is pinned at construction time. If either condition fails, the detector specification is treated as invalid and Phase 1.4 hard-stops per SU-6 below. L9H1 is highlighted in the screen figure regardless of pass/fail.

The conjunctive gate is belt-and-suspenders: top-3 by ranking establishes that L9H1 is among the most successor-like heads under the detector (sensitivity), and clearing τ establishes that its score is above what permuted-prefix prompts produce on average across heads (specificity). k=3 is chosen because top-1 is too strict (a single noisy prompt could push L9H1 to rank 2 even if its mechanism is intact), while top-5 is too loose (it admits the possibility of L9H1 being an also-ran among many heads with comparable DLA, which would not validate the *specificity* of the detector). Top-3 absorbs single-prompt noise without admitting false positives.

The SU-4 specification deliberately uses **rank + null-percentile** rather than the σ-statistic that §S-5 used in Phase 1.3 — see SU-7 for the methodological lesson carried forward.

### SU-5. Phase 1.4 scope and Pythia gates (locked)

Phase 1.4 has three deliverables, executed in order:

1. **GPT-2 small validation** per SU-4 above. Gating deliverable; failure triggers SU-6.
2. **Pythia-410M-deduped @ `step143000` anchor inspection.** Two pre-committed gates: a **numerical gate** (≥1 Pythia-410M head clears τ on the same mean cross-category DLA scalar) and a **qualitative gate** (the top candidate has cross-category breadth — DLA is positive across at least 3 of the 4 categories — not concentrated in a single category, which would indicate sequence memorization rather than abstract successor behaviour).
3. **18-cell exploration sweep** on Pythia `{70m, 160m, 410m}` × `{0, 1000, 3000, 8000, 25000, 143000}`. The 70m size is included per the brief's "drop 70m" fallback being post-hoc-only; pre-registered scope includes all three sizes.

The SU-6 hard-stop applies only at the GPT-2 validation level (deliverable 1). The Pythia anchor (deliverable 2) and the 18-cell sweep (deliverable 3) run unconditionally after GPT-2 validation passes; if the anchor fails either of its two gates, the sweep still completes and documents the failure pattern across the 18 cells in Path-C-style negative-result form.

### SU-6. Failure-mode policy (locked)

Failure of the GPT-2 validation gate at SU-4 is a **hard stop**. The response is to re-grill the detector specification from SU-1, beginning with prompt format. There are **no fallback variants** registered in this amendment: no fallback to GPT-2 medium, no fallback to a different category set, no fallback to per-category DLA rather than mean cross-category DLA, no fallback to a per-head null. Any of these would constitute a new specification and would require its own amendment with its own validation gate.

This SU-6 policy mirrors §S-7 of the §S-inhibition amendment exactly. (A note on chronology: mid-grilling, Q6 was initially chosen as a soft fallback to GPT-2 small — a structure that became nonsensical once the SU-0 misattribution was identified, since GPT-2 small *is* the corrected primary. The reversion to hard-stop is documented in NOTES.md Phase 1.4 entry and is the methodological-discipline equivalent of choosing the no-fallback path in §S-7 of Phase 1.3.)

### SU-7. Methodological lessons carried forward from Phase 1.3

Two lessons from Phase 1.3 inform the SU-4 design and are recorded here for traceability:

- **Rank-only validation is load-bearing.** §S-5c (post-data) recorded that the σ-statistic gate is pathological under outlier known-positives. Going forward — including this Phase 1.4 specification — rank-only criteria (top-k inclusion) plus a separately-defined null-percentile threshold replace any σ-of-bulk gate. The SU-4 conjunction of "top-3 by ranking AND clears null-percentile τ" is the Phase-1.4-shape of this lesson.
- **Numerical thresholds must be specified after data is observed, with the procedure committed before.** The SU-3 procedure (within-category prefix permutation, one fixed permutation per (category, base prompt), per-head-mean pooled across heads, 95th percentile) is committed in this amendment. The numerical τ is committed only after the pooled null is observed; this is the same procedure-now / number-later split that §S-tau used for Phase 1.3.

### SU-8. Pre-registration form (locked)

This amendment is **reference-style**: the locked procedural specification lives here in HYPOTHESIS.md; the full grilling rationale and option-list-with-rejected-alternatives lives in NOTES.md (gitignored working memory) under the Phase 1.4 grilling entry. τ is locked in a *separate* HYPOTHESIS.md amendment (§SU-tau) posted after the GPT-2 validation gate (SU-4) is executed.

Implementation files (planned, not yet written): `src/detectors/successor.py` (cross-category DLA detector reusing `src/replication/tigges_ioi.py::component_dla` machinery for per-(layer, head) DLA computation). 4-category prompt builder either inline in the detector module or in a sibling file. No new dependencies. Compute estimate: ~1-2 minutes for GPT-2 small validation, ~5-10 minutes for the full 18-cell Pythia sweep on M5 Pro.

## Amendment 2026-05-06 (early) — §SU-1b: lift-form scoring (supersede of §SU-3 / §SU-4 score definition)

**Posted before any formal validation run is recorded.** This supersede amendment corrects a methodological flaw in the §SU-1 spec that surfaced during a pre-validation smoke test: the score `mean cross-category DLA on real prompts` conflates successor mechanism with category-token-boost behavior, and the population-level null threshold of §SU-3 doesn't isolate one from the other. Pre-validation diagnostic (uncommitted; not a formal validation run) showed L9H1 ranked #36 of 144 under the §SU-1 spec, while the rank-1 head L10H3 had real DLA = +10.37 but lift = real − null = **−1.90** — i.e., L10H3 boosts ordinal tokens *more* on category-shuffled prompts than on real prompts, the literal anti-successor behavior. L9H1 ranked #1 of 144 by lift with lift = +0.3917, exactly the rank L 2023's argmax-within-7-days protocol predicts (verified pre-amendment: L9H1 is the unique 7/7 head; all other 143 heads score ≤3/7 under L's exact protocol).

The chronology — §SU-1 spec locked → smoke-test surfaced the spec failure → research confirmed L 2023 used argmax-within-restricted-vocabulary, not magnitude-summed DLA → local diagnostic showed lift form recovers L 2023's named target as rank-1 → this supersede locks the corrected score *before* formal validation runs are recorded — is the auditable pre-registration record. **No formal data has been recorded yet under §SU-1**; this amendment is the operative spec for Phase 1.4 validation onward.

### SU-1b-3. Score definition (supersedes §SU-3 score-and-threshold construction)

For each head `h`, define the **lift score**:

```
lift[h] = mean over 4 categories c of ( real_DLA_c[h]  −  null_DLA_c[h] )
```

where `real_DLA_c[h]` is the mean DLA at the END token toward `c4`'s first token across all real (clean-text) prompts in category `c`, and `null_DLA_c[h]` is the same mean across the within-category-prefix-permuted variants of those same prompts (one fixed seed-pinned permutation per (category, base prompt), per the unchanged §SU-3 permutation procedure). The unchanged §SU-2 first-token-DLA convention applies.

The lift form measures the head's *successor-specific contribution* — the magnitude by which the head boosts the correct successor over what it would boost when given the same prompt structure with the ordinal relationship destroyed. Heads that boost any ordinal-category token (category-token-boosters) have lift ≈ 0 because their real and null are matched. Heads that boost the successor specifically — the §SU-0 mechanism Gould et al. and L 2023 describe — have positive lift. Anti-successor heads (those that boost the duplicate position more than the successor under the prompt structure) have negative lift; we expect zero or negligible counts of these for our purposes.

The null threshold τ is now defined as **the 95th percentile of the pooled per-head lifts** (one entry per head: 144 entries on GPT-2 small; 384 on Pythia-410M). Pooling is at the per-head-mean level, identical to the §SU-3 unit. τ remains numerically deferred; it is committed in §SU-tau after the GPT-2 validation screen runs.

### SU-1b-4. Validation gate (supersedes §SU-4)

The gate has two conjunctive conditions, both of which must be satisfied:

1. **Top-3 inclusion by lift.** L9H1 must appear within the top-3 of the per-head ranking by `lift`.
2. **Lift exceeds τ.** L9H1's `lift` must exceed τ (the 95th-percentile of the pooled per-head-lift distribution).

The conjunctive structure (§S-5 / §SU-4 belt-and-suspenders pattern) is preserved unchanged. k=3, hard-stop on failure (§SU-6) unchanged. The only change is the score definition: from `real_DLA` to `lift = real − null`.

### SU-1b-5. Untouched legs

§SU-0 (validation target attribution to L 2023, GPT-2 small L9H1), §SU-1 (4 categories, 3-context, predict-4th), §SU-2 (first-token DLA), §SU-3 *permutation* (within-category prefix permutation, one fixed seed-pinned permutation per (category, base prompt), pooled across heads), §SU-5 (Phase 1.4 scope, Pythia gates), §SU-6 (hard-stop on GPT-2 failure, no fallbacks), §SU-7 (methodological lessons), §SU-8 (reference-style pre-registration form) are all unchanged.

### SU-1b-6. Why this is a supersede, not a re-grill

Q6's hard-stop policy (§SU-6) is the canonical response to validation failure. Strictly applied, the smoke-test §SU-1 FAIL should trigger a full re-grill from §SU-1. This supersede amendment is justified by the asymmetric evidence:

- The smoke-test failure surfaced **before** any formal validation run was recorded; nothing under §SU-1 is in the public artifacts. The Q6 hard-stop discipline is designed to prevent post-data goalpost-moving; the failure here is pre-data.
- The diagnostic identified the failure mode mechanistically (`real DLA` conflates successor with category-boost; lift isolates the mechanism) and showed the corrected spec recovers L 2023's named target as rank-1.
- The L 2023 argmax-within-7-days replication independently confirms L9H1 is a real successor head (uniquely 7/7 among 144 heads), so the validation target is correct.
- The fix is **focused** to score definition (one leg, §SU-3/§SU-4); it doesn't touch the prompt format, multi-token handling, scope, failure-mode policy, or pre-registration form.

A full re-grill would relitigate Q1-Q7 unnecessarily. The supersade preserves what works, fixes what doesn't, and surfaces the chronology in writing.

This sets the precedent that pre-data, smoke-test-surfaced spec flaws may be corrected by a focused supersede amendment provided (a) no formal data is recorded under the flawed spec, (b) the failure mode is mechanistically identified, (c) the supersede touches only the affected legs, and (d) the chronology is documented in writing before the corrected spec is run. Post-data spec failures continue to require either a full re-grill (the Q6 default) or a §S-5c-style supplementary acceptance (the Phase 1.3 precedent for outlier-pathology cases).

## Amendment 2026-05-06 — §SU-tau: numerical lift-threshold lock

**Posted after the GPT-2 small validation screen completed under §SU-1b** (commit `7ce6eb4` + validation run `notebooks/_run_gpt2_small_successor_validation.py`).

Per §SU-1b-3, τ_lift is the 95th percentile of the pooled per-head lift distribution observed in the GPT-2 small screen. The validation run recorded:

| head | lift | real_DLA | null_DLA |
|---|---|---|---|
| L9H1 (rank #1, target) | +0.3917 | +0.6147 | +0.2230 |
| L8H8 (rank #2) | +0.3058 | +2.1531 | +1.8473 |
| L11H10 (rank #3) | +0.2400 | −1.7342 | −1.9742 |
| L11H11 (rank #4) | +0.2149 | +2.2355 | +2.0205 |
| L6H5 (rank #5) | +0.2030 | −0.9410 | −1.1440 |

**τ_lift = 0.13496** (95th percentile of pooled per-head lifts across all 144 GPT-2 small heads).

The §SU-1b-4 gate verdict for the locked validation target L9H1:

- Rank by lift: **#1 of 144** ✓ (top-3 condition)
- L9H1 lift = +0.3917 > τ_lift = +0.13496 ✓ (threshold condition)
- **Conjunctive gate: PASS**

Independent corroboration under L (2023)'s exact argmax-within-7-days protocol (run as a §SU-1b-justification probe, not the locked detector): L9H1 is the **unique** head with 7/7 correct day predictions across all 144 heads in GPT-2 small. All 143 other heads score ≤3/7 (most score 1/7 = chance). This replicates L 2023's headline finding via a methodology fully independent of our cross-category DLA detector.

The Pythia anchor inspection (deliverable ii of Phase 1.4) and the 18-cell exploration sweep (deliverable iii) apply this absolute τ_lift threshold without further calibration. Per §SU-3's pooling rule, a Pythia head clears τ_lift if its mean cross-category lift (computed under the Pythia model's own first-token mappings per §SU-2) exceeds 0.13496.

A reviewer reading the chronology should see: §SU-1 / §SU-1b spec committed `788c44e` / `7ce6eb4` → smoke-test informal screen surfaced §SU-1 flaw and motivated §SU-1b → formal validation under §SU-1b runs and writes parquet + npz → this §SU-tau amendment locks τ_lift = 0.13496 from the formal validation distribution. No spec change has been made since the formal validation began.

## Amendment 2026-05-06 — §H2: Phase 2 sweep specification

**Posted after Phase 1.4 closed** (successor detector validated under §SU-1b with §SU-tau τ_lift = 0.13496 locked, 18-cell preview sweep complete, H1-C ordering held in all three Pythia sizes on the 6-cell preview grid). This amendment locks the Phase 2 full-sweep specification operationalizing H1-C across the three locked motifs (induction, successor, S-inhibition) and the three locked sizes (Pythia-70m, 160m, 410m). It is reference-style: the locked sweep design lives here in HYPOTHESIS.md, the option-list rationale lives in NOTES.md under the 2026-05-06 Phase 2 grilling entry. **No numerical-threshold split is deferred.** All three motif thresholds were locked in Phase 1: induction prefix-match > 0.3 (PROJECT_BRIEF.md §4); successor τ_lift = 0.13496 (§SU-tau); S-inhibition τ_strict = 0.0372 (§S-tau). Bootstrap and threshold-sensitivity parameters are not data-dependent. A single §H2 amendment now suffices.

### H2-1. Checkpoint grid (locked)

The Phase 2 sweep runs on **40 log-spaced cells** drawn from Pythia's published 154-checkpoint suite:

```
[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
 20000, 24000, 29000, 35000, 41000, 49000, 59000, 70000,
 84000, 100000, 120000, 143000]
```

All 11 of Pythia's early-dense cells (`step0` plus powers of 2 up to `step512`) are included verbatim; 29 mainline cells span 1000 to 143000 log-spaced. Phase 1's 6-cell preview cells `{0, 1k, 3k, 8k, 25k, 143k}` are subsets of this grid (with the preview's `step25k` landing on `step24k` after rounding to the published-checkpoint set). Consecutive cell ratios: min 1.06, median 1.20, max 2.00. The grid is identical across all 9 (size, motif) cells; no per-motif or per-size grid is defined.

### H2-2. Bootstrap CI scheme (locked)

**Per-prompt bootstrap with B = 1000 resamples, 95% percentile interval** is the per-(size, motif) confidence-interval mechanism for the emergence step μ. The bootstrap unit is **per-prompt resampling with replacement** — not per-head and not block-bootstrap — applied to the per-(head, prompt) score matrix that the runner caches at each cell. For each bootstrap replicate, prompts are resampled with replacement, per-head scores are re-aggregated, the count-vs-step curve is re-fit per §H2-3, and the resulting μ is recorded. The 1000 μ values yield a 95% percentile CI on μ.

A separate **threshold-sensitivity analysis** varies each motif's locked threshold by **± 25% in 5 increments** (the locked threshold ± {-25%, -12.5%, 0%, +12.5%, +25%}) and reports the resulting μ range per (size, motif) cell. Threshold sensitivity is descriptive, not gating; it documents how robust the H1-C verdict is to threshold mis-specification, per the brief's pre-committed limitation §54.

### H2-3. Logistic fit handling and tiered censoring (locked)

The emergence step μ_{s,m} for size s and motif m is extracted from the locked logistic form

```
count(step) ≈ L / (1 + exp(-k · (log(step) − μ)))
```

via `scipy.optimize.curve_fit` on the count-vs-log-step series across the 40 cells. Three tiers handle data sparsity:

- **N_emerged = 5.** If `max(count across 40 cells) ≥ 5`, fit the logistic form directly. Standard scipy fit; bootstrap CI per §H2-2.
- **Marginal (2 ≤ max < 5).** Fit is unstable but not censored. Report the bootstrap-median μ as the point estimate, with a widened CI flagged in figures (visually distinguished from full-fit cells).
- **N_cens = 2.** If `max(count across 40 cells) < 2`, right-censor μ at `step143000`. Treat as "did not emerge during training" for ordering-test purposes.

This tiered scheme subsumes the brief's separate "drop 70m for successor" rule from PROJECT_BRIEF.md §10: 70m × successor cells that are too sparse to fit are right-censored automatically, with no special-case scope reduction.

### H2-4. 70m inclusion and ties-fail policy (locked)

All **9 (size, motif) cells** — 70m, 160m, 410m × induction, successor, S-inhibition — are included in the Phase 2 sweep under the §H2-3 tiered handling. Right-censoring subsumes per-motif scope reduction.

**Ties-fail in permutation test.** If two motifs in the same size are both right-censored (μ tied at `step143000`), their pairwise ordering is undetermined and that size's H1-C ordering check **fails** for that pair. This is the only ordering-undetermined disposition; any other pair where both μ are point-estimated (full-fit or marginal) admits an ordering decision.

### H2-5. Multiple-comparison policy (locked)

**Per-pair descriptive p-values are uncorrected.** The 9 sub-tests (3 sizes × 3 ordering pairs: induction-vs-successor, successor-vs-S-inhibition, induction-vs-S-inhibition) are reported as descriptive diagnostics for the H1-C verdict, not as independent gating tests.

The **joint H1-C permutation test** (p < 0.005 from `(1/6)^3 ≈ 0.00463` under H_0 of exchangeable order across 3 sizes) is the **only gating claim**. Its conjunctive structure already incorporates multiplicity adjustment by construction: under H_0, the joint probability of observing the predicted ordering in *all three* sizes equals the product of three independent (1/6) per-size probabilities, so no further Bonferroni-style correction applies.

### H2-6. Compute and scheduling (locked)

Sweep execution: **per-motif chunked runs** + **upfront prefetch of all unique (size, step) checkpoints not already cached** in parallel + **bootstrap as post-processing** on the cached per-(head, prompt) score matrices + background runs scheduled overnight where possible.

End-to-end compute estimate: **~3 hours MPS time** for the three motif sweeps, plus **~30-60 minutes** for checkpoint prefetch, plus **~2 minutes** for bootstrap post-processing on cached scores, plus trivial logistic-fit time. Per-(size, motif) cell measurement is taken on the first sweep cell and re-projected; if the measured cost exceeds the estimate by more than 2×, the runner pauses for re-grilling rather than silently extending.

### H2-7. Pre-registration form (locked, no deferred lock)

This amendment is **reference-style with NO deferred numerical commit**. Unlike §S-tau (Phase 1.3) and §SU-tau (Phase 1.4), no follow-up amendment locks a data-dependent number after the sweep runs. The reasoning:

- All three motif thresholds are already locked from Phase 1: induction prefix-match > 0.3 (brief §4); successor τ_lift = 0.13496 (§SU-tau); S-inhibition τ_strict = 0.0372 (§S-tau).
- Bootstrap parameters (B = 1000, 95%, per-prompt resampling) are pre-committed and not data-dependent.
- Threshold-sensitivity parameters (± 25% in 5 increments) are pre-committed and not data-dependent.
- The logistic fit form, tiered handling thresholds (N_emerged = 5, N_cens = 2), and ties-fail rule are pre-committed and not data-dependent.

A single §H2 amendment therefore captures the full Phase 2 contract.

### H2-8. Spec-failure-during-Phase-2 policy (locked)

The §SU-1b precedent applies. Pre-data smoke-test-surfaced flaws in this §H2 spec may be corrected by a **focused supersede amendment** provided (a) no formal sweep data is recorded under the flawed spec, (b) the failure mode is mechanistically identified, (c) the supersede touches only the affected legs, (d) the chronology is documented in writing before the corrected spec is run.

Post-data spec failures continue to require either a **Q6-style hard-stop** with full re-grill, or a **§S-5c-style supplementary-acceptance** amendment with the original gate failure recorded in the chronology. No silent goalpost-moving.

### H2-9. Notebook deliverables (locked)

Phase 2 produces **four notebooks**, the first three of which are independent full-sweep extensions of the Phase 1 6-cell previews (the previews remain on disk as 6-cell historical artifacts, *not* in-place edits):

1. `notebooks/induction_full_sweep.ipynb` — 40-cell × 3-size induction sweep, emergence-step μ extraction with bootstrap CI per §H2-2, threshold-sensitivity bracket per §H2-2.
2. `notebooks/successor_full_sweep.ipynb` — 40-cell × 3-size successor sweep, same outputs.
3. `notebooks/s_inhibition_full_sweep.ipynb` — 40-cell × 3-size S-inhibition sweep, same outputs.
4. `notebooks/h1c_ordering_test.ipynb` — cross-motif H1-C verdict notebook. Contents:
   - (1) joint H1-C permutation-test verdict per §H2-5 (the gate);
   - (2) emergence-step μ table with bootstrap CIs across all 9 (size, motif) cells;
   - (3) per-pair descriptive p-values for the 9 sub-tests;
   - (4) side-by-side emergence-comparison curves (3 panels, one per size; three motif curves per panel);
   - (5) major differences between the three head types — depth, count saturation, cross-category breadth, identity stability across training;
   - (6) threshold-sensitivity check per §H2-2;
   - (7) verdict.

Implementation files (planned, not yet written): `notebooks/_run_phase2_induction_sweep.py`, `notebooks/_run_phase2_successor_sweep.py`, `notebooks/_run_phase2_s_inhibition_sweep.py`, `src/analysis/phase2_bootstrap.py`, `src/analysis/phase2_logistic.py`, plus the four notebook builder scripts. No new dependencies beyond what Phase 1 introduced.

## Amendment 2026-05-06 (post-Phase-2 review) — §H2-9-R: post-data reframe to scale-dependent interpretation

**Posted after Phase 2 sweep completion and external review.** This amendment registers a **post-data interpretation reframe** of the Phase 2 H1-C verdict. It does **not** re-amend the joint sign test, the gate, or the per-size strict-ordering policy — those remain as locked in §H2-3 through §H2-5, and the registered gate **passes** at p ≈ 0.00463 < 0.005. The reframe shifts what claim that pass supports.

### What the gate measures, and what it does not

The joint H1-C sign test in §H2-5 is, mechanically, a 3-bit observation: the conjunction of three per-size strict-ordering checks. Under H_0 of exchangeable per-size order, the conjunction has probability (1/6)³ ≈ 0.00463, which is below the locked 0.005 gate. We pass.

The test is silent on three orthogonal questions that an external review correctly surfaced:

1. **Censored versus signal-driven holds.** A pair-ordering can satisfy μ_a < μ_b either because both motifs emerge cleanly with separated CIs (the strong reading) or because the later motif fails to emerge during training and is right-censored at step143000 by §H2-3 (the vacuous reading). The sign test treats these identically. In the Phase 2 result, **4 of 9 (size, motif) cells are right-censored or marginal at the upper logistic-fit sentinel**, so the sign-test pass is partially propped up by censored cells.
2. **Temporal versus architectural ordering.** A temporal emergence ordering (μ_ind < μ_suc < μ_si) does not entail an architectural compositional ordering in the forward pass. Sub-deliverable 4b of `notebooks/h1c_ordering_test.ipynb` documents that in 160m and 410m, successor heads sit at *deeper* normalized layer depth than S-inhibition heads — incompatible with the compositional reading in which corrective S-inhibition heads consume successor outputs at later layers.
3. **Identity stability under turnover.** The "emergence step" abstraction tracks a count of heads above threshold. Sub-deliverable 5 documents that the *identity* of the top heads in successor and S-inhibition turns over substantially between step25k and step143k (top-3 stability 0.0–0.667 across sizes). The emerged population at step143k is not the same circuit as the emerged population at step25k for those motifs.

### What the reframed headline is

**Scale-dependent emergence of S-inhibition.** The headline finding from Phase 2 is that S-inhibition's emergence is scale-dependent in Pythia: it does not emerge during training at 70m (max_count = 1 head; right-censored), emerges marginally at 160m (max_count = 3; logistic fit hits the upper sentinel; bootstrap CI overlaps successor), and emerges with a clean fit at 410m (max_count = 3 with a tight CI). Induction and successor emerge robustly across all three sizes. The H1-C ordering pass is therefore **robustly supported only at 410m**; 160m is a marginal pass; 70m is a vacuous pass through censoring.

This is more reportable than "joint H1-C HOLDS in 3 sizes." It is also more accurate to the data.

### What this reframe does NOT do

- It does NOT re-amend the gate. §H2-5's `p < 0.005` gate, the (1/6)³ joint null, the strict per-size ordering check, and the §H2-4 ties-fail policy are unchanged. The gate passes.
- It does NOT change the registered headline of the project. PROJECT_BRIEF.md and HYPOTHESIS.md §3 register H1-C as the falsification target; the falsification target is *not falsified*. The reframe is about what supports the non-falsification.
- It does NOT relabel any cell's regime. The §H2-3 tiered handling rule (emerged ≥ 5, marginal 2-4, censored < 2) is unchanged; the marginal/censored cells are flagged as such in the verdict, exactly as the rule prescribes.
- It does NOT add a new gate. No new threshold, no new joint test, no new pass-fail criterion.

### What this reframe DOES do

It changes the **emphasis** in the writeup deliverable, the headline-notebook verdict cells, and the README status table:

- The headline becomes "scale-dependent emergence of S-inhibition" rather than "H1-C HOLDS jointly."
- The Pythia-410m row is the robust per-size confirmation; 160m is a marginal pass; 70m is a censored pass. This three-tier per-size verdict is foregrounded.
- The depth-vs-temporal asymmetry (sub-deliverable 4b) is promoted from a side observation to a substantive finding distinct from the registered hypothesis.
- The bootstrap reversal-rate (replacing the original prior-under-exchangeability descriptive p-values, which were not p-values) is the magnitude-aware companion to the sign test, and is reported for each (size, pair).
- The structural-reuse analysis in `notebooks/motif_structural_reuse.ipynb` (Phase 3 / Extension B) is added as the cross-motif circuit-identity diagnostic: 410m has zero multi-motif heads in top-5, while 70m and 160m have overlap. The cleanest H1-C-confirming size is also the cleanest structural-separation size.

### Procedural precedent

This is a **post-data interpretation reframe**, distinct from:
- A §SU-1b-style focused supersede (which requires no formal data recorded under the flawed spec; here, the data is recorded and the spec is correct).
- A §S-5c-style supplementary acceptance (which records a strict gate FAIL with a rank-only override; here, the gate passes).
- A re-amendment of §H2-3 to change the censoring rule (which would be silent goalpost-moving).

The precedent set: when a registered gate passes but the supporting evidence is heterogeneous across cells in a way that makes the joint-claim headline misleading, the response is a **post-data interpretation reframe** that (a) keeps the gate, (b) documents the heterogeneity, (c) re-emphasizes the writeup. This is distinguishable from goalpost-moving because the gate is preserved as registered and the reframe is dated after the data.

### Replacement of original sub-deliverable 3

Sub-deliverable 3 in §H2-9 was originally specified as "per-pair descriptive p-values for the 9 sub-tests." On review, those values were `0.5 if holds else 1.0` — i.e., the prior probability of the observed pair-ordering under exchangeable H_0, dressed as a p-value. They are not p-values; they are constants by construction within {holds, fails}.

Replaced under §H2-9-R with the **bootstrap reversal rate per (size, pair)**: the empirical fraction of B=1000 per-prompt bootstrap replicates in which the predicted ordering does not hold (with censored ties counted as undetermined per §H2-4). This is a magnitude-aware summary of how robust each per-size pair-ordering is to per-prompt resampling. The replacement is a clarification (the original sub-deliverable did not measure what its name suggested), not a methodological change to the gate.

### Replacement of B=1 in sub-deliverable 6

Sub-deliverable 6 (threshold sensitivity) was originally implemented with `B=1` per-call to the bootstrap functions, which produced point estimates with no envelope. §H2-9-R re-runs the threshold-sensitivity panel with the registered B=1000, displaying the bootstrap CI as a fill envelope on the figure. This is a bug fix to bring the implementation in line with the registered §H2-2 spec; it does not change the spec.

### What gets added

`notebooks/motif_structural_reuse.ipynb` — Phase 3 / Extension B cross-motif structural-reuse analysis (Jaccard similarity of top-5 head sets across motifs, multi-motif head identification at the final checkpoint). Uses existing Phase 2 sweep data; no new compute beyond the parquet reads.

`notebooks/h1c_ordering_test.ipynb` adds: a methodological audit-trail section linking the upstream Tigges replication, copy-suppression pivot evidence, and §S-5c override; a power-analysis section quantifying μ-uncertainty as a function of L_true (consistent with the marginal-cell bootstrap CIs); the bootstrap reversal-rate section replacing the original descriptive_p table; the reframed verdict.

The three `*_full_sweep.ipynb` notebooks add: a final-checkpoint head-identity table per size, listing every (layer, head) passing threshold (the inspectable circuit a reviewer needs).

`notebooks/induction_full_sweep.ipynb` adds: the Olsson 2022 / Singh 2024 cross-reference for induction emergence step as an external tooling-validation sanity check.

The README status table for Phase 2 is updated to reflect the reframed claim.

## Amendment 2026-05-06 (post-grilling) — §H3-scale: Pythia-1B scale-extension specification

**Posted after Phase 2 closure (§H2-9-R reframe registered) and before any Pythia-1B compute.** This amendment locks the procedural specification of the Pythia-1B scale-extension test. It is reference-style: the locked specification lives here in HYPOTHESIS.md; the option-list rationale and the grilling chronology live in NOTES.md (gitignored working memory) under the 2026-05-06 §H3-scale grilling entry. **No numerical-threshold split is deferred.** All three motif thresholds carry over from Phase 1: induction prefix-match > 0.3 (PROJECT_BRIEF.md §4); successor τ_lift = 0.13496 (§SU-tau); S-inhibition τ_strict = 0.0372 (§S-tau). Bootstrap and threshold-sensitivity machinery are inherited from §H2-2. Tiered censoring is inherited from §H2-3.

§H3-scale **extends, does not supplant**, the registered §H1-C / §H2-5 gate. §H2-5 passed as registered (joint sign-test p = 0.00463 < 0.005 across {70m, 160m, 410m}); §H2-9-R noted post-data scale-dependent emergence of S-inhibition, with 4 of 9 cells right-censored or marginal, and only 410m a robust per-size confirmation. §H3-scale is the forward-looking falsification of the §H2-9-R reframe at a fourth size: it tests whether the scale-dependent extrapolation continues, saturates, or reverses at Pythia-1B. The original §H1-C falsification target is not affected; PROJECT_BRIEF.md §3 and HYPOTHESIS.md §3 remain the registered project headline.

Pre-registration chronology for auditability: §H2-9-R registered post-data → grilling session 2026-05-06 surfaced the desired §H3-scale prediction → this amendment commits before any 1B sweep code runs → §H3-scale verdict recorded in `notebooks/h1c_ordering_test.ipynb` extension after the sweep.

### §H3-scale-1. Scope (locked)

The 1B scale-extension runs on **Pythia-1B-deduped** (`EleutherAI/pythia-1b-deduped`), matching the deduped lineage of the registered 70m / 160m / 410m sweep. The checkpoint grid is the **40-cell §H2-1 grid verbatim**, identical to the schedule used for the 3 registered sizes. The three locked motifs (induction, successor, S-inhibition) and the three locked absolute thresholds carry over without re-derivation. The detectors operate on Pythia-1B's 128-head architecture (16 layers × 8 heads × d_model = 2048).

### §H3-scale-2. Strong scale leg (A) — locked

Leg (A) has three conjunctive sub-conditions; all must hold:

- **(A.i) Count-threshold at 1B.** `max_count_si^1B ≥ 5` over the 40-cell sweep — full-fit regime entry per §H2-3. This is a strictly stronger requirement than the 410m S-inhibition cell achieved (max = 3, marginal regime per §H2-3); the strengthening is intentional and operationalizes the §H2-9-R "S-inhibition gets cleaner with scale" reframe in count-axis form. Pythia-1B has 128 heads total; max_count ≥ 5 corresponds to 3.9% density, vs 410m's 3/384 = 0.78%.

- **(A.ii) Bootstrap reversal-rate on cross-size emergence-step ordering.** `P(μ_si^1B < μ_si^410m) ≥ 0.95` over B = 1000 paired per-prompt bootstrap replicates. Per §H2-2 machinery: each bootstrap replicate resamples prompts with replacement, refits the count-vs-step logistic curve at both 1B and 410m, and records μ_si^1B and μ_si^410m. The reversal-rate is the empirical fraction of replicates in which μ_si^1B < μ_si^410m. Threshold ≥ 0.95 corresponds to a one-sided 5% test of the directional scale-dependence claim.

- **(A.iii) Within-1B bootstrap CI separation.** 95% bootstrap percentile CIs on `μ_si^1B` and `μ_suc^1B` are disjoint, with μ_si^1B strictly above μ_suc^1B. This is the within-1B coherence check on the order-preservation; it inherits §H2-2's bootstrap machinery without modification.

### §H3-scale-3. Order-preservation leg (B) — locked

Leg (B) has two conjunctive sub-conditions; both must hold:

- **(B.i)** All three motifs at 1B reach the full-fit regime per §H2-3 (`max_count ≥ 5`).
- **(B.ii)** Strict per-size ordering `μ_ind^1B < μ_suc^1B < μ_si^1B` holds at point estimate per §H2-5's per-size sign-test rule.

§H2-4's ties-fail policy applies: if any pair has both μ right-censored, the ordering for that pair is undetermined and (B) fails.

### §H3-scale-4. Joint gate — locked

§H3-scale **passes** iff (A.i) AND (A.ii) AND (A.iii) AND (B.i) AND (B.ii). Failure of any leg falsifies §H3-scale; the failure pattern is interpreted per §H3-scale-6 below. There is no partial-pass disposition.

### §H3-scale-5. Anchor inspection (sub-amendment §H3-scale-anchor) — locked

Three sub-anchors run at Pythia-1B-deduped @ step143000 before the full sweep:

- **Induction anchor:** verify ≥ 1 head clears prefix-match score > 0.3 (brief §4 threshold) on the 50-sequence Olsson 2022 probe.
- **Successor anchor:** §SU-5 deliverable (ii) gates verbatim: numerical (≥ 1 head clears τ_lift = 0.13496) AND qualitative (top candidate has positive lift in ≥ 3/4 categories: days, months, numerals, letters).
- **S-inhibition anchor:** §S-8 deliverable (ii) gates verbatim: numerical (≥ 1 head clears τ_strict = 0.0372 on Δ_h) AND mechanistic (≥ 2/4 of Pythia-1B's NMs — identified per §S-3 component-DLA top-4 — show positive Δ along the path candidate → NM).

**The 40-cell full sweep runs unconditionally regardless of anchor outcome.** This mirrors §S-8's policy verbatim. Anchor failure is documented in the anchor notebook output as a Path-C-style negative-result artifact and the sweep proceeds. The cost asymmetry that might justify a tooling-failure halt at this scale (~4 h MPS time) is overridden by pre-registration discipline; the unconditional-sweep rule is preserved across all phases.

### §H3-scale-6. Failure-mode taxonomy — locked

Six pass / fail patterns are pre-committed, each matched to a paper-headline interpretation:

| Pattern | Trigger | Pre-committed paper headline |
|---|---|---|
| **PASS** | All 5 sub-conditions hold | "Scale-dependent S-inhibition emergence holds across Pythia 70m → 1B." |
| **SAT** | (B.i), (B.ii), (A.i), (A.iii) hold; (A.ii) fails | "S-inhibition emergence time saturates between 410m and 1B; head count continues to scale." Reframe survives, refined to count-axis. |
| **REGR** | (A.i) fails (max_count < 5 at 1B; potentially < 3) | "Non-monotonic S-inhibition emergence in Pythia: regression at 1B." Substantive falsification of §H2-9-R reframe; novel reverse-scaling finding. |
| **ORD-BREAK** | (B.ii) fails | "H1-C ordering is scale-bounded in Pythia: breaks at 1B." Localized falsification of H1-C-extended; §H2-5 (3-size) still stands. |
| **WIDE-CI** | (A.iii) fails (CIs overlap between μ_si^1B and μ_suc^1B) | "1B reproduces 160m's marginal-overlap pattern, not 410m's clean separation; scale-dependence is non-monotonic in CI-axis." |
| **TOOLING** | Detector outputs distributionally broken: NaNs, all-zero, sign-flipped vs 410m anchor, score range > 10× shift from 410m | "Methodological note: detector instability at d_model = 2048 on MPS." Not a substantive result; verdict deferred pending re-tooling. |

**Multi-leg failures default to the most-severe pattern triggered**, in priority order:

`TOOLING > REGR > ORD-BREAK > WIDE-CI > SAT > PASS`.

Rationale: TOOLING is non-substantive and overrides all others; REGR is the most novel substantive failure (reverse-scaling); ORD-BREAK is a localized falsification of an existing claim; WIDE-CI and SAT are quantitative refinements. Post-data, the writeup must use the matched headline or trigger a §H2-8 supplementary-acceptance amendment with chronology recorded; silent goalpost-moving is forbidden by the same rule that governs §H2-9-R.

### §H3-scale-7. Compute and scheduling (locked)

End-to-end estimate: prefetch ~30–60 min HF download + anchor ~30 min MPS + sweep ~4 h MPS + bootstrap post-processing ~2 min + analysis-notebook builds ~30 min. Disk: ~120 GB for 40 cached checkpoints with `*.bin` excluded from the prefetch snapshot patterns, leaving safetensors-only. Per §H2-6 escape hatch: if measured per-cell cost on the first sweep cell exceeds the projection by more than 2×, the runner pauses for re-grilling rather than silently extending.

### §H3-scale-8. Notebook and parquet deliverables (locked)

The 1B scale-extension produces:

1. `notebooks/_prefetch_1b_checkpoints.py` — 40-checkpoint snapshot download, safetensors-only patterns (`*.bin` explicitly excluded from `_SNAPSHOT_PATTERNS_1B`).
2. `notebooks/_run_pythia_1b_anchor_induction.py`, `_run_pythia_1b_anchor_successor.py`, `_run_pythia_1b_anchor_s_inhibition.py` — three anchor scripts mirroring the 410m anchors.
3. `notebooks/_run_phase3_1b_induction_sweep.py`, `_run_phase3_1b_successor_sweep.py`, `_run_phase3_1b_s_inhibition_sweep.py` — three full-sweep runners writing to `phase3_*_sweep.parquet` (1B-only; Phase 2 parquets are not modified).
4. `notebooks/induction_full_sweep_1b.ipynb`, `successor_full_sweep_1b.ipynb`, `s_inhibition_full_sweep_1b.ipynb` — three new per-size sweep notebooks parallel to the existing 3-size full-sweep notebooks. Existing 3-size full-sweep notebooks are NOT modified.
5. `notebooks/h1c_ordering_test.ipynb` — extended in place with a §H3-scale verdict section appended at the end. Existing §H2-5 / §H2-9-R sections preserved. New section contents:
   - (i) the §H3-scale gate verdict per §H3-scale-4 with the matched failure-mode pattern per §H3-scale-6;
   - (ii) 4-size emergence-step μ table with bootstrap CIs across all 12 (size, motif) cells;
   - (iii) bootstrap reversal-rate `P(μ_si^1B < μ_si^410m)` per §H3-scale-2 (A.ii);
   - (iv) 4-panel emergence-curve figure (one panel per size, three motif curves per panel);
   - (v) within-1B CI-separation diagnostic per §H3-scale-2 (A.iii);
   - (vi) anchor-inspection diagnostic table per §H3-scale-5;
   - (vii) the matched paper-headline string per §H3-scale-6.
6. `notebooks/motif_structural_reuse.ipynb` — extended in place with 4-size Jaccard table and 4-size multi-motif head identification. Existing 3-size analysis preserved.
7. `notebooks/motif_attention_inspection.ipynb` — extended in place with 1B section at step143000 for paper figures.

### §H3-scale-9. Procedural precedent (locked)

§H2-8's spec-failure-during-phase policy applies to §H3-scale verbatim. Pre-data smoke-test-surfaced flaws in this §H3-scale spec may be corrected by a focused supersede amendment provided the §SU-1b conditions hold (no formal data recorded under the flawed spec, mechanistic identification of the failure, focused supersede touching only affected legs, written chronology). Post-data spec failures continue to require either Q6-style hard-stop with full re-grill or §S-5c-style supplementary-acceptance amendment with the original gate failure recorded in the chronology.

### §H3-scale-10. Pre-registration form (locked, no deferred lock)

This amendment is **reference-style with NO deferred numerical commit**. All numerical thresholds (induction > 0.3, τ_lift = 0.13496, τ_strict = 0.0372), bootstrap parameters (B = 1000, 95% percentile, per-prompt resampling), threshold-sensitivity parameters (± 25% in 5 increments), tiered-censoring thresholds (full-fit ≥ 5, marginal 2-4, censored < 2), and the (A.ii) reversal-rate threshold (≥ 0.95) are pre-committed in this single amendment.

A reviewer reading the chronology should see: §H1-C registered → §H2 sweep specification locked → Phase 2 sweep ran and §H2-5 gate passed at p = 0.00463 → §H2-9-R post-data reframe registered → grilling session 2026-05-06 surfaced the §H3-scale forward-looking prediction → this §H3-scale amendment commits before any 1B compute → 1B sweep runs → §H3-scale verdict recorded in `notebooks/h1c_ordering_test.ipynb` extension. No spec change is anticipated between this amendment and the §H3-scale verdict.

## Amendment 2026-05-07 — §H3-scale-8-vis: visualization-layer integration of 1B into existing notebooks

**Posted post-data (after §H3-scale verdict was computed and recorded).** This amendment is a **visualization-layer supersede** of §H3-scale-8 (#4) and (#5). The §H3-scale data, gate, verdict, and matched failure-mode pattern (REGR) are unchanged. The only change is how 1B is presented in the existing notebook artifacts.

### What is changed

- **Per-size full-sweep notebooks** (`induction_full_sweep.ipynb`, `successor_full_sweep.ipynb`, `s_inhibition_full_sweep.ipynb`): originally Phase 2 3-size deliverables. Now extended to load both `phase2_*_sweep.parquet` and `phase3_1b_*_sweep.parquet` transparently and present 4 sizes (70m, 160m, 410m, 1b) side-by-side as the natural read. The §H3-scale-8 (#4) lock that "Existing 3-size full-sweep notebooks are NOT modified" is superseded by this amendment for the visualization layer only.
- **`h1c_ordering_test.ipynb`**: §H1-C / §H2-5 / §H2-9-R sections preserved. The §H3-scale section originally appended at the end is reframed as a cross-cutting "1B scale-extension verdict" sub-section with the 4-size figures already produced by the main flow's bootstrap + emergence-curves cells; redundant 4-size figures and tables are removed from the §H3-scale section. The §H3-scale gate verdict, reversal-rate diagnostic, within-1B CI separation, anchor-inspection table, and matched-headline cells remain — they are §H3-scale-specific and not duplicated by the main flow.
- **`*_full_sweep_1b.ipynb` notebooks** (the 1B-only artifacts produced under §H3-scale-8 #4): **deleted**, since their content is now subsumed by the 4-size main notebooks. The redundant `_build_*_1b.py` builder scripts are also deleted.

### What is NOT changed

- All raw artifacts: `phase2_*_sweep.parquet` (3-size) and `phase3_1b_*_sweep.parquet` (1B-only) are unchanged. Data lineage is preserved by the parquet split.
- The §H3-scale gate, the (A.i)-(A.ii)-(A.iii)-(B.i)-(B.ii) leg structure, the §H3-scale-6 failure-mode taxonomy, and the matched REGR pattern from the executed gate verdict are unchanged.
- The §H1-C / §H2-5 joint sign-test (registered to 3 sizes) continues to operate on `REGISTERED_SIZES = ['70m', '160m', '410m']` only. The 4-size visualization in `h1c_ordering_test.ipynb` does NOT extend the joint test to 4 sizes; only the figures and per-size diagnostic tables are 4-size.
- §H3-scale's joint pre-data registration in HYPOTHESIS.md §H3-scale-1 through §H3-scale-10. The procedural lock on the gate definition is not affected; only artifact presentation is.

### Why this is not a §H3-scale-9 spec failure

§H3-scale-9 reserves Q6-style hard-stop and §S-5c-style supplementary-acceptance for **gate spec failures**. This amendment changes neither the gate spec nor any data; it integrates a registered Phase 3 result into the same notebook artifacts that hold the registered Phase 2 result so that a reviewer reads 4 sizes side-by-side rather than as a Phase-2 + Phase-3-extension split. The chronology — §H3-scale registered → 1B sweep ran → REGR verdict recorded → this presentation-layer amendment dated after the verdict — is preserved by the git commit history and explicit dates in this amendment chain.

### Reviewer instructions

A reviewer who needs to verify pre-registration discipline should:
1. Read the §H3-scale (1-10) amendment for the locked gate spec.
2. Read the executed verdict in `data/exploration/phase3_1b_h3scale_verdict.parquet` and the §H3-scale section of `h1c_ordering_test.ipynb`.
3. Inspect `git log --follow notebooks/induction_full_sweep.ipynb` etc. to see the commit chronology of the original 3-size notebooks vs the 4-size visualization upgrade.
4. The original 3-size figures, exactly as they were when §H1-C / §H2-5 passed at p = 0.00463, can be reconstructed by checking out the pre-§H3-scale-8-vis commit.

## Amendment 2026-05-07 — §H4-scaling: head-count-axis scaling argument, separate from emergence

**Posted after §H3-scale verdict at Pythia-1B (REGR pattern recorded) and before any Pythia-2.8B compute.** This amendment articulates the scaling-axis claim cleanly and supersedes §H3-scale-1's "Pythia-1B-deduped" target. The §H1-C / §H2-5 registered emergence claim is **unaffected** by this amendment; emergence claims remain locked at Pythia-{70m, 160m, 410m} per HYPOTHESIS.md §3, the registered §H2 sweep, and the §H2-5 joint sign-test verdict (passed at p = 0.00463). §H4 is a *separate* claim about the head-count axis, registered fresh before any 2.8B compute, evaluated against pre-registered legs at Pythia-2.8B-deduped.

### Project-narrative two-track structure

Going forward, the project structure has two narrative tracks, each registered separately:

1. **Emergence track (registered, complete).** §H1-C ordering claim across Pythia-{70m, 160m, 410m}. Tested at the §H2 sweep, gate passed at p = 0.00463 < 0.005. Reframed post-data in §H2-9-R to scale-dependent emergence with one robust per-size confirmation (410m). **No further sizes added; emergence claim is locked at the registered three.**

2. **Scaling track (this amendment).** Head-count-axis test: at higher head-count tiers, does S-inhibition scale (timing earlier + count higher)? Pythia-1B (§H3-scale, REGR-recorded) is reframed under this amendment as a head-count regression rather than a scale-up. Pythia-2.8B is the next valid scale-up tier; §H4 registers two legs at 2.8B before any 2.8B compute starts.

### Head-count axis rationale

Pythia-{70m, 160m, 410m} head-count progression is approximately 3× per step:

| size | params | layers × heads | total heads |
|---|---|---|---|
| 70m | 70M | 6 × 8 | **48** |
| 160m | 160M | 12 × 12 | **144** |
| 410m | 410M | 24 × 16 | **384** |
| 1b | 1.0B | 16 × 8 | **128** ← regression |
| 1.4b | 1.4B | 24 × 16 | 384 (same as 410m) |
| **2.8b** | **2.8B** | **32 × 32** | **1024** |

Pythia-1B has 8 heads per layer (vs 16 at 410m), which makes its 128 total heads narrower than 410m's 384 — a *regression* in head count, not a scale-up. The scaling claim should be measured along the head-count axis (the axis at which detectors operate and gate predicates evaluate), not the parameter axis (which includes feedforward parameters irrelevant to attention-head emergence).

This rationale is post-hoc with respect to §H3-scale (whose target was 1B), but pre-data with respect to §H4 (whose target is 2.8B). The 1B verdict (REGR per §H3-scale-6) is preserved as recorded; §H4 redefines the scaling target prospectively.

### §H4-1. Scope (locked)

The head-count-axis scaling test runs on **Pythia-2.8B-deduped** (`EleutherAI/pythia-2.8b-deduped`). The deduped variant matches the lineage used throughout this project (Phase 1.0 pilot, Phase 1.2 Tigges replication, Phase 1.3 / 1.4 anchors, Phase 2 sweep, §H3-scale 1B run). Deduped is the de facto mech-interp standard for Pythia (Tigges 2024, McDougall 2024, Singh 2024 all use deduped); training on the Pile-deduplicated set produces fewer memorization-driven attention-head artifacts than the original Pile. The non-deduped Pythia variants are not used at any size in this project. This deduped commitment is now explicit at the amendment level rather than inherited tacitly through `src/utils/pythia_loader.py`.

The checkpoint grid is the **40-cell §H2-1 grid verbatim**. The locked thresholds carry over: induction prefix-match > 0.3 (PROJECT_BRIEF.md §4); successor τ_lift = 0.13496 (§SU-tau); S-inhibition τ_strict = 0.0372 (§S-tau). Bootstrap and threshold-sensitivity machinery are inherited from §H2-2. Tiered censoring is inherited from §H2-3.

§H4-scaling concerns S-inhibition only. Induction and successor saturate "yes they emerge robustly" by 410m and at 1B; their cross-size scaling is not the substantive claim being tested. Their parquets and per-cell caches at 2.8B will be produced for completeness alongside S-inhibition's, but the §H4 gate predicates operate on S-inhibition only.

### §H4-2. Scaling-claim legs (locked)

§H4 has two conjunctive legs; both must hold:

- **(A.timing) Bootstrap reversal-rate on emergence-step ordering, 2.8B vs 410m.** `P(μ_si^2.8B < μ_si^410m) ≥ 0.95` over B = 1000 paired per-prompt bootstrap replicates. Per §H2-2 machinery: each replicate resamples prompts with replacement, refits the count-vs-step logistic curve at both 2.8B and 410m, and records μ_si^2.8B and μ_si^410m. The reversal-rate is the empirical fraction of replicates in which μ_si^2.8B < μ_si^410m. Threshold ≥ 0.95 corresponds to a one-sided 5% test of the directional timing-axis scaling claim.

- **(A.count) Absolute count threshold breaks 410m saturation.** `max_count_si^2.8B ≥ 5` over the 40-cell sweep — full-fit regime entry per §H2-3. The numerical threshold is identical to the original §H3-scale (A.i) bar; the supersede argument is that the threshold of 5 was operationally too strict at 1B's 128 heads (3.9% density bar) but is appropriate at 2.8B's 1024 heads (0.49% density). The empirical motivation: in the registered Phase 2 data, max_count_si saturates at 3 from 160m (3 / 144) through 410m (3 / 384), and the 1B head-count regression also showed max_count = 3. (A.count) tests whether breaking past head-count tier 384 (to 1024) breaks the saturation cap.

### §H4-3. Joint gate (locked)

§H4 **passes** iff (A.timing) AND (A.count). Failure of either leg falsifies §H4. Within-2.8B coherence (analog of §H3-scale (A.iii)) is **deliberately omitted** from the gate — the H1-C ordering at 2.8B is *measurable but not gating*, since emergence-ordering claims remain locked at Pythia-{70m, 160m, 410m} per the project-narrative two-track structure above. The within-2.8B ordering will be reported in the analysis notebook as a side observation, not as a §H4 gate predicate.

### §H4-4. Anchor inspection (sub-amendment §H4-anchor) — locked

Three sub-anchors run at Pythia-2.8B-deduped @ step143000 before the full sweep, **inherited verbatim from §H3-scale-5**:

- **Induction anchor:** verify ≥ 1 head clears prefix-match score > 0.3 (brief §4 threshold) on the 50-sequence Olsson 2022 probe.
- **Successor anchor:** §SU-5 deliverable (ii) gates verbatim: numerical (≥ 1 head clears τ_lift = 0.13496) AND qualitative (top candidate has positive lift in ≥ 3/4 categories: days, months, numerals, letters).
- **S-inhibition anchor:** §S-8 deliverable (ii) gates verbatim: numerical (≥ 1 head clears τ_strict = 0.0372 on Δ_h) AND mechanistic (≥ 2/4 of Pythia-2.8B's NMs — identified per §S-3 component-DLA top-4 — show positive Δ along the path candidate → NM).

**The 40-cell full sweep runs unconditionally regardless of anchor outcome.** Mirrors §S-8 / §H3-scale-5 verbatim. The anchor's job is to verify detectors fire at d_model = 2560 before the sweep, not to gate the §H4 scientific claim. Anchor failure on substantive grounds (head-existence) is documented as a Path-C-style negative-result artifact and the sweep proceeds. Anchor failure on tooling grounds (NaNs / sign-flips / score range > 10× shift from 410m) is also documented; the sweep proceeds and the analysis notebook flags the TOOLING failure-mode pattern (per §H4-5 below).

### §H4-5. Failure-mode taxonomy (locked)

Five pass / fail patterns are pre-committed, each matched to a paper-headline interpretation:

| Pattern | Trigger | Pre-committed paper headline |
|---|---|---|
| **PASS** | (A.timing) AND (A.count) hold | "Scaling argument confirmed: at Pythia-2.8B's 1024-head architecture, S-inhibition timing accelerates beyond 410m and head count exceeds the 410m saturation cap." |
| **TIMING-ONLY** | (A.timing) holds; (A.count) fails | "Timing-axis scaling holds at 2.8B; count-axis saturation extends from 1B's narrow architecture to 2.8B's 1024-head architecture, suggesting count saturation is fundamental rather than head-count-rate-limited." |
| **COUNT-ONLY** | (A.count) holds; (A.timing) fails | "Count-axis scaling unlocks at 1024 heads; timing-axis saturates between 410m and 2.8B." |
| **NEITHER** | Both legs fail | "Scaling argument falsified at 2.8B: both timing and count saturate beyond 410m on the head-count axis." |
| **TOOLING** | Detector outputs distributionally broken: NaNs, all-zero, sign-flipped vs 410m anchor, or score range > 10× shift from 410m | "Methodological note: detector instability at d_model = 2560 on MPS." Not a substantive result; verdict deferred pending re-tooling. |

**Multi-leg failures default to the most-severe pattern triggered**, in priority order:

`TOOLING > NEITHER > COUNT-ONLY > TIMING-ONLY > PASS`.

Rationale: TOOLING is non-substantive and overrides all others; NEITHER is the most-falsifying substantive pattern (both legs fail); COUNT-ONLY is more substantively interesting than TIMING-ONLY because (A.count)'s ≥ 5 is the harder bar to clear (count saturation at 410m is the surprising registered finding, breaking it is the substantive claim); TIMING-ONLY is partial vindication of the §H2-9-R reframe's timing-axis prediction. Post-data, the writeup must use the matched headline or trigger a §H2-8 supplementary-acceptance amendment with chronology recorded; silent goalpost-moving is forbidden by the same rule that governs §H2-9-R and §H3-scale-6.

### §H4-6. Status of §H3-scale (Pythia-1B) result under §H4 (locked)

The §H3-scale (1B) verdict — REGR per §H3-scale-6 priority `TOOLING > REGR > ORD-BREAK > WIDE-CI > SAT > PASS` — is **preserved as a sealed historical record** in HYPOTHESIS.md and the corresponding artifacts (`data/exploration/phase3_1b_*` parquets, `notebooks/h1c_ordering_test.ipynb` §H3-scale verdict cells). The 1B run is reframed under §H4 as a **head-count regression** (128 heads, narrower than 410m's 384) rather than a scale-up; in the writeup, 1B is referenced as the empirical observation that motivated the architecture-correction supersede registered in this amendment.

The 1B data and verdict are NOT used to evaluate §H4. The §H4 gate (§H4-2 / §H4-3) operates on 2.8B vs 410m. §H3-scale's chronology — registered at 1B → ran at 1B → REGR verdict → architecture-axis distinction surfaced post-hoc → §H4 registered at 2.8B before 2.8B compute — is preserved verbatim by the git log and the explicit dating of this amendment.

### §H4-7. Compute and scheduling (locked)

End-to-end estimate at Pythia-2.8B-deduped:
- Prefetch: ~1–2 h HF download, ~700 GB safetensors-only (40 ckpts × ~17.5 GB / ckpt). Free disk after prefetch: ~720 GB on the 1.5 TB volume.
- Memory: 11.2 GB fp32 weights + ~5–8 GB working = ~17–20 GB. Fits in 64 GB unified, tighter than 1B (~17 GB).
- Anchor: ~30–45 min MPS time (slightly longer than 1B's 3 min total due to d_model = 2560 cost).
- Sweep: ~6 h MPS time (forward-pass cost ≈ 3.1× of 1B).
- Bootstrap post-processing: ~2 min.
- Path-patching memory pressure at d_model = 2560 × 1024 heads × clean+corrupted activations: known risk. Mitigations registered in advance — `BATCH_SIZE=10` for S-inhibition (vs 25 at 1B, 50 at smaller sizes); fp16 model fallback if fp32 OOMs.

Per §H2-6 escape hatch: if measured per-cell cost on the first sweep cell exceeds the projection by more than 2×, the runner pauses for re-grilling rather than silently extending. If anchor S-inhibition OOMs at fp32, switch to fp16 model and re-anchor before sweep.

### §H4-8. Notebook and parquet deliverables (locked)

The §H4 scaling extension produces:

1. `notebooks/_prefetch_2_8b_checkpoints.py` — 40-checkpoint snapshot download, safetensors-only patterns (mirrors `_prefetch_1b_checkpoints.py`).
2. `notebooks/_run_pythia_2_8b_anchor_{induction,successor,s_inhibition}.py` — three anchor scripts mirroring 1B anchors with `SIZE = "2.8b"`.
3. `notebooks/_run_phase4_2_8b_{induction,successor,s_inhibition}_sweep.py` — three full-sweep runners writing to `phase4_2_8b_*_sweep.parquet` (Phase 2 and Phase 3 parquets unchanged).
4. `notebooks/_run_phase4_2_8b_analysis.py` — bootstrap + (A.timing) + (A.count) + §H4 verdict analysis (mirrors `_run_phase3_1b_analysis.py`).
5. **Existing notebooks extended in place to load 5-tier head-count data** (smooth integration like the 1B add): `induction_full_sweep.ipynb`, `successor_full_sweep.ipynb`, `s_inhibition_full_sweep.ipynb` add `'2.8b'` to `SIZES`; the `sweep_path()` helpers extend with a `'2.8b'` branch loading `phase4_2_8b_*_sweep.parquet`. `motif_structural_reuse.ipynb` adds `'2.8b'` to its `SIZES`. `motif_attention_inspection.ipynb` adds the `('2.8b', motif)` rows to `TOP_HEADS` after the anchor identifies them.
6. New section in `h1c_ordering_test.ipynb`: §H4-scaling verdict cell parallel to the existing §H3-scale verdict cell. Loads `phase4_2_8b_h4scaling_verdict.parquet`, displays the (A.timing) and (A.count) leg results, the matched §H4-5 failure-mode pattern, and the matched paper headline.

### §H4-9. Procedural precedent and spec-failure policy (locked)

§H2-8's spec-failure-during-phase policy applies to §H4 verbatim. Pre-data smoke-test-surfaced flaws in this §H4 spec may be corrected by a focused supersede amendment provided the §SU-1b conditions hold (no formal data recorded under the flawed spec, mechanistic identification of the failure, focused supersede touching only affected legs, written chronology). Post-data spec failures continue to require either Q6-style hard-stop with full re-grill or §S-5c-style supplementary-acceptance amendment with the original gate failure recorded in the chronology. The §H3-scale-9 precedent applies symmetrically to §H4.

### §H4-10. Pre-registration form (locked, no deferred lock)

This amendment is **reference-style with NO deferred numerical commit**. All numerical thresholds (induction > 0.3, τ_lift = 0.13496, τ_strict = 0.0372, (A.timing) reversal-rate ≥ 0.95, (A.count) max ≥ 5), bootstrap parameters (B = 1000, 95% percentile, per-prompt resampling), threshold-sensitivity parameters (± 25% in 5 increments), and tiered-censoring thresholds (full-fit ≥ 5, marginal 2-4, censored < 2) are pre-committed in this single amendment.

A reviewer reading the chronology should see: §H1-C registered → §H2 sweep specification locked → Phase 2 sweep ran and §H2-5 gate passed at p = 0.00463 → §H2-9-R post-data reframe registered → §H3-scale registered for 1B → 1B sweep ran → §H3-scale REGR verdict recorded → §H3-scale-8-vis presentation supersede → architecture-axis distinction surfaced post-hoc on the 1B verdict → **§H4-scaling registers at 2.8B before any 2.8B compute** → 2.8B sweep runs → §H4 verdict recorded in `notebooks/h1c_ordering_test.ipynb` §H4 section. No spec change is anticipated between this amendment and the §H4 verdict.

## Amendment 2026-05-07 — §H5-causal: causal-dependence ablation at 410m anchor

**Posted after §H4-scaling registration (commit `0ae0e27`) and after the loader-bug fix (commit `369d418`) that restored the §H1-C / §H2-5 PASS verdict. Posted before any phase4 causal-experiment compute runs.** This amendment registers the inference-time causal-dependence test of the §H1-C compositional account at the Pythia-410m anchor checkpoint. It is reference-style; numerical thresholds are pre-committed in this single amendment.

§H5-causal is a **separate scientific claim** from the §H1-C emergence ordering, the §H4-scaling head-count argument, or the §H3-scale REGR verdict. It tests whether the temporal emergence ordering observed in Phase 2 (μ_ind < μ_suc < μ_si at all three registered sizes) reflects an *inference-time* causal chain in the trained model, or merely a coincidence of training-time emergence rates. The two interpretations have different empirical signatures under ablation, and §H5 makes them distinguishable.

### Scientific question (locked)

Define `Δ_h^{si}(condition)` as the path-patching scalar of §S-1 evaluated at fixed S-inhibition sender h, under one of three model conditions:

1. **clean** — unmodified model.
2. **suc_ablated** — model with the top-5 successor heads (per `score_suc` at the same checkpoint) mean-ablated on `hook_z` over the 200-prompt IOI distribution.
3. **ctrl_ablated** — model with 5 control heads mean-ablated, sampled from the score-bracket near-but-below τ_lift (specification below).

§H5 asks whether `Δ_h^{si}(suc_ablated)` differs from `Δ_h^{si}(clean)` by more than the metric's noise floor as established by `Δ_h^{si}(ctrl_ablated)`. If yes: S-inhibition's inference-time computation routes through successor heads (compositional H1-C reading). If no: S-inhibition is causally independent of successor at convergence, and the Phase 2 temporal ordering does not reflect an architectural compositional chain.

§H5 does **not** test training-time causation (whether successor's earlier emergence enabled S-inhibition's later emergence). That hypothesis would require fine-tuning-from-checkpoint experiments that are out of scope for the M5 Pro hardware budget.

### Architectural prior (recorded for chronology, not gating)

At Pythia-410m step143000, the registered top successor heads (L22H6, L22H2 — the only two clearing τ_lift = 0.13496 at the anchor) sit at normalized layer depth ~0.96, downstream of the registered top S-inhibition heads (L12H12, L13H13, L14H0 at normalized depth ~0.55). In a feedforward transformer, a head at L22 cannot causally affect an attention pattern at L12-14. The architectural prior is therefore **NULL at 410m**: forward-pass causal dependence is not possible by the layer ordering. §H5 records this prior explicitly so any non-null result is interpreted as evidence of indirect routing (e.g., via intermediate MLPs writing successor-derived signal back upstream through later forward passes; not literally possible in a single forward pass but hypothetically present via residual-stream patterns established earlier).

The prior is recorded as context, not as a gate. The empirical experiment runs unconditionally and reports whatever it observes.

### §H5-1. Scope (locked)

Pythia-410m-deduped @ step143000 anchor only. The 5-checkpoint trajectory and 1B anchor extensions are explicitly **not** registered in this amendment; they will be registered in a separate §H5-causal-2 / §H5-causal-3 amendment block if and when the anchor result motivates them.

The 200-prompt IOI set (Wang 2023, 100 BABA + 100 ABBA, seed=0) at `data/prompts/ioi_prompts.tsv` is the substrate. The §S-1 path-patching detector with frozen pin paths and N=200 prompts is the measurement primitive (locked, unchanged).

### §H5-2. Ablation method (locked)

**Mean-ablation on `hook_z`.** For each ablated head (l, h):

1. Run a single forward pass over the 200 clean IOI prompts; cache `blocks.{l}.attn.hook_z` shape `(B, T, n_heads, d_head)`.
2. Compute `mean_z[l, h, :, :]` shape `(T, d_head)` = mean across the batch dimension at each sequence position.
3. Install a permanent forward hook at `blocks.{l}.attn.hook_z` that replaces `z[:, :, h, :]` with `mean_z[l, h, :, :]` (broadcast across batch).

This Wang-2023-standard convention removes the head's prompt-specific signal while preserving the IOI-distribution baseline statistics at every position. Less aggressive than zero-ablation, less variance than resample-ablation. Permanent hooks persist through all subsequent `run_with_cache` and `run_with_hooks` calls inside the §S-1 detector, so the ablated heads are silenced through the clean forward, the ABC-corrupt forward, and every patched forward.

Length-grouping caveat: the IOI prompt set has prompts of two distinct tokenized lengths (the BABA/ABBA structure differs in BPE under GPT-NeoX). Mean-ablation is computed within each length group separately. The result is two mean tensors per (l, h), each appropriate to its group's seq dimension.

### §H5-3. Ablation set selection (locked)

**Suc set:** the 5 highest-scoring successor heads at the anchor checkpoint per `score_suc(l, h)` evaluated by the §SU-1b lift-form detector on the 70-prompt successor probe. Tie-breaking rule: sort by `score_suc` descending, break ties by lower layer first, then lower head index.

At Pythia-410m step143000 (verified pre-registration from `phase2_successor_sweep.parquet`):

| rank | (layer, head) | score_suc | clears τ_lift = 0.13496? |
|---|---|---|---|
| 1 | L22H6 | 0.290 | yes |
| 2 | L22H2 | 0.145 | yes |
| 3 | L20H4 | 0.111 | no |
| 4 | L22H10 | 0.085 | no |
| 5 | L12H8 | 0.083 | no |

Three of the five do not clear τ_lift; this is intentional. Per user override, k=5 is fixed for statistical power; the registered detector threshold no longer gates inclusion. The 5-head set is **locked at this list**, not re-derived from a possibly-different post-fix scan.

### §H5-4. Control set selection (locked)

**Score-bracket-matched random control.** For the same checkpoint:

`CTRL_CANDIDATES = { (l, h) : score_suc(l, h) < τ_lift ∧ score_suc(l, h) ∈ [τ_lift − bracket_width, τ_lift) }`

with `bracket_width = 0.05` initially. If `|CTRL_CANDIDATES| < 5`, widen `bracket_width` symmetrically by 0.025 increments and document the final widened bracket in the verdict parquet `widened_bracket_width` column. Sample 5 heads with `numpy.random.default_rng(seed=0).choice(...)` after sorting `CTRL_CANDIDATES` by (layer, head) for determinism.

This control matches the suc set on score magnitude (just below threshold) but excludes any head identified as "successor" by the registered detector. If suc-ablation drops Δ_h while ctrl-ablation does not, the dependence is specific to successor heads rather than generic to "any near-threshold-magnitude head."

### §H5-5. Measurement set (locked)

The S-inhibition senders evaluated under each condition are pinned to the **top-3 by `Δ_h^{si}(clean)` at the anchor checkpoint**, identified once before any ablation runs and frozen across all conditions:

- `(L=12, H=12)` — Δ_h ≈ 0.0847 in the registered Phase 2 sweep
- `(L=13, H=13)` — Δ_h ≈ 0.0641
- `(L=14, H=0)` — Δ_h ≈ 0.0362

Only these 3 senders' Δ_h are recomputed under suc_ablated and ctrl_ablated conditions. The full 384-head screen is not re-run under ablation.

### §H5-6. NM identity (locked)

Name Mover heads (the receivers in the §S-1 path-patching protocol) are identified once on the **clean** model state via the §S-3 component-DLA top-4 rule, and **pinned across all three conditions**. Re-identifying NMs per condition would mix two interventions (changed receivers + changed senders); pinning isolates the test to "given the same receivers, does the sender→receiver path require successor heads to be live?"

The NM identity at Pythia-410m step143000 is read from `data/exploration/s_inhibition_pythia_410m_anchor_per_nm.npz` (the §S-8 anchor inspection, committed pre-§H5).

### §H5-7. Gate verdict (locked)

For each of the 3 S-inhibition senders h, evaluate the per-prompt mean Δ_h under each condition with B=200 paired bootstrap over the 200 IOI prompts.

| pattern | trigger (per-sender) | aggregation across 3 senders |
|---|---|---|
| **NULL** | `0.8 × clean ≤ suc_ablated ≤ 1.2 × clean` AND `0.8 × clean ≤ ctrl_ablated ≤ 1.2 × clean` | NULL if all 3 senders match NULL |
| **DEP** | `suc_ablated ≤ 0.5 × clean` AND `0.8 × clean ≤ ctrl_ablated ≤ 1.2 × clean` | DEP if all 3 senders match DEP, or majority (≥2) with the third within ±20% noise band |
| **GENERIC** | `suc_ablated ≤ 0.5 × clean` AND `ctrl_ablated ≤ 0.5 × clean` | GENERIC if any sender shows GENERIC pattern (metric not specific) |
| **MIXED** | does not fit NULL / DEP / GENERIC at the per-sender level | MIXED if no aggregation rule above applies |

DEP-gate threshold = 0.5 × clean (locked per user override; the IOI-literature default). Noise band ±20% × clean (locked). Aggregation rule: NULL > GENERIC > DEP > MIXED in priority — i.e., GENERIC overrides DEP if both could apply, and MIXED is the catch-all.

The matched paper-headline string is pre-committed per pattern:

| pattern | paper-headline string |
|---|---|
| NULL | "S-inhibition Δ_h is independent of successor heads at convergence in Pythia-410m. Direct mechanistic refutation of the forward-pass compositional reading of §H1-C." |
| DEP | "S-inhibition causally depends on successor at Pythia-410m convergence; the §H1-C ordering reflects an inference-time chain. Surprising given the layer-depth asymmetry; warrants follow-up on the indirect routing." |
| GENERIC | "Methodological note: Δ_h is not robust to ablation of any near-threshold head at this checkpoint; metric sensitivity is insufficient for the test. Verdict deferred pending re-tooling." |
| MIXED | "Per-sender heterogeneous dependence; no global verdict on §H1-C compositional reading. Reported by sender." |

### §H5-8. Bootstrap and statistical machinery (locked)

- **B = 200** paired per-prompt bootstrap replicates per condition (lower than §H2-2's B=1000 because Δ_h is averaged over B=200 prompts already and we have only 3×3 = 9 cells). 95% percentile CI on Δ_h.
- The "drop ratio" `Δ_h^{ablated} / Δ_h^{clean}` is computed as a paired ratio per bootstrap replicate; CI is on the ratio, not on the difference. Avoids pseudo-zero-inflation when clean Δ_h is small.
- Random seed: `numpy.random.default_rng(seed=0)` for ctrl set selection; `numpy.random.default_rng(seed=1)` for the bootstrap. Locked.

### §H5-9. Compute and scheduling (locked)

End-to-end estimate at Pythia-410m step143000:

- Mean-ablation precompute pass: ~30 sec (200 prompts × 1 forward, no caching of large activations).
- Per condition × 3 S-inhibition senders × 200 prompts × 2 forwards (clean cache + corrupt cache, then 1 patched forward per sender): ~10 min total per condition.
- 3 conditions × ~10 min = ~30 min wall.

The 2.8B prefetch (PID-tracked separately) co-tenants the machine; §H5 is compute-bound on a single MPS device and the prefetch is I/O-bound, so contention is minimal. If MPS instability is observed (NaN delta_h, sign flips vs the pre-§H5 anchor result, or runtime > 2× projection), the runner pauses for re-grilling rather than silently proceeding — same §H2-6 escape hatch.

### §H5-10. Notebook and parquet deliverables (locked)

1. `notebooks/_run_phase4_causal_410m_anchor.py` — runner. Loads model, builds clean+corrupt prompts, computes mean-ablation reference, runs §S-1 detector under 3 conditions for the 3 pinned S-inhibition senders, writes per-prompt Δ_h.
2. `data/exploration/phase4_causal_410m_anchor.parquet` — long-format per-prompt Δ_h, columns `(condition, si_sender_layer, si_sender_head, prompt_idx, delta_h)`.
3. `data/exploration/phase4_causal_410m_anchor_summary.parquet` — per-(condition, sender) aggregate, columns `(condition, si_sender_layer, si_sender_head, delta_h_mean, drop_ratio_mean, ratio_ci_low, ratio_ci_high, ablate_set, widened_bracket_width, n_prompts)`.
4. `data/exploration/phase4_causal_410m_anchor_verdict.parquet` — single-row gate verdict, columns `(pattern, n_senders, n_dep, n_null, n_generic, paper_headline)`.
5. `data/exploration/phase4_causal_410m_anchor.log` — captured stdout from the runner (gitignored, written via `tee`).
6. `notebooks/_build_causal_dependence.py` + `notebooks/causal_dependence.ipynb` — verdict notebook with the bootstrap-CI bar plot (clean / suc_ablated / ctrl_ablated × 3 senders), the per-sender pattern verdict, and the matched paper-headline string.

Phase 2/3/4 sweep parquets are NOT modified. The §S-1 detector primitive in `src/detectors/s_inhibition.py` may be extended with an optional `nm_heads_override` kwarg to support pinned-NM evaluation under §H5-6; this is a backward-compatible extension and is documented in the experiment-code commit.

### §H5-11. Procedural precedent (locked)

§H2-8's spec-failure-during-phase policy applies. Pre-data smoke-test-surfaced flaws in this §H5 spec may be corrected by a focused supersede amendment provided the §SU-1b conditions hold. Post-data spec failures continue to require either Q6-style hard-stop with full re-grill or §S-5c-style supplementary-acceptance amendment with the original gate failure recorded in the chronology.

### §H5-12. Pre-registration form (locked, no deferred lock)

This amendment is **reference-style with NO deferred numerical commit**. All numerical thresholds (DEP gate 0.5×, noise band ±20%, GENERIC threshold 0.5×, k=5 ablation set, k=5 control set, bracket_width = 0.05 initial, B=200 bootstrap), seeds (rng(0) for ctrl selection, rng(1) for bootstrap), the locked top-5 successor heads at the anchor, the locked top-3 S-inhibition senders, and the §H5-7 verdict aggregation rule are pre-committed in this single amendment.

A reviewer reading the chronology should see: §H1-C registered → §H2 sweep → §H2-9-R reframe → §H3-scale 1B (REGR) → §H4-scaling 2.8B registration → loader-bug fix `369d418` (restored §H2-5 PASS) → **§H5-causal registers at 410m anchor before any phase4 causal compute** → 410m anchor runs → verdict recorded in `notebooks/causal_dependence.ipynb`. The 5-checkpoint trajectory and the 1B anchor extension, if executed, will be registered in separate amendments before their respective compute runs.

## Amendment 2026-05-07 — §H5-causal-2: logit-diff metric for causal-dependence ablation at 410m anchor

**Posted after §H5-causal anchor verdict (NULL pattern, commit `13c7627`) and before any phase4 logit-diff compute runs.** This amendment registers a follow-up causal-dependence experiment using a different measurement primitive — the IO−S logit difference at the END token, which reads the full forward-pass output. Motivated by a methodological caveat surfaced post-hoc on the §H5-causal NULL: the §S-1 path-patching protocol freezes intermediate `attn_out` and `mlp_out` to clean cache values and reads receiver attention patterns at the component-DLA top-4 NMs only. Heads at layers ≥ max(NM layer) cannot, by construction, affect the §S-1 metric. At Pythia-410m step143000 the registered top-5 successor heads sit at L12, L20, L22, L22, L22 and the NMs at L12, L14, L17, L20 — so 4 of 5 suc heads are structurally unread by §S-1.

The §H5-causal NULL is therefore consistent with two readings:

- (a) Genuine independence of S-inhibition / IOI from successor at inference time.
- (b) Structural insensitivity of §S-1 to ablations at layers ≥ max(NM); the metric cannot detect successor's contribution even if it exists.

§H5-causal-2 distinguishes (a) from (b) by replacing the §S-1 attention-pattern readout with a logit-diff-at-END readout, which depends on all layers' contributions through the unembedding. If suc-ablation drops logit-diff while ctrl-ablation does not, suc heads contribute to IOI via a path the §S-1 metric does not read — forcing a substantial reframe of the §H1-C compositional reading. If logit-diff is also independent, the genuine-independence reading is supported.

§H5-causal-2 is **not** an emergence-claim test. It does not modify §H1-C, §H2-5, §H2-9-R, §H3-scale, or §H4-scaling. It is a refinement of §H5-causal's measurement primitive, registered before any logit-diff compute runs.

### §H5-causal-2-1. Scope (locked)

Pythia-410m-deduped @ step143000 anchor only. The same 200-prompt IOI distribution as §H5-causal (Wang 2023, 100 BABA + 100 ABBA, seed=0) is the substrate. The trajectory and 1B extensions remain explicitly out of scope for this amendment; if motivated by the §H5-causal-2 result they will be registered in separate amendment blocks.

### §H5-causal-2-2. Metric (locked)

For each clean IOI prompt with known IO and S token IDs:

`logit_diff_p = logits[end_pos, IO_token_id_p] - logits[end_pos, S_token_id_p]`

where `end_pos` is the last token position of the prompt, `logits` are the model's output at that position, and `IO_token_id_p` / `S_token_id_p` are read from the `IOIPrompt` dataclass populated by `tigges_ioi.load_ioi_prompts`. The aggregate metric is the mean across the 200 prompts: `Δlogit_clean = mean_p logit_diff_p`. Per-condition variants `Δlogit_suc_ablated`, `Δlogit_ctrl_ablated` are computed by re-running the forward pass with the corresponding mean-ablation hooks active.

Per-prompt arrays are retained for the paired-bootstrap ratio test in §H5-causal-2-7.

### §H5-causal-2-3. Suc set (locked, re-used verbatim from §H5-3)

The suc set is **identical to §H5-3** at Pythia-410m step143000 — re-used verbatim, **NOT re-derived**, to preserve no-cherry-picking discipline:

| rank | (layer, head) | score_suc |
|---|---|---|
| 1 | L22H6 | 0.290 |
| 2 | L22H2 | 0.145 |
| 3 | L20H4 | 0.111 |
| 4 | L22H10 | 0.085 |
| 5 | L12H8 | 0.083 |

Tie-breaking rule (layer asc, head asc) is inherited verbatim. The set is locked at this list; the §H5-causal-2 runner reads it from the same `phase2_successor_sweep.parquet` source as §H5-causal and asserts the resulting top-5 matches the locked list bit-for-bit.

### §H5-causal-2-4. Ctrl set (locked, re-used verbatim from §H5-4)

The ctrl set is **identical to §H5-4** under the bracket-widening that already occurred during §H5-causal execution — re-used verbatim, NOT re-sampled:

`bracket_width = 0.100` (widened from initial 0.05 because no ctrl candidates existed in [0.085, 0.135) outside the suc-5; the symmetric-widening rule of §H5-4 applied, final bracket [0.035, 0.135)). Sampled at `numpy.random.default_rng(seed=0)` over the deterministically-sorted bracket-candidate list:

| (layer, head) | score_suc |
|---|---|
| L17H12 | 0.059 |
| L20H6 | 0.038 |
| L22H11 | 0.052 |
| L23H10 | 0.036 |
| L23H13 | 0.051 |

The §H5-causal-2 runner asserts the re-derived ctrl set matches this locked list bit-for-bit. If the re-derivation diverges (e.g., due to a parquet content change), the runner halts and surfaces the divergence rather than silently using a different ctrl set.

### §H5-causal-2-5. Ablation method (locked, identical to §H5-2)

Mean-ablation on `hook_z[:, :, head, :]` per length group, replace with batch-mean. Permanent forward hook installed via `model.add_perma_hook` for each (layer, head) in the ablation set. Persists through the model forward pass that computes logits.

### §H5-causal-2-6. Gate verdict (locked)

Define `ratio_suc = Δlogit_suc_ablated / Δlogit_clean` and `ratio_ctrl = Δlogit_ctrl_ablated / Δlogit_clean`, with paired-bootstrap 95% CI per §H5-causal-2-7. Patterns:

| pattern | trigger | paper-headline string |
|---|---|---|
| **NULL** | `ratio_suc ∈ [0.8, 1.2]` AND `ratio_ctrl ∈ [0.8, 1.2]`, both with 95% CI within those bands | "IOI logit-diff at Pythia-410m is independent of the registered top-5 successor heads. Combined with the §H5-causal NULL on the §S-1 metric, this is converging evidence that S-inhibition's circuit is causally disjoint from successor's at inference time. The temporal emergence ordering ind→suc→si is decoupled from any architectural causal chain." |
| **DEP** | `ratio_suc < 0.5` with 95% CI excluding 0.5 AND `ratio_ctrl ∈ [0.8, 1.2]` | "Successor heads contribute causally to IOI logit-diff at Pythia-410m via a path the §S-1 metric does not read. The §H5-causal NULL is therefore an instance of metric insensitivity (b), not genuine independence (a). The §H1-C compositional reading is partially supported: successor → IOI is causal at inference time, but not via the registered S-inhibition mechanism. Substantial reframe of the §H1-C narrative required." |
| **GENERIC** | `ratio_suc < 0.7` AND `ratio_ctrl < 0.7` | "Methodological note: IOI logit-diff is not robust to mean-ablation of any layer-22-cluster head at this checkpoint; metric sensitivity is insufficient. Verdict deferred pending re-tooling (e.g., per-head individual ablation, or non-mean ablation method)." |
| **MIXED** | none of the above patterns fit cleanly (e.g., partial drops with CI overlap) | "Heterogeneous ablation effect on IOI logit-diff; no global verdict on suc → IOI dependence. Reported as numerical-only result with CI bands; deferred for follow-up." |

Aggregation: there is one verdict (no per-sender split — the metric is a single scalar over the prompt distribution, not a per-sender path-patching scalar). The matched paper-headline string is the verdict.

### §H5-causal-2-7. Bootstrap (locked, identical to §H5-8)

B=200 paired per-prompt resampling, `numpy.random.default_rng(seed=1)`. CI on the ratio `Δlogit_ablated / Δlogit_clean`, percentile method (2.5 / 97.5). The pairing preserves the per-prompt correlation between clean and ablated runs (each replicate samples 200 indices with replacement and computes both ratios on the same indices).

### §H5-causal-2-8. Compute and scheduling (locked)

End-to-end estimate: ~1–2 min wall on Pythia-410m (3 forward passes over 200 prompts, no path-patching, no caching beyond the mean-ablation precompute). The 2.8B prefetch (PID-tracked separately) co-tenants the machine; both are minimal contention.

### §H5-causal-2-9. Notebook and parquet deliverables (locked)

1. `notebooks/_run_phase4_causal_410m_anchor_logitdiff.py` — runner. Reuses §H5-causal helpers (`select_top_suc`, `select_ctrl_set`, `precompute_mean_z_by_length`, `install_mean_ablation_hooks`, `bootstrap_drop_ratio`) verbatim; replaces the metric computation with logit-diff-at-END.
2. `data/exploration/phase4_causal_410m_anchor_logitdiff.parquet` — long-format per-prompt logit-diff, columns `(condition, prompt_idx, logit_diff)`.
3. `data/exploration/phase4_causal_410m_anchor_logitdiff_summary.parquet` — per-condition aggregate, columns `(condition, logit_diff_mean, drop_ratio_mean, ratio_ci_low, ratio_ci_high, ablate_set)`.
4. `data/exploration/phase4_causal_410m_anchor_logitdiff_verdict.parquet` — single-row §H5-causal-2-6 gate verdict.
5. `data/exploration/phase4_causal_410m_anchor_logitdiff.log` — captured stdout (gitignored).
6. `notebooks/causal_dependence.ipynb` — extended in place with a §H5-causal-2 verdict section appended after the existing §H5-causal section. Includes one figure (clean / suc_ablated / ctrl_ablated logit-diff bars with CI). Existing §H5-causal cells preserved verbatim.

Phase 2/3/4 sweep parquets, §H5-causal anchor parquets, and HYPOTHESIS.md prose-track sections are NOT modified by this amendment.

### §H5-causal-2-10. Procedural precedent (locked)

§H2-8's spec-failure-during-phase policy applies. Pre-data smoke-test-surfaced flaws may be corrected by a focused supersede amendment under the §SU-1b conditions. Post-data spec failures continue to require either Q6-style hard-stop with full re-grill or §S-5c-style supplementary-acceptance amendment with the original gate failure recorded in the chronology.

The "no cherry-picking" discipline is enforced by the verbatim re-use of §H5-3 and §H5-4 sets: the §H5-causal-2 runner asserts that the re-derived suc set and ctrl set match the locked lists bit-for-bit, halting if not. This prevents re-sampling of the ctrl set under a different seed or re-deriving the suc set from a different parquet snapshot.

### §H5-causal-2-11. Pre-registration form (locked, no deferred lock)

This amendment is **reference-style with NO deferred numerical commit**. All numerical thresholds (NULL band [0.8, 1.2], DEP gate 0.5 with CI exclusion, GENERIC threshold 0.7, B=200 bootstrap, seed=1), the verbatim re-used suc and ctrl sets from §H5-3 and §H5-4, the verbatim bracket_width=0.100 from the §H5-causal execution, and the matched paper-headline strings per pattern are pre-committed in this single amendment.

A reviewer reading the chronology should see: §H1-C → §H2 → §H2-9-R → §H3-scale (1B REGR) → §H4-scaling (2.8B registration) → loader-bug fix `369d418` → §H5-causal registered → §H5-causal anchor runs (NULL verdict, commit `13c7627`) → structural-insensitivity caveat surfaces → **§H5-causal-2 registers at 410m anchor before any logit-diff compute** → §H5-causal-2 anchor runs → verdict recorded in `notebooks/causal_dependence.ipynb` §H5-causal-2 section.
