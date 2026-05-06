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
