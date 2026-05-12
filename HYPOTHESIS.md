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

## Amendment 2026-05-08 — §H4-7-supersede: 2.8B S-inhibition sweep halted, §H4 verdict DEFERRED

**Posted after 2.8B chain partial-completion.** §H4-7 registered a per-cell-cost escape hatch: *"if measured per-cell cost on the first sweep cell exceeds the projection by more than 2×, the runner pauses for re-grilling rather than silently extending."* The 2.8B S-inhibition sweep reached steady-state per-cell wall time of ~57 minutes (3389–3401 s observed across cells 1–8 of the sweep), against the §H4-7 projection of ~6 min/cell. **Actual: ~10× projection, exceeding the 2× pause threshold by 5×.** Per the registered discipline, the chain was halted at user instruction.

### What was completed

- **§H4 anchor inspection** (§H4-4): all 3 sub-anchors completed at Pythia-2.8B-deduped @ step143000.
- **Induction sweep** (`phase4_2_8b_induction_sweep.parquet`): full 40-cell sweep complete.
- **Successor sweep** (`phase4_2_8b_successor_sweep.parquet`): full 40-cell sweep complete.
- **S-inhibition sweep**: **8 of 40 cells complete**, all early-training (`step0, step1, step2, step4, step8, step16, step32, step64`). All 8 show top Δ_h ≈ 0.0000 — consistent with no S-inhibition emergence in this range. Per-prompt .npz cache files preserved at `data/exploration/phase4_2_8b_s_inhibition_per_prompt/`. Sweep parquet was NOT written (the runner writes parquet at end-of-loop only); the 8 cells are recoverable from the .npz cache for any future supplementary analysis.

### What is deferred

The §H4-2 / §H4-3 conjunctive gate operates on S-inhibition only:
- **(A.timing)** `P(μ_si^2.8B < μ_si^410m) ≥ 0.95` — undeterminable: μ_si^2.8B requires logistic fit across the full count-vs-step curve, which requires the full 40-cell sweep.
- **(A.count)** `max_count_si^2.8B ≥ 5` — undeterminable: max is computed across all 40 cells, and the emergence-relevant cells (steps 5000+) were not sampled.

Per §H4-9 / §H2-8 spec-failure-during-phase policy, this is a **post-data spec failure under the registered escape hatch**. The §H4 verdict is therefore **DEFERRED** rather than PASS / TIMING-ONLY / COUNT-ONLY / NEITHER / TOOLING — none of the §H4-5 patterns describe a halt-on-time-budget. DEFERRED is a sixth pattern, registered here:

| Pattern | Trigger | Pre-committed paper headline |
|---|---|---|
| **DEFERRED** | S-inhibition sweep halted before completion under §H4-7 escape hatch; (A.timing) and (A.count) undeterminable | "§H4-scaling verdict deferred at 2.8B: S-inhibition sweep halted at 8/40 cells (steps 0–64, all pre-emergence) under the §H4-7 per-cell-cost escape hatch (~57 min/cell observed vs ~6 min/cell projected). Induction and successor sweeps complete and analyzed; their cross-size emergence trajectories at 2.8B are reported as side observations. The §H4 conjunctive gate (A.timing AND A.count) is not evaluated and the §H4-5 PASS / TIMING-ONLY / COUNT-ONLY / NEITHER / TOOLING verdict is not assigned. Future re-attempts must register a fresh §H4-* amendment with revised per-cell-cost projections and budget — either by reduced grid (§H4-supersede), by accepting longer wall time, or by an alternate detector primitive that doesn't require 4096 patched forward passes per cell at d_model=2560." |

DEFERRED is added to the §H4-5 priority ordering as the highest-priority *non-substantive* pattern alongside TOOLING:

`DEFERRED > TOOLING > NEITHER > COUNT-ONLY > TIMING-ONLY > PASS`.

### What is preserved as supplementary

- **Induction at 2.8B**: full sweep, can be compared cross-size to {70m, 160m, 410m, 1b}. Reported in the analysis but does not contribute to a §H4 gate (§H4-1: "induction and successor saturate 'yes they emerge robustly' by 410m and at 1B; their cross-size scaling is not the substantive claim being tested"). Side observation only.
- **Successor at 2.8B**: full sweep, same supplementary-only status.
- **S-inhibition at 2.8B (early-training only, 8 cells)**: per-prompt .npz cache preserved. Top-head Δ_h ≈ 0 across all 8 cells — confirms the smaller-model finding that S-inhibition does not emerge in early training.

### What this amendment does NOT do

- It does NOT re-derive or weaken the §H4-2 gate predicates. The (A.timing) and (A.count) numerical thresholds remain locked. Any future re-attempt that produces full 40-cell S-inhibition data at 2.8B may be evaluated against the original gate without further amendment.
- It does NOT amend the §H1-C / §H2-5 verdict (Track 1 emergence claim, registered, complete).
- It does NOT amend the §H3-scale 1B REGR verdict (Track 2 1B head-count regression, sealed historical).
- It does NOT amend the §H5-causal / §H5-causal-2 NULL verdicts (Track 3 inference-time causal-dependence, complete).
- It does NOT amend the project-narrative two-/three-track structure. Track 2 (head-count-axis scaling) is now in a **DEFERRED** state with respect to its 2.8B leg; it is not abandoned.

### Procedural precedent

This amendment is the **first §H4-7 escape-hatch invocation in the project**. The escape hatch was pre-registered in §H4-7 specifically to handle this scenario (under-projected per-cell costs). The chronology — §H4-7 escape hatch registered before any 2.8B compute → 2.8B S-inhibition sweep observed at 10× projection → halt invoked → §H4-7-supersede registered post-halt with partial-data scope — preserves pre-registration discipline. The DEFERRED pattern is registered into §H4-5 in this amendment so future re-attempts (or different 2.8B sweep approaches) can match cleanly.

### Next steps (registered for future work)

A future §H4-2 re-attempt at Pythia-2.8B requires either:
1. **A reduced-grid §H4-supersede amendment** registered before re-running the sweep, with a smaller cell count (e.g., 10 well-chosen cells in the emergence-likely range step5000–step50000) and a refined per-cell-cost projection.
2. **Accepting a multi-day wall time** at the full 40-cell grid (~38 hours) — this requires no new amendment but requires the user to commit to the wall-time budget.
3. **An alternate detector primitive** that doesn't require 4096 patched forward passes per cell at d_model=2560 — this would constitute a substantial methodological change requiring a fresh detector-validation chain (§S-1 through §S-tau analog).

None of these are scheduled at the time of this amendment; the 2.8B leg is parked.

---

## Amendment 2026-05-10 — §H5-causal-3-record: post-data documentation of §H5-causal extension to Pythia-1B

**Posted as a chronology-canonicalization amendment, not a pre-registration.** Pythia-1B step143000 anchor compute under the §H5-causal / §H5-causal-2 protocols was executed in a feature worktree on 2026-05-07 (parquet timestamps confirm). At the time of that run, the canonical `HYPOTHESIS.md` on `main` ended at §H5-causal-2; no amendment registered the 1B extension before the compute. This amendment **records the verdicts as-found** so the canonical pre-reg chain matches the on-disk data, and explicitly marks the 1B result as **post-data documentation** rather than a registered finding. The pre-reg-discipline gap is noted and not papered over.

### §H5-causal-3-record-1. What was run (post-data)

- **Pythia-1B-deduped @ step143000** (head-count regression cell per §H4-1; chosen as the next-available size at which both successor and S-inhibition sweeps were complete).
- **Metric A (§S-1 path-patching Δ_h)**: protocol identical to §H5-causal at 410m (mean-ablation on `hook_z` per length group, suc set = top-5 successor heads with §H5-3 tie-break, ctrl set = bracket-widening with seed=0 + NM-exclusion clause, NMs pinned to component-DLA top-4 from clean anchor, B=200 paired bootstrap with seed=1).
- **Metric B (logit-diff at END)**: protocol identical to §H5-causal-2 at 410m.
- Outputs: `phase4_causal_1b_anchor{,_summary,_verdict,_logitdiff,_logitdiff_summary,_logitdiff_verdict,_h5causal3_verdict}.parquet`.

### §H5-causal-3-record-2. Recorded verdicts

| Cell | Pattern | Key numbers |
|---|---|---|
| 1B Metric A (path-patching) | **NULL** (3/3 senders) | suc={L11H6, L14H2, L12H3, L15H7, L15H1}; ctrl={L11H1, L11H3, L11H4, L12H2, L13H0}; SI senders={L8H7, L9H1, L10H4}; NMs={L11H0, L11H5, L14H2, L11H2}; widened_bracket_width=0.125 |
| 1B Metric B (logit-diff) | **MIXED** | ratio_suc = 0.790, ratio_ctrl = 0.797 — **both ablations drop ~21% similarly**. Read: no successor-specific dependence, but the logit-diff readout becomes generically ablation-sensitive at 1B |
| 1B cross-metric | **MIXED** | "Heterogeneous Metric A and B verdicts at 1B; no global conclusion on suc → si causal dependence. Reported per-metric with per-sender CIs; deferred for follow-up." |

### §H5-causal-3-record-3. Per-size structural caveat (post-data observation)

At 1B step143000, suc-set head L14H2 is **also** in the pinned NM set L14H2 (dual-role). The §H5-causal protocol's downstream-NM filter (`nl > sl`) excludes a sender's own layer from its Metric A readout, so L14H2 as suc contributes to ablation but its corresponding NM is filtered out of the path-patching scalar — Metric A is structurally insensitive to L14H2's contribution at this checkpoint. Metric B reads at END and is fully sensitive. This caveat is registered here as a **per-size empirical observation**, not a methodological change to §H5-causal / §H5-causal-2.

### §H5-causal-3-record-4. Pre-reg-discipline gap (acknowledged)

The 1B compute was run before any 1B-specific amendment. The cleanest possible chronology would have been: §H5-causal at 410m → §H5-causal-2 at 410m → §H5-causal-3-1b amendment (pre-data) → §H5-causal-3 1B compute. What actually happened: §H5-causal at 410m → §H5-causal-2 at 410m → §H5-causal-3 1B compute on a worktree → §H5-causal-3-record canonicalization (this amendment). The downstream paper section that cites the 1B verdicts MUST disclose this — the 1B verdict is **converging confirmation that survives because the protocols (suc set derivation, ctrl set bracket-widening + seed, NM identification, ablation method, bootstrap) were all locked at 410m before 1B compute, leaving no degrees of freedom**, but it is not a pre-registered confirmation in the same sense as 410m. The same is true for the 1B locked-set assertions in the logitdiff runner: they were locked in the worktree, not in canonical HYPOTHESIS.md, until this amendment.

### §H5-causal-3-record-5. Effect on the §H5-causal narrative

The single-paper-headline claim at 410m was *"S-inhibition's circuit is causally disjoint from successor's at inference time; the temporal emergence ordering ind→suc→si is decoupled from any architectural causal chain."* The 1B result modifies this to: *"At 410m the disjointness holds across both metrics; at 1B Metric A (the §S-1 path-patching readout) replicates the NULL but Metric B (logit-diff) shows generic ablation sensitivity that does not distinguish suc from ctrl. The 410m converging-evidence claim is therefore size-specific; at 1B we have no successor-specific dependence (Metric A holds, ctrl drops the same as suc on Metric B) but cannot make a clean cross-metric NULL claim. Paper must report per-(size, metric) verdicts and explain the 1B Metric B sensitivity."*

This amendment does NOT modify §H5-causal / §H5-causal-2 (the 410m result), §H1-C / §H2-5 (Track 1), the §H3-scale 1B REGR verdict, or the §H4-7-supersede DEFERRED record.

---

## Amendment 2026-05-10 — §H5-causal-3-2.8b: pre-registered extension of §H5-causal / §H5-causal-2 to Pythia-2.8B

**Posted before any Pythia-2.8B-deduped step143000 ablation compute under §H5-causal / §H5-causal-2 protocols.** This amendment registers the extension of both metrics (Metric A = §S-1 path-patching, Metric B = logit-diff) to Pythia-2.8B-deduped step143000. All gate thresholds, ablation methods, bootstrap parameters, and the per-sender / per-metric verdict taxonomies are **inherited verbatim** from §H5-causal and §H5-causal-2. Only per-size derivation rules (suc, ctrl, SI senders, NMs) are stated here, with the per-size sets locked by deterministic re-derivation from sealed input parquets — no cherry-picking degree of freedom remains.

### §H5-causal-3-2.8b-1. Scope (locked)

- **Pythia-2.8B-deduped @ step143000** (head-count tier 1024 per §H4-1 table; this is the largest size in the project).
- **Both metrics run**: Metric A (§S-1 path-patching Δ_h, per §H5-causal protocol) AND Metric B (logit-diff at END, per §H5-causal-2 protocol).
- Inputs (all sealed prior to this amendment):
    - `data/exploration/phase4_2_8b_successor_sweep.parquet` (40-cell sweep, complete per §H4-7-supersede record).
    - `data/exploration/s_inhibition_pythia_2_8b_anchor.parquet` (full 1024-head Δ_h matrix at step143000, schema `(model, step, layer, head, metric, value)` with `metric="delta_h"` rows).
    - `data/exploration/s_inhibition_pythia_2_8b_anchor_per_nm.npz` (component-DLA top-4 NMs at clean step143000).

### §H5-causal-3-2.8b-2. Suc set (locked by §H5-3 procedure re-applied at 2.8B)

Top-5 by `score_suc` at (size="2.8b", step=143000), ties broken by layer asc then head asc (verbatim §H5-3 tie-break). Deterministic re-derivation from `phase4_2_8b_successor_sweep.parquet` yields:

```
suc = [(15, 14), (28, 17), (27, 13), (13, 10), (29, 28)]
scores = [2.126, 0.726, 0.303, 0.302, 0.281]
```

All 5 ≥ τ_lift = 0.13496. Runner asserts this list bit-for-bit; mismatch halts per §H2-8 spec-failure policy.

### §H5-causal-3-2.8b-3. Ctrl set (procedure-locked, derived at runtime)

§H5-4 procedure verbatim: random sample (`rng = np.random.default_rng(0)`) from bracket `[τ_lift - bw, τ_lift)` with `bw = 0.05` initial and `+= 0.025` widening step, NM-exclusion clause from §H5-causal-3-7 (worktree precedent) applied to forbid candidates that match any pinned NM. The runner derives the ctrl set at execution time and writes both the final `widened_bracket_width` and the chosen 5 heads to the verdict parquet for audit.

The procedure is fully deterministic given the seed and the sealed input parquet; this amendment locks the *procedure*, not a pre-computed list, mirroring the §H5-causal (Metric A) 410m precedent.

### §H5-causal-3-2.8b-4. SI senders (locked by §H5-5 procedure re-applied at 2.8B)

Top-3 by Δ_h at clean step143000 from `s_inhibition_pythia_2_8b_anchor.parquet` (`metric="delta_h"` rows), ties broken by layer asc then head asc:

```
si_senders = [(11, 29), (11, 5), (13, 9)]
scores = [0.148, 0.121, 0.105]
```

Runner asserts this list bit-for-bit.

### §H5-causal-3-2.8b-5. NM identity (locked by §H5-6 — read from sealed npz)

Component-DLA top-4 from the clean §S-8 2.8B anchor, pinned across all conditions:

```
nm_heads = [(11, 29), (17, 12), (22, 31), (13, 9)]
```

Read verbatim from `s_inhibition_pythia_2_8b_anchor_per_nm.npz`. This matches the §H5-causal 410m precedent (NMs read from sealed clean-state npz, not re-derived at ablation time).

### §H5-causal-3-2.8b-6. Per-size structural caveat (pre-data observation, registered)

At 2.8B step143000:
- **NM-SI overlap**: L11H29 and L13H9 appear in BOTH the SI senders set AND the NM set. The §H5-causal downstream-NM filter (`nl > sl`) handles this:
    - (11,29) as SI sender → downstream NMs = {(17,12), (22,31)} (2 of 4 NMs)
    - (11,5) as SI sender → downstream NMs = {(17,12), (22,31), (13,9)} (3 of 4 NMs)
    - (13,9) as SI sender → downstream NMs = {(17,12), (22,31)} (2 of 4 NMs)
  All 3 SI senders retain at least 2 downstream NMs for the Metric A readout. Acceptable, but the per-sender readouts use different NM subsets — this is verbatim §H5-causal behaviour, not a deviation.
- **Suc-NM layer overlap (structural insensitivity for Metric A)**: 3 of 5 suc heads — (27,13), (28,17), (29,28) — sit at layers > max(NM layer)=22, so their ablation does NOT propagate to any pinned NM through the Metric A path-patching readout. Only (15,14) and (13,10) have any downstream NM impact via Metric A. **Metric A is structurally insensitive to (27,13), (28,17), (29,28)**. Metric B reads at END and is fully sensitive to all 5 suc ablations. This is the same caveat A2 flagged for 1B (L14H2 dual-role); it is registered here as a per-size empirical observation, not a methodological change.

### §H5-causal-3-2.8b-7. Gate verdict (inherited verbatim from §H5-7 and §H5-causal-2-6)

- **Metric A**: per-sender classifier {NULL, DEP, GENERIC, MIXED} from §H5-7 (DEP_THRESHOLD=0.5, NULL_BAND=0.20); aggregate via §H5-7 priority `GENERIC > NULL > DEP > MIXED`.
- **Metric B**: §H5-causal-2-6 classifier {NULL, DEP, GENERIC, MIXED} (NULL band [0.8, 1.2], DEP < 0.5, GENERIC threshold 0.7).
- **Cross-metric** (mirrors §H5-causal-3-record for 1B): if both metrics NULL → cross-metric NULL (converging-evidence claim). If both DEP → cross-metric DEP. Any mixed → cross-metric MIXED with per-metric breakdown reported.

Paper-headline strings for cross-metric NULL / DEP / MIXED at 2.8B are inherited from §H5-causal-2 / §H5-causal-3-record with size string substituted.

### §H5-causal-3-2.8b-8. Bootstrap and statistical machinery (inherited verbatim from §H5-8)

B = 200 paired per-prompt bootstrap, seed = 1, 95% percentile CI on drop ratio. Identical to 410m and 1B precedents.

### §H5-causal-3-2.8b-9. Compute and scheduling (locked)

- Metric A anchor: BATCH_SIZE = 10 (per §H4-7 precedent for path-patching at d_model = 2560; halved from 1B's 25). Estimated ~5-8 h wall time.
- Metric B anchor: forward-pass-only, ~30-60 min wall time.
- fp16 fallback registered per §H4-7 if fp32 OOMs.
- HF_HUB_OFFLINE = 1 set at module top of all runners per §H4-7 / §H4-7-supersede precedent.
- Per-cell-cost escape hatch (§H4-7 style) does NOT apply — Metric A here is a single-cell anchor, not a sweep; if it doesn't fit overnight on a single attempt, re-grilling happens at the next session.

### §H5-causal-3-2.8b-10. Notebook and parquet deliverables (locked)

- `notebooks/_run_phase4_causal_2_8b_anchor.py` — Metric A (path-patching) runner; mirrors `_run_phase4_causal_410m_anchor.py`. Reuses §H5-causal helpers (`bootstrap_drop_ratio`, `install_mean_ablation_hooks`, `precompute_mean_z_by_length`, `run_condition`, `classify_per_sender`, `aggregate_verdict`) via import from the 410m source; redefines `select_top_suc` / `select_ctrl_set` / `select_top_si` locally with SIZE="2.8b" / STEP=143000 to avoid the SIZE-closure bug.
- `notebooks/_run_phase4_causal_2_8b_anchor_logitdiff.py` — Metric B (logit-diff) runner; mirrors `_run_phase4_causal_410m_anchor_logitdiff.py`. Same import + redefinition pattern; bit-for-bit assertion against the locked suc + SI + NM lists (no locked-set assertion on ctrl set, which is procedure-locked).
- `data/exploration/phase4_causal_2_8b_anchor.parquet` + `_summary` + `_verdict` (Metric A).
- `data/exploration/phase4_causal_2_8b_anchor_logitdiff.parquet` + `_summary` + `_verdict` (Metric B).
- `data/exploration/phase4_causal_2_8b_anchor_h5causal3_verdict.parquet` — cross-metric verdict aggregator (analog of the 1B `h5causal3_verdict.parquet`).
- New section in `causal_dependence.ipynb` reporting per-(size, metric) verdicts side-by-side (410m, 1B, 2.8B).

### §H5-causal-3-2.8b-11. Pre-registration form (locked, no deferred lock)

This amendment is **reference-style with NO deferred numerical commit**. All numerical thresholds — DEP_THRESHOLD=0.5, NULL_BAND=0.20, NULL_LO/HI=[0.8,1.2], GENERIC_THRESHOLD=0.7, B=200, seed=0 (ctrl rng) / seed=1 (bootstrap rng), τ_lift=0.13496, τ_strict=0.0372, k_suc=5, k_si=3, k_nm=4, bracket_width_init=0.05, bracket_width_step=0.025 — are inherited verbatim from §H5-causal / §H5-causal-2 / §H5-3 / §H5-4 / §H5-5 / §H5-6 / §H5-7 / §H5-8 / §H5-causal-2-6 / §H5-causal-2-7, all locked before any 2.8B ablation compute. The per-size derived sets (suc, SI senders, NMs) above are computed deterministically from sealed pre-existing parquets / npz; the runner asserts bit-for-bit match and halts on mismatch per §H2-8. The §H5-causal-3-2.8b amendment is committed **before any §H5-causal-3-2.8b ablation compute starts**.

A reviewer reading the chronology should see: §H5-causal at 410m → §H5-causal-2 at 410m → §H5-causal-3 1B compute on worktree → §H5-causal-3-record canonicalization (post-data) → **§H5-causal-3-2.8b registers at 2.8B before any 2.8B ablation compute, with locked sets and inherited gate** → 2.8B Metric A + Metric B runs (one overnight session) → §H5-causal-3-2.8b verdict recorded in `causal_dependence.ipynb`. No spec change is anticipated between this amendment and the §H5-causal-3-2.8b verdict.

This amendment does NOT modify §H5-causal / §H5-causal-2 (the 410m result), §H5-causal-3-record (the 1B result), §H1-C / §H2-5, §H3-scale, or §H4-7-supersede.

---

## Amendment 2026-05-10 — §H4-supersede: reduced-grid re-attempt of §H4-scaling at Pythia-2.8B

**Posted after §H4-7-supersede (DEFERRED verdict, 8/40 S-inhibition cells) and before any further Pythia-2.8B S-inhibition compute.** This amendment registers a reduced-grid re-attempt of the §H4-scaling gate at Pythia-2.8B-deduped, scoped to S-inhibition only. It is reference-style; all numerical thresholds are inherited verbatim from §H4-2 and §H4-5, and no further deferred lock is introduced. §H4-supersede follows option 1 of the §H4-7-supersede "Next steps" register: *"a reduced-grid §H4-supersede amendment registered before re-running the sweep, with a smaller cell count (e.g., 10 well-chosen cells in the emergence-likely range step5000–step50000) and a refined per-cell-cost projection."*

**Paper status note (registered here):** §H4-supersede is positioned as a **scaling appendix / secondary result**, not a paper headline. The paper's main claims rest on Track 1 (§H1-C ordering + §H2-9-R reframe) and Track 3 (§H5-causal / §H5-causal-2 410m + §H5-causal-3-record 1B + §H5-causal-3-2.8b 2.8B). If §H4-supersede lands PASS / TIMING-ONLY / COUNT-ONLY / NEITHER, it is reported as an appendix scaling result. **If §H4-supersede DEFERS a second time** under its own per-cell-cost escape hatch (§H4-supersede-4), the paper ships unchanged on Tracks 1 + 3; a fresh §H4-** amendment for any third attempt is permitted but not required for submission. The §H4 leg does NOT block the paper.

§H4-supersede is **S-inhibition only**. The 2.8B induction and successor full-sweep parquets (both 40 cells) are sealed supplementary artifacts per §H4-7-supersede and are not re-run. This amendment does NOT modify §H1-C / §H2-5 (Track 1, complete), the §H3-scale 1B REGR verdict (sealed historical), the §H5-causal / §H5-causal-2 NULL verdicts (Track 3 at 410m, complete), the §H5-causal-3-record 1B verdicts (Track 3 at 1B, canonicalized), the §H5-causal-3-2.8b 2.8B pre-reg (Track 3 at 2.8B, pending), or the §H4-7-supersede DEFERRED record (preserved verbatim). It exits the DEFERRED state for Track 2's 2.8B leg by registering a fresh, in-budget compute plan against the same locked gate predicates.

### §H4-supersede-1. Scope (locked)

The §H4-supersede re-attempt runs on **Pythia-2.8B-deduped** (`EleutherAI/pythia-2.8b-deduped`) at the following **10-cell checkpoint grid**, locked in this amendment before any §H4-supersede compute:

```
[5000, 7000, 10000, 14000, 20000, 29000, 41000, 49000, 59000, 70000]
```

Every step is drawn from the §H2-1 40-cell grid, so cross-size comparisons against {70m, 160m, 410m, 1b} are exact-step-matched at those 10 indices. The 10-cell choice is motivated by the emergence-likely range observed in the registered Phase 2 / §H3-scale 1B sweeps: the S-inhibition logistic midpoint μ_si falls between step ~7000 and step ~50000 at all registered sizes, with 410m's μ_si ≈ step 14000–29000 anchoring the upper edge of the (A.timing) reversal-rate test. Steps 5000 / 7000 bracket the early shoulder; 10000–29000 spans the mid-emergence region most informative for the 2.8B vs 410m timing comparison; 41000–70000 covers the late-saturation tail used to fit the upper plateau of the count-vs-step logistic. Pre-step-5000 cells are omitted: §H4-7-supersede recorded 8 cells over steps 0–64 with top Δ_h ≈ 0.0000, and those `.npz` caches are preserved and may be appended to the analysis logistic as fixed pre-emergence anchors if useful for fit stability. The upper bound (step 70000) extends slightly beyond the §H4-7-supersede "step5000–step50000" suggestion to provide one late-saturation cell past 410m's plateau; the cell budget remains 10. Per-cell `.npz` per-prompt caches are written to a new directory `data/exploration/phase4_2_8b_s_inhibition_supersede_per_prompt/`, distinct from the §H4-7-supersede partial-cache directory, so the 8 prior `.npz` files are preserved untouched.

### §H4-supersede-2. Gate-predicate inheritance (verbatim from §H4-2)

The §H4-supersede gate predicates are inherited **verbatim** from §H4-2 — no re-derivation, no weakening, no re-tuning. Both legs must hold for §H4-supersede to PASS:

- **(A.timing) Bootstrap reversal-rate on emergence-step ordering, 2.8B vs 410m.** `P(μ_si^2.8B < μ_si^410m) ≥ 0.95` over B = 1000 paired per-prompt bootstrap replicates. Per §H2-2 machinery: each replicate resamples prompts with replacement, refits the count-vs-step logistic curve at both 2.8B and 410m, and records μ_si^2.8B and μ_si^410m. The reversal-rate is the empirical fraction of replicates in which μ_si^2.8B < μ_si^410m. Threshold ≥ 0.95 corresponds to a one-sided 5% test of the directional timing-axis scaling claim.

- **(A.count) Absolute count threshold breaks 410m saturation.** `max_count_si^2.8B ≥ 5` over the §H4-supersede 10-cell sweep — full-fit regime entry per §H2-3. Max is taken across the 10 cells in §H4-supersede-1; pre-step-5000 cells from the §H4-7-supersede partial cache do not contribute (top Δ_h ≈ 0 there).

### §H4-supersede-3. Failure-mode taxonomy inheritance (verbatim from §H4-5 + §H4-7-supersede DEFERRED)

The §H4-supersede failure-mode taxonomy is inherited **verbatim** from §H4-5 (PASS, TIMING-ONLY, COUNT-ONLY, NEITHER, TOOLING) and §H4-7-supersede (DEFERRED), including the priority ordering:

`DEFERRED > TOOLING > NEITHER > COUNT-ONLY > TIMING-ONLY > PASS`.

The five §H4-5 substantive / tooling patterns and their pre-committed paper headlines apply unchanged to the §H4-supersede verdict. DEFERRED carries over as the highest-priority non-substantive pattern: if the §H4-supersede sweep itself triggers a §H4-7-style per-cell-cost escape hatch (defined per §H4-supersede-4 below), the verdict is DEFERRED a second time and the paper still ships on Tracks 1 + 3 per the paper-status note above; a third attempt is permitted but not required. Silent goalpost-moving remains forbidden by the §H2-9-R / §H3-scale-6 rule.

### §H4-supersede-4. Compute estimate (locked)

End-to-end estimate at Pythia-2.8B-deduped, 10-cell S-inhibition sweep:

- **Per-cell cost baseline**: ~57 min/cell (steady-state across cells 1–8 of the §H4-7-supersede partial sweep, 3389–3401 s wall time). This is the *measured* cost, not the original §H4-7 projection of ~6 min/cell.
- **Sweep wall time**: 10 × ~57 min ≈ **9.5 h**, rounded to **~10 h** as the overnight budget. Compared to a §H4-scaling full-grid re-attempt at 40 × ~57 min ≈ **~38 h**, §H4-supersede is ~4× cheaper and fits in a single overnight window.
- **Bootstrap post-processing**: ~2 min (inherited from §H4-7).
- **Prefetch**: not re-incurred — the 40-checkpoint snapshots from §H4-scaling are on disk and the §H4-supersede grid is a subset. Memory and `BATCH_SIZE=10` constraints inherited from §H4-7 apply unchanged.

The §H4-7 per-cell-cost escape hatch applies to §H4-supersede with the **measured ~57 min/cell as the new baseline**: if §H4-supersede's first sweep cell exceeds ~115 min/cell (2× baseline), the runner pauses for re-grilling. fp16 model fallback if fp32 OOMs is inherited unchanged.

### §H4-supersede-5. Notebook and parquet deliverables (locked)

The §H4-supersede re-attempt produces:

1. `notebooks/_run_phase4_2_8b_s_inhibition_supersede_sweep.py` — 10-cell S-inhibition sweep runner. Mirrors `_run_phase4_2_8b_s_inhibition_sweep.py` but iterates only over the §H4-supersede-1 grid and writes to a distinct parquet path. The original 40-cell runner is preserved unmodified for chronology.
2. `data/exploration/phase4_2_8b_s_inhibition_supersede_sweep.parquet` — single new parquet, 10 rows. The abandoned `phase4_2_8b_s_inhibition_sweep.parquet` slot remains absent (per §H4-7-supersede: "Sweep parquet was NOT written").
3. `data/exploration/phase4_2_8b_s_inhibition_supersede_per_prompt/` — new per-cell `.npz` cache directory, 10 files. Distinct from the §H4-7-supersede partial-cache directory, which remains preserved.
4. `notebooks/_run_phase4_2_8b_supersede_analysis.py` — bootstrap + (A.timing) + (A.count) + §H4-supersede verdict analysis. Mirrors `_run_phase4_2_8b_analysis.py` and writes `phase4_2_8b_h4supersede_verdict.parquet`. Reported as a **scaling-appendix-grade result**, not a paper headline (per the amendment header).
5. New section in `h1c_ordering_test.ipynb`: §H4-supersede verdict cell, parallel to the existing §H3-scale and §H4-scaling-DEFERRED verdict cells. Loads `phase4_2_8b_h4supersede_verdict.parquet`, displays the (A.timing) and (A.count) leg results, the matched §H4-5 failure-mode pattern (or DEFERRED if the §H4-supersede-4 escape hatch is invoked), and the matched paper headline. The §H4-7-supersede DEFERRED cell is preserved verbatim above the new §H4-supersede cell.

Phase 2, Phase 3 (1B), and §H4-scaling supplementary parquets (induction / successor 2.8B) are NOT modified by this amendment.

### §H4-supersede-6. Pre-registration form (locked, no deferred lock)

This amendment is **reference-style with NO deferred numerical commit**. All numerical thresholds — (A.timing) reversal-rate ≥ 0.95, (A.count) max ≥ 5, τ_strict = 0.0372 (§S-tau), B = 1000, 95% percentile, per-prompt resampling, tiered-censoring full-fit ≥ 5 / marginal 2–4 / censored < 2 — are inherited verbatim from §H4-2 / §H4-3 / §H4-5, locked there before any 2.8B compute and preserved unchanged through §H4-7-supersede. The §H4-supersede-1 10-cell grid, the §H4-supersede-4 ~10 h compute estimate and ~115 min/cell escape-hatch threshold, the §H4-supersede-5 deliverable paths, and the inheritance of §H4-5 + §H4-7-supersede DEFERRED into the §H4-supersede taxonomy are pre-committed in this single amendment **before any §H4-supersede compute starts**.

A reviewer reading the chronology should see: §H1-C → §H2 → §H2-5 PASS (p = 0.00463) → §H2-9-R reframe → §H3-scale (1B REGR) → §H4-scaling registration (2.8B, 40-cell grid) → 2.8B anchor / induction / successor sweeps complete → S-inhibition sweep halts at 8/40 (~57 min/cell vs ~6 min projected) → §H4-7 escape hatch invoked → §H4-7-supersede registered (DEFERRED) → §H5-causal-3-record (1B causal-dependence canonicalization, post-data) → §H5-causal-3-2.8b (2.8B causal-dependence pre-data lock) → **§H4-supersede registers at 2.8B before any §H4-supersede compute**, with the 10-cell grid and the inherited §H4-2 / §H4-5 gate locked here → §H4-supersede sweep runs (~10 h overnight) → §H4-supersede verdict recorded in `notebooks/h1c_ordering_test.ipynb` §H4-supersede section against the verbatim §H4-2 gate. No spec change is anticipated between this amendment and the §H4-supersede verdict.

---

## Amendment 2026-05-11 — §writeup-conv: post-data convention + framing reframe (non-numerical, analog of §H2-9-R)

**Posted after §H4-supersede PASS verdict + §H5-causal-3-2.8b NULL × NULL verdict (both 2026-05-10/11).** This is a documentation-hygiene amendment, non-numerical, that locks downstream presentation conventions and rectifies two chronological notes in earlier amendment text. **No numerical thresholds, gate predicates, or detector specifications are modified.** All registered gates (§H1-C / §H2-5 PASS, §H5-causal NULL × NULL at 410m + 2.8B, §H4-supersede PASS) remain locked as registered. This amendment is the documentation analog of §H2-9-R: a post-data reframe that does not move any goalpost.

### §writeup-conv-1. Track-numbering convention (locked)

The amendment chain has drifted in its Track-N labeling. Earlier amendments (e.g., §H4-scaling line 654, §H4-supersede header line 1209) referenced Track 2 = Scaling and Track 3 = Causal-dependence in chronologically-organized framing. The paper-narrative document `WRITEUP.md` (and the downstream `README.md` status table) has settled on a reader-facing convention that orders by the paper's argumentative flow rather than registration chronology:

- **Track 1 — Emergence** (§H1-C / §H2 / §H2-9-R reframe)
- **Track 2 — Causal-disjointness** (§H5-causal / §H5-causal-2 / §H5-causal-3-record / §H5-causal-3-2.8b)
- **Track 3 — Scaling (head-count axis)** (§H3-scale / §H4-scaling / §H4-7-supersede / §H4-supersede)

§writeup-conv-1 locks this convention as the canonical Track-N → topic mapping going forward. Earlier amendment text that uses Track 2 = Scaling / Track 3 = Causal is **historically accurate as registered** but is superseded by §writeup-conv-1 for all downstream presentation (WRITEUP.md, README.md, notebook markdown cells, paper draft, LessWrong post). No earlier amendment text is edited retroactively; the §writeup-conv-1 lock is the canonical override.

The reordering reflects the paper's narrative — Track 1 (emergence ordering observed) → Track 2 (the natural compositional reading is *falsified* at convergence) → Track 3 (head-count-axis scaling confirms the emergence pattern at 1024 heads). It does NOT reflect registration chronology, which is the order: §H1-C → §H2 → §H3-scale → §H4-scaling → §H5-causal → §H5-causal-2 → §H4-7-supersede → §H5-causal-3-record → §H5-causal-3-2.8b → §H4-supersede.

### §writeup-conv-2. §H4-supersede framing reframe (post-PASS, locked)

The §H4-supersede amendment header (line 1209) was registered pre-data and positioned §H4-supersede as a *"scaling appendix / secondary result, not a paper headline"* — a conservative framing to ensure the paper would ship on Tracks 1 + 2 even if §H4-supersede DEFERED a second time. The pre-data framing was the correct registration discipline: do not over-commit to a result that has not yet been observed.

**Post-data observation (2026-05-11):** §H4-supersede PASSED on both legs — (A.timing) reversal_rate = 1.000 over B = 1000 paired bootstrap (gate ≥ 0.95) and (A.count) max_count_si^2.8B = 5 at step 29000 (gate ≥ 5). With both legs cleared cleanly, the paper-narrative framing in WRITEUP.md upgrades §H4-supersede from "scaling appendix" to "third converging substantive result". This is analogous to the §H2-9-R post-data reframe of §H1-C: the registered gate is unchanged, the falsification target is not weakened, but the paper's headline framing is upgraded after observing the result.

The framing upgrade does NOT modify:
- (A.timing) reversal_rate ≥ 0.95 gate predicate (locked §H4-2, inherited verbatim by §H4-supersede-2).
- (A.count) max_count ≥ 5 gate predicate (same).
- The §H4-5 + §H4-7-supersede failure-mode taxonomy (PASS / TIMING-ONLY / COUNT-ONLY / NEITHER / TOOLING / DEFERRED).
- The §H4-supersede-1 10-cell grid `[5000, 7000, 10000, 14000, 20000, 29000, 41000, 49000, 59000, 70000]`.

The framing upgrade DOES authorize:
- WRITEUP.md / README.md / notebook prose referring to §H4-supersede as a "third converging substantive result" rather than "scaling appendix".
- The paper's three-track headline structure (Track 1 + Track 2 + Track 3 all hitting registered targets).
- Reporting §H4-supersede in the main paper text rather than appendix-only.

A registered-trajectory caveat (post-data observation, also locked here): the §H4-supersede 2.8B count trajectory crosses (A.count) ≥ 5 transiently at step 29000 and dips back to 4 by step 70000 (trajectory: 1 → 2 → 2 → 3 → 3 → 5 → 4 → 4 → 4 → 4). The §H4-2 (A.count) gate is defined on `max` over the grid, not terminal-cell count, so PASS holds as registered. The dip caveat is registered here to ensure all downstream documents (WRITEUP / README / notebook §H4-supersede cell / paper / LessWrong) disclose it.

### §writeup-conv-3. Chronology rectification (documentation correction, no compute or threshold change)

Two chronological summaries in earlier amendment text contained errors of omission and ordering:

1. **§H4-supersede preamble chronology summary (HYPOTHESIS.md line 1266)** stated: *"§H4-7-supersede registered (DEFERRED) → §H5-causal-3-record → §H5-causal-3-2.8b → §H4-supersede"*. This summary OMITTED §H5-causal (2026-05-07) and §H5-causal-2 (2026-05-07), which were both registered BEFORE §H4-7-supersede (2026-05-08). The correct chronology should include them between §H4-scaling (2026-05-07) and §H4-7-supersede (2026-05-08).

2. **WRITEUP.md "Pre-registration chain" section (positions 11–12)** listed §H5-causal and §H5-causal-2 AFTER §H4-7-supersede (position 10). This is the inverse of the actual commit date order.

§writeup-conv-3 records the canonical chronology by amendment dates (the dispositive record):

1. §H1-C / pivot — pre-pilot.
2. Phase 1.0 pilot, Path C locked (`PILOT_RESULTS.md`, 2026-05-05).
3. Validation reframing amendment (2026-05-05, HYPOTHESIS.md line 62).
4. §S-inhibition Phase 1.3 detector spec (2026-05-05 evening, line 101).
5. §S-5b/c + §S-tau (2026-05-05 post-GPT-2-validation, line 159).
6. §SU (2026-05-05 later evening, line 208).
7. §SU-1b lift-form supersede (2026-05-06 early, line 289).
8. §SU-tau τ_lift lock (2026-05-06, line 335).
9. §H2 Phase 2 sweep spec (2026-05-06, line 363).
10. §H2-9-R post-data reframe (2026-05-06 post-Phase-2, line 456).
11. §H3-scale 1B scale-extension pre-reg (2026-05-06 post-grilling, line 524).
12. §H3-scale-8-vis (2026-05-07, line 621).
13. §H4-scaling pre-reg (2026-05-07, line 650).
14. **§H5-causal** (2026-05-07, line 766).
15. **§H5-causal-2** (2026-05-07, line 909).
16. **§H4-7-supersede** DEFERRED registration (2026-05-08, line 1014).
17. §H5-causal-3-record post-data 1B canonicalization (2026-05-10, line 1070).
18. §H5-causal-3-2.8b pre-data 2.8B (2026-05-10, line 1105).
19. §H4-supersede pre-data 2.8B reduced-grid (2026-05-10, line 1205).
20. **§writeup-conv** (this amendment, 2026-05-11).

The two chronological errors in earlier text (line 1266 omission and WRITEUP.md positions 11–12 inversion) are preserved as written for chronological auditability; §writeup-conv-3 is the canonical override. WRITEUP.md and any downstream cross-references should align to the §writeup-conv-3 chronology.

### §writeup-conv-4. §H4-supersede-vis: 5-size view authorization for per-motif sweep notebooks (analog of §H3-scale-8-vis)

§H3-scale-8-vis (2026-05-07) authorized integrating Pythia-1B as a 4th-size column into the per-motif `*_full_sweep.ipynb` notebooks alongside the registered 3 sizes (70m, 160m, 410m), with the strict caveat that the 1B column is presentation-only and does NOT extend the §H1-C registered emergence claim. §writeup-conv-4 extends the same authorization to **Pythia-2.8B as a 5th-size column** in the three per-motif sweep notebooks (`induction_full_sweep.ipynb`, `successor_full_sweep.ipynb`, `s_inhibition_full_sweep.ipynb`), subject to identical caveats:

- The 2.8B column in `induction_full_sweep.ipynb` and `successor_full_sweep.ipynb` displays the full 40-cell sweep parquets (`phase4_2_8b_induction_sweep.parquet`, `phase4_2_8b_successor_sweep.parquet`), which were sealed at §H4-7-supersede.
- The 2.8B column in `s_inhibition_full_sweep.ipynb` displays the §H4-supersede 10-cell reduced grid (`phase4_2_8b_s_inhibition_supersede_sweep.parquet`). The 8 cells from the §H4-7-supersede partial cache (steps 0–64, top Δ_h ≈ 0) are also displayed if useful for visual continuity at the left edge, marked as such. The remaining cells in the 40-cell §H2-1 grid that are absent from §H4-supersede's 10-cell grid are displayed as gaps in the count-vs-step trace, not interpolated.
- All visual presentation respects the convention that 2.8B is **not part of the registered §H1-C emergence claim**. §H1-C remains locked to the 3 registered sizes (70m, 160m, 410m) per §H2-1, with §H2-9-R as the canonical reframe. 1B remains a head-count regression per §H4-1. 2.8B's emergence trajectory is reported as supplementary cross-size data on the head-count axis, with the §H4-supersede PASS verdict providing the substantive scaling-axis claim.

§writeup-conv-4 does NOT modify §H1-C / §H2-5 / §H2-9-R / §H3-scale / §H4-scaling / §H4-supersede / §H5-causal / §H5-causal-2 / §H5-causal-3-record / §H5-causal-3-2.8b. The original 3-size figures, exactly as they were when §H1-C / §H2-5 passed at p = 0.00463, can be reconstructed by checking out the pre-§H3-scale-8-vis commit (the §H4-supersede-vis 5-size view is an additive presentation layer, not a replacement).

### §writeup-conv-5. Pre-registration form (locked, documentation-only)

This amendment is **reference-style and documentation-only**. No numerical thresholds, gate predicates, ablation methods, or detector specifications are introduced or modified. The §writeup-conv-1 track convention, §writeup-conv-2 framing reframe, §writeup-conv-3 chronology, and §writeup-conv-4 5-size visualization authorization are committed as a single documentation-hygiene amendment immediately after the §H4-supersede PASS verdict + §H5-causal-3-2.8b NULL × NULL verdict (both recorded 2026-05-11). The amendment is analogous to §H2-9-R: a post-data reframe that does not move any gate.
## Amendment 2026-05-11 — §H6-causal: induction-root causal-dependence ablation (pilot at 70M + 410M)

**Posted after §H4-supersede PASS (2026-05-11) + §H5-causal-3-2.8b NULL × NULL (2026-05-11) + §writeup-conv (2026-05-11), and before any §H6-causal compute.** This amendment registers a new inference-time causal-dependence test orthogonal to §H5-causal-family. §H5 falsified the *successor → S-inhibition* link on two metrics at 410M and 2.8B (NULL × NULL). §H6 tests the un-tested branch: **does induction sit at the root of both downstream motifs?** If ablating top-K induction heads at convergence disrupts successor and/or S-inhibition readouts, the temporal emergence ordering (μ_ind < μ_suc < μ_si) is *at least partially* an inference-time architectural ordering rooted in induction; if all readouts remain NULL, the ordered emergence is fully decoupled from architectural causal structure (a strong universal claim).

§H6-causal is a **separate scientific claim** from Track 1 (§H1-C / §H2-9-R), the §H5-causal-family suc → si branch of Track 2, and Track 3 (§H3-scale / §H4-scaling / §H4-supersede). Under §writeup-conv-1, §H6-causal **extends Track 2** with the second branch: ind → {suc, si}. Prior amendments are not modified.

§H6-causal is **pilot-first**: Pythia-70M + Pythia-410M @ step143000 only are pre-registered here. Extension to {160M, 1B, 2.8B} is registered separately as §H6-causal-3, conditional on the §H6-causal-10 trigger.

### §H6-causal-1. Scope (locked)

- **Pilot anchors:** Pythia-70M-deduped @ step143000 AND Pythia-410M-deduped @ step143000. Two anchors, one ablation experiment per anchor.
- **Three readouts run jointly per anchor** in a single ablation experiment:
    - **Readout A** — aggregate `lift_dla` across the 70-prompt successor probe per §SU-1b, over the four category subsets {days, months, numerals, letters}.
    - **Readout B** — §S-1 path-patching Δ_h on the locked top-3 SI senders at this anchor (per §H5-causal protocol).
    - **Readout C** — IO−S logit-diff at the END token over the 200-prompt IOI set (per §H5-causal-2 protocol).
- **Conditional extension to {160M, 1B, 2.8B}** — pre-registered procedurally in §H6-causal-10 below; per-size K values for the extended sizes are locked in §H6-causal-2 below (table) but the per-size suc / ctrl / SI / NM identities at extended sizes are derived at follow-up §H6-causal-3 amendment time.
- The 200-prompt IOI distribution (Wang 2023, 100 BABA + 100 ABBA, seed=0) at `data/prompts/ioi_prompts.tsv` and the 70-prompt successor probe at `data/prompts/successor_prompts.tsv` are the substrates, both unmodified from §H5-causal / §SU-1b. No new prompt sets.

### §H6-causal-2. Suc set — induction heads to ablate (procedure-locked, per-size K-scaling)

**Top-K_size induction heads** at (size, step=143000) by Olsson prefix-match score (the §IND-1 / Phase 1.2 detector score, sealed in `data/exploration/induction_full_sweep.parquet` and analogous 1B / 2.8B parquets), with the §H5-3 tie-break inherited verbatim (score desc, layer asc, head asc), **minus** any heads in the exclusion clauses below.

**Per-size K-scaling rule (locked).** K_size is set per anchor by the rule

```
K_size = min(20, max(5, ceil(0.33 × n_induction_detected_at_step143000_at_size)))
```

where `n_induction_detected` is the count of heads with Olsson prefix-match score ≥ 0.30 at (size, step=143000) in the sealed induction sweep parquet. The pre-data per-size K values, locked here from the sealed parquets, are:

| Size | n_induction (step143k) | K_size |
|---|---|---|
| 70M | 6 | 5 (floor enforced; near-exhaustion at this size) |
| 160M | 17 | 6 |
| 410M | 19 | 7 |
| 1B | 9 | 5 (floor enforced) |
| 2.8B | 34 | 12 |

These K values are part of the amendment lock (asserted bit-for-bit by the runners against the sealed parquets at runtime). The **ctrl set size matches the ablation K_size per size** (5 / 6 / 7 / 5 / 12). The pilot scope (§H6-causal-1) at 70M uses K=5 and at 410M uses K=7; the extended-size K values (160M, 1B, 2.8B) take effect only on §H6-causal-10 trigger.

Rationale. 410M's induction-detected population is 19 heads; ablating only 5 (26%) is a small sample and might not detect a real causal effect if induction's contribution is distributed across many heads. The §H5 precedent ablated 5/7 = 71% of successor's detected population at 410M; the §H6 rule ablates a smaller fraction (33%) of induction's detected population to keep the floor=5 at narrow-architecture sizes (70M, 1B) while scaling meaningfully at wider ones. At 2.8B with 34 induction-detected heads, K=12 (35%) is substantial. The min(20, ·) cap is a defensive ceiling for hypothetical larger populations — not load-bearing at any current §H6 size.

**Exclusion clauses.** The top-K_size induction list is filtered to exclude:

1. The **locked SI-sender set** at this anchor (§H5-causal-3 per-size lock; for 410M this is `{(12,12), (13,13), (14,0)}` per §H5-5; for 70M derived deterministically from `s_inhibition_pythia_70m_anchor.parquet` at runtime and asserted in the verdict parquet).
2. The **locked NM set** at this anchor (§H5-6 per-size lock; for 410M read from `s_inhibition_pythia_410m_anchor_per_nm.npz`; for 70M read from `s_inhibition_pythia_70m_anchor_per_nm.npz`, which is produced by the new §H6-6 prerequisite step).

If exclusion reduces the available top-K_size below K_size, **widen the candidate pool one rank at a time** (top-(K+1), top-(K+2), …) until K_size non-conflicting induction heads are identified, recording `widened_top_k` in the verdict parquet for audit. **Halt-rule (hard minimum):** halt and report if `K_effective < 4` — i.e., fewer than 4 induction heads survive all exclusions even after exhausting the candidate pool. The hard minimum K_effective ≥ 4 is a separate concept from the soft floor K_locked = 5 (the per-size locked value from the K-scaling table): the soft floor is the *target* per-anchor K, while the hard minimum is the *operational red line* below which the §H6 ablation experiment is not informative. When `K_effective < K_locked` but `K_effective ≥ 4`, the runner proceeds (per Decision (a), see 70M caveat below) with a registered `structural_caveat_k_exhausted = True` flag in the verdict parquet; when `K_effective < 4` the runner halts. Halt on the hard-minimum trigger is treated as a registered DEFERRED-style outcome per §H4-7-supersede precedent: a `pattern = HALT-COEXTENSIVE` verdict row is written, no headline is assigned, and the §H6 amendment chain awaits a fresh sub-amendment before re-attempting. The prior 10-rank-widening cap is subsumed by the K_effective ≥ 4 hard minimum: a population that cannot supply 4 surviving heads is essentially co-extensive with SI's circuit at this anchor.

**70M structural caveat (pre-data, registered; Decision (a) lock):** Only 6 induction heads clear the §IND-1 detection threshold at 70M step143000. The §H6-2 + §H6-2-bis 3-way exclusion (NM + SI-senders + suc-receivers) on this 6-head induction-detected population **exhausts to 4 surviving heads**. The K_locked = 5 floor cannot be reached without violating an exclusion clause. **Per Decision (a) (user-locked, this revision)**, the §H6 70M runner proceeds at **K_effective = 4** (with K_locked = 5 retained as the soft-floor target that was not reachable), with the verdict parquet column `structural_caveat_k_exhausted = True` and a logged-warning indicating the exhaustion. The cross-readout aggregate at 70M is evaluated on K_effective = 4 — interpretation of the 70M verdict acknowledges the smaller sample explicitly. Readouts A, B, and C are all uncontaminated under this rule (the 3-way exclusion is preserved); only the sample size is reduced. The ctrl set at 70M is correspondingly sized at K_effective = 4 (matching the ablation set per §H6-causal-3 / §H6-causal-5). The hard minimum K_effective ≥ 4 (§H6-2 halt-rule) is met — the 70M case **does not trigger HALT**; only K_effective < 4 (i.e., severe exhaustion) would HALT.

Beyond the K-exhaustion caveat: the suc / ctrl distinction at 70M is less informative than at larger sizes because ctrl is sampled from the score-bracket *below* threshold (§H6-causal-3), so the bracket population is the "near-induction-but-not-detected" tail rather than a magnitude-comparable population. The verdict parquet records `n_induction_above_threshold = 6`, `K_locked = 5`, `K_effective = 4`, `structural_caveat_k_exhausted = True`, and the per-exclusion `n_excluded_*` counts at 70M for audit.

### §H6-causal-2-bis. Suc-receiver-exclusion clause (locked)

**§H6-2-bis. Suc-receiver-exclusion clause (locked).** In addition to NM + SI senders exclusions in §H6-2, exclude any head that is also in the locked top-5 successor-receiver set at this anchor (the suc top-5 used as Readout A's receivers per §H5-3 tie-break). This prevents contamination of Readout A, where the suc receivers are the readout targets — ablating one as part of the induction set would conflate the headline "does induction-ablation move suc lift?" with "does ablating a suc receiver move its own lift?"

Rationale (pre-data, registered). At 70M, the locked top-5 successor-receiver set includes L4H0 / L4H1 (§H5-3 70M lock), adjacent layers to top induction heads. §H6-2 exclusion-widening alone could pull the ablation set into the suc-receiver pool, contaminating Readout A. The bis-clause ranks suc-receiver exclusion as a third equally-mandatory exclusion alongside NM and SI senders. The §H6-2 10-rank widening cap applies to the combined exclusion; the verdict parquet records `n_excluded_suc_receivers` per anchor for audit.

### §H6-causal-3. Ctrl set (procedure-locked)

Random sample of K_size heads via `rng = np.random.default_rng(0)` from the bracket `[INDUCTION_THRESHOLD − bw, INDUCTION_THRESHOLD)` with `bw = 0.05` initial and `+= 0.025` widening step (inherited verbatim from §H5-4 procedure with the score axis changed from `score_suc` to Olsson prefix-match score). The §H6-2 NM-and-SI-sender exclusions AND the §H6-2-bis suc-receiver exclusion apply to the ctrl candidate pool. `INDUCTION_THRESHOLD` is read from §IND-1 / Phase 1.2 (the registered induction detection threshold = 0.30); the runner writes the final `widened_bracket_width` to the verdict parquet for audit.

The ctrl bracket is on Olsson score (not on τ_lift). The bracket procedure, the seed (0), and the widening-step (0.025) are inherited from §H5-4 verbatim; only the score axis, threshold, and the per-size sample size K_size change. If `|CTRL_CANDIDATES| < K_size` after maximum widening (e.g., the score axis is too sparse below threshold at 70M), the runner halts and writes `pattern = HALT-CTRL-EMPTY`; analogous to §H6-causal-2's HALT-COEXTENSIVE rule.

### §H6-causal-4. Receivers — pinned per-anchor (locked)

The receivers for each readout are inherited verbatim from §H5-causal-3 per-anchor locks:

- **Readout A** (successor `lift_dla`): the top-5 successor heads at this anchor — i.e., the §H5-3 suc set, re-used **but here as receivers**, not as the heads being ablated. (At 70M and 410M, these are derived deterministically from `phase2_successor_sweep.parquet` per §H5-3, with §H5-3 tie-break. The 410M set is `[(22,6), (22,2), (20,4), (22,10), (12,8)]` per the §H5-3 lock; the 70M set is derived at runtime and asserted in the verdict parquet.) These heads are simultaneously the **suc-receiver-exclusion set** for §H6-2-bis. `lift_dla` is aggregated by mean across the 70 successor prompts within each of {days, months, numerals, letters} and then averaged across categories — identical aggregation rule to §SU-1b unchanged.
- **Readout B** (§S-1 Δ_h): the top-3 SI senders and the top-4 NMs at this anchor, pinned across all conditions per §H5-5 / §H5-6. NMs are read from the sealed `*_anchor_per_nm.npz` artifact for each size and pinned across conditions — no re-derivation under ablation.
- **Readout C** (IO−S logit-diff): no per-head receivers; the metric is a scalar over the 200-prompt IOI distribution per §H5-causal-2-2.

The §H6 receiver sets are **identical** to the §H5 receiver sets at each anchor so the §H6 verdicts are directly comparable to the §H5 verdicts at the same size. The only differential between §H6 and §H5 is the identity of the heads being ablated (induction top-K_size vs successor top-5).

### §H6-causal-5. Ablation method (locked, inherited verbatim from §H5-2)

Mean-ablation on `hook_z[:, :, head, :]` per length group, replace with batch-mean from a single clean forward pass over the 200 IOI prompts (Readouts B, C) or 70 successor prompts (Readout A). Permanent forward hook via `model.add_perma_hook` per (layer, head); persists through every `run_with_cache` / `run_with_hooks`. §H5-2 length-grouping caveat unchanged. For Readout A the substrate is the successor probe (single length group per category), so the within-group rule degenerates to a single mean tensor per (l, h).

Three conditions per anchor: **clean**, **suc_ablated** (induction top-K_size with §H6-2 + §H6-2-bis exclusions), **ctrl_ablated** (K_size score-bracket-matched controls, same exclusions). §H6 inherits the §H5 naming convention `suc_ablated` = "the ablation-set ablation" (NOT successor-head ablation) — preserved for code-level reuse of §H5-causal helpers without renaming. The verdict parquet records `ablate_kind = "induction_topK"` and `K_size`.

### §H6-causal-6. NM identity, receivers, and the 70M anchor prerequisite (locked, Strategy B)

For all anchors, the locked NM set used as Readout B receivers is the component-DLA top-4 NMs at the clean model per the §S-8 anchor-inspection protocol, persisted to `data/exploration/s_inhibition_pythia_{size}_anchor_per_nm.npz`.

- **For 410M**, the §S-8 anchor inspection already exists and the file `s_inhibition_pythia_410m_anchor_per_nm.npz` is the sealed artifact pinned across conditions per §H5-6 (unchanged from §H5-causal).
- **For 70M**, no pre-existing §S-8 anchor inspection exists. A new `_run_pythia_70m_anchor_s_inhibition.py` runner is registered to be executed BEFORE the §H6 70M compute; it produces `s_inhibition_pythia_70m_anchor_per_nm.npz` with the component-DLA top-4 NMs at clean 70M step143000. This pre-step is itself pre-registered with all numerical thresholds inherited verbatim from §S-1 / §S-tau / §S-6 / §S-7. The §H6 70M runner reads the resulting npz and asserts the NMs bit-for-bit at runtime.

The §S-8 protocol is inherited unchanged: load Pythia-70M-deduped @ step143000, run §S-1 path-patching at the §S-tau anchor, compute component-DLA per head per §S-6 / §S-7, select top-4 by aggregate component-DLA at END with §H5-6 tie-break (DLA desc, layer asc, head asc). The npz schema mirrors the 410M file (keys: `nm_layer`, `nm_head`, `nm_dla`, `nm_rank`) so downstream readers consume it without code-path differentiation.

The prerequisite is registered as a §H6-11 deliverable. Halt if fewer than 4 NMs clear the §S-7 floor (the §S-tau anchor at 70M is censored per §H1-C, so this is a pre-data risk worth flagging): write `pattern = HALT-NM-INSUFFICIENT` to `s_inhibition_pythia_70m_anchor_per_nm_halt.parquet` and pause §H6 70M compute for a sub-amendment.

### §H6-causal-7. Three readouts, verdict bands (locked, inherited from §H5-7 / §H5-causal-2-6 / §SU-1b)

For each readout, the per-readout pattern is one of {NULL, DEP, GENERIC, MIXED}, with thresholds inherited verbatim from the §H5 amendments. Ratios are computed paired per-prompt against the clean condition and bootstrapped per §H6-causal-9.

- **Readout A — successor `lift_dla` (aggregate):**
    - NULL if `ratio_suc_lift ∈ [0.8, 1.2]` AND `ratio_ctrl_lift ∈ [0.8, 1.2]`, both with 95% CI within the band.
    - DEP if `ratio_suc_lift < 0.5` with CI excluding 0.5 AND `ratio_ctrl_lift ∈ [0.8, 1.2]`.
    - GENERIC if both `ratio_suc_lift < 0.7` AND `ratio_ctrl_lift < 0.7`.
    - MIXED otherwise.
- **Readout B — §S-1 Δ_h (per-sender, then aggregate):** Per-sender classifier {NULL, DEP, GENERIC, MIXED} from §H5-7 verbatim (DEP_THRESHOLD = 0.5 × clean, NULL_BAND = ±0.20 × clean, GENERIC = both < 0.5). Aggregate across the 3 pinned SI senders via §H5-7 priority `GENERIC > NULL > DEP > MIXED` verbatim.
- **Readout C — IO−S logit-diff (scalar):** §H5-causal-2-6 classifier verbatim — NULL band `[0.8, 1.2]`, DEP `< 0.5` with CI exclusion, GENERIC `< 0.7` for both ratios.

**70M S-inhibition censoring caveat (pre-data, registered).** Per §H1-C / Phase 2, S-inhibition at 70M step143000 is censored (`max_count = 1` per §S-tau / §S-7). The §H5-5 procedure yields 3 heads at 70M, but magnitudes are at the noise floor; the ±0.20 × clean NULL band becomes effectively absolute (~10⁻³, comparable to bootstrap CI width). Pre-data expectation: Readout B at 70M will classify NULL or MIXED on noise alone regardless of true causal structure. The 70M verdict is therefore **driven primarily by Readouts A and C**; Readout B at 70M is recorded but not load-bearing for the cross-readout aggregate. The verdict parquet records `readout_B_below_noise_floor = True` at 70M.

### §H6-causal-7-agg. Cross-readout aggregate (locked, 5 patterns, OR semantics for B/C)

The three per-readout verdicts combine into a single cross-readout pattern per anchor. **OR semantics for the B/C aggregate are locked here**: `DEP_BC_only` fires when Readout B OR Readout C returns DEP (not require both). Rationale: B and C are converging metrics on the same underlying claim ("does induction-ablation disrupt S-inhibition"); either firing DEP is informative. A future sub-pattern `DEP_BC_strong` could be registered to distinguish AND-DEP from OR-DEP if needed; not registered here pre-data.

| Cross-readout pattern | Trigger | Paper-headline string (pre-committed) |
|---|---|---|
| **NULL × NULL × NULL** | A=NULL AND B=NULL AND C=NULL | "Induction heads are NOT a root for either successor or S-inhibition at inference time in Pythia-{size} step143000. Combined with the §H5-causal-family NULL × NULL on the suc → si branch, the ordered temporal emergence (μ_ind < μ_suc < μ_si) is fully decoupled from inference-time architectural causal structure at this anchor. Strong direct-mechanistic refutation of the §H1-C compositional reading; the registered emergence pattern is consistent with convergent training dynamics, not with a forward-pass causal chain." |
| **DEP on A only** | A=DEP AND B=NULL AND C=NULL | "Induction → successor is a real causal chain at inference time in Pythia-{size} step143000; induction → S-inhibition is not. Combined with the §H5-causal-family suc → si NULL × NULL, the tree story is partially confirmed on the suc branch but cleanly falsified on the si branch. Successor inherits its inference-time computation from induction; S-inhibition does not." |
| **DEP_BC_only** | A=NULL AND (B=DEP OR C=DEP) (OR semantics, not AND) | "Induction → S-inhibition is a real causal chain at inference time in Pythia-{size} step143000; induction → successor is not. The §H1-C compositional reading is partially confirmed on the si branch. Surprising given the structural-reuse data (ind ∩ suc nearly empty), and supportive of the A12 deep-dive observation that ind ∩ si is non-empty at every size larger than 70M." |
| **DEP on A AND (B or C)** | A=DEP AND (B=DEP OR C=DEP) | "Induction sits at the root of the full successor / S-inhibition tree at inference time in Pythia-{size} step143000. The temporal emergence ordering ind → {suc, si} reflects a true architectural ordering rooted in induction. The §H5-causal-family suc → si NULL stands — the two branches are parallel descendants of induction, not sequential." |
| **MIXED / GENERIC anywhere (not matching the four patterns above)** | any readout = GENERIC OR MIXED, and none of the four clean patterns above match | "Per-readout heterogeneous dependence at Pythia-{size} step143000; no global verdict on §H6-causal at this anchor. Reported per-readout with CIs. Readout-sensitivity caveat applies — particularly relevant if Readout C drops generically as observed in §H5-causal-3-record at 1B." |

**Verdict priority** when more than one pattern could match: `GENERIC > DEP-multiple > DEP-one > NULL³ > MIXED`. GENERIC on any readout overrides DEP / NULL (a generic-ablation-sensitive readout cannot distinguish induction-specific from any-head dependence). DEP on multiple readouts outranks DEP on a single readout. NULL × NULL × NULL is the substantive null. MIXED is the catch-all.

**Documentation requirement for K_effective < K_locked (locked, sub-case to the 5-pattern taxonomy).** When `structural_caveat_k_exhausted = True` at a given anchor (i.e., the §H6-2 + §H6-2-bis 3-way exclusion drops K below the soft-floor K_locked but K_effective ≥ 4 so the runner proceeds), the cross-readout aggregate **still maps to one of the existing 5 patterns** (NULL_all, DEP_A_only, DEP_BC_only, DEP_A_and_BC, GENERIC/MIXED) — no new verdict pattern is introduced. However, the paper headline at any size where `structural_caveat_k_exhausted = True` MUST disclose **both K_effective and the structural reason** (the surviving-head count after exclusion plus the named exclusion clauses consumed). Concretely, the headline string is prefixed with: "[K_effective = {K_effective} < K_locked = {K_locked}; structural exhaustion of induction-detected population by §H6-2 + §H6-2-bis exclusions] " before the pre-committed pattern headline above. This is a documentation requirement, not a verdict-modification requirement: the substantive claim is unchanged; only the sample-size disclosure is mandatory. For the 70M pilot anchor under Decision (a), this prefix fires unconditionally on every headline string at 70M.

### §H6-causal-8. (Intentionally vacated — content folded into §H6-causal-7-agg above.)

Subsection number preserved for cross-reference stability with the prior amendment draft; the cross-readout aggregate now lives in §H6-causal-7-agg.

### §H6-causal-9. Bootstrap and statistical machinery (inherited verbatim from §H5-8 / §H5-causal-2-7)

B = 200 paired per-prompt bootstrap replicates per condition, per readout. Pairing preserves the per-prompt correlation between clean and ablated runs (each replicate samples N indices with replacement and computes both clean and ablated metrics on the same indices; N = 200 for Readouts B and C, N = 70 for Readout A). 95% percentile CI on each ratio (Readout A: ratio on aggregated lift_dla; Readout B: per-sender ratio on Δ_h; Readout C: ratio on mean logit-diff). RNG: `numpy.random.default_rng(seed=0)` for ctrl-set selection (§H6-causal-3); `numpy.random.default_rng(seed=1)` for the bootstrap (§H6-causal-9). Both seeds inherited verbatim from §H5-4 / §H5-8 / §H5-causal-2-7.

### §H6-causal-9-compute. Compute estimate and escape hatch (locked)

End-to-end estimate per anchor (all three readouts in one runner pass):

- **Pythia-70M @ step143000:** ~30 min wall time for the main §H6 runner (small model, fast forward passes; Readout B's path-patching is the dominant cost at d_model = 512 but is still cheap). **Plus** the §H6-6 prerequisite step `_run_pythia_70m_anchor_s_inhibition.py` ≈ ~10–15 min wall time (single-condition §S-1 path-patching at 70M).
- **Pythia-410M @ step143000:** ~5–7 h wall time (path-patching at d_model = 1024 dominates; Readout A's 70 prompts × 3 conditions is negligible incremental cost; Readout C's logit-diff is forward-pass-only ≈ 1–2 min). Conservative upper bound that includes the precompute pass for Readout A's mean-ablation reference, the bootstrap on all three readouts, and BATCH_SIZE = 25 path-patching. K_size = 7 at 410M (vs. 5 prior draft) adds a marginal ~10% to path-patching cost per condition since the mean-ablation hook applies head-mask-wise. The runner records wall time per readout in the verdict parquet.

The §H4-7 / §H5-causal-3-2.8b-9 per-condition wall-time escape hatch applies: **if measured wall time on the first (clean) condition exceeds 2× the projected per-condition cost, the runner pauses for re-grilling**. At 70M the escape threshold is ~20 min; at 410M ~4 h. fp16 fallback inherited from §H4-7. `HF_HUB_OFFLINE = 1` at module top of all runners per §H4-7-supersede / §H5-causal-3-2.8b. Halt-on-escape produces a `pattern = HALT-COMPUTE-OVERRUN` verdict row; no headline assigned; sub-amendment required for re-attempt.

### §H6-causal-10. Conditional extension to {160M, 1B, 2.8B} (procedurally locked; per-size identities deferred to §H6-causal-3 follow-up; K_size pre-locked in §H6-causal-2 table)

Pilot anchors (§H6-causal-1) are 70M + 410M only. Extension to {160M, 1B, 2.8B} is **not pre-registered as compute**; it is registered procedurally with the per-size K values already locked in the §H6-causal-2 table.

- **Trigger (locked, unchanged from prior pass):** pilot results show **DEP on any of the three readouts at 70M OR 410M with 95% CIs supporting the DEP classification under §H6-causal-7 / §H6-causal-7-agg** (ratio < 0.5 with CI excluding 0.5 for Readouts A or C; §H5-7 per-sender DEP aggregation for Readout B; non-NULL substantive pattern under §H6-causal-7-agg). The trigger is **unchanged by Decision (a)**: a DEP verdict at 70M under K_effective = 4 still fires the extension just as a DEP verdict at 410M under K_effective = K_locked = 7 would. At extended sizes (160M, 1B, 2.8B) K_effective is expected to equal K_locked because the induction-detected populations are larger (17 / 9 / 34 heads), but the `structural_caveat_k_exhausted` flag remains available and the K_effective ≥ 4 hard minimum still applies.
- **Action on trigger:** post `§H6-causal-3` (analog of §H5-causal-3-2.8b) **before any extended-size compute**, with per-size suc, ctrl, SI, and NM identities locked from the sealed parquets and asserted bit-for-bit. Thresholds, ablation, bootstrap, escape hatch, and aggregate inherited verbatim from §H6-causal-1 through §H6-causal-9-compute. Per-size K values (160M: 6, 1B: 5, 2.8B: 12) inherited from the §H6-causal-2 table without re-derivation.
- **Non-trigger:** extended-size sweep is **not registered**. Pilot result stands as a 2-size finding.

This procedural-lock-with-deferred-numerical-lock pattern mirrors §H5-causal's original 410M-only scope. It is **not** a §H5-causal-2-style "trigger-on-NULL" pattern — §H6-causal-3 is registered to extend a DEP finding to additional sizes. The asymmetry is intentional: a NULL × NULL × NULL pilot result is already a strong universal claim (combined with §H5's NULL × NULL on the suc → si branch, it constitutes full causal-disjointness across the registered tree). A DEP pilot result is locally informative but requires cross-size replication to support a generalized claim.

### §H6-causal-11. Notebook and parquet deliverables (locked)

**Prerequisite step (registered, must complete BEFORE §H6 70M compute):**

0. `notebooks/_run_pythia_70m_anchor_s_inhibition.py` — Pythia-70M §S-8 anchor inspection runner (Strategy B per §H6-6). Loads Pythia-70M-deduped @ step143000, runs §S-1 path-patching at the §S-tau anchor, computes component-DLA per §S-6 / §S-7, persists top-4 NMs to `data/exploration/s_inhibition_pythia_70m_anchor_per_nm.npz` (schema matches the 410M file: keys `nm_layer`, `nm_head`, `nm_dla`, `nm_rank`). Runner writes a `.log` (gitignored, via `tee`). On halt (fewer than 4 NMs above §S-7 floor), writes `s_inhibition_pythia_70m_anchor_per_nm_halt.parquet` with `pattern = HALT-NM-INSUFFICIENT` and §H6 70M compute is paused.

**Main §H6 runners (per anchor; one runner produces the parquets for all three readouts in a single ablation experiment):**

1. `notebooks/_run_phase4_h6_induction_70m_anchor.py` — Pythia-70M runner. Three readouts in one runner pass. Bit-for-bit assertion on locked suc set (derived at runtime from `phase2_successor_sweep.parquet`), locked SI senders (derived from `s_inhibition_pythia_70m_anchor.parquet`), locked NMs (read from `s_inhibition_pythia_70m_anchor_per_nm.npz` produced by deliverable 0 above), and the induction top-K with K_locked = 5 and K_effective = 4 after §H6-2 + §H6-2-bis 3-way exclusion (derived from `induction_full_sweep.parquet` with §H6-causal-2 procedure). Runner sets `structural_caveat_k_exhausted = True` and emits a logged warning per Decision (a). Hard-minimum check `K_effective ≥ 4` is asserted before any forward pass — failure halts with `pattern = HALT-COEXTENSIVE`.
2. `notebooks/_run_phase4_h6_induction_410m_anchor.py` — Pythia-410M runner. Same three-readouts-in-one-pass structure; bit-for-bit assertion on the §H5-3 / §H5-5 / §H5-6 410M locks plus the runtime-derived induction top-K_size = 7 with §H6-2 + §H6-2-bis exclusions.
3. Per anchor: three readout parquets (per-prompt long-format), three summary parquets (per-(condition, readout) aggregates with bootstrap CIs), and **one cross-readout verdict parquet** (single row: §H6-causal-7-agg pattern, paper headline, per-readout sub-verdicts, and caveats `n_induction_above_threshold`, `readout_B_below_noise_floor`, `widened_top_k`, `widened_bracket_width`, `n_excluded_suc_receivers`, `n_excluded_nm`, `n_excluded_si_senders`, `K_locked`, `K_effective`, `structural_caveat_k_exhausted` (bool, per size; fires when the §H6-2 + §H6-2-bis 3-way exclusion drops K below the locked floor — pre-data fires at 70M only per Decision (a)), `ablate_kind = "induction_topK"`). Naming: `phase4_h6_induction_{70m,410m}_anchor{,_lift,_logitdiff}{,_summary,_verdict}.parquet` and `phase4_h6_induction_{70m,410m}_anchor_h6causal_verdict.parquet`.
4. `.log` per runner (gitignored, via `tee`).
5. `notebooks/_build_causal_dependence.py` + `notebooks/causal_dependence.ipynb` extended in place with a §H6-causal section after §H5-causal-3-2.8b. Per-anchor figure: clean / suc_ablated / ctrl_ablated bar chart with CI bands for each readout. §H5 cells preserved verbatim.

Phase 2, Phase 3, §H4-scaling, §H4-supersede, and §H5-causal-family parquets are NOT modified. The §S-1 path-patching primitive in `src/detectors/s_inhibition.py` is unchanged (the §H5-6 `nm_heads_override` kwarg is reused for §H6 Readout B and for the new 70M anchor npz from deliverable 0).

### §H6-causal-12. Pre-registration form (locked, no deferred numerical lock)

This amendment is **reference-style with NO deferred numerical commit**. All numerical thresholds — DEP threshold 0.5 with CI exclusion, NULL band [0.8, 1.2] (Readouts A and C) / ±0.20 × clean (Readout B per-sender), GENERIC threshold 0.7 (Readouts A and C) / 0.5 (Readout B per §H5-7), B = 200 bootstrap, seed = 0 (ctrl rng) / seed = 1 (bootstrap rng), bracket_width_init = 0.05 with step 0.025 (§H6-causal-3), per-size K_size from the §H6-causal-2 table (70M=5, 160M=6, 410M=7, 1B=5, 2.8B=12), §H5-7 per-sender aggregation priority `GENERIC > NULL > DEP > MIXED` for Readout B, §H6-causal-7-agg cross-readout priority `GENERIC > DEP-multiple > DEP-one > NULL³ > MIXED`, OR-semantics for the B/C aggregate of DEP_BC_only, escape-hatch threshold 2× per-condition projection, halt-rule 10-rank-widening for §H6-causal-2 (including §H6-2-bis), halt-rule HALT-CTRL-EMPTY for §H6-causal-3, halt-rule HALT-NM-INSUFFICIENT for §H6-causal-6 — are inherited verbatim from §H5-causal / §H5-causal-2 / §H5-3 / §H5-4 / §H5-7 / §H5-causal-2-6 / §H5-8 / §H4-7 / §H4-7-supersede / §S-1 / §S-6 / §S-7 / §S-8, all locked there before this amendment, or locked in the §H6-causal-2 / §H6-causal-2-bis / §H6-causal-7-agg new-lock subsections of this amendment.

The only **§H6-new numerical / procedural locks** are:

- The identity of the ablation-set selection criterion (top-K_size induction by Olsson prefix-match score with NM, SI-sender, and suc-receiver exclusions — §H6-2 + §H6-2-bis).
- The **per-size K-scaling rule** `K_size = min(20, max(5, ceil(0.33 × n_induction_detected)))` and the resulting locked per-size K_locked values (5 / 6 / 7 / 5 / 12 at 70M / 160M / 410M / 1B / 2.8B). Decision (a) (this revision) registers the 70M K_effective = 4 exception with `structural_caveat_k_exhausted = True`; K_locked is unmodified.
- The §H6-causal-7-agg cross-readout aggregate taxonomy (5 patterns + priority + OR semantics for B/C) and its 5 pre-committed paper-headline strings, plus the K_effective < K_locked documentation-requirement prefix (this revision).
- The §H6-causal-6 Strategy-B prerequisite step (new §S-8 anchor inspection at 70M produces `s_inhibition_pythia_70m_anchor_per_nm.npz`).
- The §H6-causal-10 conditional-extension trigger (DEP on any readout at either pilot anchor with CI support).
- The **§H6-2 hard-minimum halt-rule** `K_effective ≥ 4` (this revision; replaces the prior 10-rank-widening halt), with the soft-floor K_locked retained as a separate concept. The verdict parquet adds `structural_caveat_k_exhausted` (bool, per size) alongside the existing `widen_depth` / `n_excluded_*` columns.

No deferred lock exists; all locks are committed before any §H6-causal compute. Chronology: §H1-C → §H2 → §H2-9-R → §H3-scale (1B REGR) → §H4-scaling → §H5-causal (NULL @ 410M) → §H5-causal-2 (NULL @ 410M) → §H4-7-supersede (DEFERRED) → §H5-causal-3-record (1B post-data) → §H5-causal-3-2.8b (NULL × NULL @ 2.8B) → §H4-supersede (PASS @ 2.8B) → §writeup-conv → **§H6-causal pilot registered at 70M + 410M**, with §H6-6 prerequisite step run BEFORE 70M compute and extension to {160M, 1B, 2.8B} deferred to §H6-causal-3 → pilot run → verdict in `causal_dependence.ipynb` §H6-causal section.

§H2-8's spec-failure-during-phase policy applies. The §writeup-conv-1 track convention (Track 2 = Causal-disjointness) is canonical for §H6-causal in WRITEUP.md, README.md, paper-draft, and LessWrong-post.

This amendment does NOT modify §H1-C / §H2-5, §H5-causal-family (suc → si branch, NULL × NULL @ 410M & 2.8B preserved), §H3-scale, §H4-7-supersede, §H4-supersede, or §writeup-conv. It extends Track 2 with the un-tested ind → {suc, si} branch, as a pilot before any §H6 compute.

### Revision log (changes from prior draft)

- **Decision 1 — Added §H6-causal-2-bis (suc-receiver-exclusion clause).** New sub-clause immediately after §H6-2 excludes any head also in the locked top-5 successor-receiver set at this anchor from the induction-ablation pool. Prevents contamination of Readout A at 70M (where suc receivers L4H0 / L4H1 sit in adjacent layers to top induction heads and could otherwise be pulled in by §H6-2 exclusion-widening). The exclusion is also propagated to the §H6-causal-3 ctrl candidate pool and the §H6-causal-5 condition definitions, and a new `n_excluded_suc_receivers` field is added to the verdict parquet schema in §H6-11.
- **Decision 2 — Strategy B for 70M NM identity (§H6-causal-6 rewritten; new prerequisite deliverable in §H6-11).** §H6-causal-6 (renumbered from the prior draft's compute-estimate section, which moved to §H6-causal-9-compute) now registers a new `_run_pythia_70m_anchor_s_inhibition.py` runner as a §H6-11 deliverable-0 prerequisite that produces `s_inhibition_pythia_70m_anchor_per_nm.npz` before any §H6 70M compute, with all numerical thresholds inherited from §S-1 / §S-tau / §S-6 / §S-7 and a HALT-NM-INSUFFICIENT halt rule if fewer than 4 NMs clear the §S-7 floor at 70M.
- **Decision 3 — OR semantics for B/C aggregate locked in §H6-causal-7-agg.** The cross-readout aggregate (renumbered from §H6-causal-7 to §H6-causal-7-agg; §H6-causal-7 now holds the per-readout verdict bands) explicitly states that `DEP_BC_only` fires when Readout B OR Readout C returns DEP (not AND). A future `DEP_BC_strong` is noted as a possible sub-pattern but is not registered pre-data. The §H6-causal-12 lock-summary lists OR semantics as a new §H6 lock.
- **Decision 4 — Per-size K-scaling rule replaces fixed K=5 in §H6-causal-2.** §H6-causal-2 now uses `K_size = min(20, max(5, ceil(0.33 × n_induction_detected)))` yielding 70M=5, 160M=6, 410M=7, 1B=5, 2.8B=12. Ctrl set sizes (§H6-causal-3) match K_size per size. The pilot scope (§H6-causal-1) remains 70M + 410M only; pilot uses K=5 at 70M and K=7 at 410M (up from K=5 in prior draft at 410M). §H6-causal-9-compute notes the marginal ~10% path-patching cost increase at 410M from K=5→7. §H6-causal-10 inherits the table verbatim for the extended sizes on trigger.

#### Cross-clause conflicts resolved during revision

- The prior draft's §H6-causal-7 (cross-readout aggregate) and §H6-causal-8 (bootstrap) were renumbered to §H6-causal-7-agg and §H6-causal-9 respectively to make room for the new §H6-causal-6 (Strategy B prerequisite, replacing the old §H6-causal-6 receiver-tail text which is now in §H6-causal-4) and to keep the 12-sub-clause skeleton intact. §H6-causal-8 is intentionally vacated with a pointer to §H6-causal-7-agg to preserve cross-reference stability with any documents (paper-draft, WRITEUP.md) that already cite §H6-causal-7 / §H6-causal-8.
- The fixed K=5 in the prior draft appeared in §H6-causal-2, §H6-causal-3, §H6-causal-5, §H6-causal-9-compute (compute estimate), §H6-causal-11 (deliverables 1 and 2), and §H6-causal-12 (lock summary); all six references have been updated to `K_size` with the per-size table, with the pilot values (70M=5, 410M=7) explicitly stated at each site.
- The §H6-causal-12 lock-summary list of "§H6-new numerical / procedural locks" was expanded from 3 items to 5 to include the per-size K rule and the Strategy-B prerequisite step.

### Revision log — final pass (user Decision (a) on 70M K-exhaustion)

This is the **final revision** before commit to HYPOTHESIS.md. Appends to the prior pass; does not modify Decisions 1–4 above.

- **Decision (a) — 70M K_effective = 4 accepted with structural-exhaustion flag (§H6-causal-2 + 70M structural caveat rewritten).** The §H6-2 + §H6-2-bis 3-way exclusion (NM + SI-senders + suc-receivers) on the 6-head induction-detected population at 70M exhausts to 4 surviving heads — K_locked = 5 cannot be reached without violating an exclusion clause. Per Decision (a), the §H6 70M runner proceeds at K_effective = 4 with `structural_caveat_k_exhausted = True` and a logged warning. Readouts A, B, and C remain uncontaminated (all 3 exclusions enforced); only the sample size is reduced. The 70M structural-caveat paragraph in §H6-causal-2 was rewritten to make the K_effective = 4 disclosure explicit (not a footnote).
- **§H6-2 halt-rule rewritten: hard minimum K_effective ≥ 4 (replaces 10-rank-widening cap).** The new halt-rule is "halt if K_effective < 4 (hard minimum)". The soft floor K_locked = 5 (from the per-size K-scaling table) is preserved as a separate concept. When `K_effective < K_locked` but `K_effective ≥ 4`, the runner proceeds with `structural_caveat_k_exhausted = True`. When `K_effective < 4` the runner halts with `pattern = HALT-COEXTENSIVE`. The 70M K_effective = 4 case meets the hard minimum and **does not** HALT.
- **§H6-causal-7-agg documentation-requirement sub-case (K_effective < K_locked).** Not a new verdict pattern — the cross-readout aggregate still maps to one of the existing 5 patterns. New requirement: the paper headline at any size where `structural_caveat_k_exhausted = True` MUST be prefixed with `[K_effective = {K_effective} < K_locked = {K_locked}; structural exhaustion of induction-detected population by §H6-2 + §H6-2-bis exclusions]`. For 70M under Decision (a), this prefix fires unconditionally.
- **§H6-causal-11 verdict-parquet schema addition.** New column `structural_caveat_k_exhausted` (bool, per size) added alongside the existing `widen_depth` and `n_excluded_*` columns; fires when the §H6-2 + §H6-2-bis 3-way exclusion drops K below the locked floor. Pre-data fires at 70M only. Companion columns `K_locked` and `K_effective` (ints) are also explicit in the schema.
- **§H6-causal-10 conditional-extension trigger unchanged.** Confirmed: DEP on any readout at 70M OR 410M still triggers the {160M, 1B, 2.8B} extension. A DEP verdict at 70M under K_effective = 4 still fires the extension. At extended sizes K_effective is expected to equal K_locked (larger populations: 17 / 9 / 34 heads), but the `structural_caveat_k_exhausted` flag and the K_effective ≥ 4 hard minimum remain available.

#### Cross-clause renumbering required (final pass)

None. The 5 sub-changes integrate into existing sub-clauses §H6-causal-2, §H6-causal-7-agg, §H6-causal-10, §H6-causal-11, and §H6-causal-12 without renumbering. Sub-clause numbering from the prior pass (12-skeleton with §H6-causal-2-bis and the vacated §H6-causal-8) is preserved bit-for-bit. The deliverable-0 prerequisite (§H6-causal-6 / §H6-11 deliverable 0) is unchanged.


---

## Amendment 2026-05-11 — §H6-causal-ksweep: K-dose-response sweep at Pythia-410M

**Posted after §H6-causal pilot completion (2026-05-11 19:30:41) — 70M = GENERIC_any (weak-readout, anticipated), 410M = aggregate MIXED with Readout A NULL + Readout C NULL + Readout B per-sender NULL;MIXED;NULL.** The 410M aggregate MIXED is driven by L13H13's per-sender MIXED, where `r_ind=0.805` and `r_ctrl=0.791` are essentially identical — the same artifact pattern as §H5-causal-3-record's 1B Metric B (control behaving like the experimental condition; ablating 7 random near-threshold heads in a layer cluster generically perturbs the readout). The §H6-causal-10 extension trigger (DEP on any readout) did NOT fire; the pilot is the sealed verdict.

This amendment registers a **dose-response K-sweep at the 410M anchor only**, exploring how the three readouts move as K (the number of induction heads ablated) varies. The purpose is exploratory characterisation, not a new gate: if induction has a small but real causal contribution that K=7 didn't catch, larger K should show a coherent trend on Readout A or C; if induction is genuinely causally disjoint (the §H6 pilot's reading), readouts should stay NULL or show only the generic-ablation noise pattern Readout B already exhibits.

### §H6-causal-ksweep-1. Scope (locked)

Pythia-**410M**-deduped @ step 143000 only. No other size. The 70M pilot's GENERIC_any verdict already establishes that 70M readouts are too noisy for a K-sweep to be informative.

### §H6-causal-ksweep-2. K values (locked)

```
K ∈ {5, 10, 15, 19}
```

Rationale:
- **K = 5**: §H5-causal precedent (5 heads at 410M); the §H5 successor-ablation experiment used K=5 on a 7-head detected population (71%). For induction at 410M (19 heads), K=5 is 26% of the population.
- **K = 10**: ~half the induction-detected pool (53%).
- **K = 15**: matches §H5's 71% fraction-of-detected-population at 410M (15/19 = 79%).
- **K = 19**: all induction-detected heads at 410M (100%). Maximal possible signal if induction has any causal role.

§H6-causal-2 K-locked value (7) was already run today; its verdict is already in `phase4_h6_induction_410m_anchor_verdict.parquet`. The K-sweep does NOT re-run K=7; analysis incorporates it as a 5th data point in the dose-response curve.

### §H6-causal-ksweep-3. Exclusion clauses (inherited verbatim)

§H6-causal-2 NM + SI exclusion + §H6-causal-2-bis suc-receiver exclusion apply at every K. At 410M none of these catch any of the top-19 induction heads (cf. drafts/H6_locked_sets.md), so K can range up to 19 without exhaustion. Ctrl set is matched to K via the §H6-causal-3 bracket-widening procedure (seed=0); at larger K the ctrl bracket may need to widen further than at K=7.

### §H6-causal-ksweep-4. Readouts and verdict bands (inherited verbatim)

Three readouts and per-readout classifiers identical to §H6-causal-7. The K-sweep does NOT register a new aggregate verdict — each K gets its own per-readout (A, B, C) verdict, and the analysis presents these as a 4-row table plus a dose-response plot of `ratio_ind` and `ratio_ctrl` vs K for each readout. No cross-readout aggregate is locked per-K; the per-readout trend is the deliverable.

### §H6-causal-ksweep-5. Compute estimate (locked)

Single runner, model loaded once, 1 clean condition shared across all K, then 2 × 4 = 8 ablated conditions (ind / ctrl for each of K ∈ {5, 10, 15, 19}). The 410M pilot ran 3 conditions in ~16 min wall time; the K-sweep runs 9 conditions, projected ~50–70 min wall time. Per-condition wall escape hatch inherits from §H4-7 (halt at 2× projection).

### §H6-causal-ksweep-6. Deliverable (locked)

A single new script `notebooks/_run_phase4_h6_induction_410m_ksweep.py` writes:

- `data/exploration/phase4_h6_induction_410m_ksweep.parquet` — per-(K, condition, readout, sender_or_receiver, prompt_idx) values + the 4 verdict rows (one per K). Sufficient to reconstruct the dose-response curve.
- `data/exploration/phase4_h6_induction_410m_ksweep_verdict.parquet` — 4-row table (one per K) with per-readout verdicts + ratios + CIs + bracket-widening per K.
- `data/exploration/phase4_h6_induction_410m_ksweep.log` — captured stdout.

No new figure is committed by the runner; the verdict notebook (`causal_dependence.ipynb`) will render the dose-response plot post-compute.

### §H6-causal-ksweep-7. Pre-registration form (locked)

This amendment is reference-style; all thresholds inherited verbatim from §H6-causal / §H5-causal / §H5-causal-2 / §SU-1b. The only locks introduced here are: (a) the K set {5, 10, 15, 19}; (b) the 410M-only scope; (c) the deliverable file paths. No aggregate verdict, no new gate. The K-sweep is a characterisation experiment.

---

## Amendment 2026-05-11 — §H6-causal-ksweep-2.8b: K-dose-response sweep at Pythia-2.8B

**Posted after §H6-causal-ksweep at 410M completion (2026-05-11 21:43, ~1.3 min wall time).** The 410M K-sweep showed `ratio_ind` and `ratio_ctrl` track each other across K ∈ {5, 7, 10, 15, 19}; at K=19 the control 19-head ablation has a *larger* effect on Readout A than the induction ablation. **No K reveals an induction-specific causal effect** at 410M, confirming the §H6 pilot's NULL across the dose-response.

This amendment extends the K-sweep to Pythia-2.8B step143000 (head-count tier 1024) to test whether the 410M dose-response NULL replicates at the wider architecture. Same procedure, different K-set scaled to 2.8B's larger induction-detected population (34 heads).

### §H6-causal-ksweep-2.8b-1. Scope (locked)

Pythia-**2.8B**-deduped @ step 143000 only. Single anchor.

### §H6-causal-ksweep-2.8b-2. K values (locked)

```
K ∈ {5, 12, 18, 25, 34}
```

Rationale (parallel to 410M):
- **K = 5**: §H5-causal precedent (5/34 = 15%).
- **K = 12**: K-scaling rule for 2.8B (`ceil(0.33 × 34) = 12`; 35%).
- **K = 18**: ~half the induction-detected pool (53%).
- **K = 25**: §H5's 71% fraction-of-detected-population analog (25/34 = 74%).
- **K = 34**: all induction-detected heads at 2.8B (100% target). Two heads in the top-34 are excluded by §H6-2 NM clause (L13H9 is NM+SI dual-role; L17H12 is NM), so K_effective will be 32 with `structural_caveat_k_exhausted=True`. This is the analog of the 410M K=19 high-K row (which returned K_eff=18 of 19 attempted).

### §H6-causal-ksweep-2.8b-3. Exclusion clauses, readouts, ctrl set procedure, bootstrap, classifiers

All inherited verbatim from §H6-causal-ksweep at 410M. The 2.8B-specific locked receivers are inherited from §H5-causal-3-2.8b:
- NMs: `[(11,29), (17,12), (22,31), (13,9)]`
- SI senders: `[(11,29), (11,5), (13,9)]`
- Suc receivers (Readout A targets): `[(15,14), (28,17), (27,13), (13,10), (29,28)]` per §H5-3 suc top-5 at 2.8B.

The §H5-causal-3-2.8b-6 structural-insensitivity caveat (3 of 5 suc heads sit at layers > max(NM layer)=22) applies to Readout B only at 2.8B; it does NOT affect Readout A (lift readout at the suc heads themselves) or Readout C (logit-diff at END).

### §H6-causal-ksweep-2.8b-4. Compute estimate (locked)

Per the §H5-causal-3-2.8b runs (~1.2 min total for Metric A + Metric B at single-cell anchor) and the 410M K-sweep timing (~1.3 min for 9 conditions), the 2.8B K-sweep at 11 conditions is projected at ~5-15 min wall time. Per-condition wall escape hatch inherits from §H4-7 (halt at 2× projection).

### §H6-causal-ksweep-2.8b-5. Deliverable (locked)

`notebooks/_run_phase4_h6_induction_2_8b_ksweep.py` writes:
- `data/exploration/phase4_h6_induction_2_8b_ksweep.parquet` — per-(K, condition, readout, prompt) values.
- `data/exploration/phase4_h6_induction_2_8b_ksweep_verdict.parquet` — 5-row table.
- `data/exploration/phase4_h6_induction_2_8b_ksweep.log` — captured stdout.

### §H6-causal-ksweep-2.8b-6. Pre-registration form (locked)

Reference-style; no new thresholds. The 2.8B K-sweep is paired with the 410M K-sweep for the dose-response cross-size comparison reported in `causal_dependence.ipynb`. If `ratio_ind` and `ratio_ctrl` continue to track each other across K at 2.8B (matching 410M), the §H6 NULL is **scale-stable across head-count tiers 384 and 1024 on the dose-response axis**, parallel to §H5-causal-3-2.8b's NULL × NULL on the per-metric-at-single-K axis.

---

## Amendment 2026-05-11 — §H4-fullgrid: complete the 40-cell §H4-scaling S-inhibition sweep at Pythia-2.8B

**Posted after §H4-supersede PASS (2026-05-11) and as a paper-strengthening completion of the §H4-scaling chain.** §H4-supersede ran a 10-cell reduced grid at steps {5000, 7000, 10000, 14000, 20000, 29000, 41000, 49000, 59000, 70000} and PASSed both legs (A.timing reversal_rate = 1.000; A.count max_count = 5 at step 29000). The §H4-7-supersede partial cache from the original 40-cell halt holds 8 cells at steps {0, 1, 2, 4, 8, 16, 32, 64} (all top Δ_h ≈ 0). Combined existing: **18 of 40 §H2-1 grid cells**. The remaining **22 cells** at 2.8B are the deliverable of this amendment.

§H4-fullgrid is a paper-strengthening completion of the §H4-scaling 40-cell schedule. It does NOT modify the §H4-supersede PASS verdict (sealed as the registered Track 3 result). It re-evaluates the §H4-2 conjunctive gate on the merged 40-cell data and resolves the count-trajectory dip caveat (§writeup-conv-2) by adding the late-saturation tail (steps 84000–143000).

### §H4-fullgrid-1. Scope (locked)

Pythia-**2.8B**-deduped @ the following **22 cells**:

```
[128, 256, 512,
 1000, 2000, 3000, 4000,
 6000, 8000, 9000,
 11000, 12000, 13000, 15000, 16000, 17000,
 24000, 35000,
 84000, 100000, 120000, 143000]
```

These are exactly the §H2-1 40-cell grid minus the 18 cells already on disk (8 §H4-7-supersede partial + 10 §H4-supersede). The merged 40-cell artifact uses union of all three sources.

### §H4-fullgrid-2. Gate inheritance + re-evaluation (locked)

The §H4-2 conjunctive gate is inherited **verbatim**:
- (A.timing) `P(μ_si^2.8B < μ_si^410m) ≥ 0.95` over B = 1000 paired per-prompt bootstrap.
- (A.count) `max_count_si^2.8B ≥ 5` over the full 40-cell sweep.

Re-evaluation on the merged 40-cell data is the registered acceptance criterion. Two possible outcomes:
- **PASS re-confirmation**: the §H4-supersede 10-cell verdict is robust to the full grid; the count-trajectory dip caveat is resolved by the late-tail data.
- **FAIL re-evaluation**: the §H4-supersede 10-cell PASS was a window-selection artifact; this triggers a §H4-fullgrid-amendment-N follow-up to investigate.

The §H4-supersede 10-cell sealed verdict stands as historical record regardless of the §H4-fullgrid outcome.

### §H4-fullgrid-3. Dip-trajectory analysis (locked deliverable)

The §H4-supersede count trajectory `1 → 2 → 2 → 3 → 3 → 5 → 4 → 4 → 4 → 4` crosses (A.count) ≥ 5 at step 29000 and dips back to 4 by step 70000. The §H4-fullgrid late-tail cells {84000, 100000, 120000, 143000} resolve whether the count:
- **Recovers to 5** by step 143000 — the dip was a mid-emergence fluctuation.
- **Sustains at 4** through convergence — the dip is the steady-state for 2.8B's 1024-head architecture.
- **Drops below 4** — the count saturates further, indicating §H4-supersede was a transient peak.

The §writeup-conv-2 count-trajectory caveat is updated post-data based on the late-tail observation.

### §H4-fullgrid-4. Compute estimate (locked)

22 cells × ~57 min/cell = **~21 hours wall time** at the measured §H4-7-supersede per-cell cost. Two overnight runs (cells 1–11, then cells 12–22). §H4-7-style per-cell-cost escape hatch inherits: halt if any cell exceeds ~115 min/cell (2× baseline).

### §H4-fullgrid-5. Deliverables (locked)

1. `notebooks/_run_phase4_2_8b_s_inhibition_fullgrid_sweep.py` — 22-cell sweep runner.
2. `data/exploration/phase4_2_8b_s_inhibition_fullgrid_sweep.parquet` — 22-row parquet at the new cells (1024 heads × 22 cells).
3. `data/exploration/phase4_2_8b_s_inhibition_fullgrid_per_prompt/` — per-cell `.npz` cache directory for the 22 new cells.
4. `notebooks/_run_phase4_2_8b_fullgrid_analysis.py` — merges 8 §H4-7-supersede + 10 §H4-supersede + 22 §H4-fullgrid cells into a single 40-cell parquet (`phase4_2_8b_s_inhibition_40cell_merged.parquet`); re-runs the §H4-2 gate; produces the count-trajectory dip-resolution report.
5. New section in `h1c_ordering_test.ipynb` §H4-fullgrid verdict cell.

### §H4-fullgrid-6. Pre-registration form (locked)

Reference-style, all numerical thresholds inherited verbatim from §H4-2 / §H4-3 / §H4-5. No new locks beyond the 22-cell list, the merged-parquet schema, and the dip-resolution analysis procedure. Pre-committed before any §H4-fullgrid compute.

A reviewer reading the chronology should see: §H4-scaling → §H4-7-supersede DEFERRED → §H4-supersede PASS → §H4-fullgrid (this amendment) completes the original 40-cell schedule and re-evaluates the gate on the full grid. Three nested registered artifacts: (1) the 8-cell partial cache (§H4-7-supersede), (2) the 10-cell reduced grid PASS (§H4-supersede), (3) the merged 40-cell re-gate (§H4-fullgrid).

---

## Amendment 2026-05-11 — §H1-C-altdetectors: cross-readout consistency of detectors at the §H1-C registered sizes

**Posted as a §H1-C strengthening exercise after §H5-causal NULL × NULL + §H6-causal NULL/artifact-MIXED/NULL findings.** The §H5 + §H6 causal-disjointness results comprehensively falsify the inference-time causal-chain reading of the temporal-emergence ordering. The §H1-C joint sign-test at p = 0.00463 < 0.005 remains as the registered verdict for the temporal claim, but the claim's *interpretation* is now constrained: the ordering reflects training-dynamics, not architectural causation.

A residual question: **is the temporal ordering an artifact of the specific detectors chosen, or does it survive alternative operational definitions of each motif?** This amendment registers a cross-readout consistency exercise to address that question. For each motif, an alternative detector (taken from published prior art) is locked, applied to GPT-2 small to derive a threshold, then applied to Pythia 70M / 160M / 410M at all 40 §H2-1 cells. The §H2-5 joint sign-test is then re-run under each alternative-detector-triple.

If the joint sign-test passes (p < 0.005) under each of the three alternatives, the **temporal-ordering claim is detector-invariant** — a much stronger version of §H1-C. If any alternative produces a different ordering at any size, the claim is **detector-dependent** and disclosed as such.

### §H1-C-altdetectors-1. Alternative detectors (locked)

For each motif, the alternative detector is published prior art that is mechanistically distinct from the locked detector:

**Induction** (locked: Olsson 2022 prefix-matching score, threshold > 0.30 — the QK criterion only).
Alternative: **Olsson 2022 OV-circuit verification (the missing half of the two-criterion definition).** Per-head OV-score = mean, over second-half positions in 50 random-token-repetition sequences (length 100; repeat at position 50; same prompts as the locked QK detector), of the head's direct-logit-attribution magnitude on the prior-occurrence-next-token direction. Mechanically distinct: tests the OV side of induction rather than the QK side.

**Successor** (locked: §SU-1b lift_dla cross-category, threshold τ_lift = 0.13496).
Alternative: **L (2023) argmax-K-of-7 (graded version of the GPT-2 small validation protocol).** For each head, count how many of the 7 day-of-week transitions (`Mon→Tue, Tue→Wed, ..., Sun→Mon`) the head's direct-logit-attribution correctly argmaxes the target day. Per-head K_score ∈ {0, 1, ..., 7}. Mechanically distinct: a binary-per-day argmax test, not a continuous lift magnitude.

**S-inhibition** (locked: §S-1 path-patching Δ_h, threshold τ_strict = 0.0372).
Alternative: **Component-DLA on the (IO−S) direction at the S2 token position** (Wang 2023 §3 structural definition). Per-head score = head's direct-logit-attribution magnitude on the (IO−S) direction, evaluated at the S2 token position, averaged over the same 200-prompt IOI set (BABA + ABBA, seed=0) used by the locked detector. Mechanically distinct: a direct DLA readout, not a frozen-path patching perturbation.

### §H1-C-altdetectors-2. Threshold-locking rule (c1-uniform, locked)

Per-motif threshold derivation, uniform across all three alternatives:

1. **Reference set** = heads passing the **locked** detector for that motif in GPT-2 small. (For induction: prefix-match > 0.30; for successor: lift_dla ≥ 0.13496; for S-inhibition: path-patching Δ_h ≥ 0.0372.)
2. **Alternative threshold** = minimum alternative-detector score across the reference-set heads in GPT-2 small.

Concretely, the three locked thresholds are:

| Motif | Alternative threshold name | Derivation | Locked value (post-GPT-2-validation) |
|---|---|---|---|
| Induction | τ_ind_OV | min OV-score across reference set | **TBD post-validation** |
| Successor | K_min | min K_score across reference set | **TBD post-validation** (integer in {1, ..., 7}) |
| S-inhibition | τ_si_DLA | min Component-DLA at S2 across reference set | **TBD post-validation** |

The three TBD values are locked by a separate amendment update (`§H1-C-altdetectors-2-locked`) immediately after the GPT-2 small validation runner completes, but before any Pythia application of the alternative detectors.

### §H1-C-altdetectors-3. Pythia application (locked)

Apply each alternative detector at Pythia-{70M, 160M, 410M} × all 40 §H2-1 cells. For each (size, motif, step) cell: count heads with alternative-score ≥ threshold. Fit logistic per (size, motif) per §H2-3 tiered handling (emerged ≥ 5, marginal 2-4, censored < 2). Compute μ_alternative.

### §H1-C-altdetectors-4. Acceptance gate (locked)

The §H2-5 joint sign-test is re-run under each alternative-detector-triple:
- Under each triple `(alt_ind, locked_suc, locked_si)`, `(locked_ind, alt_suc, locked_si)`, `(locked_ind, locked_suc, alt_si)`, AND `(alt_ind, alt_suc, alt_si)` (the all-alternative triple).
- For each triple: does the joint sign-test pass at p < 0.005? (i.e., does the directional ordering `μ_ind < μ_suc < μ_si` hold at each of the 3 registered sizes?)

**Acceptance gate**: all four triples pass at p < 0.005.

Failure modes (pre-committed):
- **Full survival**: all four triples PASS. The temporal-ordering claim is detector-invariant.
- **Partial survival**: 1–3 triples PASS. The claim is detector-dependent at the specific failing triple; failure is disclosed in the paper with the specific detector that breaks the ordering.
- **Full failure**: 0 triples PASS. The locked-detector PASS at p = 0.00463 was a detector-specific result; the temporal claim is fundamentally detector-dependent and disclosed as such.

### §H1-C-altdetectors-5. Compute estimate (locked)

GPT-2 small validation: ~1-2 hours wall time (model load + three alt-detector implementations + threshold derivation).
Pythia alt-detector sweeps: 3 alternatives × 3 sizes × 40 cells. Per-cell wall ~5-30s at 70M/160M, ~30-90s at 410M. Total: ~5-10 hours wall time, chainable.
§H2-5 joint sign-test re-runs: ~5-10 minutes (cheap, bootstrap on existing data).

### §H1-C-altdetectors-6. Deliverables (locked)

1. `notebooks/_run_pythia_anchor_altdetectors_validation.py` — GPT-2 small validation runner producing per-head OV-scores, K-scores, and Component-DLA-at-S2 scores. Outputs `data/exploration/gpt2_small_altdetector_validation.parquet`.
2. Three Pythia sweep runners (or one runner with `(motif, size)` loop):
   - `notebooks/_run_phase4_h1c_alt_induction_ov.py`
   - `notebooks/_run_phase4_h1c_alt_successor_argmax.py`
   - `notebooks/_run_phase4_h1c_alt_s_inhibition_compdla.py`
3. `notebooks/_run_phase4_h1c_alt_analysis.py` — re-runs §H2-5 joint sign-test under each alternative-detector-triple; outputs `phase4_h1c_alt_joint_verdict.parquet`.
4. New §H2-5-altdetectors section in `h1c_ordering_test.ipynb`.

### §H1-C-altdetectors-7. Pre-registration form (locked, with deferred numerical thresholds)

This amendment is reference-style for the detector procedures, gate predicates, sweep grids, and Pythia-side application. The three threshold values (τ_ind_OV, K_min, τ_si_DLA) are deferred-locked: their derivation procedure (min alt-score across reference set in GPT-2 small per §H1-C-altdetectors-2) is locked here, but the numerical values are derived from GPT-2 small post-pre-data-commit and registered in a small follow-up amendment (`§H1-C-altdetectors-2-locked`) before any Pythia application. This deferred-lock pattern matches the §SU-tau / §S-tau precedent.

A reviewer reading the chronology should see: §H1-C registered → §H2 sweep specification → §H2-5 PASS at p = 0.00463 with the LOCKED detectors → §H2-9-R scale-dependence reframe → §H5 + §H6 causal-disjointness findings → §H1-C-altdetectors registers the cross-readout consistency test → GPT-2 validation locks the three alt thresholds → Pythia sweeps under each alt detector → §H2-5 joint sign-test re-run under each alt-detector-triple. The §H1-C verdict is then either re-affirmed as detector-invariant, or qualified as detector-dependent with explicit disclosure.

## Amendment 2026-05-12 — §H1-C-altdetectors-2-r-supersede: replace (c1-uniform) with (c2-percentile) threshold rule

**Posted post-GPT-2-small-validation, pre-Pythia-sweep.** The §H1-C-altdetectors-2 (c1-uniform) rule — alt-threshold = min alt-score across heads passing the locked detector in GPT-2 small — yielded **trivially permissive thresholds** on the GPT-2 small validation run (see `data/exploration/gpt2_small_altdetector_validation.parquet` produced by `notebooks/_run_pythia_anchor_altdetectors_validation.py`):

| Motif | (c1-uniform) threshold | Mechanism of failure |
|---|---|---|
| Induction (τ_ind_OV) | −21.186 | Reference set (QK prefix-match > 0.30, 15 heads) is contaminated by Negative-Name-Movers (L10H7 OV=−21.19, L11H10 OV=−13.16) that have induction-pattern QK but suppress the induction prediction at OV. Min is dominated by NNM. |
| Successor (K_min) | 0 | Reference set (lift_dla ≥ 0.13496, 8 heads) includes lift-passing heads (L6H5) that argmax 0 of 7 day-of-week transitions. The argmax-K-of-7 protocol is much stricter than the lift_dla cross-category aggregate. No head in GPT-2 small exceeds K=3 of 7. |
| S-inhibition (τ_si_DLA) | −0.098 | Reference set (Δ_h ≥ 0.0372, 3 heads) includes L8H6 whose CompDLA-at-S2 is −0.098: S-inhibition is mechanistically a suppression operation, so the head's direct contribution to (IO − S) at S2 can be negative. |

The (c1-uniform) rule failed because the locked detectors and the alt detectors do not have a clean inclusion relationship: the locked detectors are **behavioral screens** (broad), and the alt detectors are **mechanism verifications** (strict and narrow). They identify overlapping but non-identical head populations in GPT-2 small. Taking the **min** alt-score across the locked-reference-set is dominated by edge-case heads where the alt-detector strongly disagrees with the locked detector — yielding thresholds so permissive that ~100% of heads pass at any Pythia checkpoint, which would make the §H2-5 joint sign-test re-run statistically uninformative.

This supersede amendment registers a corrected threshold rule **before any Pythia application of the alt detectors** (no data has been collected under the new rule at the time of this lock). The original §H1-C-altdetectors amendment (which specified the alt-detector procedures and acceptance gate) remains in force; only §H1-C-altdetectors-2 (threshold-locking rule) is superseded here.

### §H1-C-altdetectors-2-r-1. (c2-percentile) rule (locked)

For each alt-detector, the threshold is the **95th percentile** of the alt-score distribution pooled across all heads in GPT-2 small. This matches the §SU-3 precedent (95th-percentile-of-pooled-null for the locked successor threshold) and produces meaningful, roughly-comparable head counts across alt-detectors (~7–9 heads each at GPT-2 small).

Concretely:

| Motif | Alt-threshold | Derivation | Locked numerical value |
|---|---|---|---|
| Induction | τ_ind_OV | 95th percentile of OV-score across 144 GPT-2 small heads | **+13.592629** |
| Successor | K_min | ceil(95th percentile of K-score across 144 GPT-2 small heads) | **2** (raw 95th-pct=2.0; K_score ∈ {0,…,7} integer-valued; ceil to next integer if non-integer) |
| S-inhibition | τ_si_DLA | 95th percentile of CompDLA-at-S2 across 144 GPT-2 small heads | **+0.247095** |

Numerical values are derived in `data/exploration/gpt2_small_altdetector_validation.parquet`. A Pythia head passes the alt-detector iff its alt-score ≥ the locked threshold (strict-greater-than for QK induction is unchanged; alt-detectors all use ≥).

### §H1-C-altdetectors-2-r-2. Pass-count + cross-tab sanity at GPT-2 small (locked, informative)

At the (c2-percentile) thresholds, GPT-2 small head-counts are:

- Alt-induction (OV ≥ +13.59): **8 heads pass**
- Alt-successor (K ≥ 2): **9 heads pass**
- Alt-S-inhibition (CompDLA-S2 ≥ +0.247): **8 heads pass**

Cross-tab with locked detectors on GPT-2 small (out of 144 heads):

| Motif | Both pass | Only locked | Only alt | Neither |
|---|---|---|---|---|
| Induction | 8 | 7 | 0 | 129 |
| Successor | 2 | 6 | 7 | 129 |
| S-inhibition | 2 | 1 | 6 | 135 |

Induction shows alt ⊂ locked (alt is a strict mechanism-verified subset). Successor and S-inhibition show substantial divergence — alt and locked identify substantially different head populations. This is the right shape for a detector-invariance test: the Pythia §H2-5 re-run will be diagnostic rather than trivial.

### §H1-C-altdetectors-2-r-3. Reason this is a supersede, not a violation

The original §H1-C-altdetectors-2 (c1-uniform) rule's pre-registration was wrong about how to operationalize alt-thresholds for cross-readout consistency. The failure is in the **threshold rule** (a methodological-detail), not in the **acceptance gate or scientific claim**. The acceptance gate (§H1-C-altdetectors-4: all four detector-triples pass the joint sign-test at p < 0.005) is unchanged. No data has been collected under either (c1-uniform) or (c2-percentile) thresholds for Pythia — both are pre-registered before any Pythia compute. The supersede is registered openly with full disclosure of the (c1-uniform) failure mode and the GPT-2-small validation evidence.

The §H1-C-altdetectors-2-r-supersede precedent matches the §H4-supersede / §H5-causal-3-supersede pattern: a method-detail correction registered before data collection, with the original rule retained in the chronology and the supersede traceable to its motivation.

### §H1-C-altdetectors-2-r-4. Effect on later sections (locked)

- §H1-C-altdetectors-3 (Pythia application): unchanged in scope. Apply each alt-detector at Pythia-{70M, 160M, 410M} × all 40 §H2-1 cells; for each (size, motif, step) cell, count heads with alt-score ≥ the (c2-percentile) threshold.
- §H1-C-altdetectors-4 (acceptance gate): unchanged. Joint sign-test re-run under four triples; all four must pass p < 0.005 for full survival.
- §H1-C-altdetectors-5–7: unchanged.

### §H1-C-altdetectors-2-r-5. Provenance

Validation runner: `notebooks/_run_pythia_anchor_altdetectors_validation.py` (committed pre-data).
Validation outputs (committed): `data/exploration/gpt2_small_altdetector_validation.parquet`, `data/exploration/gpt2_small_altdetector_per_head.npz`.
Threshold derivation: `df[col].quantile(0.95)` over all 144 heads, with K_min ceil-rounded to the next integer.

## Amendment 2026-05-12 — §H1-C-altdetectors-2-rr-supersede: reframe as post-hoc robustness with within-Pythia frozen thresholds

**Posted post-data (after the three Pythia × 40-cell alt-detector sweeps committed under §H1-C-altdetectors-2-r-supersede ran to completion).** The (c2-percentile) thresholds derived from GPT-2 small (τ_ind_OV = +13.59, K_min = 2, τ_si_DLA = +0.247) did not transfer to Pythia. At all three Pythia sizes, the GPT-2-calibrated absolute thresholds produced **trivially degenerate Pythia sweeps**:

| Motif (alt-detector) | 70m max alt-score over training | 160m max | 410m max | GPT-2 threshold |
|---|---|---|---|---|
| Induction (OV) | +6.39 | +15.70 | +7.20 | +13.59 (only 1 head ever crosses, at 160m step 20k) |
| Successor (argmax-K) | K=3 max | K=7 max | K=7 max | K=2 (but 73 heads at 410m step 0 already pass — *de-emergence*) |
| S-inhibition (CompDLA-S2) | +0.0035 | (TBD) | +0.190 | +0.247 (almost no head ever crosses) |

Two distinct failure modes:

1. **Cross-family magnitude transfer fails.** OV-score and CompDLA-S2 magnitudes depend on unembedding/layernorm scale and on how many heads divide the circuit work. Pythia has different |W_U| and 2.7× more heads at 410M than GPT-2 small at the same nominal head count threshold — so individual head magnitudes are systematically smaller. The GPT-2-95th-percentile threshold sits *above the maximum-ever* alt-score at most Pythia sizes.

2. **Argmax-K-of-7 is de-emergent, not emergent.** At random initialization, ~26% of heads pass K ≥ 2 by chance (expected count ~100 at 410M's 384 heads; observed 73 at step 0). As training progresses, heads specialize → most heads' DLA toward day tokens converges to ~0 → argmax becomes noise-driven on tiny values → accidental K=2 hits stop happening. n_pass *decreases* with training, opposite to the emergence direction.

This is a real, scientific finding about cross-family detector transfer (see also Tigges et al. 2024, who normalize component scores over checkpoints rather than transferring raw cross-model thresholds; Pythia's shared-data-order design (Biderman et al. 2023) is specifically meant for within-family training-dynamics work, not cross-family threshold transfer). It is not evidence against §H1-C — the locked-detector joint sign-test passes at **p = 0.00463** and remains the primary result.

This amendment supersedes §H1-C-altdetectors-2 and §H1-C-altdetectors-2-r by reframing the alt-detector exercise as **post-hoc robustness / measurement-invariance sensitivity**, not as a confirmatory pre-registered second proof. The pre-validation analysis below + the within-Pythia frozen-threshold scheme is registered as the operational definition of "robustness" for the §H1-C-altdetectors-VERDICT writeup.

### §H1-C-altdetectors-2-rr-1. Status reclassification (locked)

The §H1-C-altdetectors test is reclassified as:

- **Primary finding (unchanged)**: `induction → successor → S-inhibition` at p = 0.00463 with the LOCKED detectors. Status: PRE-REGISTERED & PASS.
- **Robustness appendix (new framing)**: measurement-invariance sensitivity analysis using three alt-detectors under within-Pythia frozen thresholds. Status: POST-HOC & exploratory. Disclosed as such in writeup.

The acceptance gate in §H1-C-altdetectors-4 (all four detector-triples pass at p < 0.005) is **withdrawn** as a confirmatory test. The joint sign-test is still computed under each triple as an exploratory data point, but no triple has confirmatory weight.

### §H1-C-altdetectors-2-rr-2. Final-checkpoint pre-validation (locked, exploratory)

Before any alt-detector trajectory is meaningful, the alt-detector must demonstrate at the **final checkpoint** (step 143000) that it identifies the same motif as the locked detector. The pre-validation metric is **top-K overlap**, where K = locked-detector's pass-count at step 143000 for that (size, motif).

Final-checkpoint pre-validation (observed):

| Motif × size | K_locked | top-K overlap with alt | Disposition |
|---|---|---|---|
| Induction 70m | 6 | 5/6 (83%) | usable |
| Induction 160m | 17 | 11/17 (65%) | usable |
| Induction 410m | 19 | 11/19 (58%) | usable |
| Successor 70m | 2 | 0/2 (0%) | unreliable |
| Successor 160m | 3 | 3/3 (100%) | usable |
| Successor 410m | 2 | 0/2 (0%) | unreliable |
| S-inh 70m | 1 | 0/1 (0%) | unreliable |
| S-inh 160m | 3 | 1/3 (33%) | partial |
| S-inh 410m | 2 | 1/2 (50%) | partial |

Disposition rule: ≥ 50% overlap = usable; 30–50% = partial; < 30% = unreliable. Robustness trajectory is reported only for usable cells; unreliable cells are disclosed in the writeup but their joint-sign-test contribution is annotated as "alt-detector ≠ locked-motif at convergence."

### §H1-C-altdetectors-2-rr-3. Frozen-threshold scheme (locked, exploratory)

For each (size, motif, alt-detector), the frozen threshold is **the K-th-largest alt-score at step 143000**, where K = locked-detector's final pass-count. (The exact threshold value is the alt-score of the K-th-ranked head at convergence; heads at rank ≤ K at convergence pass; everything below doesn't.)

The frozen threshold is **applied backward** over all 40 §H2-1 checkpoints — every earlier cell uses the same absolute cutoff. This avoids two failure modes:
- It does *not* recompute the threshold at every checkpoint (which would force every cell to have heads "passing" and erase emergence structure).
- It does *not* use a cross-family threshold (which fails per §H1-C-altdetectors-2-rr-0).

A sensitivity check at top-5% (K' = ceil(n_layers × n_heads × 0.05)) is computed in parallel as a robustness-of-the-robustness.

### §H1-C-altdetectors-2-rr-4. Emergence-step definition (locked, exploratory)

Under the frozen threshold, the alt-detector emergence step for (size, motif, alt-detector) is the **smallest step where pass-count ≥ K / 2** (half the final-checkpoint count). This is the half-final-count crossing.

The temporal-ordering check then asks: at each Pythia size, is `μ_alt_ind < μ_alt_suc < μ_alt_si` (the analogous ordering on alt-emergence-steps)?

### §H1-C-altdetectors-2-rr-5. Rank-only secondary view (locked, exploratory)

Independent of any threshold, define `top_K_final` = K heads with highest alt-score at step 143000 (where K = locked-final-count). At each earlier checkpoint, compute the **rank overlap**:

`overlap_t = |alt_top_K_at_step_t ∩ top_K_final| / K`

The rank-only emergence step is the smallest step where `overlap_t ≥ 0.5`. This view is **magnitude-free** — it asks "when does the mature motif population become rank-identifiable?" rather than "when does the magnitude cross a threshold?"

### §H1-C-altdetectors-2-rr-6. Reporting (locked)

The writeup reports:
1. Pre-validation table (final-checkpoint overlap per cell).
2. Per-(size, motif) emergence step under frozen-threshold scheme.
3. Per-(size, motif) emergence step under rank-only scheme.
4. Per-(size, motif) emergence step under top-5% scheme (sensitivity).
5. Joint-sign-test under each scheme × the four detector triples — annotated as exploratory.
6. Cells with "unreliable" pre-validation are visually flagged but reported.

Interpretation rules:
- If the temporal ordering holds under both **frozen-threshold (final-K)** and **rank-only** schemes at all three sizes → robustness PASSES. Strong evidence the §H1-C ordering is detector-invariant.
- If ordering holds under rank-only but not frozen-threshold → partial robustness. Reported as such.
- If ordering fails under both → §H1-C ordering may be detector-dependent; documented openly. Note this does not invalidate the p = 0.00463 primary result; only the robustness claim is qualified.

### §H1-C-altdetectors-2-rr-7. Provenance + literature context

- Tigges et al. 2024 (NeurIPS): tracks component emergence within Pythia, normalizes scores over checkpoints — within-family, not cross-family.
- Biderman et al. 2023 (Pythia paper): designed for within-family training-dynamics work.
- Gould et al. 2024: documents successor heads across GPT-2/Pythia/Llama at *convergence*, does not claim cross-family threshold transfer.
- TransformerLens documentation: logit attribution magnitudes depend on layernorm and unembedding choices; raw projection scales are not model-family invariant.

The within-Pythia frozen-threshold scheme matches the within-family precedent. Cross-family threshold transfer (the (c2-percentile)-on-GPT-2-small attempt under §H1-C-altdetectors-2-r) is now documented as an open methodological problem with no clean solution in the public literature.

### §H1-C-altdetectors-2-rr-8. What changes in writeup vs. original framing

The original §H1-C-altdetectors-7 promise was "either re-affirmed as detector-invariant, or qualified as detector-dependent with explicit disclosure." That binary is now refined:
- Within-Pythia robustness (frozen-threshold + rank-only): can be tested → answers the within-family detector-invariance question.
- Cross-family robustness: documented as an unsolved methodological problem; deferred.

The §H1-C verdict in writeup remains: PRE-REGISTERED & PASS at p = 0.00463 with locked detectors. The §H1-C-altdetectors-VERDICT is added as a robustness appendix, post-hoc, with the disposition determined by the analysis runner's output.
