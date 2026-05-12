# Temporal Emergence Order is Not an Architectural Causal Chain: Pre-Registered Evidence from Pythia 70M–2.8B

**Author.** Cameron Berger, camberger203@gmail.com  
**Version.** Workshop draft v0.2 (2026-05-11). Not yet submitted.
**Target venue.** BlackboxNLP @ EMNLP 2026 (fallback ICLR MI Workshop 2027).
**Status.** Pre-registered (`HYPOTHESIS.md`, anchor commit prior to any pilot code). All three tracks have landed on registered targets: Track 1 emergence PASS at p ≈ 0.00463 (§H2-5); Track 2 causal-disjointness NULL × NULL at Pythia-410M + Pythia-2.8B (§H5-causal / §H5-causal-2 / §H5-causal-3-2.8b); Track 3 head-count-axis scaling PASS at Pythia-2.8B (§H4-supersede, reversal_rate = 1.000, max_count = 5).

---

## Abstract

**Temporal emergence order is real, but it is not a simple architectural causal chain.** Across Pythia 70M–2.8B (head-count tiers 48–1024), we pre-register and test three falsifiable tracks. **Track 1** registers an emergence-ordering claim across 3 Pythia sizes: μ_induction < μ_successor < μ_S-inhibition, with a locked joint sign-test gate at p < 0.005. The gate passes (p ≈ 0.00463); the strength of confirmation scales with model size (vacuously satisfied at 70M through right-censoring, marginal at 160M, robust at 410M; §H2-9-R reframe). **Track 2** asks whether the temporal ordering reflects an inference-time causal chain. We mean-ablate the top-5 successor heads at convergence and read S-inhibition's response on two converging metrics: the §S-1 path-patching scalar and the IO−S logit difference. At Pythia-410M and Pythia-2.8B, both metrics return NULL with tight bootstrap CIs — successor and S-inhibition are causally disjoint at inference time. At Pythia-1B (a head-count regression from 410M's 384 heads to 128 heads), Metric A NULL replicates while Metric B becomes generically ablation-sensitive (suc and ctrl both drop ~21%); we interpret this as a narrow-architecture readout-specificity artifact, not a falsification of disjointness. **Track 3** asks whether the scale-dependent S-inhibition emergence pattern continues on a head-count axis. At Pythia-2.8B (1024 heads, head-count tier ~3× above 410M), S-inhibition emergence accelerates by ~2.7× (μ_si^2.8B ≈ 9,021 vs μ_si^410m ≈ 24,088) and the head count crosses the registered (A.count) ≥ 5 gate transiently at step 29,000 (paired bootstrap reversal_rate = 1.000 over B = 1,000 replicates; §H4-supersede). Three further findings tighten the interpretation: a depth–temporal asymmetry where S-inhibition sits *shallower* than successor at 160M / 410M; identity churn for the later motifs; and structural-reuse top-5 overlap that vanishes at 410M. The §H1-C compositional reading — *"later-emerging motifs sit at deeper layers and consume earlier motifs' outputs"* — is directly falsified by the Track 2 NULL × NULL at 410M + 2.8B. The temporal ordering is real and the head-count scaling is real, but the inference-time chain is not. As a side finding (Appendix G), McDougall-style copy-suppression heads are absent in Pythia-410M and Pythia-1B, answering an open question from McDougall et al. (2024). All code, sweep parquets, verdict parquets, and notebooks are released; a single notebook reproduces each main figure.

---

## 1. Introduction

The mechanistic-interpretability literature has produced an increasingly rich catalogue of named circuits: induction heads (Olsson et al., 2022), successor heads (Gould et al., 2024), copy-suppression heads (McDougall et al., 2024), the Indirect Object Identification circuit and its S-inhibition component (Wang et al., 2023), and others. A parallel line of work studies how these circuits *come into existence* during training. Olsson et al. (2022) and Singh et al. (2024) characterize the formation of induction heads as a phase transition; Edelman et al. (2024) develop a statistical theory of induction-head emergence; Tigges et al. (2024) trace the IOI and greater-than circuits across the entire Pythia checkpoint suite. Three things are conspicuously absent from the second body of work. First, except for IOI, the named multi-component circuits and the named single-head motifs have been studied in isolation; we lack a controlled comparison of *when* different motifs emerge in the *same* model. Second, the existing emergence-ordering observations have never been mechanistically tested at convergence — when a paper reports "motif A emerges before motif B," that is consistent with two distinct readings: (i) an architectural causal chain where B reads A's output in the forward pass, or (ii) convergent training dynamics where A and B form in order for optimizer-path reasons but operate independently at inference time. Third, McDougall et al. (2024) explicitly named *"do copy-suppression heads exist in Pythia and Llama?"* as future work; this has not been answered.

This paper does all three. We pre-register and test three converging tracks across Pythia (70M, 160M, 410M, 1B, 2.8B). **Track 1 (emergence ordering)** registers a falsifiable joint sign-test on the ordering μ_induction < μ_successor < μ_S-inhibition across the 3 small sizes; the gate passes (p ≈ 0.00463 < 0.005). **Track 2 (inference-time causal-disjointness)** mean-ablates the top-5 successor heads at convergence and reads S-inhibition's response on two converging metrics; the result is NULL × NULL at Pythia-410M and Pythia-2.8B (head-count tiers 384 and 1024). **Track 3 (head-count-axis scaling)** registers a conjunctive gate on whether S-inhibition's emergence accelerates and its count saturation breaks at Pythia-2.8B's 1024-head architecture; the gate passes on both legs (reversal_rate = 1.000, max_count = 5). The synthesis is the title: temporal emergence order is real (Track 1), but it is not a simple architectural causal chain (Track 2), and the head-count-axis scaling extends the emergence pattern beyond the 3 registered sizes (Track 3).

**Five notes on form.** *(i) The path pivot.* The pre-registration named two paths. Path A studied copy-suppression as the third motif; Path C substituted S-inhibition. The Week-1 pilot applied the McDougall two-criterion detector to Pythia-410M-deduped at step 143000 on a 7.5k-token canonical corpus; one head (L2H8) numerically passed both strict thresholds, but qualitative inspection showed it functions as a previous-token / induction-precursor head — its corpus-wide ablation effect on duplicate-token logits is −0.009 (i.e., it *promotes* duplicates), the opposite sign of GPT-2 small L10H7's textbook +0.032. The decision rule (`HYPOTHESIS.md` §"Pilot decision rule (Week 1)") registered Path C. The paper therefore tests H1-C; the negative result on Path A is reported in Appendix G as an answer to McDougall's open question.

*(ii) Pre-registration with dated amendments.* The hypothesis document carries 20 dated amendments tracking detector-specification work, threshold locks, three post-data interpretation reframes (§H2-9-R, §H4-7-supersede, §writeup-conv), and the three-track structure described above. All amendments are committed before the analyses they govern, with one explicitly acknowledged exception: §H5-causal-3-record canonicalizes a Pythia-1B causal-dependence run that was executed on a feature worktree before the corresponding amendment was canonicalized; the gap is disclosed in §H5-causal-3-record-4 and re-disclosed here. The full audit trail is in Appendix C.

*(iii) The 1B head-count regression.* Pythia-1B has 128 attention heads (16 layers × 8 heads), narrower than Pythia-410M's 384 heads (24 × 16). Head count is non-monotonic in parameter count across the Pythia suite; §H4-1 argues that for head-level circuit analysis, head count — not parameter count — is the operationally relevant scaling axis. We treat Pythia-1B as a head-count regression rather than a scale-up, and Pythia-2.8B (32 × 32 = 1024 heads) as the next head-count tier above 410M. This framing is registered in §H4-scaling.

*(iv) Scale-dependence in Track 1.* The pre-registered joint sign-test gate **passes**: p ≈ 0.00463 is below the locked 0.005 threshold and the predicted ordering holds in all three sizes. But the supporting evidence is *not* uniformly strong across sizes — 4 of 9 (size, motif) cells right-censor or hit the upper logistic-fit sentinel, and the cleanest per-size confirmation is at 410M. We register this heterogeneity in the post-data interpretation reframe (§H2-9-R) and report both the joint-claim verdict and the scale-dependent decomposition. The reframe does *not* change the gate; it changes the emphasis in the writeup.

*(v) Track 3 framing reframe (post-PASS).* §H4-supersede was registered pre-data as a "scaling appendix / secondary result"; the conservative framing was to ensure the paper would ship on Tracks 1 + 2 even if §H4-supersede DEFERRED a second time. Post-PASS, §writeup-conv-2 (analog of §H2-9-R) upgrades the framing to "third converging substantive result" — the registered gate is unchanged, the falsification target is not weakened, but the paper headline now includes Track 3 as a converging result. The framing upgrade is registered post-data, before any prose was written claiming a third-track headline.

**Contributions.**

1. We pre-register and test a cross-motif emergence-ordering claim across three Pythia scales, and report a sign-test pass (p ≈ 0.00463) on a 40-cell × 3-size × 3-motif sweep with bootstrap-95% CIs on the emergence step μ. The strength of confirmation scales with model size: induction and successor emerge robustly across all three sizes; S-inhibition emerges robustly only at 410M, marginally at 160M, and not at all (right-censored) at 70M during training.
2. **We directly test whether the temporal emergence ordering corresponds to an inference-time causal chain at convergence, and falsify the chain on two converging metrics across two head-count tiers.** Mean-ablating the top-5 successor heads at Pythia-410M and Pythia-2.8B convergence leaves the §S-1 path-patching scalar and the IO−S logit difference both within their NULL bands (ratio_suc ≈ ratio_ctrl ≈ 1.0; tight bootstrap CIs). The §H1-C compositional reading — *"corrective mechanisms emerge after the copying behaviors they correct"* — is directly refuted in the forward-pass-routing sense. At Pythia-1B (a 128-head head-count regression), Metric A NULL replicates but Metric B is MIXED; we interpret this as a narrow-architecture readout-specificity artifact, not a falsification of disjointness.
3. **We pre-register and test a head-count-axis scaling argument at Pythia-2.8B.** The conjunctive gate (paired bootstrap reversal_rate ≥ 0.95 for S-inhibition's emergence step + max_count ≥ 5 for full-fit regime entry) passes on both legs (reversal_rate = 1.000, max_count = 5). S-inhibition emerges ~2.7× faster at 2.8B's 1024-head architecture than at 410M's 384 heads (μ_si^2.8B ≈ 9,021 vs μ_si^410m ≈ 24,088).
4. We surface three findings that complicate the simplest "compositional layering" reading: a temporal-vs-architectural depth asymmetry (S-inhibition sits *shallower* than successor at 160M / 410M, the opposite of what compositional-stacking predicts); turnover of top-head identity across training for the later motifs; and disappearance of multi-motif top-5 head overlap at the largest registered Track-1 size.
5. We answer McDougall et al. (2024)'s open question (Appendix G): copy-suppression heads in McDougall's strict two-criterion sense are absent in Pythia-410M-deduped at the final checkpoint, and also absent at Pythia-1B's full 128-head population. The numerically-passing 410M head (L2H8) is mechanistically a previous-token head, not a copy-suppressor.
6. We release detector implementations, the canonical motif-sweep parquets across 5 sizes (long-format per-head per-prompt scores), the §H5-causal anchor parquets across 3 sizes × 2 metrics, the §H4-supersede 10-cell sweep + bootstrap-μ parquets, and one notebook per main figure.

## 2. Background

### 2.1 Notation and primitives

A transformer attention head is a (layer, head) pair indexed *(L, H)*. Pythia-410M has 24 layers × 16 heads = 384 attention heads; 160M has 12 × 12 = 144; 70M has 6 × 8 = 48. Each head has independent **QK** and **OV** circuits (Elhage et al., 2021). The QK circuit *(W_Q W_K^⊤)* acts on query/key residual streams to produce attention weights — it determines *where* a head attends. The OV circuit *(W_O W_V)* acts on attended values to produce the head's contribution to the residual stream — it determines *what* a head writes once it has decided where to attend.

The **residual stream** is the cumulative sum of all layer contributions at a given token position; each attention head and MLP reads from it and writes back to it. Most mechanistic-interpretability work treats the residual stream as a decomposable object.

**Direct logit attribution (DLA)** of a head on a token *t* at position *i* is the head's contribution to the final logit on *t*, computed by projecting the head's residual-stream output at position *i* onto the unembedding direction for *t*. Positive DLA means the head pushes the token's logit *up*; negative DLA means it pushes it *down*. We use DLA both to detect successor heads (which positively boost the next ordinal element) and to characterize copy-suppression and S-inhibition (which write negative DLA on a duplicate-name token).

### 2.2 The three motifs

**Induction heads** (Olsson et al., 2022) implement the *(A B … A → B)* pattern: given a token *A* whose previous occurrence was followed by *B*, the head attends from the current *A* to the position right after the previous *A*, copying *B*'s information into the residual stream. Detection is via Olsson's prefix-matching score on random repeated sequences (length 100 with a repeat at position 50; mean attention from positions 51–100 to the token following the previous occurrence of the current token). The detector validates against well-known GPT-2 small induction heads.

**Successor heads** (Gould et al., 2024) implement an abstract "+1 in ordinal space" via the OV circuit: given a context indicating an ordinal sequence (Mon, Tue, Wed, …), the head's OV writes the embedding of the next ordinal element. Detection is via cross-category direct logit attribution at the prediction position across days, months, numerals (digit and word forms), and letters. Cross-category breadth is the load-bearing requirement: it distinguishes a true successor head from a memorizer of one specific ordinal sequence. The validation target is GPT-2 small **L9H1** (L, 2023; cited by Gould et al., 2024 §5 in the cross-model successor scatter). We note that the original project brief misattributed this validation target; the corrected attribution is recorded in Amendment §SU-0.

**S-inhibition heads** (Wang et al., 2023) implement the suppression component of the IOI circuit: in prompts of the form "When *N1* and *N2* went to the *PLACE*, *N3* gave a *OBJECT* to *_*", with *N3* matching one of *{N1, N2}*, S-inhibition heads suppress attention from the END query to the duplicate (subject) name at downstream Name Mover heads, allowing the Name Movers to promote the indirect-object name. Detection is causal: a candidate sender head *h* is judged by a path-patching scalar Δ_h that measures, under ABC corruption of the second-clause subject, how much the path *h → Name Mover → output* changes the Name Movers' END-position attention to S2 versus IO. Wang's published S-inhibition heads in GPT-2 small are *{L7H3, L7H9, L8H6, L8H10}*.

### 2.3 The Pythia checkpoint suite

Pythia (Biderman et al., 2023) releases 154 checkpoints per model size, sampled at *step 0, 1, 2, 4, 8, …, 512* and then every 1000 steps from *step 1000* through *step 143000*. We use the *deduped* variants of 70M, 160M, and 410M, sampled at 40 log-spaced steps per size. Pythia is one-seed-per-size, so all variance estimates in this paper are over data resamples (per-prompt bootstrap), not seeds.

### 2.4 Prior emergence-of-circuits work

The closest direct ancestors of this paper are (i) Olsson et al. (2022), who locate the induction-head phase transition in their own training runs at the per-token loss bend; (ii) Singh et al. (2024), who decompose the prerequisites for induction-head formation and validate them on Pythia; (iii) Edelman et al. (2024), who derive a statistical-induction-heads emergence theory; and (iv) Tigges et al. (2024), who trace IOI and the greater-than circuit across the Pythia checkpoint suite. None of these compares the emergence times of qualitatively different motifs *within* a fixed model in a controlled way; that is the gap this paper fills.

Work on copy-suppression (McDougall et al., 2024) and on path patching for IOI (Wang et al., 2023; Goldowsky-Dill et al., 2023; Hanna, 2024) supplies the detector machinery we re-use without re-deriving.

## 3. Methods

### 3.1 Pre-registration

We commit `HYPOTHESIS.md` and `PILOT_RESULTS.md` (template) before any pilot code runs. The hypothesis specifies the operational definition of the emergence step μ, the falsification criterion (joint sign test, p < 0.005), the pilot decision rule that gates Path A vs Path C, and pre-committed limitations. Subsequent amendments are dated and committed before the analyses they govern; numerical thresholds that depend on observed reference distributions (τ_strict for S-inhibition, τ_lift for successor) are split into a procedure-now / number-later commit pair so that pre-registration discipline is preserved without forcing a fictional pre-data threshold. The full amendment list is in Appendix C; the relevant ones for the body of the paper are:

- **§SU-1b:** the successor detector uses *lift = real DLA − null DLA* rather than raw real DLA. The supersede was triggered by a pre-formal-validation smoke test showing that the raw-DLA detector ranked L9H1 at #36 of 144 because category-token-boost behavior was conflated with successor mechanism; the lift form recovers L9H1 as rank #1.
- **§S-tau:** τ_strict = 0.0372 (the minimum Δ_h across Wang's four GPT-2 S-inhibition heads), locked after the GPT-2 small validation screen.
- **§SU-tau:** τ_lift = 0.13496 (the 95th percentile of pooled per-head lifts in GPT-2 small), locked after the successor validation screen.
- **§S-5c:** the S-inhibition validation passed rank-only (Wang's four = ranks #1–#4 of 144) but failed the σ-separation criterion by 0.019σ because L8H6's outlier Δ_h inflates bulk SD when included per the locked no-leave-one-out rule. We record the failure, accept the detector on rank-strength grounds, and drop the σ leg from forward use.
- **§H2-9-R:** the post-data interpretation reframe documented after Phase 2 review. The registered gate passes, but the writeup foregrounds scale-dependence, the depth-temporal asymmetry, and identity churn instead of the joint claim.
- **§H3-scale, §H4-scaling, §H4-7-supersede, §H4-supersede:** the head-count-axis scaling chain registered before any 1B / 2.8B compute. §H3-scale targets Pythia-1B and returns the REGR pattern (1B is a head-count regression, not a scale-up). §H4-scaling targets Pythia-2.8B (the next head-count tier above 410M). §H4-7-supersede halts the original 40-cell 2.8B S-inhibition sweep at 8/40 cells under the §H4-7 per-cell-cost escape hatch (DEFERRED). §H4-supersede registers a reduced 10-cell grid `[5000, 7000, 10000, 14000, 20000, 29000, 41000, 49000, 59000, 70000]` against the verbatim §H4-2 gate predicates.
- **§H5-causal, §H5-causal-2, §H5-causal-3-record, §H5-causal-3-2.8b:** the inference-time causal-dependence chain registered before any §H5 compute (with one acknowledged exception, §H5-causal-3-record, which canonicalizes a Pythia-1B run that predated its amendment; the discipline gap is explicit). The protocol mean-ablates the top-5 successor heads on hook_z per length group and reads back two converging metrics: the §S-1 path-patching scalar (§H5-causal) and the IO−S logit difference at END (§H5-causal-2). Both metrics are run at Pythia-410M (the converging-evidence anchor), Pythia-1B (head-count regression), and Pythia-2.8B (head-count tier 1024).
- **§writeup-conv:** post-data documentation-hygiene amendment (2026-05-11). Locks the Track-numbering convention (Track 1 = Emergence, Track 2 = Causal-disjointness, Track 3 = Scaling). Upgrades §H4-supersede's pre-data "scaling appendix" framing to "third converging substantive result" post-PASS, analogous to §H2-9-R. Rectifies two chronology errors in earlier amendment text. No numerical thresholds are introduced or modified.

### 3.2 Detectors

**Induction.** Olsson prefix-matching score on 50 random-token repeated sequences (length 100, repeat at position 50). A head's score is the mean attention from positions 51–100 to the position right after the previous occurrence of the current token. Threshold *>0.3* (PROJECT_BRIEF §4); the 0.3 cut validates against well-known GPT-2 small induction heads. Day-1 validation on Pythia-410M @ step 143000 finds 11 heads with score *>0.5* and 19 with score *>0.3*; the top three (L11H14, L11H2, L7H1) cause +0.401 NLL increase on the random-repetition task when L11H14 is ablated alone, and +0.742 when ablated together (Appendix B.1).

**Successor (lift form, §SU-1b).** For each head, lift = mean over four ordinal categories (days, months, numerals, letters) of *(real DLA − null DLA)* on three-context "{c1}, {c2}, {c3}, " prompts predicting the next ordinal element (first-token DLA per §SU-2 to handle multi-token items consistently across BPE variants). The null is computed by within-category prefix permutation (one fixed seed-pinned permutation per (category, base prompt)). Threshold τ_lift *= 0.13496* — the 95th percentile of the pooled per-head lifts on GPT-2 small. Validation: L9H1 ranks #1 of 144 by lift (lift = +0.392; real DLA = +0.615; null DLA = +0.223). Independent corroboration: under L (2023)'s exact argmax-within-7-days protocol, L9H1 is the unique 7-of-7 head among 144 in GPT-2 small (Appendix B.2).

**S-inhibition (path patching, §S-1..§S-4).** Wang-style path patching with frozen paths on a 200-prompt IOI set (100 BABA + 100 ABBA, seed 0). Receiver heads (Name Movers) are identified per-model as the top-4 by component-DLA on the *(IO − S)* logit difference; on GPT-2 small this yields *{L9H9, L9H6, L10H0, L10H6}*, three of which match Wang's published Name Movers (the additional L10H6 is recorded as the §S-3-anticipated divergence). For a candidate sender *h*, the corruption is ABC at position 3 only (replacing N3 with a fresh single-token name); the Δ_h scalar averages over the four NMs the change *(patched − clean) attn(NM_END → S2)* minus *(patched − clean) attn(NM_END → IO)*. Threshold τ_strict *= 0.0372* (the minimum Δ_h across Wang's four heads). Validation passes by rank: Wang's four heads occupy ranks #1–#4 of 144 in GPT-2 small (top-8 inclusion gate satisfied; σ-separation gate fails by 0.019σ and is supplementarily accepted per §S-5c, Appendix B.3). Wang's min Δ_h (L7H3 = 0.0372) exceeds the non-Wang max (L9H4 = 0.0279) by a factor of 1.33×.

### 3.3 Phase 2 sweep design

The full sweep covers 40 log-spaced checkpoints per Pythia size:

```
0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000,
20000, 24000, 29000, 35000, 41000, 49000, 59000, 70000,
84000, 100000, 120000, 143000.
```

All 11 of Pythia's early-dense cells (step 0 + powers of 2 up to step 512) are included verbatim. Consecutive cell ratios are min 1.06, median 1.20, max 2.00. The grid is identical across all 9 (size, motif) cells.

**Logistic fit.** For each (size, motif) cell, the count of detected heads above the locked threshold across the 40 cells is fit by

$$\text{count}(\text{step}) \approx \frac{L}{1 + \exp(-k(\log\text{step} - \mu))}$$

via `scipy.optimize.curve_fit`. The emergence step μ is the half-rise. We tier the fit handling by max count across the 40 cells:

- *Emerged* (max count ≥ 5): direct logistic fit, bootstrap CI.
- *Marginal* (2 ≤ max count ≤ 4): fit reported as the bootstrap-median μ with a widened CI flagged in figures.
- *Censored* (max count < 2): right-censor μ at step 143000.

Right-censoring subsumes the brief's earlier "drop 70M for successor" rule (PROJECT_BRIEF §10); marginal/censored cells are flagged as such in all figures and tables.

**Bootstrap.** Per-prompt bootstrap, B = 1000 resamples, 95% percentile interval on μ per (size, motif). Threshold sensitivity is reported separately by varying each motif's locked threshold over *±25%* in five increments and reporting the resulting μ range.

**Joint sign test.** The pre-registered gate is the joint sign test on per-size strict ordering. Under H_0 of exchangeable per-size order, the conjunction of three predicted orderings has probability *(1/6)³ ≈ 0.00463*. The locked threshold is *p < 0.005*. The conjunctive structure incorporates multiplicity by construction. Per-pair descriptive p-values are uncorrected.

### 3.4 Phase 1.2 prerequisite — Tigges replication

Before Phase 2, we replicate Tigges et al. (2024)'s IOI emergence curve on Pythia-410M-deduped to validate the M5 Pro / MPS tooling stack. Gate: max absolute difference < 0.10 vs Tigges 2024 at the directly-shared checkpoints. Result: max diff 0.0664 at step 1000 (sample-size variance), all other shared steps within 0.010 (Appendix B.4). The make-or-break gate clears.

### 3.5 §H5-causal causal-dependence ablation (Track 2)

The Track 1 emergence-ordering result is consistent with two distinct mechanistic readings: (a) a forward-pass compositional chain where successor's outputs are read by S-inhibition, with the temporal ordering reflecting an architectural dependence; and (b) convergent training dynamics where the motifs form in order for optimizer-path reasons (gradient routing, sub-circuit prerequisites) but are causally disjoint at convergence. Track 2 distinguishes (a) from (b) by direct ablation.

**Ablation method (locked, §H5-2).** We mean-ablate the top-5 successor heads at convergence on the `hook_z` activation, per length group; the replacement is the batch-mean of `hook_z` for that (head, length) over the clean condition. The ablation is installed as a permanent forward hook for the duration of the ablation condition.

**Suc set (locked per-size, §H5-3).** Top-5 successor heads at the anchor checkpoint by per-head score (cross-category lift, threshold τ_lift = 0.13496), with §H5-3 tie-break (score descending, layer ascending, head ascending). Sets are derived deterministically from sealed sweep parquets and asserted bit-for-bit at runtime; any drift halts the run.

**Ctrl set (procedure-locked, §H5-4 + §H5-causal-3-7).** Random sample of 5 heads (`rng = np.random.default_rng(0)`) from the bracket `[τ_lift − bw, τ_lift)`, with bw initialized at 0.05 and widened by 0.025 if fewer than 5 candidates qualify. At per-size scale the candidate pool excludes pinned Name Movers (§H5-causal-3-7 NM-exclusion clause). The bracket choice score-matches the ctrl set to suc at the near-threshold edge — successful detection of suc's causal role requires that the ablation effect be larger than for the bracket-matched ctrl set.

**Receiver (locked, §H5-6).** Name Movers are pinned to the top-4 by component-DLA on the *(IO − S)* logit difference of the clean model at the anchor checkpoint. The same NM set is frozen across all three conditions (clean / suc-ablated / ctrl-ablated).

**Metric A — §S-1 path-patching scalar.** For each of the top-3 S-inhibition senders (selected by Δ_h at the clean anchor, §H5-5), we measure the path-patching Δ_h on the senders→NMs route under each ablation condition. The per-sender drop ratio is `ratio_X = Δ_h^{X-ablated} / Δ_h^{clean}`. Per-sender classification {NULL, DEP, GENERIC, MIXED} uses DEP_THRESHOLD = 0.5 and NULL_BAND = ±0.20 (§H5-7); aggregate verdict uses the §H5-7 priority `GENERIC > NULL > DEP > MIXED`.

**Metric B — IO−S logit-diff at END (§H5-causal-2).** A converging readout that addresses the §S-1-structural-insensitivity caveat: at some checkpoints, suc heads sit at layers ≥ max(NM layer), so the §S-1 senders→NMs path-patching scalar is structurally mute to those ablations. Metric B reads the logit difference *(IO − S)* at the END position under each condition; it sees every contribution of the ablated head through the residual stream to the unembedding, regardless of layer. Verdict thresholds: NULL band [0.8, 1.2], DEP < 0.5, GENERIC < 0.7 (§H5-causal-2-6).

**Statistical machinery.** Paired per-prompt bootstrap, B = 200 replicates with `rng = np.random.default_rng(1)`, 95% percentile CI on the drop ratio per metric and per sender. The same prompt indices resample at every condition (paired bootstrap).

**Locked sets per size.** All four sets — suc, ctrl, SI senders, NMs — are deterministic functions of sealed sweep parquets and clean-state anchor npz files. The runners assert bit-for-bit matches at startup. Per-size lock tables are in Appendix C; for Pythia-2.8B these are: suc = `[(15,14), (28,17), (27,13), (13,10), (29,28)]`; SI senders = `[(11,29), (11,5), (13,9)]`; NMs = `[(11,29), (17,12), (22,31), (13,9)]`; ctrl (procedure-derived, bw = 0.075, NM-excluded) = `[(13,5), (13,8), (13,27), (20,29), (24,25)]`.

**Structural caveat.** At Pythia-2.8B, 3 of 5 suc heads — (27,13), (28,17), (29,28) — sit at layers > max(NM layer) = 22, so Metric A is *structurally mute* to those 3 ablations. Only (15,14) and (13,10) propagate to NMs through Metric A. Metric B reads at END and is fully sensitive to all 5 suc ablations. Both metrics returning NULL is therefore the robust outcome, since Metric A's reading rests on only the 2 visible heads while Metric B sees all 5.

### 3.6 §H4-scaling / §H4-supersede head-count-axis scaling (Track 3)

**Head-count rationale (§H4-1).** Head count is non-monotonic in parameter count across the Pythia suite (70M=48, 160M=144, 410M=384, 1B=128, 1.4B=384, 2.8B=1024); head-level circuit analysis at the same parameter axis lumps an architectural regression (1B) with three architectural scale-ups (160M, 410M, 2.8B). §H4-1 argues that for head-level claims, head count is the operationally relevant axis. Pythia-1B is treated as a head-count regression (§H3-scale REGR verdict); Pythia-2.8B is treated as the next head-count tier above 410M.

**Conjunctive gate (locked verbatim §H4-2 / §H4-supersede-2).** Two legs, both must hold for PASS:
- **(A.timing)** `P(μ_si^2.8B < μ_si^410m) ≥ 0.95` over B = 1,000 paired per-prompt bootstrap replicates. Each replicate resamples prompts with replacement, recomputes per-head Δ_h, refits the count-vs-step logistic curve at both 2.8B and 410m, and records the (μ_si^2.8B, μ_si^410m) pair. The reversal rate is the empirical fraction of replicates in which μ_si^2.8B < μ_si^410m.
- **(A.count)** `max_count_si^2.8B ≥ 5` over the §H4-supersede 10-cell sweep — full-fit regime entry per §H2-3. Max is over the 10-cell grid; pre-step-5000 cells from the §H4-7-supersede partial cache (all Δ_h ≈ 0) do not contribute.

**Sweep grid (§H4-supersede-1).** 10 cells in the emergence-likely range: `[5000, 7000, 10000, 14000, 20000, 29000, 41000, 49000, 59000, 70000]`. Every step is drawn from the §H2-1 40-cell grid, so cross-size comparisons against {70M, 160M, 410M, 1B} are exact-step-matched at those 10 indices. The 10-cell choice is motivated by the emergence-likely range observed in the smaller sizes (μ_si typically falls between step ~7,000 and step ~50,000; 410M's μ_si ≈ step 14,000–29,000 anchors the upper edge of the (A.timing) reversal-rate test).

**Failure-mode taxonomy (locked §H4-5 + §H4-7-supersede DEFERRED, inherited verbatim by §H4-supersede-3).** Six patterns: PASS / TIMING-ONLY / COUNT-ONLY / NEITHER / TOOLING / DEFERRED, in priority order `DEFERRED > TOOLING > NEITHER > COUNT-ONLY > TIMING-ONLY > PASS`. DEFERRED was added to the taxonomy at §H4-7-supersede after the original full-grid sweep was halted at 8/40 cells; §H4-supersede inherits the taxonomy verbatim, and a second DEFERRED on the reduced grid would not falsify the scaling claim — the paper would ship on Tracks 1 + 2 (§writeup-conv-2 pre-PASS framing).

## 4. Results

### 4.1 H1-C joint sign-test verdict

The pre-registered joint sign test rejects exchangeable ordering at the locked threshold:

> **Joint H1-C verdict: PASS.** All three Pythia sizes show the predicted strict ordering μ_induction < μ_successor < μ_S-inhibition. Joint p ≈ 0.00463 < 0.005 locked gate.

Table 1 reports the emergence-step point estimates and bootstrap 95% CIs per (size, motif) cell. Induction emerges at step ~600–6000 across the three sizes; successor emerges at step ~10000 across the three sizes; S-inhibition emerges at step ~25000 in 410M, marginally near step 12000–35000 in 160M, and is right-censored at step 143000 in 70M.

**Table 1 — Per-cell emergence steps (point estimate; 95% bootstrap CI; regime).**


| Size | Induction                  | Successor                    | S-inhibition                                 |
| ---- | -------------------------- | ---------------------------- | -------------------------------------------- |
| 70M  | 641 [640, 641]; emerged    | 8670 [4400, 2×10⁶]; marginal | 143000 [143000, 143000]; **censored**        |
| 160M | 6087 [4480, 9053]; emerged | 9999 [7786, 15119]; marginal | ≥ upper sentinel [7485, 2×10⁶]; **marginal** |
| 410M | 1621 [1537, 1707]; emerged | 10360 [9450, 11584]; emerged | 24919 [22206, 40767]; marginal               |


**Table 2 — Bootstrap reversal rate per (size, ordering pair), B = 1000 per-prompt resamples.** Reversal rate is the empirical fraction of bootstrap replicates in which the predicted pair-ordering does *not* hold (with right-censored ties counted as undetermined per §H2-4).


| Size | Ind→Suc | Suc→S-inh       | Ind→S-inh |
| ---- | ------- | --------------- | --------- |
| 70M  | 0.000   | 0.378 (73 ties) | 0.000     |
| 160M | 0.024   | 0.018           | 0.012     |
| 410M | 0.000   | 0.000           | 0.000     |


**Figure 1 — Three-panel emergence comparison.** *(Placeholder; rendered from `notebooks/h1c_ordering_test.ipynb`.)* Each panel is one Pythia size (70M, 160M, 410M, left to right). x-axis is log training step; y-axis is count of detected heads above the locked threshold. Three lines per panel (induction, successor, S-inhibition) with B = 1000 bootstrap CI envelopes and per-fit logistic curves. Right-censored cells are shown as a dashed line at the 143000 sentinel; marginal cells are shown with a hatched envelope.

### 4.2 Confirmation strength scales with model size

Reading Table 1 and Figure 1 together, the per-size strength of the joint-claim confirmation is highly heterogeneous:

- **70M.** Induction emerges sharply at step ~640. Successor crosses the τ_lift threshold for 2 heads at most across the 40 cells, fitting a marginal logistic with point μ ≈ 8700 but bootstrap CI spanning more than two orders of magnitude. S-inhibition reaches a max count of 1 across all 40 cells (only L4H2 at step 143000 with Δ_h = 0.0433) — right-censored at the sentinel. The H1-C ordering is satisfied at 70M, but the suc → S-inhibition leg is held up by the censoring of the third motif rather than by any clear S-inhibition signal during training.
- **160M.** Induction emerges with 17 heads at the final step; successor reaches a max count of 4 (logistic fit marginal). S-inhibition reaches max count 3 but the logistic fit hits the upper sentinel; max Δ_h peaks earlier (step 12000–35000 at 0.18–0.19) and *decays* to 0.068 by step 143000, a non-monotonic emergence inconsistent with stable circuit assembly. Bootstrap reversal rate for suc → S-inhibition is 0.018 — H1-C ordering holds in expectation but with a wider envelope than at 410M.
- **410M.** Induction emerges with 23 heads (point μ ≈ 1621, tight CI). Successor emerges cleanly with 7 heads at step 41000 (peak), point μ ≈ 10360 with CI [9450, 11584]. S-inhibition reaches max count 3 (point μ ≈ 24919, CI [22206, 40767]). All three pair-ordering bootstrap reversal rates are 0.000. **Pythia-410M is the only size where all three motifs emerge with logistic fits supported by ≥5 heads at the threshold *or* with bootstrap CIs that do not include the upper sentinel.**

The pattern is what one would expect if a third, "later" mechanism is *available to be assembled* only at sufficient model capacity. At 70M the H1-C joint claim is *vacuously* satisfied through censoring of the third motif; at 160M it is *marginally* satisfied with a wide bootstrap envelope and a non-monotonic emergence trajectory; at 410M it is *robustly* satisfied. The pre-registered joint gate passes — we report it that way — but the strength of that pass scales monotonically with model size, and we believe the scale-dependent reading is the more accurate description of the data.

### 4.3 Temporal vs architectural ordering — the depth asymmetry

The compositional reading of the H1-C ordering is "later-emerging motifs sit at deeper layers and consume earlier motifs' outputs." This reading does not survive the data. Table 3 reports the normalized layer depth (layer index ÷ total layers) of the top-scoring head per (size, motif) at step 143000:

**Table 3 — Top-head normalized depth at step 143000.**


| Size | Induction     | Successor    | S-inhibition  |
| ---- | ------------- | ------------ | ------------- |
| 70M  | 0.60 (L3H1)   | 0.80 (L4H0)  | 0.80 (L4H2)   |
| 160M | 0.36 (L4H6)   | 0.82 (L9H10) | 0.55 (L6H2)   |
| 410M | 0.48 (L11H14) | 0.96 (L22H6) | 0.52 (L12H12) |


In 160M and 410M, the top successor head sits *deeper* than the top S-inhibition head (0.82 > 0.55 and 0.96 > 0.52 respectively). The compositional reading would predict the opposite (the corrective S-inhibition mechanism consumes successor outputs at later layers). The temporal ordering "successor emerges before S-inhibition in training" is therefore not a surface readout of an architectural ordering "successor sits before S-inhibition in the forward pass."

A weaker reading survives: the temporal ordering reflects something about *which mechanisms can form first as training progresses*, not about how those mechanisms are stacked at convergence. We discuss this in §6.

### 4.4 Identity churn for the later motifs

For each (size, motif), we compute the top-3 head identity overlap (Jaccard) between step 25000 and step 143000:

**Table 4 — Top-3 head identity stability, step 25000 → step 143000.**


| Size | Induction | Successor | S-inhibition |
| ---- | --------- | --------- | ------------ |
| 70M  | 1.00      | 0.00      | 0.33         |
| 160M | 1.00      | 0.33      | 0.33         |
| 410M | 0.67      | 0.67      | 0.33         |


Induction's top-3 heads are stable across the second half of training (1.00 in 70M and 160M; 0.67 in 410M with one new head joining). Successor's identity is stable only at 410M; in 70M the top-3 turns over completely. S-inhibition's identity turns over in *all three sizes* — only one of three top heads at step 25000 is still in the top 3 at step 143000.

The "emergence step" abstraction in Table 1 tracks a *count of heads above threshold*, not the identity of those heads. For successor and especially S-inhibition, the population of heads passing threshold at step 143000 is materially different from the population passing at step 25000. This means the bootstrap CIs on μ in Table 1 do not capture identity churn; the emergence-step quantity is a quantity over *populations*, not over *circuits*.

### 4.5 Structural reuse vanishes at scale

We take Phase 3 / Extension B from the project plan as a direct test of the compositional account: do the top-K=5 heads of each motif at step 143000 overlap, or are the three motifs implemented by disjoint head populations?

**Table 5 — Multi-motif top-5 heads at step 143000.**


| Size | Multi-motif heads (motif₁ ∩ motif₂ in top-5)                     |
| ---- | ---------------------------------------------------------------- |
| 70M  | L3H5 (induction ∩ S-inhibition); L4H0 (successor ∩ S-inhibition) |
| 160M | L8H9 (successor ∩ S-inhibition)                                  |
| 410M | **none**                                                         |


The cleanest H1-C-confirming size is also the cleanest *structurally separated* size. At 410M the three motifs are implemented by disjoint top-5 head populations; at smaller sizes there is non-trivial reuse, especially across the suc–S-inhibition pair. The figure-counterpart of Table 5 (Figure 4) tracks the three-pair Jaccard over training: at 410M, all pair-Jaccards trend to 0; at smaller sizes they do not.

The two readings consistent with this pattern are: (i) larger models have more attention heads to distribute roles across, so disjoint allocation is a capacity effect; or (ii) larger models develop *cleaner specialization* during training. The data here cannot distinguish the two; we flag this for future work in §6.

**Cross-size, cross-step structural-reuse extension.** Beyond the step-143000 top-5 view, we compute the full (size, step) overlap trajectory for the three pairwise intersections at the *detection-population* level (all heads above their motif's locked threshold, not just top-5). The successor–S-inhibition pair is structurally disjoint across essentially every cell: 0/40 at 70M, 1/40 at 160M (head (8,9) at step 143000 only), 2/40 at 410M (head (13,13) at steps 41,000 and 70,000), 0/40 at 1B, 1/10 at 2.8B (head (13,8) at step 29,000 only). The induction–S-inhibition pair, by contrast, has a small recurring overlap at the larger sizes: (17,10) at 410M IS a Name Mover; (8,7) at 1B IS the top SI sender; (13,9) at 2.8B is induction-detected AND SI sender AND NM — a triple-role head. **At every §H5 anchor (410M / 1B / 2.8B step 143000), the locked top-5 successor heads and the SI-detected population intersect on zero heads; the locked top-3 SI senders and the suc-detected population also intersect on zero heads.** Track 2's NULL × NULL is therefore not "ablation-resistant" — it is the natural readout when two structurally disjoint populations are independently probed. Critically, the 1B Metric B MIXED (§4.7) cannot be a structural-reuse artifact: 1B has zero suc ∩ si overlap across all 40 steps, so the readout's loss of suc-specificity at 1B is an architecture property, not a sign that overlap heads are confounding the ablation. Full per-(size, step) tables in Appendix I.

### 4.6 Track 2 — §H5-causal at Pythia-410M: NULL × NULL on two converging metrics

§4.3's depth-temporal asymmetry already suggests the §H1-C compositional reading is in trouble — successor heads at 410M sit at normalized depth 0.96, far *below* (in the forward pass, after) the S-inhibition heads at depth 0.52. But that observation is structural, not mechanistic: it shows the successor → S-inhibition compositional chain *cannot* be implemented as a clean layer-cascade in 410M, but does not rule out an indirect routing. §4.6–§4.8 test the chain directly by ablation.

We mean-ablate the top-5 successor heads on hook_z per length group (§3.5) and read S-inhibition's response on two converging metrics: Metric A (§S-1 path-patching Δ_h on the top-3 SI senders) and Metric B (IO−S logit difference at END).

**Table 6 — Pythia-410M causal-dependence verdict, both metrics.**

| Metric | Pattern | ratio_suc [95% CI] | ratio_ctrl [95% CI] | Note |
| --- | --- | --- | --- | --- |
| **A (§S-1 path-patching)** | **NULL** (3/3 senders) | 1.000 [tight] | 1.000 [tight] | All 3 SI senders (L12H12, L13H13, L14H0) return per-sender NULL pattern. |
| **B (logit-diff)** | **NULL** | 0.986 [0.978, 0.992] | 0.979 [0.968, 0.991] | Both ablations leave IO−S logit-diff within the [0.8, 1.2] NULL band. |

Both metrics return NULL. The bootstrap CIs are tight at both legs — Metric A's per-sender CIs all sit within ±0.005 of 1.0; Metric B's CIs are width ~0.01–0.02. The score-bracket-matched control set (selected by procedure §H5-4, random sample with seed = 0 from heads with score in [τ_lift − 0.10, τ_lift), bracket widened post-data to 0.10 by the §H5-4 widening rule) produces the same near-zero effect, ruling out the "any 5 layer-22-cluster ablations break the model" reading.

**Mechanistic reading.** Successor's mature heads at convergence do not feed S-inhibition's mature heads in either the §S-1 path-patching path or the END logit-diff readout. The temporal ordering induction → successor → S-inhibition at Pythia-410M is *decoupled* from any architectural causal chain at inference time. The §H1-C compositional reading is directly refuted at 410M.

We checked one more reading that the NULL × NULL alone does not rule out: that the §S-1 readout is *structurally insensitive* to suc ablations because some suc heads sit at layers ≥ max(NM layer) at the anchor. At 410M, 4 of 5 suc heads sit at layers ≥ max(NM) = 22 (suc set: L22H6, L22H2, L20H4, L22H10, L12H8; NMs at layers 12, 14, 17, 20). Only L12H8 routes through the §S-1 NM-receiver readout. Metric B exists specifically to address this caveat: it reads at END and sees every contribution of every ablated head through the residual stream regardless of layer. Both metrics returning NULL is the robust reading.

### 4.7 §H5-causal-3-record — Pythia-1B replicates Metric A NULL; Metric B turns generic at 128 heads

§H5-causal-3-record canonicalizes a Pythia-1B step143000 anchor run that was executed on a feature worktree (2026-05-07) before the corresponding amendment was formally registered. The pre-reg discipline gap is disclosed in §H5-causal-3-record-4 of the canonical record and again here. All protocols (suc set derivation, ctrl bracket-widening with seed=0, NM identification, mean-ablation, bootstrap) were locked at 410M before any 1B compute, so the run had no degree of freedom to cherry-pick; only the chronological order of the amendment register vs the compute violates strict pre-data discipline.

**Table 7 — Pythia-1B causal-dependence verdict.**

| Metric | Pattern | ratio_suc [95% CI] | ratio_ctrl [95% CI] | Note |
| --- | --- | --- | --- | --- |
| A (§S-1 path-patching) | **NULL** (3/3 senders) | per-sender ratios all ≈ 1.0 | similar | Replicates the 410M NULL on path-patching. |
| **B (logit-diff)** | **MIXED** | 0.790 [0.773, 0.809] | 0.797 [0.776, 0.813] | Both ablations drop IO−S logit-diff by ~21%, *similarly* — no successor-specific dependence. |
| Cross-metric (§H5-causal-3-record-2) | **MIXED** | — | — | Heterogeneous; no global verdict on suc → SI causal dependence. |

Metric A's NULL replicates 410M's converging-evidence finding at 1B's 128-head architecture. Metric B's MIXED is the interesting case: the IO−S logit-diff drops by ~21% under suc ablation *and* under ctrl ablation. The drop is not suc-specific. We read this as a **narrow-architecture readout-specificity artifact**: at 1B's 128-head architecture (a head-count regression from 410M's 384, §H4-1), the IO−S logit-diff readout itself becomes generically ablation-sensitive — it can no longer distinguish suc-specific effects from generic 5-head ablations. The Metric A path-patching scalar, which routes through specific NM heads, remains specific and returns NULL.

The §4.8 result at Pythia-2.8B is the test of this interpretation: at the wider 1024-head architecture, Metric B should recover NULL (if the 1B MIXED was narrow-architecture-specific) or remain MIXED (if it was a genuine cross-size DEP signal that 410M happened to miss). §4.8 returns the former.

A further per-size structural caveat is registered in §H5-causal-3-record-3: at 1B step143000, suc-set head L14H2 is *also* a pinned NM at L14H2 (dual-role); Metric A's downstream-NM filter excludes L14H2's contribution from the path-patching scalar but Metric B reads at END and is fully sensitive to it. This does not change the cross-metric MIXED reading, but it is recorded for transparency.

### 4.8 §H5-causal-3-2.8b — Pythia-2.8B recovers NULL × NULL at head-count 1024

§H5-causal-3-2.8b registered both metrics at Pythia-2.8B step143000 anchor *before* any 2.8B ablation compute (2026-05-10). Locked sets (asserted bit-for-bit at runtime): suc = `[(15,14), (28,17), (27,13), (13,10), (29,28)]`; SI senders = `[(11,29), (11,5), (13,9)]`; NMs = `[(11,29), (17,12), (22,31), (13,9)]`; ctrl (procedure-locked, bw = 0.075, NM-excluded) = `[(13,5), (13,8), (13,27), (20,29), (24,25)]`.

**Table 8 — Pythia-2.8B causal-dependence verdict.**

| Metric | Pattern | ratio_suc [95% CI] | ratio_ctrl [95% CI] | Note |
| --- | --- | --- | --- | --- |
| **A (§S-1 path-patching)** | **NULL** (3/3 senders) | 1.001 [tight] | 0.997 [tight] | All 3 SI senders return per-sender NULL with CI widths ~0.001–0.005. |
| **B (logit-diff)** | **NULL** | 0.984 [0.980, 0.987] | 0.984 [0.978, 0.989] | Both ratios essentially clean; CIs sit well inside the [0.8, 1.2] NULL band. |
| Cross-metric | **NULL** | — | — | Converging-evidence claim extends to head-count tier 1024. |

Both metrics return NULL with tight CIs. The 1B Metric B MIXED is *not* reproduced at 2.8B; the narrow-architecture readout-specificity reading from §4.7 is supported. The cross-metric paper headline pre-committed in §H5-causal-3-2.8b-7 reads: *"Replicates the 410M NULL on both metrics at head-count tier 1024; scale-stable causal-disjointness across 384-head and 1024-head architectures."*

A structural caveat is registered pre-data in §H5-causal-3-2.8b-6: 3 of 5 suc heads — (27,13), (28,17), (29,28) — sit at layers > max(NM layer) = 22, so Metric A is *structurally mute* to those 3 ablations. Only (15,14) and (13,10) propagate to NMs through Metric A. Metric B reads at END and is fully sensitive to all 5. Both metrics returning NULL is the robust reading; the structural caveat would matter only if Metric A returned NULL while Metric B returned DEP.

**Cross-size synthesis (Tracks 1 + 2).** The Track 2 verdict across the three sizes is:

| Size | Heads | Metric A | Metric B | Reading |
| --- | --- | --- | --- | --- |
| 410M | 384 | NULL | NULL | Converging-evidence anchor: causal-disjointness on two metrics. |
| 1B | 128 (regression) | NULL | MIXED | Metric A replicates; Metric B is generic-ablation-sensitive (narrow-architecture artifact). |
| 2.8B | 1024 | NULL | NULL | Replicates 410M; **scale-stable causal-disjointness on both metrics across head-count tiers 384 + 1024**. |

The §H1-C compositional reading is falsified in the strict forward-pass-routing sense at the two converging-evidence anchors (410M and 2.8B). At the 1B head-count regression the readout (Metric B) loses specificity, but the path-patching readout (Metric A) still returns NULL.

### 4.9 Track 3 — head-count rationale and the 1B head-count regression

Two observations motivate the head-count axis:

1. **Pythia's head budgets are non-monotonic in parameter count.** 70M = 6 × 8 = 48 heads; 160M = 12 × 12 = 144; 410M = 24 × 16 = 384; 1B = 16 × 8 = **128**; 1.4B = 24 × 16 = 384 (same as 410M); 2.8B = 32 × 32 = 1024. The 1B model has *fewer* heads than 410M.
2. **Our detectors operate at head granularity.** Count-of-heads-above-threshold and per-head score distributions both depend on the head budget. A scaling argument at the parameter axis lumps a head-count regression (1B) with three head-count scale-ups (160M, 410M, 2.8B).

§H4-1 registers head count as the operationally relevant scaling axis for this paper. Under that axis, Pythia-1B is a *regression* — it sits between 70M (48 heads) and 160M (144 heads) in head budget. §H3-scale (the original 1B scaling extension) returns the REGR pattern: timing accelerates (paired bootstrap reversal_rate = 1.000 for μ_si^1B < μ_si^410m on the §H3-scale (A.ii) leg, the maximum possible) but the count-saturation leg fails (max_count_si^1B = 3, same as 410M's). On the head-count axis, this is consistent: 1B has fewer heads to host S-inhibition with, so even though it trains faster (perhaps because narrower QK matrices admit faster optimization, or because the 128 heads are forced into more diverse roles), the cap on absolute count of S-inhibition heads cannot exceed 410M's. We treat §H3-scale REGR as a head-count regression confirmation, not as a scale-up falsification.

The Track 3 substantive test is therefore at Pythia-2.8B (1024 heads), the next head-count tier above 410M.

### 4.10 §H4-supersede — head-count-axis scaling PASS at Pythia-2.8B

The original §H4-scaling 40-cell S-inhibition sweep at 2.8B was halted at 8/40 cells under the §H4-7 per-cell-cost escape hatch (observed ~57 min/cell vs ~6 min/cell projected; the 2× pause threshold exceeded by 5×). §H4-7-supersede (2026-05-08) registered the DEFERRED pattern and added it to the §H4-5 priority `DEFERRED > TOOLING > NEITHER > COUNT-ONLY > TIMING-ONLY > PASS`.

§H4-supersede (2026-05-10) registered a reduced 10-cell grid against the verbatim §H4-2 gate predicates and ran overnight 2026-05-10/11 in ~5h 55min wall time (~27.5 min/cell on the reduced grid, ~half the §H4-7-supersede projected cost; the cells with non-zero S-inhibition activity were faster than the early-training cells in the original aborted sweep).

**Table 9 — §H4-supersede verdict.**

| Leg | Value | Gate | Verdict |
| --- | --- | --- | --- |
| **(A.count)** | max_count_si^2.8B = 5 at step 29,000 | ≥ 5 | **PASS** |
| **(A.timing)** | paired bootstrap reversal_rate = 1.000 (1000/1000 replicates show μ_si^2.8B < μ_si^410m; zero fit failures) | ≥ 0.95 | **PASS** |
| **Joint** | both legs hold | (A.count) ∧ (A.timing) | **PASS** |

**Point estimates.** μ_si^2.8B ≈ 9,021; μ_si^410m ≈ 24,088 — 2.8B accelerates S-inhibition's logistic-fit midpoint by **~2.7×**.

**Pre-committed paper headline (§H4-5 PASS pattern, §H4-supersede-2 inheritance).** *"§H4-supersede passes: at head-count tier 1024 (Pythia-2.8B), S-inhibition timing accelerates beyond 410M (paired bootstrap reversal-rate = 1.000 ≥ 0.95) AND count exceeds the 410M saturation cap (max_count = 5 ≥ 5). Scaling argument on the head-count axis confirmed."*

### 4.11 Count-trajectory across the §H4-fullgrid 40-cell grid

The §H4-supersede (A.count) gate is defined on *max* over the original 10-cell grid (locked §H4-2 verbatim). The follow-up §H4-fullgrid amendment registered a 22-cell completion at Pythia-2.8B (the cells in the §H2-1 40-cell grid not covered by §H4-supersede or §H4-7-supersede partial cache), filling in the early ramp (steps 128 → 17,000) and the late tail (steps 84,000 → 143,000). The merged 26-cell trajectory at Pythia-2.8B (τ_strict = 0.0372) is:

| Step | 128 | 1k | 3k | **4k** | 6k | 10k | 12k | **15k** | 16k | 17k | 20k | **29k** | 41k | 70k | 84k | 100k | 120k | 143k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Count | 0 | 0 | 0 | 1 | 2 | 2 | 2 | **5** | 3 | 3 | 3 | **5** | 4 | 4 | 4 | 4 | 4 | (TBD) |
| Top Δ_h | 0.000 | 0.001 | 0.013 | 0.067 | 0.141 | 0.256 | 0.237 | 0.200 | 0.206 | 0.162 | 0.181 | 0.168 | 0.153 | 0.136 | 0.167 | 0.134 | 0.162 | (TBD) |

**Emergence step at 2.8B is between steps 3,000 and 4,000.** L11H29 — the eventual canonical 2.8B S-inhibition head — already has top-rank position at step 3,000 with Δ_h = 0.013 (sub-threshold), and crosses τ_strict at step 4,000 (Δ_h = 0.067). The early checkpoints (≤512) show max-Δ_h ≈ 0 with the top-ranked head's identity jittering across the network (L17H20, L6H18, L12H14) — confirming a random-noise floor before any S-inhibition motif exists. The pre-emergence differentiation of L11H29 at step 3,000 is qualitative evidence that S-inhibition is not a discontinuous jump from noise but a continuous ramp that crosses a magnitude threshold.

**The count trajectory has two transient n=5 peaks at steps 15,000 and 29,000.** Between them the count dips to 3 (steps 16k–20k). After step 29,000 it relaxes to a steady-state of **n = 4 sustained from step 41,000 through step 120,000**. The earlier §H4-supersede "5 at 29k → 4 at 70k" reading was capturing the descent from the second transient peak; the §H4-fullgrid data show the count has two transient peaks of 5 separated by a brief dip to 3, then settles at 4 in the late tail. L11H29 maintains top-rank position from step 4,000 through every subsequent cell (with one brief swap to L11H5 at step 59,000), and the NM composition stabilizes to a mostly-fixed set {(11,29), (17,12), (13,9), (22,31)/(19,2)} by step 84,000.

**(A.count) gate verdict unchanged.** Max-count over the 40-cell grid is 5 at step 15,000 and step 29,000 (both transient); the §H4-2 gate (≥ 5) PASSES. The dip caveat (§writeup-conv-2) is resolved into a clearer picture: the count peaks at 5 transiently mid-emergence and relaxes to 4 in steady state, rather than asymptotically reaching 5. This is a methodological clarification, not a gate-pass concern.

### 4.12 §H1-C robustness appendix — measurement-invariance sensitivity under alternative detectors

This section is **post-hoc and exploratory**. The primary §H1-C result (§4.1) — locked-detector joint sign-test p = 0.00463 — is unchanged. This appendix asks whether the temporal ordering is *measurement-invariant*: does it survive substituting alternative detectors for each motif?

**Pre-registered design (§H1-C-altdetectors, §H1-C-altdetectors-2-r-supersede) failed under cross-family threshold transfer.** We registered three mechanism-verifying alt-detectors and an initial threshold-locking rule based on GPT-2 small's 95th-percentile alt-score per-motif (the (c2-percentile) rule). The Pythia sweeps produced trivially degenerate trajectories:

- Alt-induction OV-score: maximum OV-score ever observed at any Pythia size × any cell was +15.70 (160m, step ≥ 20,000); GPT-2-derived τ_ind_OV = +13.59 is exceeded by only **1 head ever** across 360 cells.
- Alt-successor argmax-K-of-7: K_min = 2 was already exceeded by 73 of 384 heads at Pythia-410M step 0 (random init), with the count *decreasing* with training as heads specialize away from accidental day-token argmax. This is a *de-emergence* trajectory, not an emergence trajectory.
- Alt-S-inhibition Component-DLA-at-S2: GPT-2-derived τ_si_DLA = +0.247; maximum observed at Pythia-70m ever = +0.0035, three orders of magnitude below threshold.

The cross-family magnitude-transfer failure is itself a methodological finding: alt-detector magnitudes depend on unembedding scale, layernorm, and how many heads divide the circuit work — none of which are family-invariant. Tigges et al. (2024) explicitly normalize component scores over checkpoints within Pythia, sidestepping the cross-family issue; Gould et al. (2024) document successor heads across families at convergence but do not attempt threshold transfer. We document the failure mode openly and supersede the threshold-locking rule with a within-Pythia frozen-threshold scheme (HYPOTHESIS.md §H1-C-altdetectors-2-rr-supersede).

**Within-Pythia frozen-threshold scheme (post-hoc).** For each (Pythia size, motif, alt-detector), the threshold is the K-th-largest alt-score at step 143,000, where K = locked-detector pass count at step 143,000. The threshold is then applied backward over all 40 §H2-1 checkpoints. A top-5%-of-heads variant is computed as sensitivity-of-sensitivity. A magnitude-free rank-only secondary view tracks |alt-top-K\_t ∩ top-K-final| / K over training.

**Pre-validation at step 143,000.** Before testing the trajectory, we verify the alt-detector identifies the same motif at convergence via locked-vs-alt top-K overlap:

| Motif × size | K_locked | top-K overlap (locked, alt) | Disposition |
|---|---|---|---|
| Induction 70m | 6 | 5/6 (83%) | usable |
| Induction 160m | 17 | 11/17 (65%) | usable |
| Induction 410m | 19 | 11/19 (58%) | usable |
| Successor 70m | 2 | 0/2 (0%) | unreliable |
| Successor 160m | 3 | 3/3 (100%) | usable |
| Successor 410m | 2 | 0/2 (0%) | unreliable |
| S-inhibition 70m | 1 | 0/1 (0%) | unreliable |
| S-inhibition 160m | 3 | 1/3 (33%) | partial |
| S-inhibition 410m | 2 | 1/2 (50%) | usable |

Only the **alt-induction OV detector is consistently usable across all 3 sizes**. The alt-successor argmax-K detector is usable only at Pythia-160m (pure coincidence at the two endpoints — 0/2 at 70m and 410m). The alt-S-inhibition CompDLA-at-S2 detector is usable only at Pythia-410m. **At cells with poor pre-validation, the alt-detector and locked-detector identify different head populations at convergence** — the alt-detector is not measuring the same motif there, so its trajectory cannot be a valid robustness probe.

**Joint sign-test under detector triples × 3 schemes:**

| Triple | Frozen-final-K | Top-5% | Rank-only | Locked baseline |
|---|---|---|---|---|
| (locked, locked, locked) | — | — | — | 3/3 sizes, **p ≈ 0.00463** |
| (alt-ind, locked, locked) | 2/3 sizes | 2/3 sizes | 2/3 sizes | — |
| (locked, alt-suc, locked) | 1/3 sizes | 0/3 sizes | 1/3 sizes | — |
| (locked, locked, alt-si) | 1/3 sizes | 2/3 sizes | 2/3 sizes | — |
| (alt-ind, alt-suc, alt-si) | 0/3 sizes | 0/3 sizes | 1/3 sizes | — |

**Interpretation.** Single-substitution with the well-pre-validated alt-induction OV detector preserves the ordering at 2/3 Pythia sizes under every scheme — the strongest robustness signal. Single-substitution with poorly-pre-validated detectors (alt-successor, alt-S-inhibition) yields 0–2 of 3 sizes; the failures cluster at the same (size, motif) cells where pre-validation overlap was below 50%. The all-alt triple fails at 0/3 to 1/3 sizes across schemes — but the failure is mechanically attributable to compounded measurement noise from poorly-pre-validated detectors, not to the temporal claim. Pre-validation overlap predicts the joint-sign-test pass-rate.

**Honest robustness verdict.** The §H1-C ordering survives measurement substitution **when the substituting detector is pre-validated as measuring the same motif at convergence**. It does not survive simultaneous substitution by three detectors that fail pre-validation at most sizes. This is a measurement-invariance result, not a temporal-claim result. The primary §H1-C verdict (p = 0.00463 with locked detectors) is unchanged; the robustness exercise establishes that the ordering is most plausibly a real temporal-emergence phenomenon — not a detector-specific artifact — when probed by alternative detectors that have demonstrated final-checkpoint agreement with the locked detectors. The cross-family threshold transfer attempt (the original §H1-C-altdetectors-2-r approach) is documented as an unsolved methodological problem.

### 4.13 §H1-C-2.8b-extension — 4-size ordering consistency (post-hoc)

The §H1-C joint sign-test was pre-registered at 3 Pythia sizes. The Pythia-2.8B sweeps for all three motifs exist (collected for Tracks 2 + 3) and are not pre-registered for use in §H1-C. A post-hoc 4-size extension is reported here for completeness — **not as a stronger §H1-C result**, but as a robustness data point. The §H1-C-2.8b-extension amendment (HYPOTHESIS.md §H1-C-2.8b-extension) registers the post-hoc framing explicitly.

Per-size first-≥5-pass emergence step at Pythia-2.8B (locked detectors, threshold-pass count over the merged 32-cell 2.8B trajectory: §H4-7-supersede ∪ §H4-supersede ∪ §H4-fullgrid):

| Motif | First step with ≥ 5 pass | Ordering position |
|---|---|---|
| Induction (QK > 0.30) | step 1,000 | earliest |
| Successor (lift_dla ≥ 0.13496) | step 10,000 | middle |
| S-inhibition (Δ_h ≥ 0.0372) | step 15,000 | latest |

Ordering `induction < successor < S-inhibition` HOLDS at 2.8B as well. 4-size joint consistency: ordering holds at 70m, 160m, 410m (pre-registered §H1-C) and at 2.8b (post-hoc); joint p ≈ (1/6)^4 ≈ 0.00077 under the same null-of-random-ordering. This is reported as a post-hoc consistency check, not a confirmatory test. The primary §H1-C verdict remains p = 0.00463 at the 3 pre-registered sizes.

### 4.14 Figure F6 — motif emergence vs. training loss

A visualization combining the three motif pass-count trajectories with EleutherAI's published training loss (pulled from the public W&B project; concatenated across the 2.8B restart-chain; remapped wandb \_step → published-checkpoint-step where applicable). Each Pythia size gets one stacked panel; primary y-axis = motif pass-count; secondary y-axis = training cross-entropy loss; x-axis = log training step. The figure makes the loss-descent / motif-emergence alignment legible without requiring the reader to compose two separate plots. Stored at `notebooks/figures/F6_loss_vs_emergence.png`; reproducible via `notebooks/_build_loss_vs_emergence_figure.py` from `data/exploration/pythia_training_loss_wandb.parquet`.

## 5. Limitations

These were pre-committed in the hypothesis document; we extend them for Tracks 2 + 3.

1. **Single seed per Pythia size.** No within-size variance estimate. The "consistency" claim is across-size, not across-seed. All confidence intervals in this paper are over per-prompt bootstrap; they do not capture seed variance. The 5-size results (Tracks 1 + 3) are best read as 5 *paired* observations of (head count, S-inhibition emergence step) on the same training data, not as 5 independent samples from a population of training runs.

2. **No cross-architecture universality claim.** Pythia (GPT-NeoX, deduped) only; reference papers (Olsson 2022, Wang 2023, McDougall 2024, Gould 2024, Singh 2024) all use deduped Pythia, so the architectural sample is uniform but narrow. The Track 2 causal-disjointness claim in particular is *not* claimed to hold in Llama / OLMo / Mamba; cross-architecture replication is named explicitly as future work.

3. **Detector-threshold sensitivity.** Reported with bootstrap CIs across thresholds at *±25%* in five increments per §H2-2; not pretended away. Threshold sensitivity is most pronounced for 160M induction (μ ranges 2986–10001 across the ±25% bracket) and for the marginal/censored cells; 410M is robust across the bracket. The Track 1 directional ordering (induction → successor → S-inhibition) survives every threshold variant.

4. **Causal-claim scope, refined post-§H5.** Track 2 claims causal-*disjointness* of successor and S-inhibition AT INFERENCE TIME at convergence (Pythia-410M + 2.8B). Track 2 does NOT claim training-time causal independence — whether successor's earlier emergence enables S-inhibition's later emergence as a *learning* effect is a separate question requiring fine-tuning-from-checkpoint experiments. §H5-9 explicitly excluded such experiments as out of scope for M5 Pro hardware. The §4.3 depth-temporal asymmetry remains *consistent with* a prerequisite-availability reading (Singh et al., 2024) but does not establish it.

5. **Path A is conditional.** The pilot pivoted to Path C (S-inhibition) per the registered decision rule. The third motif studied in the main paper is S-inhibition, not copy-suppression. The Path A negative result is reported in Appendix G as an answer to McDougall et al. (2024)'s open question. We extended Path A to Pythia-1B as a follow-up check (Appendix G).

6. **Pre-registration-discipline gap at 1B (Track 2).** §H5-causal-3-record canonicalizes the Pythia-1B Track 2 compute that ran on a feature worktree (2026-05-07) before the corresponding amendment was formally registered. All protocols (suc / ctrl / NM derivation, ablation method, bootstrap, verdict taxonomy) were locked at 410M before any 1B compute, so the run had no degree of freedom to cherry-pick; only the chronological order of amendment-then-compute was violated. We disclose the gap explicitly here and in §4.7. The downstream 2.8B compute (§H5-causal-3-2.8b) is pre-data and clean.

7. **Track 3 (A.count) trajectory non-monotonicity.** §H4-supersede's (A.count) gate is defined on max over the 10-cell grid (locked verbatim §H4-2). The §H4-fullgrid follow-up (40-cell grid, §4.11) shows two transient n=5 peaks at steps 15,000 and 29,000 with a steady-state of n=4 from step 41,000 through step 120,000. The gate is defined on max and PASSES as registered; the steady-state-of-4 is disclosed openly as a methodological clarification of the §writeup-conv-2 dip caveat.

8. **Structural insensitivity for Metric A at 2.8B.** 3 of 5 suc heads at the 2.8B anchor sit at layers > max(NM layer) = 22, so Metric A (§S-1 path-patching) is structurally mute to those 3 ablations. Metric B (logit-diff) reads at END and is fully sensitive. Both metrics NULL means the structural caveat does not undermine the claim, but a sceptical reviewer could argue that Metric A's NULL alone is partially underdetermined at 2.8B. The convergence with Metric B is dispositive.

9. **1B Metric B MIXED interpretation.** We read the 1B Metric B MIXED (suc and ctrl both drop ~21%) as a narrow-architecture readout-specificity artifact, recovered to NULL at 2.8B's 1024 heads. A sceptical reviewer could read this as "the 1B result is noisy and the causal-disjointness claim has a hole." Our response is the convergence at 410M + 2.8B on both metrics; the 1B point alone is not load-bearing. The interpretation is rhetorical, not air-tight.

10. **Detector validation imperfections.** The S-inhibition GPT-2 validation passed by rank but failed the σ-separation criterion by 0.019σ; we record the failure, accept the detector on rank-strength grounds, and drop the σ leg from forward use (§S-5c). The successor detector underwent a focused supersede before any formal validation run was recorded (§SU-1b). Both events are documented in Appendix C.

11. **Marginal/censored cells in Track 1.** 4 of 9 (size, motif) cells in the registered 3-size grid are right-censored at step 143000 or hit the upper logistic-fit sentinel. The §H1-C joint sign-test gate is satisfied partly through these cells. The §H2-9-R reframe promotes scale-dependence from a side observation to the headline reading, but does not relax the gate.

12. **Identity churn.** The emergence-step quantity is over populations of heads passing threshold, not over fixed circuits. Successor and S-inhibition identities turn over substantially across training (§4.4); this is reported but is not absorbed into the bootstrap CIs.

## 6. Discussion

**Synthesis.** *Temporal emergence order is real, but it is not a simple architectural causal chain.* Track 1 reports a pre-registered emergence-ordering claim with a clean gate pass (joint sign-test p ≈ 0.00463) and a scale-dependent reframe (§H2-9-R) — induction emerges before successor before S-inhibition at all 3 registered Pythia sizes, with the strength of confirmation scaling with model size. Track 2 directly tests whether that temporal ordering corresponds to a forward-pass causal chain at convergence, on two converging metrics across three head-count tiers, and *falsifies* the chain: ablating the top-5 successor heads at Pythia-410M and Pythia-2.8B leaves S-inhibition's §S-1 path-patching scalar and the IO−S logit-diff readout both within their NULL bands. The temporal ordering and the inference-time causal chain are decoupled. Track 3 confirms the emergence pattern on the head-count axis: at Pythia-2.8B's 1024-head architecture, S-inhibition emerges ~2.7× faster than at 410M's 384 heads and the count breaks the 410M saturation cap. The three tracks converge: there is something real about the temporal emergence order that scales with head budget, but the simplest architectural compositional reading of that ordering is wrong.

**On the Track 2 NULL × NULL as a contribution.** Direct mechanistic refutation is rare in the emergence-of-circuits literature. Olsson 2022, Singh 2024, Edelman 2024, and Tigges 2024 are all convergence- or training-dynamics-only; they do not run inference-time causal-dependence ablations between identified motifs. Gould 2024 introduces successor heads as a snapshot mechanism but does not test their causal role for downstream IOI-style circuits. The §H5-causal NULL is therefore — to our knowledge — the first published evidence that two named motifs whose emergence times are well-separated in training nonetheless operate causally disjointly at convergence in the same model. Combined with the depth-temporal asymmetry of §4.3 (S-inhibition sits *shallower* than successor at 160M / 410M, the opposite of what a chain reading predicts), the picture is consistent: the chain is not the right reading.

A weaker reading survives: *prerequisite-availability* (Singh et al., 2024). The temporal ordering may reflect which mechanisms are *available to be assembled* at training step *t*, not which mechanisms are stacked at convergence. The current data is consistent with this reading; an Extension A causal-ablation-and-fine-tuning experiment (ablate induction heads at the checkpoint right after their emergence; fine-tune; ask whether successor still forms) would test it directly. We register this as future work.

**On the Track 3 head-count-axis scaling PASS.** The §H4-supersede PASS is the cleanest scale-axis result in the paper. The (A.timing) bootstrap reversal rate is 1.000 over 1,000 paired replicates — every single resampling shows μ_si^2.8B < μ_si^410m. The (A.count) gate is marginal (max_count = 5 at step 29,000 then dips back to 4) but the trajectory is monotonic through the peak and the dip is consistent with redistribution across the wider head budget, not with non-emergence. The 1B head-count regression (§4.9) and the 2.8B PASS together establish that head count is the operationally relevant scaling axis for S-inhibition emergence under our detectors — the parameter-axis story is the wrong story.

**On the identity churn and structural reuse.** The identity churn (§4.4) tells us that "the successor population at step 143000" is not the same set of heads as "the successor population at step 25000," especially for the later motifs. The emergence step μ tracks when the count of qualifying heads passes threshold; it does not track when a *specific* circuit forms. The structural-reuse-at-small-scale finding (§4.5) is, to our knowledge, new. The vanishing of multi-motif top-5 head overlap from 70M (two cross-motif heads) to 160M (one) to 410M (none) is consistent with two mechanisms — capacity-driven role-distribution, or training-driven specialization — and the current data does not separate them. The Track 2 NULL × NULL at 410M further constrains the picture: even at 410M where the top-5 motif populations are disjoint, the two motifs operate causally independently. Whether the disjointness *causes* the causal-disjointness, or both are consequences of the wider head budget, is a question we cannot answer from this paper.

**On the absence of copy-suppression in Pythia.** McDougall et al. (2024) named *"do copy-suppression heads exist in Pythia and Llama?"* as future work. Our pilot answers, for Pythia-410M-deduped and Pythia-1B at step 143000: *not in McDougall's strict two-criterion sense.* The numerically-passing 410M head (L2H8) is mechanistically a previous-token head whose corpus-wide ablation effect on duplicate-token logits is in the *opposite* sign of the textbook copy-suppressor. Full reporting in Appendix G. We flag this as a non-trivial difference between GPT-2 and Pythia at this scale, and as a reason to be cautious about transferring named-circuit catalogs across architectures.

**On pre-registration discipline.** Pre-registration is what protects the paper from post-hoc storytelling. The Track 1 gate (joint sign-test p < 0.005) and the Track 3 gate ((A.count) ≥ 5 AND (A.timing) reversal ≥ 0.95) are exactly what we said we would test; both passed. The Track 2 NULL band ([0.8, 1.2] for Metric B; per-sender NULL classifier with DEP_THRESHOLD = 0.5 and NULL_BAND = ±0.20 for Metric A) is also exactly what we said we would compute; both metrics fell well inside the registered NULL band at the converging anchors. The three post-data reframes (§H2-9-R, §H4-7-supersede DEFERRED, §writeup-conv) move *emphasis* and *framing*, never numerical thresholds or gate predicates. The one acknowledged pre-reg-discipline gap (§H5-causal-3-record at Pythia-1B) is disclosed explicitly. We think this is the right way to handle the tension between locked statistical claims and the qualitative complexity of the data they cover.

**On future work and scope.** The single most informative extension is the causal-fine-tune one (Extension A): ablate induction heads at the checkpoint right after their emergence, fine-tune briefly, ask whether successor heads still form. This converts the consistency-with-compositional-account framing into a fully causal one. The second is the cross-architecture extension: OLMo-2, Llama, Mamba have distinct training data and architecture choices, and named-circuit catalog transfer should be tested rather than assumed; the Track 2 NULL × NULL is currently a within-Pythia claim. The third is a wider Track 3 sweep: Pythia-6.9B / 12B (NDIF-mediated) on the head-count axis, with the §H4-supersede reduced-grid protocol; the directional prediction is that S-inhibition timing should continue to accelerate but the count saturation pattern may either continue to scale or plateau. The fourth is identity-aware emergence statistics (Extension B refinement): report the first checkpoint at which the eventual top-K head set is populated to ≥ k for various *k*, alongside the count-aware emergence step. These four are listed in approximate decreasing return-on-investment.

## Acknowledgments

We thank Anthropic's Claude (Opus 4.6) for project-management and writing-assistance support. The hardware used was a single Apple M5 Pro (64 GB unified memory, MPS-only); TransformerLens (Nanda, 2022; v3.x) provided the underlying interpretability primitives; the Pythia checkpoint suite and `circuits-over-time` reference scaffolding were essential. Any errors are the author's.

## References

*[Reference list to be populated to ACL Rolling Review style. Anchor citations:]*

- Biderman, S., et al. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. *ICML*.
- Conmy, A., et al. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. *NeurIPS*.
- Edelman, B., et al. (2024). The Evolution of Statistical Induction Heads: In-context Learning Markov Chains. *arXiv*.
- Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. *Anthropic*.
- Goldowsky-Dill, N., et al. (2023). Localizing Model Behavior with Path Patching. *arXiv*.
- Gould, R., et al. (2024). Successor Heads: Recurring, Interpretable Attention Heads in the Wild. *ICLR*.
- Hanna, M., et al. (2023). How does GPT-2 compute greater-than? *NeurIPS*.
- Hanna, M., et al. (2024). Have Faith in Faithfulness: Going Beyond Circuit Faithfulness Metrics (EAP-IG). *arXiv*.
- Heimersheim, S., & Nanda, N. (2024). How to use and interpret activation patching. *arXiv*.
- L. (2023). Mechanistically interpreting time in GPT-2 small. *LessWrong*.
- McDougall, C., et al. (2024). Copy Suppression: Comprehensively Understanding an Attention Head. *BlackboxNLP*.
- Olsson, C., et al. (2022). In-context Learning and Induction Heads. *Anthropic*.
- Singh, A., et al. (2024). What needs to go right for an induction head? *arXiv*.
- Syed, A., Rager, C., & Conmy, A. (2023). Attribution Patching Outperforms Automated Circuit Discovery. *arXiv*.
- Tigges, C., et al. (2024). LLM Circuit Analyses Are Consistent Across Training and Scale. *NeurIPS*.
- Wang, K., et al. (2023). Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 small. *ICLR*.

---

# Appendix

## A. Reproducibility

The full project repository (`pythia-motif-emergence`) is released. Each main figure has a dedicated notebook:


| Figure                                | Notebook                                                                                                                  |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Figure 1 (3-panel emergence)          | `notebooks/h1c_ordering_test.ipynb`                                                                                       |
| Figure 2 (per-motif sweep panels)     | `notebooks/induction_full_sweep.ipynb`, `notebooks/successor_full_sweep.ipynb`, `notebooks/s_inhibition_full_sweep.ipynb` |
| Figure 3 (depth-over-training)        | `notebooks/h1c_ordering_test.ipynb` (sub-deliverable 4b)                                                                  |
| Figure 4 (Jaccard / structural reuse) | `notebooks/motif_structural_reuse.ipynb`                                                                                  |
| Figure 5 (representative attention)   | `notebooks/motif_attention_inspection.ipynb`                                                                              |


Canonical outputs:

- `data/exploration/phase2_induction_sweep.parquet`
- `data/exploration/phase2_successor_sweep.parquet`
- `data/exploration/phase2_s_inhibition_sweep.parquet`

Pilot artifacts (Path C registration):

- `data/pilot/copy_suppression_pythia_410m_step143000.parquet`
- `data/pilot/copy_suppression_pythia_410m_step143000_per_position.npz`

Hardware: Apple M5 Pro, 64 GB unified memory, MPS-only (PyTorch 2.11.0 with `PYTORCH_ENABLE_MPS_FALLBACK=1`). Python 3.12 (uv-managed). TransformerLens ≥ 3.x.

Total compute budget for the full Phase 2 sweep: ~3 hours MPS time for the three motif sweeps + ~30–60 minutes for checkpoint prefetch + ~2 minutes for bootstrap post-processing on cached scores + trivial logistic-fit time.

## B. Detector validation against GPT-2 reference heads

### B.1 Induction (Olsson prefix-matching)

All 384 heads of Pythia-410M-deduped @ step 143000 screened on 50 random repeated sequences. 11 heads have prefix-match score *>0.5*; 19 *>0.3*. Top-10:


| Rank | (L, H) | Score |
| ---- | ------ | ----- |
| 1    | L11H14 | 0.953 |
| 2    | L11H2  | 0.937 |
| 3    | L7H1   | 0.930 |
| 4    | L8H6   | 0.831 |
| 5    | L10H9  | 0.817 |
| 6    | L10H3  | 0.815 |
| 7    | L8H7   | 0.789 |
| 8    | L6H2   | 0.775 |
| 9    | L10H0  | 0.663 |
| 10   | L15H0  | 0.588 |


Causal ablation on 50 random repeated sequences: baseline NLL = 0.872, top-1 acc = 0.931. Ablating L11H14 alone increases NLL by +0.401 (acc → 0.891). Ablating top-3 jointly increases NLL by +0.742 (acc → 0.848). Control L0H9 ablation produces ΔNLL = −0.013 (noise level).

### B.2 Successor (lift form, §SU-1b)

GPT-2 small validation screen, all 144 heads. Top-5 by lift:


| Rank | (L, H)            | Lift   | Real DLA | Null DLA |
| ---- | ----------------- | ------ | -------- | -------- |
| 1    | **L9H1** (target) | +0.392 | +0.615   | +0.223   |
| 2    | L8H8              | +0.306 | +2.153   | +1.847   |
| 3    | L11H10            | +0.240 | −1.734   | −1.974   |
| 4    | L11H11            | +0.215 | +2.236   | +2.021   |
| 5    | L6H5              | +0.203 | −0.941   | −1.144   |


τ_lift (95th-pct of pooled per-head lifts) = **0.13496**. L9H1 lift = +0.392 ≫ τ_lift; rank #1 of 144. Conjunctive gate PASS.

L9H1 per-category lift breakdown: days +0.168, months +0.829, numerals +0.557, letters +0.013. The letters category is a project addition (not in Gould et al. 2024 / L 2023); L9H1 is marginal there.

Independent corroboration (L 2023 argmax-within-7-days protocol, run as a §SU-1b-justification probe): L9H1 is the unique 7-of-7 head among 144 in GPT-2 small. All 143 others score ≤3/7 (most 1/7 = chance).

§SU-1b-motivation diagnostic (raw real-DLA detector under §SU-1, since superseded): L10H3 had real DLA = +10.37 but lift = −1.90 (anti-successor); raw real-DLA detector ranked L9H1 at #36 of 144. The lift form recovers the correct ordering and is the locked detector form.

### B.3 S-inhibition (path patching, Wang-style frozen paths)

GPT-2 small validation screen, all 144 heads. Top-5 by Δ_h:


| Rank | (L, H) | Δ_h     | Wang-published? |
| ---- | ------ | ------- | --------------- |
| 1    | L8H6   | +0.220  | yes             |
| 2    | L8H10  | +0.0493 | yes             |
| 3    | L7H9   | +0.0402 | yes             |
| 4    | L7H3   | +0.0372 | yes             |
| 5    | L9H4   | +0.0279 | no              |


Wang's four heads occupy ranks #1–#4. Top-8 inclusion gate: PASS. Wang min Δ_h (L7H3 = 0.0372) exceeds non-Wang max (L9H4 = 0.0279) by factor 1.33×; non-Wang 99th percentile = 0.0194; Wang min > non-Wang 99th-pct: PASS.

σ-separation gate (locked criterion in §S-5): Wang median Δ_h (NumPy convention) = 0.04475 sits at +1.981σ above bulk mean across 144 heads — short of the locked +2σ threshold by **0.019σ**. The σ-statistic failure is driven entirely by L8H6's outlier Δ_h inflating bulk SD when included per the locked no-leave-one-out rule. Leave-Wang-out σ-separation = +4.574σ.

§S-5c documents the failure, accepts the detector on rank-strength grounds, and drops the σ leg from forward use as a pathological criterion under outlier known-positives. The pre-registration record reflects the gate FAIL with rank-only override.

Component-DLA NM identification on GPT-2 small per §S-3: top-4 = {L9H9, L9H6, L10H0, **L10H6**}. Three of four match Wang's published Name Movers; L10H6 is the §S-3-anticipated divergence and is recorded as a methodological finding. Pythia NMs are re-derived per-model (not transferred from GPT-2).

τ_strict = 0.0372 (locked in §S-tau). τ_permissive = 0.0186.

### B.4 Tigges IOI replication (Phase 1.2 gate)

Pre-Phase-2 tooling-validation milestone. Pythia-410M-deduped on N = 200 IOI prompts (100 BABA + 100 ABBA, seed 0). IOI accuracy at directly-shared checkpoints with Tigges et al. 2024 Figure 2:


| Step   | Ours   | Tigges 2024 | abs-diff |
| ------ | ------ | ----------- | -------- |
| 1000   | 0.495  | 0.4286      | 0.0664   |
| 50000  | ≈ 0.99 | ≈ 0.985     | 0.005    |
| 100000 | ≈ 0.99 | ≈ 0.985     | 0.005    |
| 143000 | 0.99   | 0.985       | 0.005    |


Max abs-diff = **0.0664** at step 1000 (sample-size variance: ours N=200, Tigges N=70). Below the locked 0.10 tolerance. Gate PASS.

Component-DLA top-positive heads at step 143000 (read on the (IO − S) logit difference): L12H12 (1.337), L17H10 (0.801), L14H0 (0.647), L20H15 (0.473), L18H12 (0.427). Negative-NM-like: L15H0 (−0.221), L19H10 (−0.185).

## C. Pre-registration audit trail

The full HYPOTHESIS.md amendment chronology:


| Date                                        | Amendment                        | Effect                                                                                                                                                                                      |
| ------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-05-04                                  | Initial commit                   | H1-A and H1-C pre-registered; pilot decision rule locked.                                                                                                                                   |
| 2026-05-05 (morning)                        | §1 (validation reframing)        | Dual-report copy-suppression validation; calibrated supplementary scheme added (later dropped).                                                                                             |
| 2026-05-05 (afternoon, Path C registration) | `PILOT_RESULTS.md` finalized     | Path A → Path C pivot. L2H8 numerical-pass overridden by qualitative inspection.                                                                                                            |
| 2026-05-05 (evening)                        | §S-1..§S-9 (S-inhibition spec)   | Wang-style path-patching detector locked. Numerical τ_strict deferred.                                                                                                                      |
| 2026-05-05 (post-validation)                | §S-5b/c, §S-tau                  | Median convention resolution; rank-only acceptance after σ-FAIL by 0.019σ; τ_strict = 0.0372 locked.                                                                                        |
| 2026-05-05 (later evening)                  | §SU-0..§SU-8 (successor spec)    | Cross-category DLA detector locked. Validation target attribution corrected (GPT-2 small L9H1 / L 2023, not GPT-2 medium / Gould 2024).                                                     |
| 2026-05-06 (early)                          | §SU-1b (lift-form supersede)     | Score definition changed from real-DLA to lift = real − null; smoke-test pre-data failure documented.                                                                                       |
| 2026-05-06                                  | §SU-tau                          | τ_lift = 0.13496 locked from validation distribution.                                                                                                                                       |
| 2026-05-06                                  | §H2 (Phase 2 sweep spec)         | 40-cell grid, B = 1000 bootstrap, ±25% threshold sensitivity, tiered censoring, ties-fail, multiple-comparison policy. No deferred number.                                                  |
| 2026-05-06 (post-Phase-2)                   | §H2-9-R (interpretation reframe) | Gate passes as registered; headline shifts to scale-dependent emergence; bootstrap reversal-rate replaces nominal "descriptive p-values"; threshold-sensitivity panel re-run with B = 1000. |
| 2026-05-06 (post-grilling)                  | §H3-scale (1B scale-extension)   | 5-leg conjunctive gate for Pythia-1B; pre-registered before any 1B compute. Verdict landed as REGR (1B is a head-count regression; A.ii reversal_rate = 1.000 but A.count fails).                                                                                              |
| 2026-05-07                                  | §H3-scale-8-vis                  | Visualization-layer supersede of the 3-size lock; 1B added as a 4th column in per-motif sweep notebooks, presentation-only, does not extend §H1-C.                                                                                                                              |
| 2026-05-07                                  | §H4-scaling (head-count axis)    | 2.8B head-count-axis test pre-registered with (A.timing) + (A.count) conjunctive gate; head-count vs parameter-count axis distinction registered explicitly.                                                                                                                  |
| 2026-05-07                                  | §H5-causal (Track 2, Metric A)   | Mean-ablation on hook_z; §S-1 path-patching readout at 410M; suc / ctrl / NM / bootstrap protocols locked pre-data.                                                                                                                                                              |
| 2026-05-07                                  | §H5-causal-2 (Track 2, Metric B) | IO−S logit-diff readout at END; converging metric registered pre-data after Metric A; verdict bands [0.8, 1.2] NULL.                                                                                                                                                              |
| 2026-05-08                                  | §H4-7-supersede (DEFERRED)       | 2.8B S-inhibition sweep halted at 8/40 cells under §H4-7 per-cell-cost escape hatch; DEFERRED pattern added to §H4-5 priority `DEFERRED > TOOLING > NEITHER > COUNT-ONLY > TIMING-ONLY > PASS`.                                                                                  |
| 2026-05-10                                  | §H5-causal-3-record              | **Post-data canonicalization** of the Pythia-1B Track 2 anchor (which ran on a feature worktree 2026-05-07 before formal amendment). Pre-reg discipline gap disclosed in §H5-causal-3-record-4 and §4.7 of this paper. Verdicts: Metric A NULL, Metric B MIXED, cross-metric MIXED. |
| 2026-05-10                                  | §H5-causal-3-2.8b                | 2.8B Track 2 pre-data lock. Sets (suc / SI / NMs) locked from sealed parquets with bit-for-bit runtime assertion; ctrl procedure-locked with NM-exclusion clause.                                                                                                                |
| 2026-05-10                                  | §H4-supersede                    | Reduced 10-cell grid `[5000, 7000, 10000, 14000, 20000, 29000, 41000, 49000, 59000, 70000]` for 2.8B S-inhibition; verbatim §H4-2 gate predicates inherited; pre-data framing locked as "scaling appendix".                                                                       |
| 2026-05-11                                  | §writeup-conv                    | Post-data documentation-hygiene amendment (analog of §H2-9-R). Locks Track-numbering convention (T1=Emergence, T2=Causal, T3=Scaling); upgrades §H4-supersede framing to "third converging substantive result" post-PASS; rectifies two chronology errors in earlier amendment text. Non-numerical.    |


## D. Threshold-sensitivity tables

For each (size, motif) cell, μ recovery is reported across the locked threshold *× {0.75, 0.875, 1.0, 1.125, 1.25}*. *(Full table to be inlined from `notebooks/h1c_ordering_test.ipynb` sub-deliverable 6 final B = 1000 panel.)* Pythia-410M cells are robust across the full bracket; Pythia-160M induction is the most sensitive (μ ranges 2986–10001). Marginal cells widen the bracket monotonically.

## E. Per-checkpoint raw counts

*(Full 9 × 40 table to be inlined from the three `*_full_sweep.ipynb` notebooks at camera-ready. Schema: (size, step, motif, count_above_threshold, max_score).)*

## F. Notebook → figure crosswalk


| Section | Claim                                       | Notebook                                                                                    |
| ------- | ------------------------------------------- | ------------------------------------------------------------------------------------------- |
| §4.1    | Joint sign-test PASS, μ table               | `h1c_ordering_test.ipynb`                                                                   |
| §4.2    | Per-size emergence, marginal/censored cells | `induction_full_sweep.ipynb`, `successor_full_sweep.ipynb`, `s_inhibition_full_sweep.ipynb` |
| §4.3    | Depth-vs-temporal asymmetry                 | `h1c_ordering_test.ipynb` (sub-deliverable 4b)                                              |
| §4.4    | Identity churn                              | `h1c_ordering_test.ipynb` (sub-deliverable 5) + per-motif sweep notebooks                   |
| §4.5    | Structural reuse Jaccard                    | `motif_structural_reuse.ipynb`                                                              |
| §4.6    | Track 2 410M NULL × NULL                    | `causal_dependence.ipynb` (§H5-causal + §H5-causal-2 sections)                              |
| §4.7    | Track 2 1B Metric A NULL, Metric B MIXED    | `causal_dependence.ipynb` (§H5-causal-3 1B verdict cell)                                    |
| §4.8    | Track 2 2.8B NULL × NULL                    | `causal_dependence.ipynb` (§H5-causal-3-2.8b verdict cell + cross-size table)               |
| §4.9    | Head-count rationale + 1B REGR              | `h1c_ordering_test.ipynb` (§H3-scale section + §H4-1 head-count-axis sub-cell)               |
| §4.10   | Track 3 §H4-supersede PASS                  | `h1c_ordering_test.ipynb` (§H4-supersede verdict cell)                                       |
| §4.11   | Count-trajectory dip caveat                 | `h1c_ordering_test.ipynb` (§H4-supersede count-vs-step printout)                             |
| §B.1    | Induction validation                        | `induction_heads_proof.ipynb`                                                               |
| §B.2    | Successor validation                        | `successor_proof.ipynb`                                                                     |
| §B.3    | S-inhibition validation                     | `s_inhibition_proof.ipynb`                                                                  |
| §B.4    | Tigges replication gate                     | `tigges_ioi_replication.ipynb`                                                              |
| §G      | Path A copy-suppression negative result     | `copy_suppression_pythia_proof.ipynb` + `PILOT_RESULTS.md`                                  |
| §H      | Per-(size, metric) Track 2 verdict table    | `causal_dependence.ipynb` (cross-size summary cell)                                          |


## G. Path A negative result — copy-suppression heads in Pythia

Following the pre-registered Path A pilot, we apply the McDougall two-criterion detector (QK > 0.3 AND OV < 0) to all 384 heads of Pythia-410M-deduped at step 143000 on a 7,521-token canonical corpus (40 Wikipedia featured-article opening passages; 2,769 eligible duplicate positions). We also extend the screen to Pythia-1B (128 heads) at the same checkpoint.

**Numerical result at Pythia-410M.** Exactly one head (L2H8) passes both strict thresholds (QK = 0.328, OV = −0.023). Both passes are marginal.

**Mechanistic check at Pythia-410M.** L2H8 has a textbook duplicate-attending QK pattern at the rank-1 worked-example position (attention from " Inside" at position 115 back to " Inside" at position 105 in the Stonehenge passage = 0.910, stronger than even GPT-2 small L10H7's typical attention-to-prior). Its corpus-wide ablation effect on duplicate-token logits across all 2,769 positions is **−0.0092** — i.e., ablating L2H8 *lowers* duplicate-token logits, the opposite sign of suppression. The reference benchmark GPT-2 small L10H7's corpus-wide ablation effect on the *same canonical corpus* is **+0.0317**.

The four supplementary candidates at 410M (top by most-negative OV with QK ≥ 0.05) all show similar negative corpus-wide d-logit (range −0.007 to −0.115). **No head in Pythia-410M-deduped at step 143000 implements McDougall's attend-then-suppress mechanism.** L2H8 is best characterized as a previous-token / induction-precursor head (per Singh et al., 2024).

**Pythia-1B extension.** Applying the strict McDougall two-criterion detector to all 128 heads of Pythia-1B at step 143000 returns **0/128 heads passing both thresholds**. The null is even cleaner at 1B than at 410M; the McDougall mechanism is absent throughout the Pythia checkpoint suite at this scale, not just at 410M.

We do not claim the absence of copy-suppression in Pythia at *all* scales — we checked 410M and 1B only — nor in the GPT-NeoX architecture in general. We note that the threshold-transfer issue (L10H7 itself fails QK > 0.3 on raw text; the threshold was inherited from filtered data in McDougall's analysis) replicates on Pythia, so the correct interpretation is *the McDougall-2024-strict mechanism is absent in Pythia-410M-deduped + Pythia-1B-deduped*, not *no head in Pythia attends to duplicates*. Pythia-2.8B / 6.9B are accessible via the existing HuggingFace cache (we already prefetched 2.8B for Tracks 2 + 3) and are obvious follow-up targets.

## H. Per-(size, metric) Track 2 verdict table

| Size | Heads | Step | Metric A (§S-1 path-patching) | Metric B (IO−S logit-diff) | Cross-metric |
| --- | --- | --- | --- | --- | --- |
| Pythia-410M | 384 | 143,000 | **NULL** (3/3 senders; ratio_suc ≈ 1.0, ratio_ctrl ≈ 1.0, tight CIs) | **NULL** (ratio_suc = 0.986 [0.978, 0.992]; ratio_ctrl = 0.979 [0.968, 0.991]) | **NULL** |
| Pythia-1B | 128 (regression) | 143,000 | **NULL** (3/3 senders) | **MIXED** (ratio_suc = 0.790 [0.773, 0.809]; ratio_ctrl = 0.797 [0.776, 0.813] — both ablations drop ~21% similarly) | **MIXED** (narrow-architecture readout artifact, not falsification) |
| Pythia-2.8B | 1024 | 143,000 | **NULL** (3/3 senders; ratio_suc = 1.001 [tight], ratio_ctrl = 0.997 [tight]) | **NULL** (ratio_suc = 0.984 [0.980, 0.987]; ratio_ctrl = 0.984 [0.978, 0.989]) | **NULL** |

**Locked sets** (asserted bit-for-bit at runtime; full per-size table in `causal_dependence.ipynb`):

| Size | suc (top-5) | SI senders (top-3) | NMs (top-4 by component-DLA) | ctrl (procedure-derived) | Bracket width |
| --- | --- | --- | --- | --- | --- |
| 410M | (22,6), (22,2), (20,4), (22,10), (12,8) | (12,12), (13,13), (14,0) | (12,12), (17,10), (14,0), (20,15) | (17,12), (20,6), (22,11), (23,10), (23,13) | 0.10 |
| 1B | (11,6), (14,2), (12,3), (15,7), (15,1) | (8,7), (9,1), (10,4) | (11,0), (11,5), (14,2), (11,2) | (11,1), (11,3), (11,4), (12,2), (13,0) | 0.125 |
| 2.8B | (15,14), (28,17), (27,13), (13,10), (29,28) | (11,29), (11,5), (13,9) | (11,29), (17,12), (22,31), (13,9) | (13,5), (13,8), (13,27), (20,29), (24,25) | 0.075 |

## I. Structural-reuse deep dive (cross-size, cross-step)

Extends §4.5's step-143000 top-5 view to the full (size, step) sweep at the detection-population level. For each cell, we count heads above each motif's locked threshold (induction > 0.30; successor lift_dla ≥ τ_lift = 0.13496; S-inhibition Δ_h ≥ τ_strict = 0.0372) and intersect the populations.

**Table I.1 — successor ∩ S-inhibition overlap across training, per size:**

| Size | Steps with suc ∩ si > 0 | Total steps | Heads ever in overlap (across all steps) |
|---|---|---|---|
| 70M | 0 | 40 | — |
| 160M | 1 | 40 | (8, 9) [step 143000 only] |
| 410M | 2 | 40 | (13, 13) [steps 41,000 and 70,000] |
| 1B | **0** | 40 | — |
| 2.8B | 1 | 10 | (13, 8) [step 29,000 only] |

The suc–SI populations are structurally disjoint across every (size, step) cell. The single overlap events at 160M / 410M / 2.8B are isolated single-head, single-step events that do not coincide with any §H5 anchor.

**Table I.2 — induction ∩ S-inhibition overlap across training, per size:**

| Size | Steps with ind ∩ si > 0 | Total steps | Top recurring head |
|---|---|---|---|
| 70M | 0 | 40 | — |
| 160M | 0 | 40 | — |
| 410M | 3 | 40 | (17, 10) — IS a §H5 Name Mover |
| 1B | 6 | 40 | (8, 7) — IS the top §H5-causal-3-record SI sender |
| 2.8B | 5 | 10 | (13, 9) — triple-role: induction + SI sender + NM |

Induction and S-inhibition share head populations at the larger sizes; the recurring overlap heads frequently serve as Name Movers or SI senders in the §H5 protocol. This is mechanistically expected: Name Movers' QK circuit attends from END to the IO position, which is a previous-occurrence-of-the-co-referent-name token — a QK pattern closely related to Olsson induction.

**Table I.3 — Cross-reference of locked §H5 sets with detector populations at the §H5 anchors:**

| Cell | suc_top5 ∩ si_pop | si_top3 ∩ suc_pop | si_top3 ∩ nm_top4 | ind_pop ∩ si_top3 | ind_pop ∩ nm_top4 |
|---|---|---|---|---|---|
| 410M step 143000 | [] | [] | (12,12), (14,0) | [] | (17,10) |
| 1B step 143000 | [] | [] | [] | [] | (11,5) |
| 2.8B step 143000 | [] | [] | (11,29), (13,9) | (13,9) | (13,9), (17,12) |

At every §H5 anchor: zero overlap between the locked-ablated suc top-5 and the SI-detected population; zero overlap between the locked-readout SI top-3 and the suc-detected population. The Track 2 NULL × NULL is the expected readout when two structurally disjoint populations are independently probed.

**Mechanistic reading.** Successor's OV writes the next-ordinal-direction; S-inhibition's OV writes a duplicate-name suppression signal that NMs consume to disambiguate IO from S. The two output directions are unrelated, so a single head's OV cannot productively write both simultaneously. The single fleeting suc ∩ si overlap at 410M (head (13,13) at steps 41k / 70k, suc-detected briefly during mid-training while it is also SI-detected) is interesting as a transient phenomenon — (13,13) is the second-ranked §H5 SI sender by step 143000, so it is doing SI work at convergence; the brief crossing of τ_lift in mid-training is consistent with role-distribution settling. Induction-and-SI overlap, by contrast, is robust at the larger sizes because the NM circuit implements a QK pattern that the Olsson detector also fires on (attend from END to position-after-previous-occurrence-of-the-co-referent-name).

**Connecting to the §H5-causal-3-record 1B Metric B MIXED.** At Pythia-1B (the head-count regression with 128 heads), suc ∩ si has *zero* overlap across all 40 steps (Table I.1). If the 1B Metric B MIXED were a structural-reuse artifact — i.e., the suc ablation accidentally hit an SI-detected head — the structural data would show that overlap. It does not. The MIXED is therefore a property of 1B's narrow architecture making the IO−S logit-diff readout generically ablation-sensitive, not a hidden structural reuse. This is the §writeup-conv-2 reframe restated mechanistically.

**Data product.** `data/exploration/structural_reuse_deep_dive.parquet` (170 rows): per-(size, step) overlap counts for all three pairwise intersections + triple intersection. Pure analytical extraction; no new compute.

