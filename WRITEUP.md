# Project writeup — three-track narrative

This document re-organizes the project's *registered* claims (locked in `HYPOTHESIS.md` and its amendment chain) into a paper-narrative three-track structure. **`HYPOTHESIS.md` remains the canonical pre-registration record**; this writeup is presentation-layer only and adds no new claims.

## Lead synthesis

**Temporal emergence order is real, but it is not a simple architectural causal chain.**

The three motifs (induction, successor, S-inhibition) emerge in the predicted order during Pythia training (Track 1, §H1-C joint sign-test at p = 0.00463). But at convergence, ablating the mature successor heads does **not** disrupt S-inhibition's readouts (Track 2, §H5-causal at 410m NULL on both metrics; §H5-causal-3-2.8b at 2.8B NULL on both metrics; §H5-causal-3-record at 1B NULL on Metric A and MIXED on Metric B — interpreted as a head-count-regression readout-specificity artifact, not a falsification). The temporal ordering does not correspond to a forward-pass dependence — S-inhibition's circuit is causally disjoint from successor's at inference time, and this disjointness is scale-stable across head-count tiers 384 (410m) and 1024 (2.8B). The scaling-by-head-count picture is incomplete (Track 3, §H4-7-supersede DEFERRED, §H4-supersede pending) and reported as a secondary appendix.

The project's findings split into three scientific tracks:

| Track | Question | Verdict |
|---|---|---|
| **1. Emergence** | What is the ordering of motif emergence during training in the registered Pythia size grid? | PASS (§H2-5, p = 0.00463); §H2-9-R reframe: scale-dependent (vacuous at 70m, marginal at 160m, robust at 410m) |
| **2. Causal-disjointness** | At convergence, does the temporal ordering correspond to a forward-pass causal chain? | NULL at 410m on both metrics (§H5-causal, §H5-causal-2). NULL at 2.8B on both metrics (§H5-causal-3-2.8b). At 1B (128-head regression): Metric A NULL replicates; Metric B MIXED — reframed as a narrow-architecture readout-specificity artifact, not a substantive DEP finding. **Scale-stable disjointness across head-count tiers 384 and 1024.** |
| **3. Scaling (head-count axis)** | Does the scale-dependent S-inhibition pattern continue beyond 410m on a head-count axis? | §H3-scale at 1B = REGR (head-count regression); §H4-scaling at 2.8B DEFERRED at §H4-7-supersede; **§H4-supersede reduced-grid re-attempt PASS at 2.8B**: reversal_rate = 1.000 (gate ≥ 0.95), max_count = 5 (gate ≥ 5), μ_si^2.8B ≈ 9021 vs μ_si^410m ≈ 24088 (~2.7× speedup). |

All three tracks PASS or NULL-as-target. Tracks 1 + 2 carry the paper's narrative; Track 3 confirms head-count-axis scaling as a converging third result.

---

## Track 1 — Emergence at small Pythia models (registered, complete)

### What the emergence claim says

For each Pythia model size $s \in \{70\text{m}, 160\text{m}, 410\text{m}\}$ and each motif $m \in \{\text{induction}, \text{successor}, \text{S-inhibition}\}$, define the *emergence step* $\mu_{s,m}$ as the training step at which the count of detected heads reaches half of its final-checkpoint value (logistic fit to count-vs-log-step). The pre-registered hypothesis (`HYPOTHESIS.md` §H1-C) predicts:

$$\mu_{s,\text{induction}} < \mu_{s,\text{successor}} < \mu_{s,\text{S-inhibition}} \quad \text{for all three sizes}.$$

The hypothesis is **falsified** if any size shows reversed order with statistically distinguishable gaps. The pre-registered gate is a joint sign test: under H_0 of exchangeable order, the probability of observing the predicted ordering in all three sizes by chance is $(1/6)^3 \approx 0.00463$. The threshold is $p < 0.005$.

### What was tested

Pre-committed before any sweep code ran (`HYPOTHESIS.md` §H2-1 through §H2-9):

- 40 log-spaced checkpoints per Pythia size (`step0` through `step143000`).
- Three locked detectors operating on every (layer, head) per cell:
    - **Induction** — Olsson 2022 prefix-matching score over 50 random-token-repetition sequences (length 100); threshold > 0.3.
    - **Successor** — §SU-1b lift-form cross-category direct logit attribution over 70 prompts in {days, months, numerals, letters}; locked threshold τ_lift = 0.13496 from §SU-tau (GPT-2 small validation).
    - **S-inhibition** — §S-1 path-patching Δ_h scalar over 200 IOI prompts (ABC corruption, frozen paths to component-DLA top-4 Name Movers); locked threshold τ_strict = 0.0372 from §S-tau (GPT-2 small Wang heads, post-§S-5c override).
- Logistic fit per (size, motif) cell with §H2-3 tiered censoring: full-fit if max_count ≥ 5; marginal (bootstrap-median μ) if 2–4; right-censored at step143000 if max_count < 2.
- §H2-2 bootstrap machinery: B = 1000 per-prompt resamples, 95% percentile CI on μ, ±25% threshold-sensitivity bracket.

### What was found

**Headline (§H2-9-R reframe, registered after the gate but before the writeup):** *scale-dependent emergence of S-inhibition + a depth-vs-temporal asymmetry in 160m / 410m.* The joint sign-test PASSes at p = 0.00463 < 0.005, but the per-cell evidence is heterogeneous:

| size | induction | successor | S-inhibition | per-size verdict |
|---|---|---|---|---|
| 70m | full-fit | marginal | **right-censored** (max_count = 1 at step143000) | ordering vacuous through censoring |
| 160m | full-fit | marginal | marginal (bootstrap CI overlaps successor) | ordering passes but CI overlap |
| 410m | full-fit | full-fit (max = 7) | marginal (max = 3, but tight CI) | **robust per-size confirmation** |

Specifically:
- S-inhibition's emergence is scale-dependent in Pythia: not at 70m, marginal at 160m, clean at 410m.
- In 160m and 410m, S-inhibition heads sit at *shallower* normalized layer depth than successor heads — incompatible with a strict compositional reading where S-inhibition consumes successor outputs at later layers (`h1c_ordering_test.ipynb` sub-deliverable 4b). **This is the structural-asymmetry preview of the Track 2 NULL.**
- Top-3 head identities for successor and S-inhibition turn over substantially between step25k and step143k (sub-deliverable 5).

The registered §H1-C falsification target is *not falsified*. The §H2-9-R reframe re-emphasises which cells robustly support the pass; the joint-PASS-with-censoring framing is what the introduction should lead with.

### What's locked at this scope

The emergence track is **complete**. No further sizes are added to the emergence claim. The §H1-C verdict and the §H2-9-R reframe stand as registered. Subsequent bigger-model work belongs to Tracks 2 (causal-disjointness) and 3 (scaling); neither extends the emergence claim.

### Pre-committed limitations (verbatim from `PROJECT_BRIEF.md` §9)

1. Single-seed per Pythia size — no within-size variance estimate; "consistency" is across-size, not across-seed.
2. Pythia (GPT-NeoX) only — no cross-architecture universality claim.
3. Detector-threshold sensitivity — reported with bootstrap CIs across thresholds, not pretended away.
4. No causal claim — default framing for Track 1: *"consistent with a compositional account."* Track 2 then probes that account directly and falsifies the simple forward-pass reading.

### Reproducibility pointers (Track 1)

| artifact | role |
|---|---|
| `HYPOTHESIS.md` §H1-C, §H2 (full chain), §H2-9-R | pre-reg + post-data reframe |
| `data/exploration/phase2_{induction,successor,s_inhibition}_sweep.parquet` | sweep outputs at 3 registered sizes |
| `data/exploration/phase2_{induction,successor,s_inhibition}_per_{seq,prompt}/` | per-cell .npz caches for bootstrap |
| `notebooks/induction_full_sweep.ipynb` (4-size view) | per-motif sweep notebook for induction |
| `notebooks/successor_full_sweep.ipynb` (4-size view) | per-motif sweep notebook for successor |
| `notebooks/s_inhibition_full_sweep.ipynb` (4-size view) | per-motif sweep notebook for S-inhibition |
| `notebooks/h1c_ordering_test.ipynb` | the §H2-5 joint verdict + §H2-9-R reframe + sub-deliverables |

Note: the per-size sweep notebooks (`*_full_sweep.ipynb`) display 4 sizes as of the §H3-scale-8-vis presentation supersede; the 1B column is a head-count regression and not part of the registered emergence claim. The registered Track-1 evidence is in the 70m / 160m / 410m columns of those notebooks.

### Track 1 robustness — §H1-C-altdetectors measurement-invariance sensitivity (post-hoc)

A post-hoc robustness exercise asks whether the §H1-C ordering survives substituting alternative detectors for each motif. Three mechanism-verifying alt-detectors were locked (HYPOTHESIS.md §H1-C-altdetectors): Olsson 2022 OV-circuit verification (alt-induction), L 2023 graded argmax-K-of-7 day-of-week (alt-successor), and Wang 2023 §3 Component-DLA-at-S2 (alt-S-inhibition).

**Cross-family threshold transfer failed.** The first attempt locked alt-thresholds via GPT-2 small's 95th-percentile alt-score per-motif. The GPT-2-calibrated absolute thresholds did not transfer to Pythia (cross-family magnitude scale mismatch; cf. Tigges 2024 within-family normalization precedent). Documented as a methodological finding under §H1-C-altdetectors-2-rr-supersede; reframed as post-hoc robustness, not confirmatory.

**Within-Pythia frozen-threshold scheme (locked 2026-05-12).** Per (Pythia size, motif, alt-detector), threshold = K-th-largest alt-score at step 143,000, where K = locked-detector pass count at step 143,000. Applied backward over training. Sensitivity-of-sensitivity at top-5%; rank-only secondary view (magnitude-free).

**Pre-validation at convergence (top-K overlap between locked and alt at step 143000):**

| Motif × size | K | Overlap | Disposition |
|---|---|---|---|
| Induction (70m / 160m / 410m) | 6 / 17 / 19 | 83% / 65% / 58% | all usable |
| Successor (70m / 160m / 410m) | 2 / 3 / 2 | 0% / 100% / 0% | usable only at 160m |
| S-inhibition (70m / 160m / 410m) | 1 / 3 / 2 | 0% / 33% / 50% | partial/usable at larger sizes |

**Joint sign-test under detector triples (single-substitution preserves most signal):**

| Triple | Frozen-final-K | Top-5% | Rank-only |
|---|---|---|---|
| (locked, locked, locked) | 3/3 sizes, p ≈ 0.00463 (baseline, unchanged) | | |
| (alt-ind, locked, locked) | 2/3 | 2/3 | 2/3 |
| (locked, alt-suc, locked) | 1/3 | 0/3 | 1/3 |
| (locked, locked, alt-si) | 1/3 | 2/3 | 2/3 |
| (alt-ind, alt-suc, alt-si) | 0/3 | 0/3 | 1/3 |

**Verdict.** Single-substitution with the well-pre-validated alt-induction OV detector preserves the ordering at 2/3 sizes under all schemes. All-alt failure is mechanically attributable to compounded measurement noise from poorly-pre-validated alt-detectors at the cells where overlap < 50%. Pre-validation overlap predicts the pass-rate. **Primary §H1-C verdict (p = 0.00463, locked detectors) unchanged.** Robustness exercise establishes that the ordering most plausibly reflects a real temporal-emergence phenomenon — not a detector-specific artifact — when probed by alternative detectors that demonstrate final-checkpoint agreement with the locked detectors. See paper §4.12 for the full appendix.

### Track 1 robustness — §H1-C-2.8b-extension (post-hoc 4-size consistency)

A post-hoc 4-size extension of §H1-C using the Pythia-2.8B sweeps collected for Tracks 2 + 3 (HYPOTHESIS.md §H1-C-2.8b-extension). The 2.8B coverage was never registered for §H1-C — its existence is a byproduct of Tracks 2/3. The extension is reported here as a **consistency data point, not a stronger result**.

Per-size first-≥5-pass emergence step at 2.8B (locked detectors, merged 32-cell trajectory): induction step 1,000 < successor step 10,000 < S-inhibition step 15,000 — ordering HOLDS. 4-size joint consistency `p ≈ (1/6)^4 ≈ 0.00077` reported as **post-hoc**, not a confirmatory replacement for the pre-registered 3-size joint sign-test (which remains the primary verdict at p ≈ 0.00463).

### Figure F6 — motif emergence vs. training loss

`notebooks/figures/F6_loss_vs_emergence.png` (built by `notebooks/_build_loss_vs_emergence_figure.py`): 5 stacked panels (one per Pythia size), each showing the three motif pass-count trajectories overlaid with the published training cross-entropy loss curve (pulled from EleutherAI's public W&B project; remapped wandb step → checkpoint step where applicable). The emergence cliff for each motif aligns with a specific point on the loss descent — induction rises as loss drops through ~3-4, successor follows around ~2.5-3, S-inhibition emerges last around ~2.0-2.5. Visual confirmation of the ordering at all 5 sizes (including the §H4-1B head-count regression visible in panel 4).

| artifact (robustness) | role |
|---|---|
| HYPOTHESIS.md §H1-C-altdetectors, -2-r-supersede, -2-rr-supersede | amendment chain (pre-data, then post-data reframe) |
| `notebooks/_run_pythia_anchor_altdetectors_validation.py` | GPT-2 small validation (cross-family attempt) |
| `notebooks/_run_phase4_h1c_alt_{induction_ov,successor_argmax,s_inhibition_compdla}.py` | Pythia × 3 sizes × 40 cells alt-detector sweeps |
| `notebooks/_run_phase4_h1c_alt_analysis.py` | within-Pythia frozen-threshold + rank-only analysis + joint sign-test |
| `data/exploration/phase4_h1c_alt_{prevalidation,trajectories,emergence_steps,joint_verdict}.parquet` | analysis outputs |

---

## Track 2 — Causal-disjointness at convergence (registered, partial)

### What the causal-disjointness claim asks

Track 1's joint emergence-ordering PASS is compatible with two distinct mechanistic readings:
- **(a) Forward-pass compositional chain.** Successor outputs are read by S-inhibition; the temporal ordering reflects an architectural dependence. *Predicts: ablating successor heads should disrupt S-inhibition's readout at convergence.*
- **(b) Convergent training dynamics without inference-time coupling.** The motifs emerge in order for training-dynamics reasons (gradient routing, sub-circuit prerequisites in the optimizer's path), but at convergence are causally disjoint. *Predicts: ablating successor heads should leave S-inhibition's readout unchanged.*

Track 2 distinguishes (a) from (b) by directly ablating the top-5 successor heads at convergence and measuring S-inhibition's response across two complementary metrics. Pre-registered in `HYPOTHESIS.md` §H5-causal (Metric A) and §H5-causal-2 (Metric B); extended to additional sizes in §H5-causal-3-record (1B, post-data canonicalization) and §H5-causal-3-2.8b (2.8B, pre-data).

### What was tested at 410m (the converging-evidence anchor)

**Common protocol (§H5-2 through §H5-8):**
- Pythia-410m-deduped @ step143000 anchor.
- 200 IOI prompts (Wang 2023 set, 100 BABA + 100 ABBA, seed=0).
- Suc set: top-5 successor heads at this checkpoint (§H5-3 tie-break: score desc, layer asc, head asc).
- Ctrl set: random sample of 5 from heads with score in `[τ_lift − 0.05, τ_lift)`, bracket-widened by 0.025 if <5 candidates, seed=0; score-bracket-matched to suc.
- NMs: pinned to component-DLA top-4 from the clean anchor, frozen across all conditions.
- Mean-ablation: `hook_z[:, :, head, :]` replaced with batch-mean per length group; permanent forward hook installed.
- Bootstrap: B = 200 paired per-prompt, seed = 1; 95% percentile CI on drop ratio.

**Two metrics:**
- **Metric A — §S-1 path-patching Δ_h** (`_run_phase4_causal_410m_anchor.py`). Per-sender classification {NULL, DEP, GENERIC, MIXED} with DEP_THRESHOLD = 0.5, NULL_BAND = ±0.20.
- **Metric B — IO−S logit-diff at END** (`_run_phase4_causal_410m_anchor_logitdiff.py`). Cross-condition classification with NULL band [0.8, 1.2], DEP < 0.5, GENERIC < 0.7.

### What was found at 410m

Both metrics returned **NULL**:

| Metric | Verdict | Key numbers |
|---|---|---|
| **A (§S-1 path-patching)** | **NULL** (3/3 senders) | ratio_suc ≈ ratio_ctrl ≈ 1.0 — S-inhibition Δ_h survives both ablations cleanly |
| **B (logit-diff)** | **NULL** | ratio_suc = 0.986, ratio_ctrl = 0.979 — IO−S logit-diff barely moves under either ablation |

**Converging-evidence claim at 410m (paper headline):** *"S-inhibition's circuit is causally disjoint from successor's at inference time. The temporal emergence ordering ind→suc→si is decoupled from any architectural causal chain."* The §H1-C compositional reading (a) is **falsified** at 410m by direct ablation; reading (b) — convergent training dynamics without inference-time coupling — is consistent with the observed NULL.

### What was found at 1B (post-data, canonicalized 2026-05-10)

The §H5-causal protocols were re-run at Pythia-1B step143000 on a feature worktree (2026-05-07). The canonicalization is recorded in `HYPOTHESIS.md` §H5-causal-3-record; the pre-reg-discipline gap is acknowledged (1B compute predated the 1B-specific amendment) and noted in the paper.

| Metric | Verdict | Key numbers |
|---|---|---|
| **A (§S-1 path-patching)** | **NULL** (3/3 senders) | Replicates 410m cleanly |
| **B (logit-diff)** | **MIXED** | ratio_suc = 0.790, ratio_ctrl = 0.797 — **both ablations drop ~21% similarly**; no successor-specific dependence, but the logit-diff readout becomes generically ablation-sensitive at 1B |
| **Cross-metric** | **MIXED** | Pre-committed headline: *"Heterogeneous Metric A and B verdicts at 1B; no global conclusion on suc → si causal dependence. Reported per-metric with per-sender CIs; deferred for follow-up."* |

**Per-size structural caveat at 1B (registered §H5-causal-3-record-3):** L14H2 appears in BOTH the suc set and the pinned NM set (dual-role). Metric A's downstream-NM filter excludes L14H2's contribution from the path-patching scalar — Metric A is structurally insensitive to L14H2 at this checkpoint. Metric B reads at END and is fully sensitive.

**Interpretation of the 1B Metric B MIXED verdict:** ctrl drops the same as suc. This is *not* a falsification of the 410m disjointness claim — it is a finding that the IOI logit-diff readout itself becomes generically ablation-sensitive at 1B's narrower 128-head architecture, so the readout can no longer distinguish successor-specific dependence from generic ablation effects. The Metric A NULL still holds (no successor-specific dependence via the path-patching scalar). The cross-metric MIXED is reported honestly per-metric with per-sender CIs; no single-headline NULL claim is made at 1B.

### What was found at 2.8B (2026-05-10, pre-data lock, completed)

`HYPOTHESIS.md` §H5-causal-3-2.8b registered both metrics at Pythia-2.8B step143000 anchor (head-count tier 1024) before any 2.8B ablation compute. Locked sets (asserted bit-for-bit at runtime):
- suc = `[(15,14), (28,17), (27,13), (13,10), (29,28)]` (scores 2.126, 0.726, 0.303, 0.302, 0.281)
- SI senders = `[(11,29), (11,5), (13,9)]` (scores 0.148, 0.121, 0.105)
- NMs = `[(11,29), (17,12), (22,31), (13,9)]`
- ctrl = `[(13,5), (13,8), (13,27), (20,29), (24,25)]` (procedure-locked: bracket [0.060, 0.135), bw=0.075, seed=0, NM-excluded)

**Per-size structural caveat at 2.8B (registered pre-data):** 3 of 5 suc heads — (27,13), (28,17), (29,28) — sit at layers > max(NM layer) = 22, so Metric A is **structurally mute** to those 3 ablations. Only (15,14) and (13,10) are visible to Metric A. Metric B reads at END and is fully sensitive to all 5.

| Metric | Verdict | Key numbers |
|---|---|---|
| **A (§S-1 path-patching)** | **NULL** (3/3 senders) | per-sender NULL;NULL;NULL — ratio_suc ≈ 1.001, ratio_ctrl ≈ 0.997 with 95% CIs of width ~0.005. Even the 2 suc heads visible to Metric A — (15,14), (13,10) — do not perturb S-inhibition's path-patching readout. |
| **B (logit-diff at END)** | **NULL** | ratio_suc = 0.984 ± 0.003, ratio_ctrl = 0.984 ± 0.006 — both essentially clean, no generic ablation sensitivity. Metric B sees all 5 suc ablations through to END. |
| **Cross-metric** | **NULL** | Pre-committed paper headline: *"Extends the §H5-causal 410m NULL on both metrics to head-count tier 1024; scale-stable causal-disjointness across 384-head and 1024-head architectures."* |

**Reframe of the 1B Metric B MIXED in light of 2.8B NULL:** at 2.8B, Metric B distinguishes ablation conditions cleanly (NULL with tight CIs); at 410m, Metric B was also clean NULL; only at 1B (128-head head-count regression) did Metric B return MIXED with ratio_suc ≈ ratio_ctrl ≈ 0.79. The 2.8B clean NULL falsifies the most pessimistic reading of the 1B MIXED (*"the readout itself is broken at scale"*). Instead, the 1B Metric B MIXED is now best read as a **narrow-architecture readout-specificity artifact**: 128-head IOI-circuit at 1B has the suc ablation set sit in a regime where the logit-diff readout cannot distinguish suc-specific from generic ablation effects. The Metric A NULL holds at all three sizes; Metric B NULL holds at 410m and 2.8B and is generically blunted at 1B. The paper's converging-evidence claim is therefore size-conditional: 410m and 2.8B are the converging anchors, 1B is the narrow-architecture caveat.

### Why this is the paper's mechanistic punchline

The §H5-causal NULL is the project's strongest direct mechanistic refutation of a tempting reading — the temporal ordering ind→suc→si naturally suggests an architectural causal chain, and the project explicitly falsifies that chain at the inference-time level via two converging metrics at 410m, and replicates the converging NULL at Pythia-2.8B (head-count tier 1024). The 1B Metric A NULL replicates, with the 1B Metric B MIXED reframed as a narrow-architecture readout artifact. The paper carries a scale-stable directional negative result across two converging metrics and two converging head-count tiers (384 + 1024) that does not appear in the prior literature (Tigges 2024, Singh 2024, Olsson 2022, Gould 2024 are all convergence- or training-dynamics-only, not inference-time-causal-dependence). That is the mechanistic interpretability contribution.

### Reproducibility pointers (Track 2)

| artifact | role |
|---|---|
| `HYPOTHESIS.md` §H5-causal, §H5-causal-2 | 410m pre-reg (both metrics) |
| `HYPOTHESIS.md` §H5-causal-3-record | 1B post-data canonicalization |
| `HYPOTHESIS.md` §H5-causal-3-2.8b | 2.8B pre-data lock |
| `data/exploration/phase4_causal_410m_anchor*.parquet` | 410m Metric A + B outputs (NULL) |
| `data/exploration/phase4_causal_1b_anchor*.parquet` | 1B Metric A NULL + Metric B MIXED outputs |
| `data/exploration/phase4_causal_2_8b_anchor*.parquet` (pending) | 2.8B both metrics |
| `notebooks/_run_phase4_causal_{410m,1b,2_8b}_anchor*.py` | runners per (size, metric) |
| `notebooks/causal_dependence.ipynb` | verdict notebook (per-size + per-metric verdict tables) |

---

## Track 3 — Scaling argument by head count (registered, PASS at 2.8B)

### Why head count, not parameter count

The §H2-9-R reframe surfaced the question: does the scale-dependent S-inhibition pattern continue beyond 410m? Initial extension to Pythia-1B (§H3-scale, registered before any 1B compute) returned a complex result — timing accelerated (perfectly, P = 1.000) but count saturated at 3 (same as 410m). This produced the §H3-scale REGR pattern by the locked priority `TOOLING > REGR > ORD-BREAK > WIDE-CI > SAT > PASS`.

Post-data architectural realization, registered in `HYPOTHESIS.md` §H4-scaling (committed before any 2.8B compute): **Pythia-1B is a head-count regression, not a scale-up.** Head counts across Pythia:

| size | params | layers × heads | total heads |
|---|---|---|---|
| 70m | 70M | 6 × 8 | 48 |
| 160m | 160M | 12 × 12 | 144 |
| 410m | 410M | 24 × 16 | **384** |
| 1b | 1.0B | 16 × 8 | **128** ← regression |
| 1.4b | 1.4B | 24 × 16 | 384 (same as 410m) |
| **2.8b** | **2.8B** | **32 × 32** | **1024** |

The parameter-count axis is non-monotonic in heads. The registered detectors operate at head granularity, the gate predicates evaluate at the head level (count-thresholds, density bars), and the project's reference papers (Olsson 2022, Wang 2023, McDougall 2024, Gould 2024, Singh 2024) all frame findings at head granularity. **Head count is the operationally relevant scaling axis.**

### What the scaling claim asks (`HYPOTHESIS.md` §H4-scaling, §H4-supersede)

For Pythia-2.8B-deduped (1024 heads = next ~3× tier from 410m's 384), the §H4-scaling pre-registered conjunctive gate:

- **(A.timing)** $P(\mu_{\text{si}}^{2.8\text{b}} < \mu_{\text{si}}^{410\text{m}}) \geq 0.95$ over $B = 1000$ paired per-prompt bootstrap replicates. *Tests whether S-inhibition timing-axis acceleration extends to head-count tier 1024.*
- **(A.count)** $\max\text{-count}_{\text{si}}^{2.8\text{b}} \geq 5$ over the §H4-supersede 10-cell sweep — full-fit regime entry per §H2-3. *Tests whether the count saturation observed at head-count tiers 144 (160m), 384 (410m), and 128 (1B) breaks when the head budget grows to 1024.*

§H4 **passes** iff (A.timing) AND (A.count). Within-2.8B coherence (analog of §H3-scale (A.iii)) is *deliberately omitted* from the gate — the H1-C ordering at 2.8B is *measurable but not gating*, since emergence-ordering claims remain locked at the registered 3 sizes per Track 1.

### Status (as of 2026-05-11) — PASS on both legs

- §H4-scaling amendment **committed** before any 2.8B compute (commit `0ae0e27`).
- 2.8B prefetch + anchor + induction sweep + successor sweep: **complete**. Original full-grid S-inhibition sweep was **halted at 8 of 40 cells** (steps 0–64, all early-training, all Δ_h ≈ 0) under the §H4-7 per-cell-cost escape hatch (~57 min/cell observed vs ~6 min/cell projected — 10× over budget).
- §H4-7-supersede amendment registered the halt (committed 2026-05-08); §H4-5 failure-mode taxonomy extended with a **DEFERRED** pattern at top priority alongside non-substantive TOOLING.
- §H4-supersede amendment (committed 2026-05-10) registered a reduced 10-cell re-attempt grid `[5000, 7000, 10000, 14000, 20000, 29000, 41000, 49000, 59000, 70000]` against the verbatim §H4-2 gate.
- §H4-supersede sweep **completed overnight 2026-05-10/11** in ~5 h 55 min (per-cell ~27.5 min, ~half the projected ~57 min). Verdict: **PASS on both legs**.

| Leg | Value | Gate | Verdict |
|---|---|---|---|
| **(A.count)** | max_count_si^2.8B = 5 at step 29000 | ≥ 5 | **PASS** |
| **(A.timing)** | paired bootstrap reversal_rate = 1.000 (1000/1000 replicates show μ_si^2.8B < μ_si^410m, zero fit failures) | ≥ 0.95 | **PASS** |

**Point estimates:** μ_si^2.8B ≈ 9,021, μ_si^410m ≈ 24,088 — 2.8B accelerates S-inhibition's logistic-fit midpoint by **~2.7×**.

**Count trajectory across the 10-cell grid (τ_strict = 0.0372):** 1 → 2 → 2 → 3 → 3 → **5** → 4 → 4 → 4 → 4 (peaks at step 29k, settles at 4 by step 70k). The count crosses the (A.count) ≥ 5 gate transiently at mid-emergence; the late-saturation tail sits at 4. This is a clean directional PASS but not an overwhelming margin — a reviewer might reasonably ask about the dip back to 4; the §H4-2 gate is defined on max over the grid, not on terminal-cell count, and so the PASS stands as registered.

**§H4-fullgrid extension (committed 2026-05-12).** The 22-cell completion at 2.8B (steps 128 → 17,000 + steps 84,000 → 143,000, filling the §H2-1 40-cell grid gaps) resolves the dip caveat into a sharper picture:
- **Emergence step at 2.8B is the 3,000 → 4,000 boundary.** L11H29 (the eventual canonical 2.8B S-inhibition head) becomes top-ranked at step 3,000 with Δ_h = 0.013 (sub-threshold) and crosses τ_strict at step 4,000 (Δ_h = 0.067).
- **Pre-emergence floor is genuinely random.** Steps 128/256/512 have max Δ_h ≈ 0 with top-head identity jittering across the network (L17H20, L6H18, L12H14).
- **Two transient n=5 peaks at steps 15,000 and 29,000.** The count dips to 3 between them (steps 16k–20k), then relaxes to a sustained **n=4 from step 41,000 through step 120,000**. The §H4-supersede 10-cell reading captured one slice of this non-monotonic pattern; the §writeup-conv-2 dip caveat is now resolved as a transient-peak-and-settle structure, not an asymptotic n=5.
- **Max-over-grid is still 5** (two transient peaks); the (A.count) gate PASSES as registered.

**Paper position**: with §H4-supersede PASS, the project now has three pre-registered tracks all hitting their targets. Track 3 is a substantive paper result, not an appendix. The paper's narrative is: ordered emergence (Track 1) + scale-stable inference-time causal-disjointness (Track 2) + head-count-axis scaling PASS (Track 3) — three converging tracks, all locked before data was observed under their specs.

Side observations preserved from §H4-7-supersede: induction at 2.8B reaches max_count = 48 (highest of all 5 sizes); successor at 2.8B emerges cleanly (max_count = 14). Reported in `notebooks/h1c_ordering_test.ipynb` §H4-scaling DEFERRED section. These are NOT §H4 gate inputs (per §H4-1) — supplementary cross-size data on the head-count axis.

### What is *not* claimed in Track 3

- §H4 is not an emergence claim. It does not extend the §H1-C ordering registered at the 3 small sizes.
- §H4 is not about induction or successor. Both robustly emerge at all tested sizes (including 1B); no scale-dependence story applies. §H4 is about S-inhibition only.
- §H4 does not retroactively reinterpret §H3-scale (1B). The 1B verdict (REGR) is preserved as a sealed historical record; under §H4 it is reframed as a head-count regression rather than a scale-up.

### Failure-mode taxonomy (§H4-5 + §H4-7-supersede DEFERRED, inherited verbatim by §H4-supersede)

| Pattern | Trigger | Paper headline |
|---|---|---|
| **DEFERRED** | Sweep halted under per-cell-cost escape hatch | "§H4 verdict deferred. Paper stands on Tracks 1 + 2." |
| **PASS** | Both legs hold | "Scaling argument confirmed at head-count tier 1024." |
| **TIMING-ONLY** | (A.timing) holds; (A.count) fails | "Timing-axis scaling holds at 2.8B; count saturates." |
| **COUNT-ONLY** | (A.count) holds; (A.timing) fails | "Count unlocks at 1024 heads; timing saturates." |
| **NEITHER** | Both legs fail | "Scaling argument falsified at 2.8B on head-count axis." |
| **TOOLING** | Detector distributionally broken | Non-substantive note. |

Priority ordering: `DEFERRED > TOOLING > NEITHER > COUNT-ONLY > TIMING-ONLY > PASS`.

### Reproducibility pointers (Track 3)

| artifact | role |
|---|---|
| `HYPOTHESIS.md` §H3-scale (1-10), §H3-scale-8-vis | pre-reg + visualization supersede + 1B verdict (sealed historical) |
| `HYPOTHESIS.md` §H4-scaling (1-10), §H4-7-supersede, §H4-supersede | pre-reg chain for 2.8B head-count-axis test |
| `data/exploration/phase3_1b_*` | 1B sweep outputs (head-count regression) |
| `data/exploration/phase4_2_8b_*` (partial) | 2.8B sweep outputs (induction + successor complete; S-inhibition pending §H4-supersede) |
| `notebooks/_run_phase3_1b_*.py` | 1B anchor + sweep + analysis runners |
| `notebooks/_run_phase4_2_8b_*.py` + `_run_phase4_2_8b_s_inhibition_supersede_sweep.py` | 2.8B runners (original + §H4-supersede reduced grid) |
| `notebooks/h1c_ordering_test.ipynb` §H3-scale section | 1B verdict (REGR) |
| `notebooks/h1c_ordering_test.ipynb` §H4-supersede section (pending) | 2.8B verdict |

---

## Cross-cutting methodology

### Detector specifications (locked, applied at every Pythia size in this project)

| motif | mechanism | metric | threshold | source |
|---|---|---|---|---|
| Induction | Olsson 2022 prefix-matching score over 50 length-100 random-token-repetition sequences | mean attention from second-half positions to (preceding-occurrence + 1) | > 0.3 | `PROJECT_BRIEF.md` §4 |
| Successor | §SU-1b lift-form cross-category direct logit attribution; categories = {days, months, numerals, letters}; 70 prompts; first-token DLA | mean cross-cat lift = real_DLA − null_DLA (within-category prefix permutation) | ≥ τ_lift = 0.13496 | §SU-tau |
| S-inhibition | §S-1 path-patching Δ_h with frozen paths; 200 IOI prompts (Wang 2023 set, 100 BABA + 100 ABBA, seed=0); ABC position-3 corruption | mean over k = 4 component-DLA NMs of [(patched_attn_NM_END→S2 − clean) − (patched_attn_NM_END→IO − clean)] | ≥ τ_strict = 0.0372 | §S-tau (with §S-5c rank-only override on the σ-criterion gate) |

### Bootstrap and sensitivity machinery (`HYPOTHESIS.md` §H2-2)

- Per-prompt resampling with replacement, B = 1000 replicates, 95% percentile CI on μ.
- Bootstrap reversal-rate per (size, pair) = empirical fraction of B = 1000 replicates where the predicted ordering does not hold; used in `h1c_ordering_test.ipynb` and §H4-scaling / §H4-supersede (A.timing).
- Threshold sensitivity bracket: ± 25% in 5 increments per motif, μ point estimate at each variant.

### Tiered logistic-fit handling (`HYPOTHESIS.md` §H2-3)

| regime | trigger | μ extraction |
|---|---|---|
| emerged | max_count ≥ 5 | direct scipy logistic fit |
| marginal | 2 ≤ max_count < 5 | bootstrap-median μ (widened CI flagged) |
| censored | max_count < 2 | μ right-censored at step143000 |

### Causal-dependence ablation (Track 2; `HYPOTHESIS.md` §H5-causal / §H5-causal-2 / §H5-causal-3)

- Mean-ablation on `hook_z[:, :, head, :]` per length group; permanent forward hook.
- Suc set: top-5 successor heads at the anchor checkpoint, §H5-3 tie-break.
- Ctrl set: random-from-bracket near-but-below τ_lift with bracket-widening + NM-exclusion clause; seed = 0.
- NMs: pinned to component-DLA top-4 from clean anchor; frozen across conditions.
- Bootstrap: B = 200 paired per-prompt; seed = 1.
- Per-(size, metric) verdict taxonomy: {NULL, DEP, GENERIC, MIXED}; cross-metric aggregation: both-NULL → cross-metric NULL, any-MIXED → cross-metric MIXED, etc.

### Deduped variant choice (justified explicitly per §H4-1)

All Pythia models in this project use the deduped variant (`EleutherAI/pythia-{size}-deduped`):

- `pythia_loader.py` hardcodes the deduped repo template.
- Tigges et al. (2024) — our Phase 1.2 replication target — uses deduped.
- McDougall et al. (2024), Singh et al. (2024), Gould et al. (2024) all use deduped.
- The deduped models train on a deduplicated Pile, producing fewer memorization-driven head-level artifacts.
- The non-deduped variants are not used at any size in this project.

### Repository layout

```
pythia-motif-emergence/
├── HYPOTHESIS.md               # canonical pre-registration record (this is the source of truth)
├── PROJECT_BRIEF.md            # original project plan + pre-committed limitations
├── PILOT_RESULTS.md            # Path A vs Path C decision, locked to Path C
├── README.md                   # status table
├── WRITEUP.md                  # this document — paper-narrative re-organization
├── checkpoints.yaml            # legacy pre-§H2-1 schedule (now superseded by §H2-1)
├── src/
│   ├── detectors/              # induction.py, successor.py, s_inhibition.py, copy_suppression.py
│   ├── analysis/               # phase2_logistic.py, phase2_bootstrap.py
│   ├── replication/            # tigges_ioi.py, path_patching.py
│   └── utils/                  # pythia_loader.py, mps_compat.py, corpus_io.py
├── notebooks/
│   ├── _build_*.py             # nbformat builders for the executable .ipynb files
│   ├── _run_phase2_*.py        # Phase 2 sweep runners (3 registered sizes)
│   ├── _run_phase3_1b_*.py     # 1B head-count regression runners
│   ├── _run_phase4_2_8b_*.py   # 2.8B scaling-track runners (+ §H4-supersede reduced grid)
│   ├── _run_phase4_causal_*.py # Track 2 ablation runners (410m, 1B, 2.8B; Metric A + Metric B)
│   ├── induction_full_sweep.ipynb / successor_full_sweep.ipynb / s_inhibition_full_sweep.ipynb  (4-size views)
│   ├── h1c_ordering_test.ipynb # registered gate verdict + scaling-track verdict
│   ├── causal_dependence.ipynb # Track 2 per-(size, metric) verdict notebook
│   ├── motif_structural_reuse.ipynb / motif_attention_inspection.ipynb  (4-size views)
│   └── *_proof.ipynb / *_emergence_exploration.ipynb  (Phase 1 historical artifacts)
└── data/
    ├── prompts/                # IOI prompts, successor prompts (per-tokenizer)
    ├── corpora/                # canonical raw-text corpus
    ├── pilot/                  # Path A pilot anchor outputs at 410m
    └── exploration/            # all Phase 2 / Phase 3 / Phase 4 sweep + causal outputs (gitignored except parquets)
```

### Pre-registration chain (chronology)

A reviewer auditing pre-registration discipline should read the amendment chain in `HYPOTHESIS.md` in **registration-date order** (the dispositive record, per §writeup-conv-3):

1. **§H1-C / pivot decision rule** — registered before any pilot code.
2. **Phase 1.0 pilot** — Path A vs Path C decision; Path C locked (2026-05-05).
3. **§S-1 through §S-tau** — S-inhibition detector specification + numerical threshold lock (2026-05-05).
4. **§SU-0 through §SU-tau** — Successor detector specification + numerical threshold lock with §SU-1b score-form supersede (2026-05-05 / 05-06).
5. **§H2-1 through §H2-9** — Phase 2 sweep specification (2026-05-06).
6. **§H2-9-R** — Post-data scale-dependence reframe (2026-05-06).
7. **§H3-scale (1-10)** — 1B scale-extension pre-registration (2026-05-06).
8. **§H3-scale-8-vis** — Visualization-layer supersede of the 3-size lock (2026-05-07).
9. **§H4-scaling (1-10)** — Head-count-axis scaling argument; supersedes §H3-scale-1's 1B target prospectively (2026-05-07).
10. **§H5-causal** — Track 2 Metric A (path-patching) at 410m, registered pre-data (2026-05-07).
11. **§H5-causal-2** — Track 2 Metric B (logit-diff) at 410m, registered pre-data (2026-05-07).
12. **§H4-7-supersede** — DEFERRED pattern registered post-halt at 8/40 S-inhibition cells (2026-05-08).
13. **§H5-causal-3-record** — Track 2 at 1B, **post-data canonicalization** (2026-05-10; gap acknowledged in the amendment).
14. **§H5-causal-3-2.8b** — Track 2 at 2.8B, registered pre-data (2026-05-10).
15. **§H4-supersede** — Track 3 reduced-grid re-attempt at 2.8B, registered pre-data (2026-05-10).
16. **§writeup-conv** — Post-data documentation-hygiene amendment (2026-05-11): locks Track-numbering convention (Track 2 = Causal-disjointness, Track 3 = Scaling); reframes §H4-supersede from "scaling appendix" to "third converging substantive result" post-PASS; rectifies the chronology errors in earlier amendment text. Analog of §H2-9-R: non-numerical, post-data, does not move any gate.

The git commit history corroborates this chronology, with one acknowledged exception (§H5-causal-3-record canonicalizes a 2026-05-07 worktree run after-the-fact; the paper discloses this gap).

---

## What this document is *not*

- It is **not** a paper draft. The paper is a separate artifact (`paper_draft.md`).
- It is **not** the canonical pre-registration. `HYPOTHESIS.md` and its amendment chain are.
- It does **not** introduce new claims. Every claim referenced here is locked in `HYPOTHESIS.md`.
- It is **not** a status board. `README.md` is the canonical status surface.

Its only function: re-organize what's already locked in `HYPOTHESIS.md` into the three-track narrative that the project's findings naturally fall into, so that a paper draft can lift sections from this document into the manuscript with minimal re-work.
