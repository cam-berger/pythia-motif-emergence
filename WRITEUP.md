# Project writeup — two-track narrative

This document re-organizes the project's *registered* claims (locked in `HYPOTHESIS.md` and its amendment chain) into a paper-narrative two-track structure. **`HYPOTHESIS.md` remains the canonical pre-registration record**; this writeup is presentation-layer only and adds no new claims.

The project's findings split cleanly into two scientific tracks:

1. **Emergence track** — what does the ordering of attention-head motifs look like during training, in the registered Pythia size grid?
2. **Scaling track** — does the registered ordering generalize, and along what axis (head count vs parameter count) does it scale?

Each track has its own pre-registration chain, its own gate, and its own verdict. The tracks are independent — a verdict on one does not invalidate the other.

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

**Registered gate result (§H2-5): PASS at p = 0.00463 < 0.005.** The strict joint sign-test holds in all three Pythia sizes.

**Post-data reframe (§H2-9-R), registered after the gate but before the writeup:** the joint sign-test pass is heterogeneous across cells. Of the nine (size, motif) cells, four are right-censored or marginal at the upper logistic-fit sentinel. Only Pythia-410m is a robust per-size confirmation:

| size | induction | successor | S-inhibition | per-size verdict |
|---|---|---|---|---|
| 70m | full-fit | marginal | **right-censored** (max_count = 1 at step143000) | ordering vacuous through censoring |
| 160m | full-fit | marginal | marginal (bootstrap CI overlaps successor) | ordering passes but CI overlap |
| 410m | full-fit | full-fit (max = 7) | marginal (max = 3, but tight CI) | **robust per-size confirmation** |

The reframe headline: *"scale-dependent emergence of S-inhibition + a depth-vs-temporal asymmetry in 160m / 410m."* Specifically:
- S-inhibition's emergence is scale-dependent in Pythia: not at 70m, marginal at 160m, clean at 410m.
- In 160m and 410m, S-inhibition heads sit at *shallower* normalized layer depth than successor heads — incompatible with a strict compositional reading where S-inhibition consumes successor outputs at later layers (`h1c_ordering_test.ipynb` sub-deliverable 4b).
- Top-3 head identities for successor and S-inhibition turn over substantially between step25k and step143k (sub-deliverable 5).

The registered §H1-C falsification target is *not falsified*; the §H2-9-R reframe re-emphasises which cells robustly support the pass.

### What's locked at this scope

The emergence track is **complete**. No further sizes are added to the emergence claim. The §H1-C verdict and the §H2-9-R reframe stand as registered. Subsequent bigger-model work belongs to the scaling track below; it does *not* extend the emergence claim.

### Pre-committed limitations (verbatim from `PROJECT_BRIEF.md` §9)

1. Single-seed per Pythia size — no within-size variance estimate; "consistency" is across-size, not across-seed.
2. Pythia (GPT-NeoX) only — no cross-architecture universality claim.
3. Detector-threshold sensitivity — reported with bootstrap CIs across thresholds, not pretended away.
4. No causal claim — default framing: *"consistent with a compositional account."*

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

---

## Track 2 — Scaling argument by head count (pre-registered, in progress)

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

### What the scaling claim says (`HYPOTHESIS.md` §H4-scaling)

For Pythia-2.8B-deduped (1024 heads = next ~3× tier from 410m's 384), the §H4-scaling pre-registered conjunctive gate:

- **(A.timing)** $P(\mu_{\text{si}}^{2.8\text{b}} < \mu_{\text{si}}^{410\text{m}}) \geq 0.95$ over $B = 1000$ paired per-prompt bootstrap replicates. *Tests whether S-inhibition timing-axis acceleration extends to head-count tier 1024.*
- **(A.count)** $\max\text{-count}_{\text{si}}^{2.8\text{b}} \geq 5$ over the 40-cell sweep — full-fit regime entry per §H2-3. *Tests whether the count saturation observed at head-count tiers 144 (160m), 384 (410m), and 128 (1B) breaks when the head budget grows to 1024.*

§H4 **passes** iff (A.timing) AND (A.count). Within-2.8B coherence (analog of §H3-scale (A.iii)) is *deliberately omitted* from the gate — the H1-C ordering at 2.8B is *measurable but not gating*, since emergence-ordering claims remain locked at the registered 3 sizes per Track 1.

### What is *not* claimed

- §H4 is not an emergence claim. It does not extend the §H1-C ordering registered at the 3 small sizes.
- §H4 is not about induction or successor. Both robustly emerge at all tested sizes (including 1B); no scale-dependence story applies. §H4 is about S-inhibition only.
- §H4 does not retroactively reinterpret §H3-scale (1B). The 1B verdict (REGR) is preserved as a sealed historical record; under §H4 it is reframed as a head-count regression rather than a scale-up.

### Failure-mode taxonomy (§H4-5)

Five pass / fail patterns are pre-committed, each matched to a paper-headline interpretation:

| Pattern | Trigger | Paper headline |
|---|---|---|
| **PASS** | Both legs hold | "Scaling argument confirmed: at 1024 heads, S-inhibition timing accelerates beyond 410m and count exceeds the 410m saturation cap." |
| **TIMING-ONLY** | (A.timing) holds; (A.count) fails | "Timing-axis scaling holds at 2.8B; count-axis saturation extends from 1B's narrow architecture to 2.8B's 1024-head architecture, suggesting count saturation is fundamental rather than head-count-rate-limited." |
| **COUNT-ONLY** | (A.count) holds; (A.timing) fails | "Count-axis scaling unlocks at 1024 heads; timing-axis saturates between 410m and 2.8B." |
| **NEITHER** | Both legs fail | "Scaling argument falsified at 2.8B: both timing and count saturate beyond 410m on the head-count axis." |
| **TOOLING** | Detector outputs distributionally broken | "Methodological note: detector instability at d_model = 2560 on MPS." Not a substantive result; verdict deferred pending re-tooling. |

Multi-leg failures default to the most-severe pattern triggered, in priority order: `TOOLING > NEITHER > COUNT-ONLY > TIMING-ONLY > PASS`.

### Status and next steps (as of 2026-05-07)

- §H4-scaling amendment **committed** before any 2.8B compute (commit `0ae0e27`).
- 2.8B prefetch + anchor + sweep + analysis: pending. Triggered explicitly by the user; no compute starts unilaterally.
- Verdict to be recorded in `notebooks/h1c_ordering_test.ipynb` §H4 section once the 2.8B sweep completes.

### Reproducibility pointers (Track 2)

| artifact | role |
|---|---|
| `HYPOTHESIS.md` §H3-scale (1-10), §H3-scale-8-vis | pre-reg + visualization supersede + 1B verdict (sealed historical) |
| `HYPOTHESIS.md` §H4-scaling (1-10) | pre-reg for 2.8B head-count-axis test |
| `data/exploration/phase3_1b_*` | 1B sweep outputs (head-count regression) |
| `data/exploration/phase4_2_8b_*` (pending) | 2.8B sweep outputs |
| `notebooks/_run_phase3_1b_*.py` | 1B anchor + sweep + analysis runners |
| `notebooks/_run_phase4_2_8b_*.py` (pending) | 2.8B anchor + sweep + analysis runners |
| `notebooks/h1c_ordering_test.ipynb` §H3-scale section | 1B verdict (REGR) |
| `notebooks/h1c_ordering_test.ipynb` §H4 section (pending) | 2.8B verdict |

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
- Bootstrap reversal-rate per (size, pair) = empirical fraction of B = 1000 replicates where the predicted ordering does not hold; used in `h1c_ordering_test.ipynb` and §H4-scaling (A.timing).
- Threshold sensitivity bracket: ± 25% in 5 increments per motif, μ point estimate at each variant.

### Tiered logistic-fit handling (`HYPOTHESIS.md` §H2-3)

| regime | trigger | μ extraction |
|---|---|---|
| emerged | max_count ≥ 5 | direct scipy logistic fit |
| marginal | 2 ≤ max_count < 5 | bootstrap-median μ (widened CI flagged) |
| censored | max_count < 2 | μ right-censored at step143000 |

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
│   ├── _run_phase4_2_8b_*.py   # 2.8B scaling-track runners (pending)
│   ├── induction_full_sweep.ipynb / successor_full_sweep.ipynb / s_inhibition_full_sweep.ipynb  (4-size views)
│   ├── h1c_ordering_test.ipynb # registered gate verdict + scaling-track verdict
│   ├── motif_structural_reuse.ipynb / motif_attention_inspection.ipynb  (4-size views)
│   └── *_proof.ipynb / *_emergence_exploration.ipynb  (Phase 1 historical artifacts)
└── data/
    ├── prompts/                # IOI prompts, successor prompts (per-tokenizer)
    ├── corpora/                # canonical raw-text corpus
    ├── pilot/                  # Path A pilot anchor outputs at 410m
    └── exploration/            # all Phase 2 / Phase 3 / Phase 4 sweep outputs (gitignored except parquets)
```

### Pre-registration chain (chronology)

A reviewer auditing pre-registration discipline should read the amendment chain in `HYPOTHESIS.md` in the following order:

1. **§H1-C / pivot decision rule** — registered before any pilot code.
2. **Phase 1.0 pilot** — Path A vs Path C decision; Path C locked.
3. **§S-1 through §S-tau** — S-inhibition detector specification + numerical threshold lock.
4. **§SU-0 through §SU-tau** — Successor detector specification + numerical threshold lock (with §SU-1b score-form supersede).
5. **§H2-1 through §H2-9** — Phase 2 sweep specification.
6. **§H2-9-R** — Post-data scale-dependence reframe (does NOT change the registered gate).
7. **§H3-scale (1-10)** — 1B scale-extension pre-registration.
8. **§H3-scale-8-vis** — Visualization-layer supersede of the 3-size lock.
9. **§H4-scaling (1-10)** — Head-count-axis scaling argument; supersedes §H3-scale-1's 1B target prospectively.

The git commit history corroborates this chronology.

---

## What this document is *not*

- It is **not** a paper draft. The paper is a separate artifact (`paper_draft.md`).
- It is **not** the canonical pre-registration. `HYPOTHESIS.md` and its amendment chain are.
- It does **not** introduce new claims. Every claim referenced here is locked in `HYPOTHESIS.md`.
- It is **not** a status board. `README.md` is the canonical status surface.

Its only function: re-organize what's already locked in `HYPOTHESIS.md` into the two-track narrative that the project's findings naturally fall into, so that a paper draft can lift sections from this document into the manuscript with minimal re-work.
