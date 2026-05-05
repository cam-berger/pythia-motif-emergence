# Pilot Results — Path A vs Path C Decision

**Status:** **FINAL** — Path C registered 2026-05-05 (Day 4 of Week 1 pilot).

This template was committed alongside `HYPOTHESIS.md` *before* the pilot ran. Fields are populated as pilot evidence accrues. Per the `HYPOTHESIS.md` 2026-05-05 amendment, calibrated supplementary analysis was dropped (validation failed), so this document reports strict-criterion results only as gating evidence.

## Metadata

- **Pilot start date:** 2026-05-04 (Day 1 induction work)
- **Pilot end date:** TBD (Day 5 path lock)
- **Model:** Pythia-410M-deduped, checkpoint `step143000` (final)
- **Detector:** McDougall two-criterion copy-suppression detector (`src/detectors/copy_suppression.py`)
- **Detector implementation commit:** TBD (recorded at Day 3 commit time)
- **Validation:** dual report per `HYPOTHESIS.md` 2026-05-05 amendment §1
  - **Strict (McDougall published threshold, QK > 0.3 AND OV < 0):** **FAIL** — L10H7 mean QK=0.019 on canonical corpus, mean OV=-0.802. Threshold-transfer issue documented in amendment §1.
  - **Calibrated (Q4 rubric, top-5 OV AND QK > 0.05):** **FAIL** — L10H7 OV rank=15 of 144, mean QK=0.019. Calibrated supplementary scheme dropped per amendment §1; Pythia application reverts to strict-only.
- **Canonical corpus:** `data/corpora/copy_suppression_corpus.txt` (40 Wikipedia featured-article opening passages, 7521 GPT-NeoX tokens; revision IDs in file header)
- **Eligible duplicate positions on Pythia-410M-deduped @ step143000:** 2769
- **Random seed:** N/A (deterministic forward pass)
- **Hardware:** Apple M5 Pro, 64 GB unified memory, MPS (PyTorch 2.11.0 + `PYTORCH_ENABLE_MPS_FALLBACK=1`)

## Per-head pilot scores

McDougall two-criterion detector applied to all 384 heads of Pythia-410M-deduped at `step143000` on the canonical corpus. Ranking rule for the top-5 table: rank 1 = strict-passing heads (sorted by most-negative OV among strict-passing); ranks 2-5 = top-by-most-negative-OV among heads with QK ≥ 0.05 (heads attending non-trivially to prior duplicates).

| Rank | Layer | Head | QK score | OV score | QK passes (>0.3) | OV passes (<0) | Both pass |
|---|---|---|---|---|---|---|---|
| 1 | 2  | 8  | 0.328 | -0.023 | ✓ | ✓ | **✓** |
| 2 | 19 | 3  | 0.059 | -0.191 | ✗ | ✓ | ✗ |
| 3 | 1  | 4  | 0.265 | -0.175 | ✗ | ✓ | ✗ |
| 4 | 0  | 15 | 0.082 | -0.103 | ✗ | ✓ | ✗ |
| 5 | 1  | 6  | 0.067 | -0.094 | ✗ | ✓ | ✗ |

**Strict-criterion summary:** 1 head (L2H8) passes both criteria of 384 total. L2H8 squeaks past both thresholds (QK = 0.328 marginally above 0.3; OV = -0.023 marginally below 0). All other top candidates fail QK > 0.3 — same dilution pattern observed on GPT-2 L10H7 (raw-text mean QK plateaus low without filtering). Strong-OV heads in late layers (L18H15: OV=-0.593, L18H5: OV=-0.310) all have QK < 0.06 and so fail the strict criterion.

**Full distributions:**
- Per-head scores: `data/pilot/copy_suppression_pythia_410m_step143000.parquet` (long format, 768 rows = 384 heads × 2 motifs).
- Per-position OV side cache: `data/pilot/copy_suppression_pythia_410m_step143000_per_position.npz` (shape `(2769, 24, 16)`).

## Manual qualitative inspection (top-5 candidates)

For each candidate from the top-5 table above, inspect attention patterns and direct logit attribution on the worked-example position(s) selected by the per-position OV ranking (Q7 rule: positions where the head's own per-position OV is most-negative). Record whether the head's behavior is *qualitatively* consistent with copy-suppression as described by McDougall.

Each row reports: (a) total attention from the rank-1 worked-example query position back to prior occurrences of `token_i`; (b) corpus-wide mean change in logit on `token_i` when this head is ablated (positive = head was suppressing the duplicate; negative = head was promoting it); (c) qualitative verdict.

| Rank | Layer | Head | Strict-pass | Attn-to-priors @ rank-1 | Corpus-wide d-logit on dup | DLA-vs-ablation consistent? | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 2  | 8  | ✓ | 0.910 | -0.0092 | ✗ (DLA -0.023, ablation -0.009: same sign but opposite to suppression mechanism) | **FAIL** |
| 2 | 19 | 3  | ✗ | 0.095 | -0.0835 | ✗ (DLA -0.191 negative, ablation -0.084 negative — head is promoting duplicates, not suppressing) | **FAIL** |
| 3 | 1  | 4  | ✗ | 0.208 | -0.0072 | ✗ (high QK + slight negative DLA but no functional suppression) | **FAIL** |
| 4 | 0  | 15 | ✗ | 0.098 | -0.0142 | ✗ (mostly self-attention; weak signal) | **FAIL** |
| 5 | 1  | 6  | ✗ | 0.000 | -0.1146 | ✗ (no attention to priors at rank-1 position; not a duplicate-attending head at all) | **FAIL** |

Verdict scale: **PASS** (clearly copy-suppression-like — high attn-to-priors AND positive d-logit on duplicate when ablated, mirroring GPT-2 L10H7's signature) / **WEAK** (one criterion passes; functional effect ambiguous) / **FAIL** (does not match copy-suppression mechanism).

Reference benchmark: GPT-2 L10H7 corpus-wide d-logit on duplicate tokens = **+0.032** (positive — the textbook suppression direction). All 5 Pythia candidates have **negative** corpus-wide d-logit (-0.007 to -0.115); ablating any of them *lowers* duplicate-token logits, the *opposite* of copy-suppression. These heads contribute positively to duplicate-token prediction; they are not corrective.

**L2H8 specifically:** the only head numerically passing the strict criterion. Its attention-to-prior pattern is textbook (0.91 attention from " Inside" at pos 115 back to " Inside" at pos 105 in the Stonehenge passage — stronger than even GPT-2 L10H7's typical attention-to-prior). But its corpus-wide ablation effect on duplicate logits is **-0.009** (negative). L2H8 is best characterized as a **previous-token / induction-precursor head** that attends to prior occurrences of the same token but functionally *promotes* rather than suppresses the duplicate — failing the OV criterion in the mechanism-defining sense.

Inspection scripts: `notebooks/_run_qualitative_inspection.py` (per-position attention + ablation per head); corpus-wide verification via inline ablation loop in the same script's results.

## Decision rule application

Apply the rule from `HYPOTHESIS.md` § *Pilot decision rule (Week 1)*, in order. Stop at the first match.

**Numerical state (post-anchor):** 1 head (L2H8) passes both strict criteria.
**Qualitative state (post-Day-4):** L2H8 fails qualitative inspection — corpus-wide d-logit on duplicates is -0.009 (opposite direction of copy-suppression). All four supplementary candidates also FAIL.

- [ ] **Strong positive → Path A.** ≥ 3 heads pass both criteria, qualitatively confirmed. *(Numerical: 1 < 3.)*
- [ ] **Weak positive → Path A with caveat.** 1–2 heads pass both criteria, qualitatively confirmed. *(Numerical: 1 in range, but **L2H8 fails qualitative inspection** — branch does not apply.)*
- [x] **Negative → Path C.** 0 heads pass both criteria, OR numerically-passing heads fail qualitative inspection. **Registered outcome.** L2H8 numerically passes but qualitatively FAILS (functional effect is opposite direction of suppression).
- [ ] **Tie / ambiguous → Path C.** Default to the cleaner motif. *(Not invoked — the Negative branch is unambiguous.)*

## Final decision

- **Path selected:** **Path C** (S-inhibition pivot)
- **Justification:** The McDougall two-criterion detector applied to Pythia-410M-deduped @ step143000 on the canonical 7.5k-token corpus identifies exactly 1 head passing both strict thresholds (L2H8: QK=0.328, OV=-0.023). Day 4 qualitative inspection shows L2H8 has a textbook duplicate-attending QK pattern (0.91 attention back to the prior occurrence at the rank-1 worked-example position) but its corpus-wide ablation effect on duplicate-token logits is **-0.009** — i.e., ablating L2H8 *lowers* duplicate-token confidence, the opposite of suppression. The four supplementary candidates (L19H3, L1H4, L0H15, L1H6) all show similar negative corpus-wide d-logit (-0.007 to -0.115). L2H8 is best characterized as a previous-token / induction-precursor head, not a copy-suppression head. By `HYPOTHESIS.md` decision rule path 3, the numerically-passing head failing qualitative inspection registers Path C. The project pivots: the registered hypothesis becomes H1-C, with S-inhibition replacing copy-suppression as the third motif. Phase 2 detector targets are induction, successor, and S-inhibition.
- **Decision recorded by:** Cam Berger (cam-berger)
- **Date:** 2026-05-05
- **Decision commit SHA:** TBD (recorded at Day 3/4 commit time)
- **Registered hypothesis going forward:** H1-C (per `HYPOTHESIS.md` §"Pivot hypothesis (H1-C)")

## Notes from pilot

Free-form notes on anything surprising, methodological hiccups, or qualitative observations that don't fit the structured fields above. Keep this section honest — surprises here are exactly the thing that should not be retconned later.

- **Calibrated supplementary scheme failed validation on GPT-2 (Day 3).** L10H7 mean QK on canonical corpus = 0.019 (rubric required > 0.05); OV rank = 15 of 144 (rubric required top-5). Per `HYPOTHESIS.md` 2026-05-05 amendment §1, the supplementary scheme is dropped and Pythia application reverts to strict-only. The failure is itself meaningful: L10H7's per-position OV is highly bimodal (rank-1 worked-example position has OV=-24 in the Antarctica passage at " Antarctic"; most other positions have OV ≈ 0), so corpus-mean averaging dilutes the signal below thresholds calibrated for filtered data. The reframed validation (dual report with strict FAIL + calibrated FAIL on the canonical corpus) is documented in the metadata above.
- **L2H8 was a previous-token head, not a copy-suppression head.** Layer 2 of 24 is unusually early for a copy-suppression analog (McDougall's L10H7 is at layer 10 of 12). Day 4 qualitative inspection confirmed: L2H8's QK pattern is textbook (0.91 attention from " Inside" at pos 115 to its prior occurrence at pos 105 in the Stonehenge passage — stronger than even L10H7's typical attention-to-prior), but its corpus-wide d-logit on duplicate tokens is **-0.009** when ablated. Ablating L2H8 *lowers* duplicate-token logits, meaning L2H8 is *promoting* duplicates, not suppressing them. The numerical strict-pass was a fluke from corpus-mean OV being marginally negative (-0.023). This kind of head — high attention to prior duplicates, positive contribution to duplicate-token logit — is the *induction-precursor / previous-token head* described in Singh 2024, not McDougall's copy-suppression motif.
- **No genuine copy-suppression heads in Pythia-410M-deduped @ step143000.** All five candidates (top-1 strict + top-4 OV-strong with QK ≥ 0.05) showed negative corpus-wide d-logit on duplicate tokens (-0.007 to -0.115 range). Reference: GPT-2 L10H7 corpus-wide d-logit on the *same canonical corpus* is +0.032 (positive — the suppression direction). Ablating any of the Pythia candidates makes the model *worse* at predicting duplicates, the opposite of what copy-suppression heads do. This isn't a threshold-calibration issue — the *functional mechanism* (attend-then-suppress) is absent in the heads that the strict criterion flagged.
- **Threshold-transfer issue replicates on Pythia.** The pattern from GPT-2 (high-QK heads have weak OV; strong-OV heads have low QK) appears identically on Pythia-410M-deduped. McDougall's QK > 0.3 threshold under-detects on raw text in both architectures, not just GPT-2 — strengthening the case that the threshold itself is data-conditional, not model-conditional.

## Supplementary analyses

This section is **non-gating**. The pilot path decision in §"Final decision" is determined exclusively by the strict-criterion result in §"Per-head pilot scores" + Day 4 qualitative inspection. The analyses below are reported per `HYPOTHESIS.md` §54 (detector-threshold sensitivity) and the 2026-05-05 amendment redefining validation. They inform the paper but do not change the registered Path C decision.

### Calibrated criterion analysis (failed validation)

The 2026-05-05 amendment locked a calibrated supplementary scheme (`OV < 0 AND QK > 0.05`) gated by a Q4 validation rubric on GPT-2: L10H7 must rank in top 5 by most-negative OV AND have mean QK > 0.05 on the canonical corpus. Day 3 calibration result:

| Test | Threshold | L10H7 measured | Verdict |
|---|---|---|---|
| L10H7 in top-5 by most-negative OV | rank ≤ 5 | rank = **15** of 144 | FAIL |
| L10H7 mean QK on canonical corpus | > 0.05 | **0.019** | FAIL |

Both rubric conditions failed. Per amendment §1, the calibrated supplementary scheme is dropped and Pythia application reverts to strict-only. The calibrated scheme would have flagged 6 of 144 GPT-2 heads (L0H8, L1H5, L1H11, L5H11, L6H6, L11H0) — *not L10H7*. Applied to Pythia-410M @ step143000 it would flag many more heads (140 of 384 have OV < 0; ~20 have OV < 0 AND QK > 0.05). Without validation against the published reference, none of those calibrated-passing heads would have been interpretable.

### Emergence sweep summary (Pythia 70M / 160M / 410M × 6 checkpoints)

Strict-criterion pass count and extreme scores per cell. Full per-head data: `data/exploration/copy_suppression_emergence_preview.parquet` (long-format canonical schema, 6912 rows).

| size  | step    | strict pass | max QK | min OV   | induction pass | induction max |
|---|---:|---:|---:|---:|---:|---:|
| 70m   | 0       | 0 | 0.039 | -0.010 | 0  | 0.015 |
| 70m   | 1000    | 0 | 0.081 | -0.012 | 6  | 0.953 |
| 70m   | 3000    | 0 | 0.250 | -0.047 | 5  | 0.978 |
| 70m   | 8000    | 0 | 0.256 | -0.178 | 5  | 0.976 |
| 70m   | 25000   | 0 | 0.229 | -0.314 | 4  | 0.838 |
| 70m   | 143000  | 1 | 0.314 | -0.779 | 6  | 0.867 |
| 160m  | 0       | 0 | 0.041 | -0.006 | 0  | 0.015 |
| 160m  | 1000    | 0 | 0.083 | -0.021 | 6  | 0.973 |
| 160m  | 3000    | 1 | 0.302 | -0.143 | 6  | 0.973 |
| 160m  | 8000    | 1 | 0.336 | -0.304 | 8  | 0.964 |
| 160m  | 25000   | 1 | 0.333 | -0.428 | 10 | 0.932 |
| 160m  | 143000  | 2 | 0.354 | -0.385 | 17 | 0.884 |
| 410m  | 0       | 0 | 0.041 | -0.003 | 0  | 0.016 |
| 410m  | 1000    | 0 | 0.110 | -0.028 | 11 | 0.879 |
| 410m  | 3000    | 1 | 0.303 | -0.074 | 14 | 0.947 |
| 410m  | 8000    | 1 | 0.306 | -0.191 | 20 | 0.943 |
| 410m  | 25000   | 1 | 0.304 | -0.242 | 22 | 0.955 |
| 410m  | 143000  | 1 | 0.328 | -0.593 | 19 | 0.953 |

**Strict-pass identity tracking:** the strict-passing head's identity is largely stable per size:
- 70M: only L0H3 passes (only at step 143000).
- 160M: L1H4 passes from step 3000 onward (consistent identity); L1H8 joins at step 143000.
- 410M: L1H4 passes at steps 3000-8000; L2H8 passes at steps 25000-143000 (identity *switches* mid-training).

**Cross-motif observation.** Induction emerges fast in all sizes (saturated by step 1000-3000). Copy-suppression strict-passes appear from step 3000 onward and grow modestly. The *temporal pattern* (induction → copy-suppression strict-pass) qualitatively matches the H1 ordering hypothesis. **However**, the Day 4 inspection of L2H8 (the rank-1 strict-passing head at the pre-reg anchor) shows the "copy-suppression strict pass" is *not* the McDougall mechanism — it's induction-precursor heads with marginal-negative corpus-mean OV. The *functional* H1 ordering for genuine copy-suppression vs induction is *not testable* on Pythia in this sweep, because the third-motif (copy-suppression) is essentially absent.

**Visualization:** `notebooks/copy_suppression_emergence_exploration.ipynb`.
