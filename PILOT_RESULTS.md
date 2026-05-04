# Pilot Results — Path A vs Path C Decision

**Status:** TEMPLATE — to be populated by Day 5 of Week 1.

This template is committed alongside `HYPOTHESIS.md` *before* the pilot runs. Filling in the fields below is the pre-registered method of recording pilot evidence; the schema commits the project to *what counts as evidence* in advance.

## Metadata

- **Pilot start date:** TBD
- **Pilot end date:** TBD
- **Model:** Pythia-410M, checkpoint `step143000`
- **Detector:** McDougall two-criterion copy-suppression detector
- **Detector implementation commit:** TBD (SHA of the commit that introduced `src/detectors/copy_suppression.py`)
- **Validation:** detector fires on GPT-2 small layer 10 head 7 — TBD (PASS / FAIL)
- **Random seed (where applicable):** TBD
- **Hardware:** Apple M5 Pro, 64 GB unified memory, MPS

## Per-head pilot scores

Apply the McDougall two-criterion detector to all 384 heads of Pythia-410M at `step143000`. Record the top-N candidates by combined score, where N ≥ 5.

| Rank | Layer | Head | QK score (attention $i \to j$ where $\text{token}_j = \text{token}_i$) | OV score (DLA to $\text{token}_i$ at position $i$) | QK passes (>0.3) | OV passes (<0) | Both pass |
|---|---|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Full distribution across all 384 heads: link to `data/pilot/copy_suppression_scores.parquet` (TBD).

## Manual qualitative inspection (top-5 candidates)

For each numerically-passing candidate, inspect attention patterns and direct logit attribution on a small set of test prompts. Record whether the head's behavior is *qualitatively* consistent with copy-suppression as described by McDougall.

| Rank | Layer | Head | Attention pattern matches expected? | DLA sign consistent across prompts? | Qualitative verdict |
|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD | TBD |
| 5 | TBD | TBD | TBD | TBD | TBD |

Test prompts used: link to `data/prompts/pilot_inspection_prompts.txt` (TBD).
Attention pattern visualizations: link to `notebooks/pilot_validation.ipynb` (TBD).

## Decision rule application

Apply the rule from `HYPOTHESIS.md` § *Pilot decision rule (Week 1)*, in order. Stop at the first match.

- [ ] **Strong positive → Path A.** ≥ 3 heads pass both criteria, qualitatively confirmed.
- [ ] **Weak positive → Path A with caveat.** 1–2 heads pass both criteria, qualitatively confirmed.
- [ ] **Negative → Path C.** 0 heads pass both criteria, OR numerically-passing heads fail qualitative inspection.
- [ ] **Tie / ambiguous → Path C.** Default to the cleaner motif.

## Final decision

- **Path selected:** TBD (A | A-with-caveat | C)
- **Justification (2–4 sentences):** TBD
- **Decision recorded by:** TBD
- **Date:** TBD
- **Decision commit SHA:** TBD (the commit that records this decision)

## Notes from pilot

Free-form notes on anything surprising, methodological hiccups, or qualitative observations that don't fit the structured fields above. Keep this section honest — surprises here are exactly the thing that should not be retconned later.

TBD.
