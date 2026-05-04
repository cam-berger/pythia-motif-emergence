# pythia-motif-emergence

Mechanistic interpretability research project characterizing the emergence dynamics of three attention-head motifs — induction, successor, and a suppressive motif (copy-suppression *or* S-inhibition, gated by Week 1 pilot) — across the Pythia training-checkpoint suite.

**Canonical project state:** [`PROJECT_BRIEF.md`](./PROJECT_BRIEF.md). This README is a thin shim; the brief is load-bearing.

**Pre-registration:** [`HYPOTHESIS.md`](./HYPOTHESIS.md) and [`PILOT_RESULTS.md`](./PILOT_RESULTS.md) are committed before any pilot code runs. Their commit timestamp is the pre-registration anchor.

## Why this project exists

Tigges et al. (NeurIPS 2024) characterized IOI and greater-than circuit emergence across Pythia checkpoints but did not study successor heads or copy-suppression. McDougall et al. (BlackboxNLP 2024) explicitly named *"do copy-suppression heads exist in Pythia and Llama?"* as future work. The infrastructure (`curt-tigges/circuits-over-time`, `hannamw/EAP-IG`) exists; the experimental design — comparing emergence ordering across induction, successor, and a suppressive motif at three Pythia scales — is the contribution.

If the ordering predicted by the hypothesis (induction → successor → suppression) holds across all three sizes, that is consistent with a compositional account in which corrective mechanisms emerge after the copying behaviors they correct. If it doesn't, the inconsistency is itself an interesting null result.

## What's planned

One paper + arXiv preprint + LessWrong post + clean GitHub repo with one notebook per main figure, submitted simultaneously. Target venue: BlackboxNLP at EMNLP 2026 (fallback ICLR MI Workshop 2027). Solo researcher, M5 Pro hardware, 16-week timeline.

The full plan — weekly milestones, decision gates, pre-committed limitations, failure-mode contingencies — is in `PROJECT_BRIEF.md`.

## Status

| Phase | Weeks | Status |
|---|---|---|
| 0 — Reading | parallel with 1–4 | not started |
| 1 — Tooling & replication | 1–4 | not started |
| 1.0 — Pilot (Path A vs Path C decision) | week 1 | not started |
| 1.1 — TransformerLens fluency on Pythia | week 2 | not started |
| 1.2 — Tigges IOI replication | week 3 | not started |
| 1.3 — Detector validation | week 4 | not started |
| 2 — Novel sweep | 5–10 | not started |
| 3 — Push past replication | 11–13 | not started |
| 4 — Writeup | 14–16 | not started |

Update this table as gates pass. Do not add findings to the README before the corresponding gate.

## Setup

```bash
uv sync
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Python 3.12 (uv-managed). MPS-only — no CUDA path supported.

## Repository layout

See `PROJECT_BRIEF.md` §6.

## License

[MIT](./LICENSE).
