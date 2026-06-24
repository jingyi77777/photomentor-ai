# Labeling Log

Records how the four-axis labels for the custom set were produced, and how
reliable each one is. This file is the honest companion to the results in the
top-level README.

## Summary

The 120-image custom set is built by `scripts/make_dataset.py` from royalty-free
Lorem Picsum photos. True aesthetic labels cannot be collected at scale without
the (descoped) user study, so labels are **engineered** from controlled
degradations or a measured proxy. Three of four axes are exact by construction;
one is a heuristic proxy.

| Axis | Method | How the label is set | Reliability |
|------|--------|----------------------|-------------|
| Sharpness   | Gaussian blur | radius → known score: 0→5, 0.8→4, 1.6→3, 3.0→2, 5.0→1 | **exact** (controlled) |
| Exposure    | Brightness shift | factor → known score (brighten or darken): 1.0→5 … 2.1/0.48→1 | **exact** (controlled) |
| Composition | Tilt + centre crop | tilt angle → known score: 0°→5, 4°→4, 9°→3, 14°→2, 20°→1 | **exact-ish** (controlled) |
| Background  | Border edge-density | gradient energy in the outer frame, quantile-binned 1–5 (busier → lower) | **proxy** (heuristic) |

For sharpness, exposure, and composition, one base photo gets exactly one
controlled degradation; the other controllable axes stay at their "good" value
(5). Background is measured on the **clean** base image (before any degradation),
so the proxy reflects scene clutter rather than the applied filter.

## Reliability notes (important for interpreting metrics)

- **Composition, exposure, sharpness — trustworthy.** Labels follow from a known
  transform, so a Spearman ρ ≈ 0.52–0.60 reflects the model genuinely learning
  blur / brightness / tilt cues.
- **Background — discount it.** ρ ≈ 0.94 is inflated: the label *is* an edge
  statistic, so the model partly re-learns that statistic. Report it, but treat
  it as a proxy result, not evidence of true background-cleanliness judgement.

## Known limitations

- Score spread is uneven for composition/exposure/sharpness (most images sit at
  5, since only the degraded subset drops lower). Background is balanced by
  quantile binning.
- Semi-synthetic labels mean the model partly learns "is this blurred / dark /
  tilted" rather than holistic aesthetic quality. This is acceptable for an MVP
  whose goal is a working, honest pipeline — not a publishable aesthetic model.
- A tilted horizon is a reasonable but imperfect stand-in for "poor composition";
  it captures the classic beginner mistake but not framing or subject placement.

## What the full research plan would do instead

Two independent raters per photo on the real 4-axis rubric, calibrated on a
30-photo set, targeting Cohen's quadratic-weighted κ ≥ 0.6 per axis before full
labelling, with disagreements resolved by discussion. Compute agreement with
`src.evaluate.rater_agreement(rater_a, rater_b)`.

| Axis        | κ (rater A vs B) | Pass (≥0.6) |
|-------------|------------------|-------------|
| composition | _to be collected_ |            |
| exposure    | _to be collected_ |            |
| sharpness   | _to be collected_ |            |
| background  | _to be collected_ |            |
