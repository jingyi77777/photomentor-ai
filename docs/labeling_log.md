# Labeling Log

Records the labeling protocol and inter-rater reliability for the custom set.

## Protocol
- Two raters per photo (developer + one second rater).
- Each photo scored 1–5 on: composition, exposure, sharpness, background.
- Calibration: both raters label a 30-photo calibration set first.
- Target inter-rater reliability: **Cohen's quadratic-weighted κ ≥ 0.6** per axis
  before full labeling begins.
- Disagreements > 1 point resolved by discussion; the agreed value is recorded
  and the disagreement noted below.

## Calibration results (fill in)
| Axis        | κ (rater A vs B) | Pass (≥0.6) |
|-------------|------------------|-------------|
| composition |                  |             |
| exposure    |                  |             |
| sharpness   |                  |             |
| background  |                  |             |

Compute with `src.evaluate.rater_agreement(rater_a, rater_b)`.

## Disagreement notes
- (date) filename — axis — A=?, B=? — resolution
