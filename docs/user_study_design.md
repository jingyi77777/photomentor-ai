# User Study Design (designed, not run in the 1-week MVP)

> For the 1-week deliverable this study is **designed and documented but not
> executed** — a 2-week practice window cannot fit in one week. The Streamlit
> app already implements both arms (sidebar toggle), so the study is runnable
> as-is once recruited.

## Question
Does explainable feedback (score + Grad-CAM + suggestion) improve beginners'
photo-quality scores more than score-only feedback over two weeks?

## Design
Two-arm randomized comparison.
- **Arm A (XAI):** score + heatmap + one-sentence tip.
- **Arm B (score-only):** numeric score only.

## Participants
12–18 beginner photographers (peers + family). Written informed consent;
parental consent for minors. No PII stored; photos deleted after blind scoring.

## Procedure
1. Day 0 — each participant submits 5 baseline photos.
2. Days 1–10 — 10 practice photos using their assigned condition.
3. Day 14 — 5 final photos.
4. Baseline + final scored **blind** by two independent judges on the 4-axis rubric.

## Outcomes & stats
- Primary: mean change in (composition + exposure) baseline → final.
- Within-arm change: Wilcoxon signed-rank.
- Between-arm difference: Mann–Whitney U.
- Effect size: Cliff's δ (or Cohen's d). 95% bootstrap CIs throughout.

## Fallbacks
- If recruitment < 12 → within-subjects design (each person uses both arms on
  different days), needs fewer people.
- If model accuracy too low → drop to 2 axes (composition + exposure).

## Ethics
Low-risk educational study. Consent form + ethics statement go in the paper
appendix (competitions such as Regeneron STS ask for this).
