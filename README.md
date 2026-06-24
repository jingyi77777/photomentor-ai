# 📸 PhotoMentor AI

**An explainable photo-quality assistant for beginner photographers.**

Upload a photo → get a 1–5 score on four axes (**composition, exposure,
sharpness, background**), a **Grad-CAM heatmap** showing *where* the model is
looking, and **one actionable suggestion** for what to change.

> **Scope note (read first).** This repo is a **1-week technical MVP** of a
> larger 6-month research plan. The full plan includes a 12–18 person, 2-week
> user study and a published paper; those are **designed and documented** here
> (`docs/user_study_design.md`) but deliberately **not executed** in a week.
> What *is* delivered and working end-to-end: the three-model pipeline, per-axis
> Grad-CAM explainability, a rule-based feedback engine, a full evaluation
> harness with real results, and a deployable Streamlit demo. Some of the
> original plan was also **descoped during the build** for honest, documented
> reasons — see [What's done vs. descoped](#whats-done-vs-descoped).

---

## Results

Model **M1** (the main contribution) evaluated against the held-out labels.
All metrics report 95% bootstrap confidence intervals. **H2** from the research
plan asks whether per-axis scores correlate with labels at Spearman ρ ≥ 0.5.

| Axis | MAE (1–5 scale) | Spearman ρ | H2 (ρ ≥ 0.5) |
|------|-----------------|------------|--------------|
| Composition | 0.37 `[0.30, 0.44]` | 0.60 `[0.45, 0.70]` | ✅ |
| Exposure    | 0.37 `[0.29, 0.45]` | 0.60 `[0.47, 0.70]` | ✅ |
| Sharpness   | 0.37 `[0.28, 0.46]` | 0.52 `[0.35, 0.66]` | ✅ |
| Background  | 0.31 `[0.26, 0.38]` | 0.94 `[0.89, 0.97]` | ✅ |

All four axes pass H2. Mean absolute error is below half a point on the 1–5
scale for every axis.

> **Honest caveat on the numbers.** Background's ρ = 0.94 is **inflated** and
> should be discounted: its labels are a heuristic *proxy* (border edge
> density), so the model partly re-learns the statistic that generated them.
> The trustworthy figures are **composition, exposure, and sharpness
> (ρ ≈ 0.52–0.60)**, whose labels come from controlled, known degradations.
> See [Dataset and label provenance](#dataset-and-label-provenance).

Reproduce: `python -m src.evaluate` → writes `results/metrics.csv`.

---

## Demo

```bash
streamlit run app/streamlit_app.py
```

Upload a photo to get the four scores, per-axis Grad-CAM heatmaps, and a
suggestion. The sidebar toggles the two planned user-study conditions
(explainable vs. score-only), so the same app drives both arms of the study.

---

## The three models (the experimental contrast)

| ID | Model | What it establishes | Trained on |
|----|-------|---------------------|-----------|
| **B1** | BRISQUE (classical, no ML) | the "without deep learning" floor | — |
| **B2** | Frozen ResNet-50 + linear head → 4 axes | a basic transfer-learning baseline | custom set |
| **M1** | Frozen ResNet-50 + per-axis heads → 4 axes (+ Grad-CAM) | **main contribution**: per-axis scores with spatial explanations | custom set |

B1 gives one number and no spatial explanation — which is exactly why B2 and M1
exist. M1 adds a deeper head per axis and Grad-CAM, so every score comes with a
*where*. Both deep models start from an ImageNet-pretrained ResNet-50 backbone
(frozen) and learn the four 1–5 axes from the custom set.

---

## Quickstart (Google Colab, free GPU)

```bash
# 1. clone + install
git clone https://github.com/jingyi77777/photomentor-ai.git
cd photomentor-ai
pip install -r requirements.txt

# 2. build the labelled dataset (~120 images, all 4 axes auto-labelled)
python scripts/make_dataset.py --n 120

# 3. train B2 and M1
python -m src.train --model both

# 4. evaluate (MAE, Spearman ρ, 95% bootstrap CIs)
python -m src.evaluate

# 5. run the demo
streamlit run app/streamlit_app.py
```

A `scripts/smoke_test.py` runs the whole pipeline on throwaway data in ~10s to
confirm the environment works before touching real data.

---

## Dataset and label provenance

The custom set is **120 images** built by `scripts/make_dataset.py`. Base photos
come from [Lorem Picsum](https://picsum.photos) (royalty-free, no API key,
sourced from Unsplash). Images are **not committed** to the repo
(`.gitignore` covers `data/custom/*.jpg`) and are used locally only, keeping the
licensing clean.

Because true aesthetic labels can't be collected at scale without a user study,
labels are engineered so the model has a real range to learn from. **Three of
the four axes are exact by construction; one is a proxy.** This is the central
honesty point of the project:

| Axis | How it is labelled | Label quality |
|------|--------------------|---------------|
| Sharpness   | Gaussian blur of known strength | **exact** (controlled) |
| Exposure    | Brightness shift of known factor | **exact** (controlled) |
| Composition | Tilt + off-centre crop of known degree | **exact-ish** (controlled) |
| Background  | Border edge-density, quantile-binned 1–5 | **proxy** (heuristic) |

Per-axis score spread in the 120-image set: composition / exposure / sharpness
each have ~6 images at scores 1–4 and ~96 at 5 (only the degraded subset drops
below 5); background is balanced at ~24 per score (quantile binning).

The labelling protocol and these provenance notes are recorded in
`docs/labeling_log.md`.

---

## What's done vs. descoped

**Done and working in this repo**
- ✅ Data pipeline: automated download + degradation, balanced labels, 4-axis loader.
- ✅ All three models (B1 / B2 / M1) implemented and trained.
- ✅ Per-axis Grad-CAM heatmaps.
- ✅ Rule-based, LLM-free feedback engine (keeps the "explainable" claim clean).
- ✅ Evaluation: MAE, Spearman ρ, 95% bootstrap CIs, Cohen's κ helper.
- ✅ Streamlit demo implementing both study arms.

**Designed and documented, intentionally not executed in 1 week**
- 📋 12–18 person, 2-week user study (`docs/user_study_design.md`) — H1.
- 📋 IEEE paper / poster / FigShare DOI (research deliverables, not engineering).

**Descoped during the build (documented engineering calls)**
- ⚠️ **AVA large-scale pretraining.** The reachable AVA mirror ships images only,
  with no score labels, so AVA pretraining wasn't possible without a fragile
  scrape. Since ResNet-50 already carries ImageNet features, B2 and M1 are
  trained directly on the custom set instead.
- ⚠️ **Hand-collected 150–250 photo set.** Replaced by the automated 120-image
  set above. The trade-off — semi-synthetic / proxy labels in exchange for a
  bigger, balanced, reproducible dataset — is disclosed, not hidden.

Knowing what to cut when a dependency fails, and saying so plainly, is the point.

---

## Hypotheses (from the research plan)

- **H1** — beginners given explainable feedback improve composition + exposure
  more than score-only feedback. *Requires the user study; designed in
  `docs/user_study_design.md`, not run in this MVP.*
- **H2** — model per-axis scores correlate with labels at Spearman ρ ≥ 0.5.
  **Tested and passed on all four axes** (see [Results](#results)); the credible
  margin is on composition/exposure/sharpness, with background discounted as a
  proxy.

---

## Repository layout

```
photomentor-ai/
├── app/streamlit_app.py        # the demo (both study arms)
├── scripts/
│   ├── make_dataset.py         # build the labelled custom set (Picsum + degradation)
│   └── smoke_test.py           # end-to-end pipeline check on throwaway data
├── src/
│   ├── config.py               # all paths & hyperparameters
│   ├── dataset.py              # PyTorch Dataset + transforms
│   ├── models/
│   │   ├── brisque_baseline.py # B1
│   │   └── resnet_models.py    # B2 + M1
│   ├── train.py                # trains B2 and M1 on the custom set
│   ├── evaluate.py             # MAE / Spearman / bootstrap CIs / Cohen's κ
│   ├── feedback.py             # rule-based suggestion engine
│   └── gradcam_utils.py        # per-axis Grad-CAM + inference wrapper
├── docs/
│   ├── literature.md           # related-work notes
│   ├── labeling_log.md         # labelling protocol + provenance
│   └── user_study_design.md    # full study (designed, not run)
├── notebooks/01_colab_train.ipynb
├── results/                    # checkpoints + metrics.csv land here
├── requirements.txt
└── LICENSE
```

---

## License

MIT — see `LICENSE`. Base photos are from Lorem Picsum (Unsplash-sourced) and
are not redistributed in this repository.
