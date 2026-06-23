# 📸 PhotoMentor AI

**An explainable photo-quality assistant for beginner photographers.**

Upload a photo → get a 1–5 score on four axes (**composition, exposure,
sharpness, background**), a **Grad-CAM heatmap** showing *where* the model is
looking, and **one actionable suggestion** for what to change.

> **Scope note (read first).** This repository is a **1-week technical MVP** of
> a larger 6-month research plan. The full plan includes a 12–18 person,
> 2-week user study and a published paper. Those are **designed and documented
> here** (`docs/user_study_design.md`) but deliberately **not executed** —
> they don't fit in one week. What *is* delivered and working: the three-model
> pipeline, per-axis Grad-CAM explainability, the rule-based feedback engine,
> the evaluation harness, and a deployable Streamlit demo. See
> [What's done vs. scoped out](#whats-done-vs-scoped-out).

---

## Demo

```bash
streamlit run app/streamlit_app.py
```

The sidebar toggles the two planned user-study conditions (XAI vs. score-only),
so the same app drives both arms of the study.

---

## The three models (the experimental contrast)

| ID | Model | What it establishes | Trained on |
|----|-------|---------------------|-----------|
| **B1** | BRISQUE (classical, no ML) | the "without deep learning" floor | — |
| **B2** | Frozen ResNet-50 + linear head | the basic transfer-learning baseline | AVA |
| **M1** | ResNet-50 + 4 regression heads | **main contribution**: per-axis scores + heatmaps | AVA → fine-tuned on custom |

B1 gives one number and no spatial explanation — which is exactly why B2 and
M1 exist. M1 adds Grad-CAM so each score comes with a *where*.

---

## Quickstart (Google Colab, free GPU)

```bash
# 1. clone
git clone https://github.com/jingyi77777/photomentor-ai.git && cd photomentor-ai
pip install -r requirements.txt

# 2. get a stratified 5k slice of AVA (streams from HuggingFace)
python -m data.download_ava --n 5000

# 3. pretrain on AVA (trains B2 + M1's backbone head)
python -m src.train --stage ava

# 4. add your custom photos + data/custom/labels.csv, then fine-tune the 4 axes
python -m src.train --stage finetune

# 5. evaluate (MAE, Spearman, 95% bootstrap CIs)
python -m src.evaluate

# 6. run the demo
streamlit run app/streamlit_app.py
```

A ready-to-run Colab notebook is in `notebooks/01_colab_train.ipynb`.

---

## Repository layout

```
photomentor-ai/
├── app/streamlit_app.py        # the demo (both study arms)
├── data/
│   ├── download_ava.py         # AVA → stratified 5k subsample
│   ├── labels_template.csv     # schema for the custom 4-axis labels
│   └── README.md               # dataset instructions + fallbacks
├── src/
│   ├── config.py               # all paths & hyperparameters
│   ├── dataset.py              # PyTorch Datasets + transforms
│   ├── models/
│   │   ├── brisque_baseline.py # B1
│   │   └── resnet_models.py    # B2 + M1
│   ├── train.py                # AVA pretrain + custom fine-tune
│   ├── evaluate.py             # MAE / Spearman / bootstrap CIs / Cohen's κ
│   ├── feedback.py             # rule-based suggestion engine
│   └── gradcam_utils.py        # per-axis Grad-CAM + inference wrapper
├── docs/
│   ├── literature.md           # 7-paper related-work notes
│   ├── labeling_log.md         # labeling protocol + κ reliability
│   └── user_study_design.md    # full study (designed, not run)
└── results/                    # checkpoints + metrics.csv land here
```

---

## What's done vs. scoped out

**Done & working in this repo**
- ✅ Data pipeline: AVA streaming + stratified subsample; custom 4-axis loader.
- ✅ All three models (B1 / B2 / M1) implemented and trainable.
- ✅ Per-axis Grad-CAM heatmaps.
- ✅ Rule-based, LLM-free feedback engine (keeps the "explainable" claim clean).
- ✅ Evaluation: MAE, Spearman ρ (tests H2), bootstrap CIs, Cohen's κ.
- ✅ Streamlit demo implementing both study arms.

**Designed & documented, intentionally not executed in 1 week**
- 📋 12–18 person, 2-week user study (`docs/user_study_design.md`).
- 📋 Full 150–250 photo labelled set (MVP uses 30–50 to prove the pipeline).
- 📋 IEEE paper / poster / FigShare DOI (research deliverables, not engineering).

This split is the deliberate part: a project mentor's job is to know what's
realistic in the time available and to leave the rest cleanly specified.

---

## Hypotheses (from the research plan)
- **H1** — beginners given explainable feedback improve composition+exposure
  more than score-only (user study; *designed, not run here*).
- **H2** — model per-axis scores correlate with human labels at Spearman ρ ≥ 0.5
  (tested by `src.evaluate`; results in `results/metrics.csv`).

## License
MIT — see `LICENSE`.
