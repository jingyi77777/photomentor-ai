# Data

This folder is intentionally (almost) empty in the repo — images are **not**
committed. Two datasets are used.

## 1. AVA (training / pretraining)

We use a **stratified 5,000-image subsample** of the AVA dataset (full AVA is
~255k images). Download it with:

```bash
python -m data.download_ava --n 5000
```

This streams from the HuggingFace mirror [`Iceclear/AVA`](https://huggingface.co/datasets/Iceclear/AVA)
(no need to download all 255k images) and writes:

```
data/ava/images/*.jpg
data/ava/ava_subsample.csv     # filename, mean_score, bin
```

**Fallback if the HuggingFace mirror is down:** Kaggle mirrors exist, e.g.
`nicolacarrassi/ava-aesthetic-visual-assessment`. Download the images + the
`AVA.txt` votes file, then adapt `download_ava.py` to read local files instead
of streaming.

## 2. Custom 4-axis set (evaluation / fine-tuning)

Beginner photos labelled on the four axes (1–5 each). Put the images directly
in `data/custom/` and a `labels.csv` alongside them. Use
`labels_template.csv` as the schema:

```
filename, composition, exposure, sharpness, background, rater, notes
```

For the 1-week MVP, 30–50 photos are enough to demonstrate the pipeline end to
end. The full plan targets 150–250 with two raters per photo (see
`docs/labeling_log.md`).

> Consent: any photo from a person other than the developer needs a signed
> consent form (parental consent for minors). Records live in
> `data/custom/consent/` (not committed). See `docs/user_study_design.md`.
