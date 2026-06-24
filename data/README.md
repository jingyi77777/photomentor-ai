# Data

Images are **not** committed to the repo (`.gitignore` covers
`data/custom/*.jpg`); they are generated locally.

## Custom set (training + evaluation)

The labelled set is built automatically by `scripts/make_dataset.py` from
royalty-free [Lorem Picsum](https://picsum.photos) photos (no API key,
Unsplash-sourced):

```bash
python scripts/make_dataset.py --n 300
```

This writes:

```
data/custom/img_*.jpg        # the images (not committed)
data/custom/labels.csv       # filename, composition, exposure, sharpness, background
data/custom/contact_sheet.png
```

All four axes are auto-labelled and balanced across 1–5. See the project
README and `docs/labeling_log.md` for how each axis is labelled (three are
controlled degradations; background is a measured proxy).

`labels_template.csv` shows the label schema if you ever want to add
hand-labelled photos of your own.
