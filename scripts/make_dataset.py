"""Build a balanced, reproducible custom dataset with auto-generated labels.

Every image gets an INDEPENDENT controlled degradation on three axes, plus a
measured proxy for the fourth, so all four axes are balanced across 1..5 and a
random train/val split is balanced on every axis:

  sharpness   <- Gaussian blur of known strength   (controlled, exact)
  exposure    <- brightness shift of known factor   (controlled, exact)
  composition <- tilt + centre crop of known degree (controlled, exact-ish)
  background  <- border edge-density on the CLEAN base, quantile-binned 1..5
                 (measured proxy; computed before any degradation)

The first three labels are exact by construction; background is a heuristic
proxy and its correlation should be discounted (see docs/labeling_log.md).
Base photos come from Lorem Picsum (royalty-free, no API key). Images are not
committed to the repo (.gitignore covers data/custom/*.jpg).

Run:  python scripts/make_dataset.py --n 120
"""
import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import requests
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.config import CUSTOM_DIR, SEED

SIZE = 512
BLUR = {5: 0.0, 4: 0.8, 3: 1.6, 2: 3.0, 1: 5.0}
EXPO = {5: 1.0, 4: 1.30, 3: 1.55, 2: 1.85, 1: 2.10}   # brighten (darken = 1/f)
TILT = {5: 0.0, 4: 4.0, 3: 9.0, 2: 14.0, 1: 20.0}     # degrees


def fetch_image_ids(n):
    ids, page = [], 1
    while len(ids) < n:
        r = requests.get("https://picsum.photos/v2/list",
                         params={"page": page, "limit": 100}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        ids += [it["id"] for it in batch]
        page += 1
    return ids[:n]


def download(image_id):
    url = f"https://picsum.photos/id/{image_id}/{SIZE}/{SIZE}.jpg"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def border_edge_energy(img):
    """Proxy for background clutter: gradient energy in the outer frame."""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    gx = np.abs(np.diff(g, axis=1))[:-1, :]
    gy = np.abs(np.diff(g, axis=0))[:, :-1]
    mag = gx + gy
    h, w = mag.shape
    b = int(min(h, w) * 0.22)
    mask = np.ones_like(mag, dtype=bool)
    mask[b:h - b, b:w - b] = False
    return float(mag[mask].mean())


def blur(img, s):
    return img.filter(ImageFilter.GaussianBlur(BLUR[s])) if BLUR[s] else img


def expose(img, s, dark):
    f = EXPO[s]
    if dark and s != 5:
        f = 1.0 / f
    return ImageEnhance.Brightness(img).enhance(f)


def tilt(img, s):
    a = TILT[s]
    if not a:
        return img
    rot = img.rotate(a, resample=Image.BICUBIC, expand=False)
    w, h = img.size
    cw, ch = int(w * 0.8), int(h * 0.8)
    l, t = (w - cw) // 2, (h - ch) // 2
    return rot.crop((l, t, l + cw, t + ch)).resize((w, h))


def balanced_targets(n, seed):
    """Return a length-n array with scores 1..5 as evenly distributed as
    possible, then shuffled. Independent per axis via different seeds."""
    arr = np.tile(np.arange(1, 6), int(np.ceil(n / 5)))[:n]
    rng = np.random.default_rng(seed)
    rng.shuffle(arr)
    return arr


def main(n):
    rng = np.random.default_rng(SEED)
    print(f"[1/4] Fetching {n} image ids from Picsum ...")
    ids = fetch_image_ids(n)

    # Independent, balanced targets so every axis spans 1..5 evenly.
    sharp_t = balanced_targets(len(ids), SEED + 1)
    expo_t = balanced_targets(len(ids), SEED + 2)
    comp_t = balanced_targets(len(ids), SEED + 3)

    print(f"[2/4] Downloading + degrading {len(ids)} images ...")
    recs = []
    for i, img_id in enumerate(ids):
        base = None
        for attempt in range(3):
            try:
                base = download(img_id)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"   skip {img_id}: {e}")
                time.sleep(1)
        if base is None:
            continue

        raw_bg = border_edge_energy(base)            # on the CLEAN base
        img = tilt(base, int(comp_t[i]))
        img = blur(img, int(sharp_t[i]))
        img = expose(img, int(expo_t[i]), dark=bool(rng.random() < 0.5))

        fname = f"img_{i:03d}.jpg"
        img.save(CUSTOM_DIR / fname, quality=92)
        recs.append([fname, int(comp_t[i]), int(expo_t[i]),
                     int(sharp_t[i]), raw_bg])

    df = pd.DataFrame(recs, columns=["filename", "composition", "exposure",
                                     "sharpness", "_bg_raw"])
    ranks = df["_bg_raw"].rank(method="first")
    df["background"] = (6 - np.ceil(ranks / len(df) * 5)).clip(1, 5).astype(int)
    df = df.drop(columns="_bg_raw")
    df["label_source"] = ("sharpness,exposure,composition=controlled;"
                          "background=edge-density proxy")
    df = df[["filename", "composition", "exposure", "sharpness",
             "background", "label_source"]]
    df.to_csv(CUSTOM_DIR / "labels.csv", index=False)
    print(f"[3/4] Wrote {len(df)} rows -> {CUSTOM_DIR / 'labels.csv'}")

    sample = df.head(25).to_dict("records")
    cols, thumb = 5, 160
    rws = (len(sample) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rws * (thumb + 18)), "white")
    d = ImageDraw.Draw(sheet)
    for j, row in enumerate(sample):
        im = Image.open(CUSTOM_DIR / row["filename"]).resize((thumb, thumb))
        x, y = (j % cols) * thumb, (j // cols) * (thumb + 18)
        sheet.paste(im, (x, y))
        d.text((x + 3, y + thumb + 3),
               f'c{row["composition"]}e{row["exposure"]}'
               f's{row["sharpness"]}b{row["background"]}', fill="black")
    sheet.save(CUSTOM_DIR / "contact_sheet.png")
    print(f"[4/4] Sample contact sheet -> {CUSTOM_DIR / 'contact_sheet.png'}")

    print("\nPer-axis score spread (should be ~even across 1..5):")
    for ax in ["composition", "exposure", "sharpness", "background"]:
        print(f"  {ax}: {dict(df[ax].value_counts().sort_index())}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=120)
    main(p.parse_args().n)
