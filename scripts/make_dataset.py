"""Build most of the custom dataset automatically.

Downloads royalty-free photos from Lorem Picsum (picsum.photos, no API key,
images sourced from Unsplash) and engineers a quality range the model can
actually learn from:

  * SHARPNESS  -- controlled by Gaussian blur, so we KNOW the label.
  * EXPOSURE   -- controlled by brightness shift, so we KNOW the label.

These two axes are auto-labelled exactly. COMPOSITION and BACKGROUND cannot be
synthesised, so they are pre-filled with a placeholder (3) and flagged
needs_review -- you set those by eye using the generated contact_sheet.png
(takes ~5 min). Honest note for the write-up: sharpness/exposure labels are
semi-synthetic by construction; composition/background are hand-labelled.

Images are NOT committed to the repo (.gitignore covers data/custom/*.jpg);
they're used locally only, which keeps the licensing clean.

Run in Colab:   !python scripts/make_dataset.py
"""
import io
import sys
import time
from pathlib import Path

import requests
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.config import AXES, CUSTOM_DIR

SIZE = 512

# Treatment recipe. Each entry becomes ONE labelled image and needs one base
# photo. Designed so sharpness and exposure each span the full 1..5 range.
# (kind, param, sharpness, exposure)
RECIPE = (
    [("clean", 0, 5, 5)] * 10 +              # crisp, well-exposed
    [("blur", 1.2, 4, 5)] * 2 +              # slightly soft
    [("blur", 2.5, 3, 5)] * 3 +              # noticeably soft
    [("blur", 5.0, 1, 5)] * 3 +              # very blurry
    [("expo", 1.4, 5, 4)] * 2 +              # a touch bright
    [("expo", 1.9, 5, 1)] * 2 +              # blown out
    [("expo", 0.7, 5, 3)] * 2 +              # a touch dark
    [("expo", 0.4, 5, 1)] * 2                # very dark
)


def fetch_image_ids(n: int) -> list[int]:
    """Grab a list of available Picsum image ids."""
    ids = []
    page = 1
    while len(ids) < n:
        r = requests.get("https://picsum.photos/v2/list",
                         params={"page": page, "limit": 100}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        ids += [item["id"] for item in batch]
        page += 1
    return ids[:n]


def download(image_id: int) -> Image.Image:
    url = f"https://picsum.photos/id/{image_id}/{SIZE}/{SIZE}.jpg"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def apply(img: Image.Image, kind: str, param: float) -> Image.Image:
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=param))
    if kind == "expo":
        return ImageEnhance.Brightness(img).enhance(param)
    return img


def contact_sheet(rows, cols=5, thumb=180):
    """Numbered grid of every image, for fast hand-labelling."""
    n = len(rows)
    r = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, r * (thumb + 20)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        im = Image.open(CUSTOM_DIR / row["filename"]).resize((thumb, thumb))
        x, y = (i % cols) * thumb, (i // cols) * (thumb + 20)
        sheet.paste(im, (x, y))
        draw.text((x + 4, y + thumb + 4), f'#{i}  {row["filename"]}', fill="black")
    out = CUSTOM_DIR / "contact_sheet.png"
    sheet.save(out)
    return out


def main():
    print(f"[1/4] Fetching image ids from Picsum ...")
    ids = fetch_image_ids(len(RECIPE))

    print(f"[2/4] Downloading + degrading {len(RECIPE)} images ...")
    rows = []
    for i, ((kind, param, sharp, expo), img_id) in enumerate(zip(RECIPE, ids)):
        for attempt in range(3):
            try:
                base = download(img_id)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"   skip id {img_id}: {e}")
                    base = None
                time.sleep(1)
        if base is None:
            continue
        out = apply(base, kind, param)
        fname = f"img_{i:03d}.jpg"
        out.save(CUSTOM_DIR / fname, quality=92)
        rows.append({
            "filename": fname,
            "composition": 3,            # placeholder -> hand-label
            "exposure": expo,            # auto
            "sharpness": sharp,          # auto
            "background": 3,             # placeholder -> hand-label
            "source": f"picsum/{img_id}",
            "degradation": f"{kind}:{param}",
            "needs_review": "composition,background",
        })
        print(f"   {fname}  {kind}({param})  sharp={sharp} expo={expo}")

    df = pd.DataFrame(rows)
    df.to_csv(CUSTOM_DIR / "labels.csv", index=False)
    print(f"[3/4] Wrote {len(df)} rows -> {CUSTOM_DIR/'labels.csv'}")

    sheet = contact_sheet(rows)
    print(f"[4/4] Contact sheet -> {sheet}")

    print("\nDONE. Sharpness + exposure are labelled. Two things left:")
    print("  1. Open contact_sheet.png and set the composition + background")
    print("     columns in labels.csv (1=poor .. 5=great). Spread the scores")
    print("     -- include some 1s and 2s, or those axes won't train.")
    print("  2. Then:  !python -m src.train --model both")
    print("\nSharpness/exposure spread:")
    print(df.groupby('sharpness').size().to_string())
    print(df.groupby('exposure').size().to_string())


if __name__ == "__main__":
    main()
