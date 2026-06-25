"""Deterministic, image-statistics scoring (1..5) for the four axes.

Unlike a model trained on synthetic degradations, these scores come straight
from real, measurable image properties, so they behave sensibly on ordinary
photos (good vs bad light, clean vs busy background) instead of failing
out-of-distribution. This is a transparent, classical scorer in the spirit of
B1 -- no training, fully explainable.

    exposure    <- luminance histogram (shadow/highlight clipping + mid-tone mass)
    sharpness   <- variance of the Laplacian (focus / motion blur)
    background  <- edge density in the image border (clutter behind the subject)
    composition <- where the salient mass sits vs the rule-of-thirds lines

All functions take a PIL.Image and return a float in [1, 5].
"""
from __future__ import annotations
import math

import numpy as np
from PIL import Image


def _gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _to_score(unit01: float) -> float:
    """Map a 0..1 'goodness' value to a 1..5 score."""
    return round(1.0 + 4.0 * _clamp(unit01), 1)


def exposure_score(img: Image.Image) -> float:
    g = _gray(img)
    dark = float((g < 0.12).mean())          # clipped / crushed shadows
    bright = float((g > 0.88).mean())         # blown highlights
    mean_l = float(g.mean())
    # clipping is the dominant signal: ~18%+ clipped -> fully penalised
    clip_pen = _clamp((dark + bright) / 0.18)
    # also penalise an overall too-dark or too-bright average
    mean_pen = _clamp((abs(mean_l - 0.5) - 0.10) / 0.30)
    good = 1.0 - (0.7 * clip_pen + 0.3 * mean_pen)
    return _to_score(good)


def sharpness_score(img: Image.Image) -> float:
    g = _gray(img) * 255.0
    # 3x3 Laplacian via numpy (no OpenCV dependency)
    lap = (-4 * g
           + np.roll(g, 1, 0) + np.roll(g, -1, 0)
           + np.roll(g, 1, 1) + np.roll(g, -1, 1))
    lap = lap[1:-1, 1:-1]                      # drop wrap-around border
    var = float(lap.var())
    # log map: ~50 = blurry, ~800 = crisp
    s = (math.log10(var + 1.0) - math.log10(50)) / (math.log10(800) - math.log10(50))
    return _to_score(s)


def _edge_mag(g: np.ndarray) -> np.ndarray:
    gx = np.abs(np.diff(g, axis=1))[:-1, :]
    gy = np.abs(np.diff(g, axis=0))[:, :-1]
    return gx + gy


def _colour_variety(img: Image.Image) -> float:
    """Fraction of the 4x4x4 RGB colour cube that the image actually uses.
    A plain wall/sky uses very few colour bins; a cluttered scene uses many."""
    arr = np.asarray(img.convert("RGB"))
    q = (arr // 64).reshape(-1, 3)               # 4 levels per channel -> 64 bins
    codes = q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2]
    return len(np.unique(codes)) / 64.0


def background_score(img: Image.Image) -> float:
    g = _gray(img)
    mag = _edge_mag(g)
    h, w = mag.shape

    # 1) Border clutter (background is usually in the outer frame).
    b = int(min(h, w) * 0.22)
    border = np.ones_like(mag, dtype=bool)
    border[b:h - b, b:w - b] = False
    border_frac = float((mag[border] > 0.06).mean())

    # 2) Grid busyness: split into a 4x4 grid and count how many cells are
    #    "busy" (lots of edges). Clutter anywhere -- centre included -- raises
    #    this, so it catches a messy desk even against a plain border.
    busy_cells, gh, gw = 0, h // 4, w // 4
    for i in range(4):
        for j in range(4):
            cell = mag[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw]
            if float((cell > 0.06).mean()) > 0.12:
                busy_cells += 1
    busy_frac = busy_cells / 16.0

    # 3) Colour variety: plain backgrounds use few colours, clutter uses many.
    colour = _colour_variety(img)

    # Combine: each term is 0 (clean) .. ~1 (busy). Border weighted highest.
    clutter = (0.35 * min(border_frac / 0.30, 1.0)
               + 0.50 * busy_frac
               + 0.15 * min(colour / 0.45, 1.0))
    return _to_score(1.0 - clutter)


def composition_score(img: Image.Image) -> float:
    g = _gray(img)
    mag = _edge_mag(g)
    ys, xs = np.nonzero(mag > mag.mean())
    if len(xs) < 50:
        return 3.0
    h, w = mag.shape
    cx, cy = xs.mean() / w, ys.mean() / h     # centroid of detail, 0..1
    thirds = np.array([1 / 3, 2 / 3])
    dx = float(np.min(np.abs(thirds - cx)))
    dy = float(np.min(np.abs(thirds - cy)))
    dist = math.hypot(dx, dy)                  # 0 = on a thirds intersection
    # reward near thirds, mild penalty for dead-centre or edge framing
    good = 1.0 - _clamp(dist / 0.33)
    return _to_score(good)


def image_scores(img: Image.Image) -> dict[str, float]:
    img = img.convert("RGB")
    return {
        "composition": composition_score(img),
        "exposure": exposure_score(img),
        "sharpness": sharpness_score(img),
        "background": background_score(img),
    }
