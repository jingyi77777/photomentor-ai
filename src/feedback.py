"""Rule-based feedback generator.

The project plan deliberately avoids an LLM here: a transparent rule table keeps
the "explainable" claim clean and makes the system trivial to evaluate. Each
rule maps a low axis score (optionally plus a cheap image statistic) to one
actionable sentence a beginner can act on.

`generate_feedback` returns at most `max_tips` suggestions, worst axis first.
"""
from __future__ import annotations
import numpy as np
from PIL import Image

from .config import AXES

# Threshold below which an axis is considered "needs work" (on the 1..5 scale).
LOW = 2.5


def _mean_brightness(img: Image.Image) -> float:
    """Mean luminance in 0..255."""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    return float(g.mean())


def _suggest_exposure(score: float, brightness: float) -> str:
    if brightness < 80:
        return "The shot looks underexposed — increase exposure or find brighter light."
    if brightness > 200:
        return "Highlights look blown out — lower exposure or avoid shooting into the light."
    return "Balance the exposure so the subject isn't lost in shadow or glare."


def _suggest_composition(score: float, _b: float) -> str:
    return "Place your subject off-centre using the rule of thirds, and watch for leading lines."


def _suggest_sharpness(score: float, _b: float) -> str:
    return "The subject looks soft — tap to focus, steady the camera, or use more light to freeze motion."


def _suggest_background(score: float, _b: float) -> str:
    return "The background is busy — change angle or move closer so clutter doesn't compete with the subject."


_RULES = {
    "composition": _suggest_composition,
    "exposure": _suggest_exposure,
    "sharpness": _suggest_sharpness,
    "background": _suggest_background,
}


def generate_feedback(scores: dict[str, float], img: Image.Image,
                      max_tips: int = 2) -> list[str]:
    """Return up to `max_tips` suggestions, worst-scoring axis first."""
    brightness = _mean_brightness(img)
    ranked = sorted(AXES, key=lambda a: scores.get(a, 5.0))
    tips: list[str] = []
    for axis in ranked:
        if scores.get(axis, 5.0) < LOW and len(tips) < max_tips:
            tips.append(_RULES[axis](scores[axis], brightness))
    if not tips:
        tips.append("Nicely balanced shot — strong across all four axes. Keep it up!")
    return tips
