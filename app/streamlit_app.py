"""PhotoMentor AI -- Streamlit demo.

Upload a photo and get:
  * four axis scores (composition / exposure / sharpness / background)
  * Grad-CAM heatmaps showing WHERE the model is looking (XAI arm)
  * one or two actionable suggestions

The sidebar toggles the two user-study conditions:
  * Arm A (XAI)        -> score + heatmap + suggestion
  * Arm B (score-only) -> numeric score only
so the same app drives both arms of the planned study.

Run locally:   streamlit run app/streamlit_app.py
Deploy:        push to GitHub, connect at share.streamlit.io
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import AXES, CHECKPOINT_DIR
from src.models.brisque_baseline import brisque_score

st.set_page_config(page_title="PhotoMentor AI", page_icon="📸", layout="wide")


@st.cache_resource
def load_mentor():
    """Load the trained M1 model once. Returns None if no checkpoint yet."""
    ckpt = CHECKPOINT_DIR / "m1_multihead.pt"
    if not ckpt.exists():
        return None
    from src.gradcam_utils import PhotoMentor
    return PhotoMentor(ckpt)


def score_bar(label: str, value: float):
    st.markdown(f"**{label.capitalize()}**")
    st.progress(min(1.0, max(0.0, (value - 1) / 4)), text=f"{value:.1f} / 5")


st.title("📸 PhotoMentor AI")
st.caption("An explainable photo-quality assistant for beginner photographers.")

arm = st.sidebar.radio(
    "Feedback condition (user study arm)",
    ["Arm A — Explainable (score + heatmap + tip)", "Arm B — Score only"],
)
show_xai = arm.startswith("Arm A")

mentor = load_mentor()
file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if file is None:
    st.info("Upload a JPG or PNG to get feedback.")
    st.stop()

img = Image.open(file).convert("RGB")
left, right = st.columns([1, 1])
left.image(img, caption="Your photo", use_container_width=True)

if mentor is None:
    right.warning(
        "No trained model checkpoint found at "
        "`results/checkpoints/m1_multihead.pt`.\n\n"
        "Train it first (see README), or this is a fresh clone. "
        "Showing the BRISQUE baseline only:")
    right.metric("BRISQUE (lower = better)", f"{brisque_score(img):.1f}")
    st.stop()

result = mentor.analyse(img, make_heatmaps=show_xai)

with right:
    st.subheader("Scores")
    for axis in AXES:
        score_bar(axis, result["scores"][axis])
    st.caption(f"BRISQUE baseline (B1): {brisque_score(img):.1f}  (lower = better)")

if show_xai:
    st.subheader("Where the model is looking (Grad-CAM)")
    cols = st.columns(4)
    for col, axis in zip(cols, AXES):
        col.image(result["heatmaps"][axis], caption=axis, use_container_width=True)

    st.subheader("Suggestions")
    for tip in result["feedback"]:
        st.markdown(f"- {tip}")
else:
    st.subheader("Suggestions")
    st.markdown("_(Hidden in the score-only condition.)_")
