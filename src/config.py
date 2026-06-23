"""Central configuration for PhotoMentor AI.

Keeping every magic number in one place makes the project easy to read and
easy for a student to tweak. Nothing here should need editing to reproduce
the baseline run except DATA_ROOT if you move the data folder.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
AVA_DIR = DATA_ROOT / "ava"                 # stratified 5k subsample lands here
CUSTOM_DIR = DATA_ROOT / "custom"           # 4-axis labelled beginner photos
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

for _d in (AVA_DIR, CUSTOM_DIR, RESULTS_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# The four quality axes (order matters: it defines the model head ordering)
# ----------------------------------------------------------------------------
AXES = ["composition", "exposure", "sharpness", "background"]
AXIS_SCALE = (1.0, 5.0)                      # human labels live on a 1..5 scale

# ----------------------------------------------------------------------------
# Image preprocessing (ImageNet stats, since we use pretrained ResNet-50)
# ----------------------------------------------------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 2
DEVICE = "cuda"                              # falls back to cpu in code if absent

# AVA pretraining (predict mean aesthetic score 1..10)
AVA_SUBSAMPLE_N = 5000
AVA_EPOCHS = 8
AVA_LR = 1e-3                                # only the head is trained

# Custom fine-tuning (predict 4 axes 1..5)
FINETUNE_EPOCHS = 30
FINETUNE_LR = 5e-4
VAL_FRACTION = 0.2

# Bootstrap settings for confidence intervals
N_BOOTSTRAP = 2000
CI = 0.95
