"""Shared settings.

Every model reads its paths, split and training budget from here, so all four
models are compared under the same protocol (same split, same metrics).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- paths
DATA_DIR = ROOT / "data"                  # not committed to git (see .gitignore)
BINARY_DIR = DATA_DIR / "binary"          # merged folders: {train,val,test}/{no_tumor,tumor}
FEATURE_DIR = DATA_DIR / "features"       # cached CNN features (.npz)
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
METRIC_DIR = RESULTS_DIR / "metrics"
DATASET_INFO_DIR = RESULTS_DIR / "dataset"
SPLIT_MANIFEST = DATASET_INFO_DIR / "split_manifest.csv"

# ---------------------------------------------------------------- dataset
KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
TUMOR_SUBTYPES = ["glioma", "meningioma", "pituitary"]   # merged into one "tumor" class
NO_TUMOR_FOLDER = "notumor"
CLASS_NAMES = ["no_tumor", "tumor"]       # label 0 = no tumor, label 1 = tumor (positive class)

# ---------------------------------------------------------------- split
SEED = 42
VAL_FRACTION = 0.20                       # 20% of the official Training folder -> validation
                                          # the official Testing folder is the held-out test set

# ---------------------------------------------------------------- inputs
CNN_IMG_SIZE = (224, 224)                 # VGG16, ResNet50 and the hybrid
MLP_IMG_SIZE = (64, 64)                   # MLP baseline: 64x64 grayscale, flattened
BATCH_SIZE = 32

# ---------------------------------------------------------------- training (same budget for every model)
MAX_EPOCHS = 30
PATIENCE = 5                              # early stopping on validation loss
HEAD_LR = 1e-3                            # Adam learning rate for the CNN classifier heads
MLP_LR = 1e-4                             # Adam learning rate for the MLP (raw pixels need a smaller step)
DENSE_UNITS = 256                         # hidden units in the CNN classifier head
DROPOUT = 0.5
