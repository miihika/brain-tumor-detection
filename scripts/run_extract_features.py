"""Background job: cache frozen lower-part feature maps for the final phase."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports

import os, sys
os.environ.setdefault("KERAS_BACKEND", "torch")
import pandas as pd
from src import config
from src.features import extract_intermediate, N_AUG_VIEWS

m = pd.read_csv(config.DATASET_INFO_DIR / "split_manifest_final.csv")
paths = m["cropped_path"].tolist()
train_paths = m.loc[m.split == "train", "cropped_path"].tolist()
for name in ["vgg16", "resnet50"]:
    extract_intermediate(name, paths, "orig")
for v in range(N_AUG_VIEWS):
    for name in ["vgg16", "resnet50"]:
        extract_intermediate(name, train_paths, f"aug{v}", aug_seed_base=100000 * (v + 1))
print("ALL DONE")
