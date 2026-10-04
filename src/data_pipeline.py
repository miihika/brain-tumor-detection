"""Dataset acquisition and preprocessing pipeline.

Steps (run once, from notebooks/01_data_preprocessing.ipynb):
  1. download    - Brain Tumor MRI Dataset (Masoud Nickparvar) from Kaggle
  2. scan/clean  - open every image, drop unreadable files, hash files to find exact duplicates
  3. encode      - glioma + meningioma + pituitary -> "tumor" (1), notumor -> "no_tumor" (0)
  4. split       - official Testing folder = held-out test set;
                   official Training folder -> stratified 80/20 train/validation (fixed seed)
  5. write       - copy images into data/binary/{train,val,test}/{no_tumor,tumor}
                   and save the split manifest (results/dataset/split_manifest.csv)

After that, Keras ImageDataGenerator loads batches from data/binary (see make_*_generator).
"""
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from . import config

try:  # Keras 2 / early Keras 3
    from keras.preprocessing.image import ImageDataGenerator
except ImportError:  # newer Keras 3 keeps ImageDataGenerator in its legacy module
    from keras.src.legacy.preprocessing.image import ImageDataGenerator


# ------------------------------------------------------------------ 1. download
def download_dataset():
    """Download (or reuse the cached copy of) the Kaggle dataset; returns its folder."""
    import kagglehub
    return Path(kagglehub.dataset_download(config.KAGGLE_DATASET))


# ------------------------------------------------------------------ 2-3. scan, clean, encode
def scan_images(raw_dir):
    """One row per image file with its source split/class, binary label, size and hash."""
    rows = []
    for source_split in ["Training", "Testing"]:
        for class_dir in sorted((Path(raw_dir) / source_split).iterdir()):
            for path in sorted(class_dir.iterdir()):
                row = {
                    "path": str(path),
                    "source_split": source_split,
                    "source_class": class_dir.name,
                    "label": "no_tumor" if class_dir.name == config.NO_TUMOR_FOLDER else "tumor",
                }
                try:
                    with Image.open(path) as im:
                        im.verify()                      # raises if the file is corrupt
                    with Image.open(path) as im:
                        row["width"], row["height"] = im.size
                        row["mode"] = im.mode
                    row["readable"] = True
                except Exception:
                    row["readable"] = False
                row["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
                rows.append(row)
    return pd.DataFrame(rows)


def mark_duplicates(df):
    """Flag exact duplicate files (same MD5). The first copy is kept, the rest are dropped.

    Also reports whether any duplicate crosses Training/Testing (that would leak test
    images into training) or has conflicting labels.
    """
    df = df.sort_values("path").reset_index(drop=True)
    df["is_duplicate"] = df.duplicated("md5", keep="first")
    groups = df[df.duplicated("md5", keep=False)].groupby("md5")
    report = {
        "duplicate_groups": int(groups.ngroups),
        "files_removed": int(df["is_duplicate"].sum()),
        "groups_crossing_train_test": int((groups["source_split"].nunique() > 1).sum()),
        "groups_with_label_conflict": int((groups["label"].nunique() > 1).sum()),
    }
    return df, report


# ------------------------------------------------------------------ 4. split
def make_split(df, seed=config.SEED, val_fraction=config.VAL_FRACTION):
    """Assign train / val / test. Stratified on the original 4 classes so that each
    tumor subtype is represented proportionally in train and validation."""
    df = df[df["readable"] & ~df["is_duplicate"]].copy()
    test = df[df["source_split"] == "Testing"].copy()
    test["split"] = "test"
    train_pool = df[df["source_split"] == "Training"]
    train, val = train_test_split(
        train_pool, test_size=val_fraction, stratify=train_pool["source_class"], random_state=seed
    )
    train, val = train.copy(), val.copy()
    train["split"], val["split"] = "train", "val"
    return pd.concat([train, val, test]).reset_index(drop=True)


# ------------------------------------------------------------------ 5. write folders
def write_binary_folders(split_df, out_dir=config.BINARY_DIR):
    """Copy images into out_dir/{split}/{label}/ -- the 'merged single folder' layout
    that flow_from_directory expects. File names keep the original class as a prefix."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    new_paths = []
    for row in split_df.itertuples():
        dst = out_dir / row.split / row.label / f"{row.source_class}_{Path(row.path).name}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row.path, dst)
        new_paths.append(str(dst.relative_to(config.ROOT)))
    split_df = split_df.copy()
    split_df["binary_path"] = new_paths
    return split_df


def split_summary(split_df):
    """Counts per split x class (+ tumor subtypes)."""
    table = pd.crosstab(split_df["split"], split_df["label"]).reindex(["train", "val", "test"])
    table["total"] = table.sum(axis=1)
    table["tumor_share_%"] = (100 * table["tumor"] / table["total"]).round(1)
    return table


def class_weights(split_df):
    """Inverse-frequency class weights from the TRAIN split: n_total / (2 * n_class)."""
    counts = split_df[split_df["split"] == "train"]["label"].value_counts()
    total = counts.sum()
    return {i: float(total / (2 * counts[name])) for i, name in enumerate(config.CLASS_NAMES)}


# ------------------------------------------------------------------ generators (Keras ImageDataGenerator)
AUGMENTATION = dict(           # used for training only (fine-tuning stage)
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True,
)


def make_cnn_generator(split, preprocess_fn, augment=False, shuffle=False):
    """224x224 RGB batches for VGG16 / ResNet50 / hybrid.

    preprocess_fn is the backbone's own keras.applications preprocess_input
    (ImageNet normalisation). Augmentation is only ever applied to the train split.
    """
    if augment and split != "train":
        raise ValueError("augmentation is only allowed on the training split")
    datagen = ImageDataGenerator(preprocessing_function=preprocess_fn, **(AUGMENTATION if augment else {}))
    return datagen.flow_from_directory(
        config.BINARY_DIR / split,
        target_size=config.CNN_IMG_SIZE,
        color_mode="rgb",                  # grayscale MRI slices are repeated to 3 channels
        classes=config.CLASS_NAMES,
        class_mode="categorical",
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        seed=config.SEED,
        interpolation="bilinear",
    )


def make_mlp_generator(split):
    """64x64 grayscale batches rescaled to [0, 1] for the MLP baseline."""
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    return datagen.flow_from_directory(
        config.BINARY_DIR / split,
        target_size=config.MLP_IMG_SIZE,
        color_mode="grayscale",
        classes=config.CLASS_NAMES,
        class_mode="categorical",
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        interpolation="bilinear",
    )


def generator_to_arrays(gen):
    """Read a whole (non-shuffled) generator into memory: X, integer labels y."""
    xs = [gen[i][0] for i in range(len(gen))]
    return np.concatenate(xs), gen.classes.copy()
