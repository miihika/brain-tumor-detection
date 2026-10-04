"""Cached feature extraction for the frozen CNN backbones.

A frozen backbone maps each image to the same pooled feature vector every epoch, so
running it once per image and training the head on the stored vectors gives the same
model as training end-to-end with the backbone frozen (without augmentation), at a
fraction of the CPU time. The hybrid model reuses the VGG16 and ResNet50 caches and
simply concatenates them.
"""
import contextlib
import gc
import time

import keras
import numpy as np

from . import config
from .data_pipeline import make_cnn_generator
from .models import BACKBONES, build_backbone


def extract_features(backbone_name, split, force=False):
    """Returns (features, labels, filenames) for one split; cached in data/features/."""
    cache = config.FEATURE_DIR / f"{backbone_name}_{split}.npz"
    if cache.exists() and not force:
        d = np.load(cache, allow_pickle=False)
        print(f"loaded cached {cache.name}: {d['X'].shape} "
              f"(extracted earlier in {float(d['seconds']) / 60:.1f} min)")
        return d["X"], d["y"], d["filenames"]

    _, preprocess_fn, _ = BACKBONES[backbone_name]
    gen = make_cnn_generator(split, preprocess_fn, augment=False, shuffle=False)
    backbone = build_backbone(backbone_name)
    start = time.time()
    # Batch-by-batch instead of backbone.predict(): with Python 3.14 + the PyTorch backend,
    # ResNet50's intermediate tensors (residual Add blocks) were held in reference cycles and
    # not freed between batches; predict() grew to 51 GB and crashed. Converting each batch to
    # NumPy and running the garbage collector keeps memory flat.
    no_grad = contextlib.nullcontext()
    if keras.backend.backend() == "torch":
        import torch
        no_grad = torch.no_grad()
    chunks = []
    with no_grad:
        for i in range(len(gen)):
            x, _ = gen[i]
            chunks.append(keras.ops.convert_to_numpy(backbone(x, training=False)))
            gc.collect()
            if (i + 1) % 20 == 0 or i + 1 == len(gen):
                print(f"  batch {i + 1}/{len(gen)}  ({time.time() - start:.0f} s)", flush=True)
    X = np.concatenate(chunks)
    seconds = time.time() - start
    y, filenames = gen.classes.copy(), np.array(gen.filenames)
    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, filenames=filenames, seconds=seconds)
    print(f"{backbone_name}/{split}: {X.shape} in {seconds / 60:.1f} min "
          f"({len(X) / seconds:.1f} images/s) -> {cache.name}")
    return X, y, filenames


def load_hybrid_features(split):
    """Hybrid input = [VGG16 features (512) | ResNet50 features (2048)] -> 2560 values."""
    Xv, yv, fv = extract_features("vgg16", split)
    Xr, yr, fr = extract_features("resnet50", split)
    assert (fv == fr).all() and (yv == yr).all(), "VGG16 and ResNet50 caches are not aligned"
    return np.concatenate([Xv, Xr], axis=1), yv, fv
