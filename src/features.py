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


# ====================================================================== final phase
# Fine-tuning the last block of each backbone. The layers *below* the fine-tuned block
# stay frozen, so their output for a given input image never changes. We therefore run
# the frozen lower part once per image (and once per cached augmented view) and store
# the intermediate feature maps; fine-tuning then trains only the upper block + head on
# them. This is the same model as end-to-end fine-tuning with a frozen lower part, at a
# fraction of the CPU cost (we had no GPU).
#   VGG16    : lower = up to block4_pool (14x14x512),       upper = block5 (3 conv) + GAP
#   ResNet50 : lower = up to conv4_block6_out (14x14x1024), upper = conv5 stage (3 blocks) + GAP
import json
from pathlib import Path

import cv2

from . import cropping

SPLIT_POINT = {"vgg16": "block4_pool", "resnet50": "conv4_block6_out"}
INTER_SHAPE = {"vgg16": (14, 14, 512), "resnet50": (14, 14, 1024)}
N_AUG_VIEWS = 2                         # cached augmented views per training image


def load_cropped_rgb(path):
    """Cropped 256x256 grayscale PNG -> 224x224x3 float32 in [0, 255] (bilinear)."""
    g = cv2.imread(str(config.ROOT / path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, config.CNN_IMG_SIZE, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.repeat(g[..., None], 3, axis=2)


def augmenter():
    from .data_pipeline import ImageDataGenerator, AUGMENTATION
    return ImageDataGenerator(**AUGMENTATION)


def build_lower(name):
    from .models import BACKBONES
    constructor, _, _ = BACKBONES[name]
    base = constructor(weights="imagenet", include_top=False, input_shape=(*config.CNN_IMG_SIZE, 3))
    return keras.Model(base.input, base.get_layer(SPLIT_POINT[name]).output, name=f"{name}_lower")


def inter_path(name, view):
    return config.FEATURE_DIR / f"{name}_{view}.f16"          # raw float16 memmap


def extract_intermediate(name, paths, view, aug_seed_base=None):
    """Writes the frozen lower-part feature maps of `paths` to a float16 memmap.
    view = 'orig' (no augmentation) or 'aug0', 'aug1', ... (one random augmentation per
    image, seeded by image index so it is reproducible). Resumable."""
    from .models import BACKBONES
    _, preprocess_fn, _ = BACKBONES[name]
    out = inter_path(name, view)
    meta = out.with_suffix(".json")
    n = len(paths)
    state = json.loads(meta.read_text()) if meta.exists() else {"n": n, "done": 0, "seconds": 0.0}
    if state["done"] >= n:
        print(f"{out.name}: already complete ({n})")
        return
    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(out, dtype=np.float16, mode="r+" if out.exists() else "w+", shape=(n, *INTER_SHAPE[name]))
    lower = build_lower(name)
    gen = augmenter() if view != "orig" else None
    no_grad = contextlib.nullcontext()
    if keras.backend.backend() == "torch":
        import torch
        no_grad = torch.no_grad()
    start, bs = time.time(), config.BATCH_SIZE
    with no_grad:
        for s in range(state["done"], n, bs):
            batch = []
            for k in range(s, min(n, s + bs)):
                x = load_cropped_rgb(paths[k])
                if gen is not None:
                    x = gen.random_transform(x, seed=aug_seed_base + k)
                batch.append(x)
            x = preprocess_fn(np.stack(batch))
            mm[s:s + len(batch)] = keras.ops.convert_to_numpy(lower(x, training=False)).astype(np.float16)
            gc.collect()
            state["done"] = s + len(batch)
            if (s // bs) % 10 == 0 or state["done"] == n:
                mm.flush()
                el = time.time() - start
                state_out = dict(state, seconds=state["seconds"] + el)
                meta.write_text(json.dumps(state_out))
                print(f"  {name}/{view}: {state['done']}/{n}  ({el / 60:.1f} min this run)", flush=True)
    mm.flush()
    state["seconds"] += time.time() - start
    meta.write_text(json.dumps(state))
    print(f"{out.name} complete: {n} images, {state['seconds'] / 60:.1f} min total")


def open_intermediate(name, view, n):
    return np.memmap(inter_path(name, view), dtype=np.float16, mode="r", shape=(n, *INTER_SHAPE[name]))
