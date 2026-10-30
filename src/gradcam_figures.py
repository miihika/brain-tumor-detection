"""Grad-CAM figures for the fine-tuned CNNs (seed 42), built from the cached feature maps."""
import json

import cv2
import matplotlib
import numpy as np

import matplotlib.pyplot as plt  # noqa: E402

from . import config, experiments as E, finetune as F, gradcam as G  # noqa: E402
from .features import open_intermediate  # noqa: E402

_CACHE = {}


def final_variant(key):
    from .analysis import FINAL
    return FINAL.get(key, "finetuned")


def load_finetuned(key, seed=42, variant=None):
    """Model of the selected final variant: fine-tuned weights, or ImageNet upper block + stage-1 head."""
    variant = variant or final_variant(key)
    if (key, seed, variant) in _CACHE:
        return _CACHE[(key, seed, variant)]
    names = ["vgg16", "resnet50"] if key == "hybrid" else [key]
    cfg = json.loads((config.RESULTS_DIR / "final" / "chosen_heads.json").read_text())[key]
    head = F.build_head(sum(E.models.BACKBONES[n][2] for n in names), cfg["units"], cfg["dropout"])
    model, uppers = F.build_finetune_model(names, head, trainable=True)
    if variant == "finetuned":
        model.load_weights(config.DATA_DIR / "models" / f"final_{key}_finetuned_s{seed}.weights.h5")
    else:
        head.load_weights(config.DATA_DIR / "models" / f"final_{key}_frozen_s{seed}.weights.h5")
    _CACHE[(key, seed, variant)] = (names, uppers, head)
    return _CACHE[(key, seed, variant)]


def cams_for(key, manifest_rows, seed=42):
    """Grad-CAM overlays (224x224 RGB) for the given manifest rows. Hybrid: mean of the two
    branch maps after upsampling (each branch map is also available from G.gradcam)."""
    m = E.manifest_for("final")
    names, uppers, head = load_finetuned(key, seed)
    maps = [np.asarray(open_intermediate(n, "orig", len(m))[np.sort(manifest_rows)], dtype=np.float32) for n in names]
    order = np.argsort(np.argsort(manifest_rows))                       # back to the caller's order
    _, cams = G.gradcam(names, uppers, head, [x[order] for x in maps])
    out = []
    for k, row in enumerate(manifest_rows):
        g = cv2.resize(cv2.imread(str(config.ROOT / m.cropped_path[row]), 0), (224, 224))
        cam = np.mean([cv2.resize(c[k], (224, 224), interpolation=cv2.INTER_CUBIC) for c in cams], axis=0)
        out.append(G.overlay(g, cam / (cam.max() + 1e-8)))
    return out


def gradcam_figure(key, seed=42, per_class=2):
    """Correctly classified test examples: per_class per tumor subtype + no tumor."""
    m = E.manifest_for("final")
    test_rows = np.flatnonzero((m.split == "test").values)
    variant = final_variant(key)
    p = E.load_prob("final", key, variant, seed)
    y = E.labels(m, "test")
    rng = np.random.default_rng(seed)
    pick = []
    for c in ["glioma", "meningioma", "pituitary", "notumor"]:
        ok = np.flatnonzero((m.source_class.values[test_rows] == c) & ((p >= 0.5) == y))
        pick += list(rng.choice(ok, per_class, replace=False))
    rows = test_rows[pick]
    ov = cams_for(key, rows, seed)
    fig, ax = plt.subplots(2, len(rows), figsize=(2.1 * len(rows), 4.6))
    for c, (row, i) in enumerate(zip(rows, pick)):
        ax[0, c].imshow(cv2.resize(cv2.imread(str(config.ROOT / m.cropped_path[row]), 0), (224, 224)), cmap="gray")
        ax[0, c].set_title(f"{m.source_class[row]}\nP(tumor)={p[i]:.2f}", fontsize=8)
        ax[1, c].imshow(ov[c])
    for a in ax.flat:
        a.axis("off")
    fig.suptitle(f"Grad-CAM, {key} ({variant}, seed {seed}), correctly classified test scans", fontsize=10)
    fig.tight_layout()
    path = config.FIG_DIR / f"final_{key}_gradcam.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
