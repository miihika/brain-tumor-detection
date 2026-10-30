"""Final-phase experiment runner (used by notebooks 07-10).

Protocols (same 4 models, same metrics in each):
  P0 'original_uncropped' : interim set-up -- original split (official Testing folder = test),
                            uncropped images. Reproduces the interim numbers.
  P1 'original_cropped'   : original split, brain-cropped images (removes the size/border cue).
  P2 'final'              : leak-free group split (no near-duplicates across train/val/test),
                            brain-cropped images.  <-- the protocol of the final comparison.

Hyper-parameters are chosen on the P2 validation split only (grid search, seed 42); the
test split is scored once per selected configuration. Each selected configuration is
re-trained with 3 seeds and reported as mean +- std.
"""
import itertools
import json

import cv2
import keras
import numpy as np
import pandas as pd

from . import config, models, training
from .evaluation import save_json

SEEDS = [42, 7, 2026]
PROTOCOLS = ["original_uncropped", "original_cropped", "final"]
RUN_DIR = config.RESULTS_DIR / "final" / "runs"

MLP_GRID = {"lr": [1e-4, 3e-4], "dropout": [(0.5, 0.3), (0.3, 0.2)]}
HEAD_GRID = {"lr": [1e-3, 3e-4], "units": [128, 256], "dropout": [0.3, 0.5]}


def manifest_for(protocol):
    m = pd.read_csv(config.DATASET_INFO_DIR / "split_manifest_final.csv")
    if protocol.startswith("original"):
        m["split"] = m["split_original"]
    return m


def labels(m, split):
    return (m.loc[m.split == split, "label"] == "tumor").astype(int).values


def class_weight(m):
    c = m.loc[m.split == "train", "label"].value_counts()
    return {0: float(c.sum() / (2 * c["no_tumor"])), 1: float(c.sum() / (2 * c["tumor"]))}


# ------------------------------------------------------------------ inputs
def mlp_arrays(protocol):
    """64x64 grayscale in [0, 1] for every image (manifest order), cached."""
    cropped = protocol != "original_uncropped"
    path = config.FEATURE_DIR / f"mlp_{'cropped' if cropped else 'uncropped'}.npy"
    if path.exists():
        return np.load(path)
    m = manifest_for(protocol)
    out = np.empty((len(m), 64, 64, 1), dtype=np.float32)
    for i, r in enumerate(m.itertuples()):
        if cropped:
            g = cv2.imread(str(config.ROOT / r.cropped_path), cv2.IMREAD_GRAYSCALE)
            g = cv2.resize(g, config.MLP_IMG_SIZE, interpolation=cv2.INTER_AREA)
        else:   # same loader as the interim ImageDataGenerator (PIL bilinear)
            g = np.asarray(keras.utils.load_img(config.ROOT / r.binary_path.replace("\\", "/"),
                                                color_mode="grayscale", target_size=config.MLP_IMG_SIZE,
                                                interpolation="bilinear"))
        out[i, ..., 0] = g / 255.0
    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


def cnn_features(model_name, protocol):
    """Pooled frozen-backbone features (manifest order). Cropped: from the cached maps.
    Uncropped: the interim cache (data/binary generators), re-ordered to the manifest."""
    from .finetune import pooled_features
    m = manifest_for(protocol)
    names = ["vgg16", "resnet50"] if model_name == "hybrid" else [model_name]
    feats = []
    for n in names:
        if protocol == "original_uncropped":
            from .features import extract_features
            parts = {}
            for s in ["train", "val", "test"]:
                X, _, files = extract_features(n, s)
                for x, f in zip(X, files):
                    parts[f"{s}/{f}".replace("\\", "/")] = x
            keys = [p.replace("\\", "/").split("binary/")[1] for p in m["binary_path"]]
            feats.append(np.stack([parts[k] for k in keys]))
        else:
            feats.append(pooled_features(n, m))
    return np.concatenate(feats, axis=1)


# ------------------------------------------------------------------ models
def build_mlp(dropout=(0.5, 0.3)):
    inputs = keras.Input(shape=(*config.MLP_IMG_SIZE, 1))
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(dropout[0])(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(dropout[1])(x)
    return keras.Model(inputs, keras.layers.Dense(2, activation="softmax")(x), name="mlp_baseline")


def build_head(dim, units, dropout):
    from .finetune import build_head as bh
    return bh(dim, units, dropout, name="head")


def fit_one(model_name, X, m, cfg, seed, verbose=0):
    keras.utils.set_random_seed(seed)
    tr, va = (m.split == "train").values, (m.split == "val").values
    if model_name == "mlp":
        net = build_mlp(cfg["dropout"])
    else:
        net = build_head(X.shape[1], cfg["units"], cfg["dropout"])
    hist = training.train(net, X[tr], labels(m, "train"), X[va], labels(m, "val"),
                          learning_rate=cfg["lr"], class_weight=class_weight(m), verbose=verbose)
    return net, hist


def grid_search(model_name, X, m, grid, seed=SEEDS[0]):
    """Selection rule: highest validation balanced accuracy, ties -> lower validation loss."""
    rows = []
    for values in itertools.product(*grid.values()):
        cfg = dict(zip(grid.keys(), values))
        net, hist = fit_one(model_name, X, m, cfg, seed)
        vm, _, _ = training.evaluate(net, X[(m.split == "val").values], labels(m, "val"))
        rows.append({**{k: str(v) for k, v in cfg.items()}, "cfg": cfg, "val_balanced_accuracy": vm["balanced_accuracy"],
                     "val_auc": vm["roc_auc"], "val_loss": float(min(hist.history["val_loss"])),
                     "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1})
    df = pd.DataFrame(rows).sort_values(["val_balanced_accuracy", "val_loss"], ascending=[False, True])
    return df.iloc[0]["cfg"], df.drop(columns="cfg")


def run_seeds(model_name, variant, protocol, X, m, cfg, seeds=SEEDS, keep_models=False):
    """Train with each seed, score val and test, save one JSON (+ probabilities) per run."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out, nets = [], {}
    for seed in seeds:
        net, hist = fit_one(model_name, X, m, cfg, seed)
        rec = {"model": model_name, "variant": variant, "protocol": protocol, "seed": seed,
               "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
               "epochs_run": len(hist.history["loss"]),
               "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
               "train_seconds": round(hist.train_seconds, 1),
               "trainable_params": int(sum(np.prod(w.shape) for w in net.trainable_weights)),
               "history": {k: [round(float(v), 4) for v in vals] for k, vals in hist.history.items()}}
        for split in ["val", "test"]:
            mm, prob, _ = training.evaluate(net, X[(m.split == split).values], labels(m, split))
            rec[f"{split}_metrics"] = mm
            np.save(RUN_DIR / f"{protocol}_{model_name}_{variant}_s{seed}_{split}_prob.npy", prob)
        (RUN_DIR / f"{protocol}_{model_name}_{variant}_s{seed}.json").write_text(json.dumps(rec, indent=2))
        out.append(rec)
        if keep_models:
            nets[seed] = net
    return (out, nets) if keep_models else out


def summarise(records, split="test", keys=("accuracy", "precision", "recall", "specificity", "f1",
                                           "balanced_accuracy", "roc_auc")):
    """mean +- std over seeds for one model/variant/protocol."""
    vals = {k: np.array([r[f"{split}_metrics"][k] for r in records]) for k in keys}
    fn = np.array([r[f"{split}_metrics"]["confusion_matrix"][1][0] for r in records])
    fp = np.array([r[f"{split}_metrics"]["confusion_matrix"][0][1] for r in records])
    s = {k: {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0} for k, v in vals.items()}
    s["FN"] = {"mean": float(fn.mean()), "std": float(fn.std(ddof=1)) if len(fn) > 1 else 0.0}
    s["FP"] = {"mean": float(fp.mean()), "std": float(fp.std(ddof=1)) if len(fp) > 1 else 0.0}
    s["n_seeds"] = len(records)
    return s


def load_runs(protocol=None, model=None, variant=None):
    recs = []
    for p in sorted(RUN_DIR.glob("*.json")):
        r = json.loads(p.read_text())
        if (protocol in (None, r["protocol"])) and (model in (None, r["model"])) and (variant in (None, r["variant"])):
            recs.append(r)
    return recs


def load_prob(protocol, model, variant, seed, split="test"):
    return np.load(RUN_DIR / f"{protocol}_{model}_{variant}_s{seed}_{split}_prob.npy")


__all__ = ["SEEDS", "PROTOCOLS", "manifest_for", "labels", "class_weight", "mlp_arrays", "cnn_features",
           "grid_search", "run_seeds", "summarise", "load_runs", "load_prob", "MLP_GRID", "HEAD_GRID",
           "save_json", "models"]
