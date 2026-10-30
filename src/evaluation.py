"""Final-phase evaluation helpers: confidence intervals, paired tests, ROC plots,
and the 'shortcut probes' that measure how much of the task can be solved without
looking at the tumor at all."""
import json

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402
from .training import compute_metrics  # noqa: E402


def bootstrap_ci(y, prob, metric, n_boot=2000, seed=config.SEED, alpha=0.05):
    """Percentile bootstrap CI of one metric over test images."""
    rng = np.random.default_rng(seed)
    y, prob = np.asarray(y), np.asarray(prob)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        vals.append(compute_metrics(y[idx], prob[idx])[metric])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def mcnemar(y, pred_a, pred_b):
    """Exact McNemar test on paired predictions: b = A right & B wrong, c = A wrong & B right."""
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    ra, rb = a == y, b == y
    n01, n10 = int((ra & ~rb).sum()), int((~ra & rb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": float(p)}


def plot_roc(curves, path, title="ROC curves (test set)"):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for name, (y, prob) in curves.items():
        fpr, tpr, _ = roc_curve(y, prob)
        auc = compute_metrics(y, prob)["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate (1 - specificity)"), ax.set_ylabel("true positive rate (recall)")
    ax.set_title(title, fontsize=10), ax.legend(fontsize=7, loc="lower right"), ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ shortcut probes
def size_rule(manifest, split):
    """Predict 'tumor' iff the ORIGINAL file is exactly 512x512 (never looks at pixels)."""
    m = manifest[manifest["split"] == split]
    y = (m["label"] == "tumor").astype(int).values
    pred = ((m["width"] == 512) & (m["height"] == 512)).astype(int).values
    return compute_metrics(y, pred.astype(float))


def histogram_features(paths, bins=32):
    """Global grey-level histogram of each (cropped) image: no spatial information at all,
    so it cannot 'see' a tumor -- only overall contrast/brightness, i.e. the scanner,
    sequence and source of the image."""
    import cv2
    feats = []
    for p in paths:
        g = cv2.imread(str(config.ROOT / p), cv2.IMREAD_GRAYSCALE)
        h = np.histogram(g, bins=bins, range=(0, 256))[0].astype(np.float32)
        feats.append(h / h.sum())
    return np.array(feats)


def histogram_probe(manifest, path_col="cropped_path", train="train", test="test"):
    tr, te = manifest[manifest["split"] == train], manifest[manifest["split"] == test]
    Xtr, Xte = histogram_features(tr[path_col]), histogram_features(te[path_col])
    ytr, yte = (tr["label"] == "tumor").astype(int).values, (te["label"] == "tumor").astype(int).values
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    return compute_metrics(yte, clf.predict_proba(Xte)[:, 1])


def save_json(obj, name):
    path = config.RESULTS_DIR / "final" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return path


def metrics_table(records, keys=("accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy", "roc_auc")):
    rows = []
    for name, m in records.items():
        rows.append({"model": name, **{k: m[k] for k in keys},
                     "FN": m["confusion_matrix"][1][0], "FP": m["confusion_matrix"][0][1]})
    return pd.DataFrame(rows).set_index("model")
