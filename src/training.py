"""Common training loop, metrics and plots, shared by all four models.

Every model is trained and scored by the same functions, so the comparison uses one
protocol: Adam, categorical cross-entropy, class weights, early stopping on validation
loss with the best weights restored, and the same metric definitions
(tumor = positive class).
"""
import json
import time

import keras
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402


# ------------------------------------------------------------------ training
def train(model, X_train, y_train, X_val, y_val, learning_rate, class_weight,
          epochs=config.MAX_EPOCHS, patience=config.PATIENCE, batch_size=config.BATCH_SIZE, verbose=2):
    """Compile + fit with early stopping on val_loss (best weights restored).
    y_* are integer labels (0 = no tumor, 1 = tumor); converted to one-hot for the softmax output."""
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    start = time.time()
    history = model.fit(X_train, keras.utils.to_categorical(y_train, 2),
                        validation_data=(X_val, keras.utils.to_categorical(y_val, 2)),
                        epochs=epochs, batch_size=batch_size, class_weight=class_weight,
                        callbacks=[stop], shuffle=True, verbose=verbose)
    history.train_seconds = time.time() - start
    return history


# ------------------------------------------------------------------ metrics
def compute_metrics(y_true, prob_tumor, threshold=0.5):
    """All metrics used in the comparison table. prob_tumor = P(tumor) from the softmax."""
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(prob_tumor) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    safe = lambda a, b: float(a / b) if b else float("nan")   # noqa: E731
    precision, recall = safe(tp, tp + fp), safe(tp, tp + fn)
    return {
        "accuracy": round(safe(tp + tn, cm.sum()), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),                          # sensitivity: tumors found
        "specificity": round(safe(tn, tn + fp), 4),          # healthy scans correctly cleared
        "f1": round(safe(2 * precision * recall, precision + recall), 4),
        "macro_f1": round(float(f1_score(y_true, pred, average="macro")), 4),
        "balanced_accuracy": round((recall + safe(tn, tn + fp)) / 2, 4),
        "roc_auc": round(float(roc_auc_score(y_true, prob_tumor)), 4) if len(set(y_true)) == 2 else float("nan"),
        "confusion_matrix": cm.tolist(),                     # rows = true [no_tumor, tumor]
        "n": int(cm.sum()),
    }


def evaluate(model, X, y, batch_size=256):
    prob = model.predict(X, batch_size=batch_size, verbose=0)[:, 1]
    metrics = compute_metrics(y, prob)
    return metrics, prob, (prob >= 0.5).astype(int)


def print_metrics(m):
    keys = ["accuracy", "precision", "recall", "specificity", "f1", "macro_f1", "balanced_accuracy", "roc_auc"]
    print("  ".join(f"{k}={m[k]:.4f}" for k in keys if k in m))
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    print(f"confusion: TN={tn} FP={fp} FN={fn} TP={tp}  (n={m['n']})")


# ------------------------------------------------------------------ plots
def plot_history(history, title, path):
    h = history.history if hasattr(history, "history") else history
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, a in zip(["loss", "accuracy"], ax):
        a.plot(h[k], label="train")
        a.plot(h[f"val_{k}"], label="validation")
        a.set_xlabel("epoch"), a.set_title(f"{title}: {k}"), a.legend(), a.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, title, path):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], config.CLASS_NAMES), ax.set_yticks([0, 1], config.CLASS_NAMES)
    ax.set_xlabel("predicted"), ax.set_ylabel("true"), ax.set_title(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ results
def save_results(name, metrics, history, model, extra=None, split="val"):
    h = history.history if hasattr(history, "history") else history
    record = {
        "model": name,
        "split_evaluated": split,
        f"{split}_metrics": metrics,
        "epochs_run": len(h["loss"]),
        "best_epoch": int(np.argmin(h["val_loss"])) + 1,
        "train_seconds": round(float(getattr(history, "train_seconds", 0.0)), 1),
        "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        "history": {k: [round(float(v), 4) for v in vals] for k, vals in h.items()},
    }
    record.update(extra or {})
    config.METRIC_DIR.mkdir(parents=True, exist_ok=True)
    (config.METRIC_DIR / f"{name}.json").write_text(json.dumps(record, indent=2))
    return record
