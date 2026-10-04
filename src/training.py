"""Common training loop and evaluation protocol used by all four models."""
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import keras
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from . import config


def train(model, X_train, y_train, X_val, y_val, learning_rate, class_weight):
    """Adam + categorical cross-entropy, class weights, early stopping on val loss
    (best weights restored). Same budget for every model (config.MAX_EPOCHS / PATIENCE)."""
    keras.utils.set_random_seed(config.SEED)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE,
                                               restore_best_weights=True)
    start = time.time()
    history = model.fit(
        X_train, keras.utils.to_categorical(y_train, 2),
        validation_data=(X_val, keras.utils.to_categorical(y_val, 2)),
        epochs=config.MAX_EPOCHS, batch_size=config.BATCH_SIZE,
        class_weight=class_weight, callbacks=[early_stop], shuffle=True, verbose=2,
    )
    history.history["train_seconds"] = time.time() - start
    history.history["best_epoch"] = int(np.argmin(history.history["val_loss"])) + 1
    return history


def evaluate(model, X, y_true):
    """Metrics with 'tumor' (label 1) as the positive class."""
    prob_tumor = model.predict(X, verbose=0)[:, 1]
    y_pred = (prob_tumor >= 0.5).astype(int)     # same as argmax for a 2-way softmax
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),            # sensitivity: tumors found
        "specificity": tn / (tn + fp),                     # healthy scans correctly cleared
        "f1": f1_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_true, prob_tumor),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n": int(len(y_true)),
    }, prob_tumor, y_pred


def save_results(name, metrics, history, model, extra=None):
    """Writes results/metrics/<name>.json (metrics on the validation split + training info)."""
    h = history.history
    record = {
        "model": name,
        "split_evaluated": "val",
        "val_metrics": {k: (round(float(v), 4) if not isinstance(v, list) else v)
                        for k, v in metrics.items()},
        "epochs_run": len(h["loss"]),
        "best_epoch": h["best_epoch"],
        "train_seconds": round(h["train_seconds"], 1),
        "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        "history": {k: [round(float(x), 4) for x in v] for k, v in h.items()
                    if isinstance(v, list)},
    }
    record.update(extra or {})
    config.METRIC_DIR.mkdir(parents=True, exist_ok=True)
    (config.METRIC_DIR / f"{name}.json").write_text(json.dumps(record, indent=2))
    return record


def plot_history(history, title, path):
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, key, label in [(axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")]:
        ax.plot(epochs, h[key], label="train")
        ax.plot(epochs, h[f"val_{key}"], label="validation")
        ax.axvline(h["best_epoch"], color="grey", ls="--", lw=1, label="best epoch")
        ax.set_xlabel("Epoch"); ax.set_ylabel(label); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def plot_confusion(cm, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], config.CLASS_NAMES); ax.set_yticks([0, 1], config.CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


def print_metrics(metrics):
    for k in ["accuracy", "precision", "recall", "specificity", "f1", "macro_f1", "roc_auc"]:
        print(f"{k:>12}: {metrics[k]:.4f}")
    print(f"{'confusion':>12}: [[TN, FP], [FN, TP]] = {metrics['confusion_matrix']}")
