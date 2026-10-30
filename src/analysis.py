"""Final comparison tables, error analysis and figures (notebook 12 and the report).
Everything is computed from the saved runs in results/final/runs/."""
import json

import cv2
import matplotlib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt  # noqa: E402

from . import config, experiments as E, evaluation as ev  # noqa: E402
from .training import compute_metrics  # noqa: E402

def _select_final():
    """Final variant per CNN = the stage (frozen head or fine-tuned) with the higher mean VALIDATION
    balanced accuracy -- the test split plays no part in this choice."""
    out = {"mlp": "trained"}
    for k in ["vgg16", "resnet50", "hybrid"]:
        best, score = "frozen", -1.0
        for v in ["frozen", "finetuned"]:
            recs = E.load_runs("final", k, v)
            if recs:
                s = float(np.mean([r["val_metrics"]["balanced_accuracy"] for r in recs]))
                if s > score:
                    best, score = v, s
        out[k] = best
    return out


FINAL = _select_final()
LABEL = {"mlp": "MLP baseline", "vgg16": "VGG16", "resnet50": "ResNet50", "hybrid": "Hybrid VGG16–ResNet50"}
KEYS = ["accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy", "roc_auc"]
OUT = config.RESULTS_DIR / "final"


def fmt(s, k, pct=True):
    v = s[k]
    if k in ("FN", "FP"):
        return f"{v['mean']:.1f} ± {v['std']:.1f}" if s["n_seeds"] > 1 else f"{v['mean']:.0f}"
    if pct and k != "roc_auc":
        return f"{100 * v['mean']:.2f} ± {100 * v['std']:.2f}" if s["n_seeds"] > 1 else f"{100 * v['mean']:.2f}"
    return f"{v['mean']:.3f} ± {v['std']:.3f}" if s["n_seeds"] > 1 else f"{v['mean']:.3f}"


def comparison(split="test"):
    rows, raw = {}, {}
    for key, variant in FINAL.items():
        recs = E.load_runs("final", key, variant)
        if not recs:
            continue
        s = E.summarise(recs, split)
        raw[key] = s
        rows[LABEL[key] + {"trained": "", "frozen": " (frozen)", "finetuned": " (fine-tuned)"}[variant]] = {**{k: fmt(s, k) for k in KEYS + ["FN", "FP"]},
                                                                        "seeds": s["n_seeds"]}
    return pd.DataFrame(rows).T, raw


def comparison_variants():
    rows = {}
    for key in ["vgg16", "resnet50", "hybrid"]:
        for variant in ["frozen", "finetuned"]:
            recs = E.load_runs("final", key, variant)
            if not recs:
                continue
            sv, st = E.summarise(recs, "val"), E.summarise(recs, "test")
            rows[f"{LABEL[key]} · {variant}"] = {
                "val balanced acc": fmt(sv, "balanced_accuracy"), "test accuracy": fmt(st, "accuracy"),
                "test recall": fmt(st, "recall"), "test specificity": fmt(st, "specificity"),
                "test F1": fmt(st, "f1"), "test ROC-AUC": fmt(st, "roc_auc"),
                "trainable params": f"{recs[0]['trainable_params']:,}",
                "train time / run (min)": f"{np.mean([r['train_seconds'] for r in recs]) / 60:.1f}", "seeds": len(recs)}
    return pd.DataFrame(rows).T


def seed42_probs(split="test"):
    m = E.manifest_for("final")
    y = E.labels(m, split)
    probs = {k: E.load_prob("final", k, v, 42, split) for k, v in FINAL.items()
             if (E.RUN_DIR / f"final_{k}_{v}_s42_{split}_prob.npy").exists()}
    return m[m.split == split].reset_index(drop=True), y, probs


def mcnemar_table():
    sub, y, probs = seed42_probs()
    keys = list(probs)
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            r = ev.mcnemar(y, probs[a] >= 0.5, probs[b] >= 0.5)
            rows.append({"A": LABEL[a], "B": LABEL[b], "A right, B wrong": r["a_right_b_wrong"],
                         "A wrong, B right": r["a_wrong_b_right"], "p-value (exact)": round(r["p_value"], 4)})
    return pd.DataFrame(rows)


def errors_by_subtype():
    sub, y, probs = seed42_probs()
    rows = {}
    for k, p in probs.items():
        pred = p >= 0.5
        r = {}
        for c in ["glioma", "meningioma", "pituitary"]:
            msk = (sub.source_class == c).values
            r[f"{c} missed"] = f"{int((~pred[msk]).sum())} / {int(msk.sum())}"
        nt = (sub.source_class == "notumor").values
        r["no-tumor false alarms"] = f"{int(pred[nt].sum())} / {int(nt.sum())}"
        r["bootstrap 95% CI, balanced acc."] = "[{:.3f}, {:.3f}]".format(*ev.bootstrap_ci(y, p, "balanced_accuracy"))
        rows[LABEL[k]] = r
    return pd.DataFrame(rows).T


def roc_figure():
    sub, y, probs = seed42_probs()
    ev.plot_roc({LABEL[k]: (y, p) for k, p in probs.items()}, config.FIG_DIR / "final_roc.png",
                title="ROC curves, test split (seed 42)")


def ablation():
    """Frozen-feature models (and the MLP) under the three protocols, mean over seeds."""
    rows = {}
    for p, pname in [("original_uncropped", "P0 original split, uncropped (interim)"),
                     ("original_cropped", "P1 original split, cropped"),
                     ("final", "P2 leak-free split, cropped (final)")]:
        for key, variant in [("mlp", "trained"), ("vgg16", "frozen"), ("resnet50", "frozen"), ("hybrid", "frozen")]:
            recs = E.load_runs(p, key, variant)
            if not recs:
                continue
            sv, st = E.summarise(recs, "val"), E.summarise(recs, "test")
            rows[(pname, LABEL[key])] = {"val accuracy": fmt(sv, "accuracy"), "test accuracy": fmt(st, "accuracy"),
                                         "test balanced acc": fmt(st, "balanced_accuracy"),
                                         "test specificity": fmt(st, "specificity"), "test ROC-AUC": fmt(st, "roc_auc"),
                                         "seeds": len(recs)}
    df = pd.DataFrame(rows).T
    df.index.names = ["protocol", "model"]
    return df


def leak_vs_clean():
    """Original split (cropped): test images that have a near-duplicate in train/val vs those that do not."""
    m = E.manifest_for("original_cropped")
    test = m[m.split == "test"].reset_index(drop=True)
    y = E.labels(m, "test")
    leak = test["leaked_in_original_split"].values
    rows = {}
    for key, variant in [("mlp", "trained"), ("vgg16", "frozen"), ("resnet50", "frozen"), ("hybrid", "frozen")]:
        f = E.RUN_DIR / f"original_cropped_{key}_{variant}_s42_test_prob.npy"
        if not f.exists():
            continue
        p = np.load(f)
        for name, msk in [("has near-duplicate in train/val", leak), ("no near-duplicate", ~leak)]:
            t = msk & (y == 1)                      # tumour images only: almost every no-tumour test image is
            missed = int((p[t] < 0.5).sum())        # leaked, so overall accuracy would mix in the class balance
            rows[(LABEL[key], name)] = {"tumour images": int(t.sum()), "missed": missed,
                                        "tumour recall %": f"{100 * (1 - missed / t.sum()):.2f}",
                                        "no-tumour images": int((msk & (y == 0)).sum())}
    df = pd.DataFrame(rows).T
    df.index.names = ["model (seed 42, frozen)", "test subset"]
    return df


def errors_gallery(key="hybrid", n=8):
    """Misclassified test scans of one model (seed 42), with Grad-CAM if it is a CNN."""
    from .gradcam_figures import cams_for
    sub, y, probs = seed42_probs()
    p = probs[key]
    wrong = np.flatnonzero((p >= 0.5) != y)
    order = wrong[np.argsort(-np.abs(p[wrong] - 0.5))][:n]
    test_rows = np.flatnonzero((E.manifest_for("final").split == "test").values)
    overlays = cams_for(key, test_rows[order]) if key != "mlp" else None
    cols = min(n, len(order))
    fig, ax = plt.subplots(2 if overlays is not None else 1, cols, figsize=(2.1 * cols, 4.6 if overlays is not None else 2.4),
                           squeeze=False)
    for c, i in enumerate(order[:cols]):
        g = cv2.resize(cv2.imread(str(config.ROOT / sub.cropped_path[i]), 0), (224, 224))
        ax[0, c].imshow(g, cmap="gray")
        ax[0, c].set_title(f"{sub.source_class[i]}\nP(tumor)={p[i]:.2f}", fontsize=8)
        if overlays is not None:
            ax[1, c].imshow(overlays[c])
    for a in ax.flat:
        a.axis("off")
    fig.suptitle(f"{LABEL[key]}: most confident test errors (seed 42)" +
                 (" — bottom: Grad-CAM" if overlays is not None else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "final_errors_gallery.png", dpi=150)
    plt.close(fig)


def build_all():
    OUT.mkdir(parents=True, exist_ok=True)
    comp, raw = comparison()
    res = {"comparison": comp, "comparison_variants": comparison_variants(), "mcnemar": mcnemar_table(),
           "errors_by_subtype": errors_by_subtype(), "ablation": ablation(), "leak_vs_clean": leak_vs_clean()}
    roc_figure()
    try:
        errors_gallery(best_model(raw))
    except Exception as e:  # noqa: BLE001
        print("errors gallery skipped:", e)
    for k, df in res.items():
        df.to_csv(OUT / f"{k}.csv")
    (OUT / "comparison_raw.json").write_text(json.dumps(raw, indent=2))
    return res


def best_model(raw):
    return max(raw, key=lambda k: (raw[k]["balanced_accuracy"]["mean"], raw[k]["roc_auc"]["mean"]))
