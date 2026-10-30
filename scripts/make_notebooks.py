"""Generates notebooks 04-12 (run from the repo root: python scripts/make_notebooks.py).
They are then executed with jupyter nbconvert so the saved copies carry their outputs."""
import nbformat as nbf
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = '''%matplotlib inline
import os, sys
os.environ.setdefault("KERAS_BACKEND", "torch")   # Keras 3 backend used for our runs (CPU); "tensorflow" also works
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))        # make the project's src/ package importable

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from IPython.display import Image, display
from src import config, data_pipeline as dp, models, features, training

keras.utils.set_random_seed(config.SEED)
print("Keras", keras.__version__, "| backend:", keras.backend.backend())'''

FINAL_SETUP = SETUP + '''
from src import experiments as E, evaluation as ev, finetune as F, gradcam as G
RERUN = False    # True = retrain everything here; False = load the saved runs produced by the same code
                 # (scripts/run_heads.py, scripts/run_finetune.py), which take hours on a CPU'''


def nb(cells):
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    book.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
    return book


def interim_cnn(name, title, owner, intro, backbones, fig_prefix):
    feats = (f'X_train, y_train, _ = features.extract_features("{backbones[0]}", "train")\n'
             f'X_val, y_val, _ = features.extract_features("{backbones[0]}", "val")'
             if len(backbones) == 1 else
             'from src.features import load_hybrid_features\n'
             'X_train, y_train, _ = load_hybrid_features("train")\n'
             'X_val, y_val, _ = load_hybrid_features("val")')
    pre = f'models.BACKBONES["{backbones[0]}"][1]'
    return nb([
        f"md:# {title}\n**Owner:** {owner}\n\n{intro}\n\n**Interim set-up (unoptimised):** ImageNet weights, backbone(s) frozen, "
        "global average pooling, then the common head Dense(256, ReLU) → Dropout(0.5) → Dense(2, softmax). "
        "The frozen backbone's pooled features are computed once and cached (`src/features.py`); only the head is trained. "
        "The last cell rebuilds the full end-to-end model and checks it gives the same predictions as the cached path.\n\n"
        "*This notebook reproduces the interim (Phase 2) result on the original split with uncropped images. "
        "The final-phase notebooks (07–12) change the protocol; see 07.*",
        SETUP,
        "md:## Pooled features from the frozen backbone(s) (cached)",
        feats + '\nmanifest = pd.read_csv(config.SPLIT_MANIFEST)\nclass_weight = dp.class_weights(manifest)\n'
                'print("train", X_train.shape, "val", X_val.shape, "| class weights", class_weight)',
        "md:## Model",
        f'head = models.build_head(models.head_input_dim({backbones!r}), name="{name}_head")\n'
        f'full_model = models.assemble_cnn({backbones!r}, head)\nfull_model.summary()',
        "md:## Train the classifier head",
        'history = training.train(head, X_train, y_train, X_val, y_val,\n'
        '                         learning_rate=config.HEAD_LR, class_weight=class_weight)',
        "md:## Validation results",
        f'training.plot_history(history, "{title.split(" · ")[1]}", config.FIG_DIR / "{fig_prefix}_curves.png")\n'
        'metrics, prob, pred = training.evaluate(head, X_val, y_val)\ntraining.print_metrics(metrics)\n'
        f'training.plot_confusion(metrics["confusion_matrix"], "{title.split(" · ")[1]} (validation)",\n'
        f'                        config.FIG_DIR / "{fig_prefix}_confusion.png")\n'
        f'record = training.save_results("{name}", metrics, history, head,\n'
        '                               extra={"total_params_full_model": int(full_model.count_params()),\n'
        '                                      "learning_rate": config.HEAD_LR})\n'
        'print("epochs run:", record["epochs_run"], "| best epoch:", record["best_epoch"],\n'
        '      "| head training time:", record["train_seconds"], "s")',
        "md:## Sanity check: full model = cached-feature path",
        f'xb, _ = dp.make_cnn_generator("val", {pre})[0]   # first 32 validation images\n'
        'diff = np.abs(full_model.predict(xb[:16], verbose=0) - head.predict(X_val[:16], verbose=0)).max()\n'
        'print(f"max |difference| in predicted probabilities: {diff:.2e}")',
    ])


def interim_comparison():
    return nb([
        "md:# 06 · Model comparison (interim, validation split)\n**Owner:** shared\n\n"
        "One table for the four interim models under the common protocol (same split, same metrics, "
        "tumor = positive class), plus a sanity check that never looks at the pixels: predict *tumor* "
        "whenever the original file is exactly 512 × 512.",
        SETUP,
        'rows = {}\nfor name, label in [("mlp_baseline", "MLP baseline"), ("vgg16", "VGG16"), ("resnet50", "ResNet50"),\n'
        '                    ("hybrid_vgg16_resnet50", "Hybrid VGG16–ResNet50")]:\n'
        '    rows[label] = json.loads((config.METRIC_DIR / f"{name}.json").read_text())["val_metrics"]\n'
        'manifest = pd.read_csv(config.SPLIT_MANIFEST)\nval = manifest[manifest.split == "val"]\n'
        'y = (val.label == "tumor").astype(int).values\n'
        'rule = ((val.width == 512) & (val.height == 512)).astype(float).values\n'
        'rows["Image-size rule (no model)*"] = training.compute_metrics(y, rule)\n'
        'from src.evaluation import metrics_table\ntable = metrics_table(rows, keys=("accuracy", "precision", "recall", "specificity", "f1", "roc_auc"))\n'
        'table.to_csv(config.METRIC_DIR / "comparison_val.csv")\ntable',
        'print("share of 512x512 files:", manifest.assign(s=(manifest.width == 512) & (manifest.height == 512))'
        '.groupby("label").s.mean().round(3).to_dict())',
        "md:*The image-size rule is not a model: it shows how much of the task can be solved from the image source alone. "
        "The ROC-AUC of a hard 0/1 rule is not meaningful and is reported only for completeness.*",
    ])


def nb07():
    return nb([
        "md:# 07 · Brain-region cropping, near-duplicate check and the leak-free split\n**Owner:** shared "
        "(cropping: Mihika; near-duplicate check: Pihu; split + shortcut probes: Udita)\n\n"
        "The interim report listed two risks for the final phase: an **image-source shortcut** (no-tumor scans come from a "
        "different source and have different file sizes and borders) and **near-duplicate slices** that exact (MD5) "
        "de-duplication cannot catch. This notebook fixes the first by cropping every slice to the head and measures the second.",
        FINAL_SETUP,
        "md:## 1. Crop every slice to the head, pad to square, resize to 256 × 256",
        'from src import cropping\nboxes_csv = config.DATASET_INFO_DIR / "crop_boxes.csv"\n'
        'boxes = cropping.build_cropped_dataset() if (RERUN or not boxes_csv.exists()) else pd.read_csv(boxes_csv)\n'
        'boxes.to_csv(boxes_csv, index=False)\nprint(len(boxes), "cropped images")',
        'import cv2\nsel = pd.concat([boxes[boxes.source_class == c].sample(2, random_state=3)\n'
        '                 for c in ["glioma", "meningioma", "pituitary", "notumor"]])\n'
        'fig, ax = plt.subplots(2, 8, figsize=(16, 4.4))\nfor k, r in enumerate(sel.itertuples()):\n'
        '    ax[0, k].imshow(cv2.imread(str(config.ROOT / r.binary_path.replace("\\\\", "/")), 0), cmap="gray")\n'
        '    ax[0, k].set_title(f"{r.source_class}\\n{r.width}×{r.height}", fontsize=9)\n'
        '    ax[1, k].imshow(cv2.imread(str(config.ROOT / r.cropped_path), 0), cmap="gray")\n'
        '    ax[1, k].set_title("cropped 256×256", fontsize=9)\nfor a in ax.flat: a.axis("off")\n'
        'fig.tight_layout(); fig.savefig(config.FIG_DIR / "crop_examples.png", dpi=150); plt.show()',
        "md:## 2. Near-duplicate check (perceptual hash + pixel correlation)\n"
        "pHash (64-bit) of every cropped image under all 8 flips/rotations; pairs within 8 bits are candidates, and a "
        "candidate is confirmed when the 64 × 64 crops correlate ≥ 0.97 under the best flip/rotation. Visual checks "
        "(below) show confirmed pairs are the same slice re-saved, flipped, rescaled or with changed contrast.",
        'from src import near_duplicates as nd\npairs_csv = config.DATASET_INFO_DIR / "near_duplicate_pairs.csv"\n'
        'if RERUN or not pairs_csv.exists():\n    H = nd.phash_table(boxes)\n'
        '    pairs = nd.confirm_pairs(boxes, nd.find_pairs(H, max_bits=8))\n    pairs.to_csv(pairs_csv, index=False)\n'
        'pairs = pd.read_csv(pairs_csv)\nsummary = json.loads((config.DATASET_INFO_DIR / "near_duplicate_summary.json").read_text())\n'
        'print(json.dumps({k: v for k, v in summary.items() if k != "final_split_counts"}, indent=1))',
        'conf = pairs[pairs["corr"] >= nd.CORR_THRESHOLD].sample(6, random_state=0)\n'
        'fig, ax = plt.subplots(2, 6, figsize=(13, 4.6))\nfor k, r in enumerate(conf.itertuples()):\n'
        '    for row, i in enumerate([r.i, r.j]):\n'
        '        b = boxes.iloc[i]\n        ax[row, k].imshow(cv2.imread(str(config.ROOT / b.cropped_path), 0), cmap="gray")\n'
        '        ax[row, k].set_title(f"{b.split} · {b.source_class}" + (f"\\ncorr {r.corr:.3f}" if row == 0 else ""), fontsize=8)\n'
        'for a in ax.flat: a.axis("off")\nfig.suptitle("Confirmed near-duplicate pairs (top/bottom)", fontsize=10)\n'
        'fig.tight_layout(); fig.savefig(config.FIG_DIR / "near_duplicate_examples.png", dpi=150); plt.show()',
        "md:## 3. Leak-free split\nEvery near-duplicate group goes entirely to train, validation or test "
        "(`StratifiedGroupKFold`, 13 folds: 3 → test, 2 → validation, 8 → train, stratified by the 4 source classes, seed 42). "
        "This split is used by all four models in the final comparison.",
        'final = pd.read_csv(config.DATASET_INFO_DIR / "split_manifest_final.csv")\n'
        'assert (final.groupby("group").split.nunique() == 1).all()          # no group crosses splits\n'
        'tab = pd.crosstab(final.split, final.label).reindex(["train", "val", "test"]); tab["total"] = tab.sum(axis=1)\n'
        'tab["tumor_%"] = (100 * tab.tumor / tab.total).round(1); print(tab)\n'
        'print("class weights (train):", E.class_weight(final))\n'
        'orig_test = final[final.split_original == "test"]\n'
        'print(f"original test images with a near-duplicate in original train/val: {orig_test.leaked_in_original_split.sum()} '
        'of {len(orig_test)}")\nprint(orig_test.groupby("source_class").leaked_in_original_split.mean().round(3))',
        "md:## 4. Shortcut probes (no tumor-specific information)\n"
        "* **Image-size rule:** tumor iff the original file is 512 × 512.\n"
        "* **Histogram probe:** logistic regression on the 32-bin grey-level histogram of the *cropped* image — no spatial "
        "information, so it can only use overall contrast/brightness (scanner, MRI sequence, source).",
        'probes = {}\nfor p in ["original_uncropped", "original_cropped", "final"]:\n    m = E.manifest_for(p)\n'
        '    probes[f"size rule · {p}"] = ev.size_rule(m, "test")\n'
        '    col = "binary_path" if p == "original_uncropped" else "cropped_path"\n'
        '    if p == "original_uncropped":\n        m = m.assign(binary_path=m.binary_path.str.replace("\\\\", "/", regex=False))\n'
        '    probes[f"histogram probe · {p}"] = ev.histogram_probe(m, path_col=col)\n'
        'ev.save_json(probes, "shortcut_probes.json")\nev.metrics_table(probes)',
        "md:**Reading:** the size rule is defeated by cropping only in the sense that the models never see the file size; "
        "the histogram probe shows that the *contrast* of no-tumor scans (mostly T2-weighted, from Br35H) still differs from "
        "tumor scans (mostly contrast-enhanced T1, from Figshare). Cropping cannot remove that; it is a property of how the "
        "public dataset was assembled, and it is discussed as a limitation in the report.",
    ])


def model_nb(num, key, title, owner, intro):
    return nb([
        f"md:# {num} · {title} (final phase)\n**Owner:** {owner}\n\n{intro}\n\n"
        "Final protocol: brain-cropped images, leak-free split (07), same metrics. Steps: (1) grid search on the validation "
        "split with seed 42, (2) the chosen configuration trained with 3 seeds, scored once on the test split, "
        + ("(3) fine-tuning of the last convolutional block from the stage-1 head, (4) Grad-CAM." if key != "mlp" else "(3) learning curves."),
        FINAL_SETUP,
        "md:## Inputs",
        ('m = E.manifest_for("final")\nX = E.mlp_arrays("final")\nprint(X.shape, "| class weights", E.class_weight(m))'
         if key == "mlp" else
         f'm = E.manifest_for("final")\nX = E.cnn_features("{key}", "final")      # frozen ImageNet pooled features\n'
         'print(X.shape, "| class weights", E.class_weight(m))'),
        "md:## 1. Hyper-parameter search (validation split only)",
        (f'grid_csv = config.RESULTS_DIR / "final" / "grid_{key}.csv"\n'
         f'if RERUN:\n    cfg, grid = E.grid_search("{key}", X, m, E.{"MLP_GRID" if key == "mlp" else "HEAD_GRID"})\n'
         'else:\n    grid = pd.read_csv(grid_csv)\n'
         f'    cfg = json.loads((config.RESULTS_DIR / "final" / "{"chosen_mlp.json" if key == "mlp" else "chosen_heads.json"}").read_text())["{key}"]\n'
         'print("chosen:", cfg)\ngrid'),
        f"md:## 2. Chosen configuration, 3 seeds ({'trained' if key == 'mlp' else 'frozen backbone'})",
        (f'variant = "{"trained" if key == "mlp" else "frozen"}"\n'
         f'recs = E.run_seeds("{key}", variant, "final", X, m, cfg) if RERUN else E.load_runs("final", "{key}", variant)\n'
         'pd.DataFrame({r["seed"]: {**{f"val_{k}": r["val_metrics"][k] for k in ["balanced_accuracy", "roc_auc"]},\n'
         '                          **{f"test_{k}": r["test_metrics"][k] for k in ["accuracy", "recall", "specificity", "f1", "roc_auc"]},\n'
         '                          "best_epoch": r["best_epoch"], "trainable_params": r["trainable_params"]} for r in recs}).T'),
        'summary = E.summarise(recs)\npd.DataFrame(summary).drop(columns="n_seeds").T.round(4)',
    ] + ([] if key == "mlp" else [
        "md:## 3. Fine-tuning the last block (stage 2)\n"
        + {"vgg16": "Unfrozen: `block5_conv1–3` (7.08 M weights).",
           "resnet50": "Unfrozen: the `conv5` stage (3 bottleneck blocks, 14.96 M weights); BatchNorm layers stay frozen in inference mode.",
           "hybrid": "Unfrozen: VGG16 `block5` **and** ResNet50 `conv5` (22.0 M weights), trained jointly through the fused head."}[key]
        + " Adam, learning rate 1e-5, ≤ 12 epochs, early stopping on validation loss (patience 3), class weights; every epoch "
        "each training image is shown as one of three cached views (original or one of two random augmentations).",
        f'ft = E.load_runs("final", "{key}", "finetuned")      # produced by scripts/run_finetune.py (same code: src/finetune.py)\n'
        'pd.DataFrame({r["seed"]: {**{f"val_{k}": r["val_metrics"][k] for k in ["balanced_accuracy", "roc_auc"]},\n'
        '                         **{f"test_{k}": r["test_metrics"][k] for k in ["accuracy", "recall", "specificity", "f1", "roc_auc"]},\n'
        '                         "epochs_run": r["epochs_run"], "best_epoch": r["best_epoch"],\n'
        '                         "minutes": round(r["train_seconds"] / 60, 1), "trainable_params": r["trainable_params"]} for r in ft}).T',
        f'h = ft[0]["history"]\ntraining.plot_history(h, "{title} fine-tuning (seed {{}})".format(ft[0]["seed"]),\n'
        f'                      config.FIG_DIR / "final_{key}_finetune_curves.png")\n'
        f'display(Image(str(config.FIG_DIR / "final_{key}_finetune_curves.png"), width=700))',
        f'cm = ft[0]["test_metrics"]["confusion_matrix"]\ntraining.plot_confusion(cm, "{title} fine-tuned (test)", '
        f'config.FIG_DIR / "final_{key}_confusion.png")\ndisplay(Image(str(config.FIG_DIR / "final_{key}_confusion.png"), width=300))',
        "md:## 4. Grad-CAM on test images (fine-tuned model, seed 42)",
        f'fig_path = config.FIG_DIR / "final_{key}_gradcam.png"\n'
        f'if RERUN or not fig_path.exists():\n    from src.gradcam_figures import gradcam_figure\n    gradcam_figure("{key}")\n'
        'display(Image(str(fig_path), width=900))',
    ]))


def nb12():
    return nb([
        "md:# 12 · Final comparison, error analysis and protocol ablation\n**Owner:** shared\n\n"
        "All numbers below come from the saved runs of notebooks 08–11 (same split, same metrics, test split scored once "
        "per selected configuration).",
        FINAL_SETUP,
        'from src.analysis import build_all\nres = build_all()          # writes results/final/*.json|csv and the figures\n'
        'res["comparison"]',
        'res["comparison_variants"]',
        "md:## Paired significance (McNemar, seed 42, test split)",
        'res["mcnemar"]',
        "md:## ROC curves",
        'display(Image(str(config.FIG_DIR / "final_roc.png"), width=480))',
        "md:## Error analysis by tumor subtype (missed tumors) and by source of false alarms",
        'res["errors_by_subtype"]',
        'display(Image(str(config.FIG_DIR / "final_errors_gallery.png"), width=900))',
        "md:## Protocol ablation: how much of the interim score came from the shortcut and from near-duplicates?",
        'res["ablation"]',
        'res["leak_vs_clean"]',
    ])


def main():
    NB.mkdir(exist_ok=True)
    books = {
        "04_resnet50.ipynb": interim_cnn(
            "resnet50", "04 · ResNet50", "Pihu Pandey (230953134)",
            "ResNet50 (He et al., 2016): 50 layers organised in bottleneck residual blocks whose skip connections "
            "let gradients flow through a much deeper network than VGG16.", ["resnet50"], "resnet50"),
        "05_hybrid_vgg16_resnet50.ipynb": interim_cnn(
            "hybrid_vgg16_resnet50", "05 · Hybrid VGG16–ResNet50", "Kumari Udita (230953094)",
            "The same image goes through both frozen backbones; the pooled features are concatenated "
            "(512 + 2,048 = 2,560) and passed to the common head, so the classifier can use both feature sets.",
            ["vgg16", "resnet50"], "hybrid"),
        "06_model_comparison.ipynb": interim_comparison(),
        "07_crop_leakage_split.ipynb": nb07(),
        "08_final_mlp.ipynb": model_nb("08", "mlp", "MLP baseline", "shared by all members",
                                       "64 × 64 grayscale (from the cropped slice), flattened; Dense 512 → 128 → 2 with dropout."),
        "09_final_vgg16.ipynb": model_nb("09", "vgg16", "VGG16", "Mihika Agarwal (230911094)",
                                         "ImageNet VGG16 → global average pooling (512) → head."),
        "10_final_resnet50.ipynb": model_nb("10", "resnet50", "ResNet50", "Pihu Pandey (230953134)",
                                            "ImageNet ResNet50 → global average pooling (2,048) → head."),
        "11_final_hybrid.ipynb": model_nb("11", "hybrid", "Hybrid VGG16–ResNet50", "Kumari Udita (230953094)",
                                          "Both backbones on the same image → concatenated pooled features (2,560) → head."),
        "12_final_comparison.ipynb": nb12(),
    }
    for name, book in books.items():
        nbf.write(book, NB / name)
        print("wrote", name)


if __name__ == "__main__":
    main()
