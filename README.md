# Brain Tumor Detection Using Convolutional Neural Networks

ICT-4442 Deep Learning mini project · School of Computer Engineering, MIT Manipal (MAHE)

| Member | Reg. No. | Model owned |
| --- | --- | --- |
| Pihu Pandey | 230953134 | ResNet50 |
| Mihika Agarwal | 230911094 | VGG16 |
| Kumari Udita | 230953094 | Hybrid VGG16–ResNet50 |
| All members | – | MLP baseline, data pipeline, evaluation |

Binary classification of brain MRI slices into **tumor** vs **no tumor**, comparing four models
under one common protocol: an MLP baseline, VGG16, ResNet50, and a hybrid that fuses VGG16 and
ResNet50 features.

**Final phase (Phase 3).** The interim numbers (98.7% validation accuracy for every CNN) turned out to be
inflated by two dataset artefacts, so the final comparison changes the protocol: every slice is **cropped to the
head** (removes the file-size/border cue), and a **near-duplicate check** (perceptual hash + pixel correlation)
showed that 1,087 of the 1,584 official test images have a near-copy in the training folder, so all models are
compared on a **leak-free group split**. See notebooks 07–12 and the final report.

## Final results (test split, leak-free, cropped; mean ± std over 3 seeds)

| Model | Accuracy % | Precision % | Recall % | Specificity % | F1 % | Balanced acc. % | ROC-AUC | Missed tumours | False alarms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP baseline | 94.10 ± 1.21 | 96.59 ± 1.98 | 95.64 ± 0.80 | 89.20 ± 6.56 | 96.10 ± 0.75 | 92.42 ± 3.00 | 0.969 ± 0.022 | 53.7 ± 9.9 | 42.0 ± 25.5 |
| VGG16 (fine-tuned) | 97.41 ± 0.88 | 98.64 ± 0.32 | 97.94 ± 1.44 | 95.72 ± 1.07 | 98.28 ± 0.59 | 96.83 ± 0.34 | 0.997 ± 0.001 | 25.3 ± 17.8 | 16.7 ± 4.2 |
| ResNet50 (fine-tuned) | 98.21 ± 0.43 | 99.10 ± 0.15 | 98.54 ± 0.49 | 97.18 ± 0.44 | 98.82 ± 0.29 | 97.85 ± 0.40 | 0.997 ± 0.000 | 18.0 ± 6.1 | 11.0 ± 1.7 |
| Hybrid VGG16–ResNet50 (fine-tuned) | 98.03 ± 0.41 | 99.04 ± 0.23 | 98.35 ± 0.61 | 97.00 ± 0.74 | 98.69 ± 0.27 | 97.67 ± 0.34 | 0.998 ± 0.000 | 20.3 ± 7.5 | 11.7 ± 2.9 |

n = 1,620 test scans (1,231 tumour, 389 no tumour). Hyper-parameters chosen on validation only; the test split
was scored once per selected configuration. Fine-tuned = last block unfrozen (VGG16 block5, ResNet50 conv5,
both in the hybrid), chosen over the frozen-backbone variant on validation balanced accuracy.

**Shortcut / leakage checks** (details in `results/final/`): on the original split the frozen CNNs reach
99.4–99.9% test specificity because 390 of the 400 no-tumour test images have a near-copy in training; on the
leak-free split it is 95–97%. A grey-level-histogram probe with no spatial information still reaches 82.2%
balanced accuracy on cropped images, so part of the signal is the MRI sequence/source, not the tumour.

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
by Masoud Nickparvar (Kaggle, version 2, CC BY 4.0): 7,200 JPEG slices in four classes, which come
as a Training folder (1,400 per class) and a Testing folder (400 per class).

Pipeline (`src/data_pipeline.py`, `notebooks/01_data_preprocessing.ipynb`):

1. Download with `kagglehub`.
2. Verify every image (0 unreadable) and remove exact duplicates by MD5 hash (187 files, 153 groups,
   none across Training/Testing) → **7,013 images**.
3. Merge glioma + meningioma + pituitary → `tumor` (1); notumor → `no_tumor` (0).
4. Official Testing folder = **held-out test set**. Training folder → stratified 80/20 train/validation (seed 42).
5. Copy into `data/binary/{train,val,test}/{no_tumor,tumor}` for `ImageDataGenerator.flow_from_directory`.
   The file-level split is saved in `results/dataset/split_manifest.csv`.

| Split | no_tumor | tumor | total |
| --- | --- | --- | --- |
| train | 1,025 | 3,318 | 4,343 |
| val | 256 | 830 | 1,086 |
| test (held out) | 400 | 1,184 | 1,584 |

Class weights (from train): no_tumor 2.12, tumor 0.65.

## Models (`src/models.py`)

| Model | Input | Architecture |
| --- | --- | --- |
| MLP baseline | 64×64 grayscale, [0, 1], flattened | Dense 512 → Dropout 0.5 → Dense 128 → Dropout 0.3 → Dense 2 (softmax) |
| VGG16 | 224×224 RGB, ImageNet `preprocess_input` | frozen VGG16 → global average pooling (512) → head |
| ResNet50 | 224×224 RGB, ImageNet `preprocess_input` | frozen ResNet50 → global average pooling (2048) → head |
| Hybrid | 224×224 RGB, ImageNet `preprocess_input` | both frozen backbones on the same image → concatenate (2560) → head |

Head (shared): Dense 256 (ReLU) → Dropout 0.5 → Dense 2 (softmax). Training (`src/training.py`): Adam,
categorical cross-entropy, batch 32, ≤ 30 epochs, early stopping on validation loss (patience 5,
best weights restored), class weights, seed 42.

**Interim set-up.** The backbones are frozen, so each image's pooled features are computed once and
cached (`src/features.py`), and only the head is trained. This is equivalent to training the full
network with a frozen backbone and no augmentation. Each CNN notebook rebuilds the full end-to-end
model and checks that its predictions match. Augmentation (rotation ±15°, shifts 10%, zoom 10%,
horizontal flip; training split only) is implemented in `src/data_pipeline.py` for the
fine-tuning stage.

## Interim results (Phase 2: original split, uncropped; validation n = 1,086)

<!-- RESULTS -->
| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Missed tumors (FN) | False alarms (FP) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP baseline | 97.15% | 97.84% | 98.43% | 92.97% | 98.14% | 0.987 | 13 | 18 |
| VGG16 | 98.71% | 99.16% | 99.16% | 97.27% | 99.16% | 0.997 | 7 | 7 |
| ResNet50 | 98.71% | 98.80% | 99.52% | 96.09% | 99.16% | 0.999 | 4 | 10 |
| Hybrid VGG16–ResNet50 | 98.71% | 98.92% | 99.40% | 96.48% | 99.16% | 0.999 | 5 | 9 |
| *Image-size rule (no model)* | 97.15% | – | 96.63% | 98.83% | 98.10% | – | – | – |

**Caution: image-source shortcut.** 93.1% of tumor images are exactly 512×512 but only 1.0% of no-tumor images are (they come from a different source, Br35H). A rule that only checks the original file size (last row) scores almost as well as the models, so these scores overstate what the models know about tumors. The final phase fixes this (see the final results above).

## Final-phase pipeline (notebooks 07–12)

| Step | Code | What it does |
| --- | --- | --- |
| Brain crop | `src/cropping.py` | head bounding box → pad to square → 256×256 PNG (`data/cropped/`) |
| Near-duplicates + split | `src/near_duplicates.py` | pHash (8 flips/rotations) ≤ 8 bits, confirmed by 64×64 correlation ≥ 0.97; groups → `StratifiedGroupKFold` split (`results/dataset/split_manifest_final.csv`: 4,313 / 1,080 / 1,620) |
| Cached feature maps | `src/features.py` | frozen lower part of each backbone run once per image (+2 augmented views per training image), float16 memmaps |
| Fine-tuning | `src/finetune.py` | VGG16 block5 / ResNet50 conv5 (BN frozen) / both, Adam 1e-5, ≤ 12 epochs, patience 3 |
| Experiments | `src/experiments.py`, `scripts/run_*.py` | grid search on validation, 3 seeds, protocols P0/P1/P2 |
| Evaluation | `src/evaluation.py`, `src/analysis.py` | bootstrap CIs, McNemar, ROC, shortcut probes, error analysis |
| Grad-CAM | `src/gradcam.py`, `src/gradcam_figures.py` | maps at block5_conv3 / conv5_block3_out |

All saved runs (one JSON + test/val probabilities per model × variant × protocol × seed) are in
`results/final/runs/`; the notebooks load them (`RERUN = False`) because re-training takes hours on a CPU.
`scripts/run_all.sh` lists the full order.

## How to run

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook   # run 01 → 12 in order (07-12 load the saved runs; see scripts/run_all.sh to retrain)
```

| Notebook | What it does | Owner |
| --- | --- | --- |
| `01_data_preprocessing.ipynb` | download, clean, de-duplicate, split, figures | all |
| `02_mlp_baseline.ipynb` | MLP baseline | all |
| `03_vgg16.ipynb` | VGG16 | Mihika |
| `04_resnet50.ipynb` | ResNet50 | Pihu |
| `05_hybrid_vgg16_resnet50.ipynb` | Hybrid VGG16–ResNet50 | Udita |
| `06_model_comparison.ipynb` | interim comparison table + image-size sanity check | all |
| `07_crop_leakage_split.ipynb` | brain crop, near-duplicate check, leak-free split, shortcut probes | all |
| `08_final_mlp.ipynb` | MLP, final protocol | all |
| `09_final_vgg16.ipynb` | VGG16: grid, seeds, fine-tuning, Grad-CAM | Mihika |
| `10_final_resnet50.ipynb` | ResNet50: grid, seeds, fine-tuning, Grad-CAM | Pihu |
| `11_final_hybrid.ipynb` | Hybrid: grid, seeds, fine-tuning, Grad-CAM | Udita |
| `12_final_comparison.ipynb` | final comparison, McNemar, ROC, error analysis, protocol ablation | all |

The notebooks set `KERAS_BACKEND=torch` (the backend we used, on CPU). On Google Colab you can
use `tensorflow` instead. The code is plain Keras 3 and runs on either.

## Repository layout

```
src/            config, data pipeline, cropping, near-duplicates, models, cached features,
                training, fine-tuning, experiments, evaluation, analysis, Grad-CAM
scripts/        long-running experiment scripts + notebook generator
notebooks/      01–12, executed with outputs
results/        dataset/  split manifest + counts
                figures/  class distribution, samples, augmentation, curves, confusion matrices
                metrics/  interim: one JSON per model + comparison_val.csv
                final/    final-phase tables (csv/json) and runs/ (every run's metrics + probabilities)
data/           (not committed) merged image folders and cached features
```

## Credits

- Model design follows the comparison in S. Ganesh and A. Anoop, "Brain tumor detection using
  convolution neural networks," CSE-3175 report, MIT Manipal, 2025. No code was copied from it.
- VGG16: Simonyan & Zisserman, ICLR 2015. ResNet50: He et al., CVPR 2016. Pretrained weights from `keras.applications`.
- Grad-CAM: Selvaraju et al., ICCV 2017. Near-duplicate leakage in brain MRI: Yagis et al., Sci. Rep. 2021.
