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

## Preliminary results (validation split, n = 1,086, tumor = positive class)

<!-- RESULTS -->
| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Missed tumors (FN) | False alarms (FP) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP baseline | 97.15% | 97.84% | 98.43% | 92.97% | 98.14% | 0.987 | 13 | 18 |
| VGG16 | 98.71% | 99.16% | 99.16% | 97.27% | 99.16% | 0.997 | 7 | 7 |
| ResNet50 | 98.71% | 98.80% | 99.52% | 96.09% | 99.16% | 0.999 | 4 | 10 |
| Hybrid VGG16–ResNet50 | 98.71% | 98.92% | 99.40% | 96.48% | 99.16% | 0.999 | 5 | 9 |
| *Image-size rule (no model)* | 97.15% | – | 96.63% | 98.83% | 98.10% | – | – | – |

**Caution: image-source shortcut.** 93.1% of tumor images are exactly 512×512 but only 1.0% of no-tumor images are (they come from a different source, Br35H). A rule that only checks the original file size (last row) scores almost as well as the models, so these scores overstate what the models know about tumors. The final phase crops each slice to the brain region and pads it to a square before resizing, then re-runs every model. The test set has not been used yet.

## How to run

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook   # run 01 → 06 in order
```

| Notebook | What it does | Owner |
| --- | --- | --- |
| `01_data_preprocessing.ipynb` | download, clean, de-duplicate, split, figures | all |
| `02_mlp_baseline.ipynb` | MLP baseline | all |
| `03_vgg16.ipynb` | VGG16 | Mihika |
| `04_resnet50.ipynb` | ResNet50 | Pihu |
| `05_hybrid_vgg16_resnet50.ipynb` | Hybrid VGG16–ResNet50 | Udita |
| `06_model_comparison.ipynb` | common comparison table + image-size sanity check | all |

The notebooks set `KERAS_BACKEND=torch` (the backend we used, on CPU). On Google Colab you can
use `tensorflow` instead. The code is plain Keras 3 and runs on either.

## Repository layout

```
src/            config, data pipeline, models, cached features, training/evaluation
notebooks/      01–06, executed with outputs
results/        dataset/  split manifest + counts
                figures/  class distribution, samples, augmentation, curves, confusion matrices
                metrics/  one JSON per model + comparison_val.csv
data/           (not committed) merged image folders and cached features
```

## Credits

- Model design follows the comparison in S. Ganesh and A. Anoop, "Brain tumor detection using
  convolution neural networks," CSE-3175 report, MIT Manipal, 2025. No code was copied from it.
- VGG16: Simonyan & Zisserman, ICLR 2015. ResNet50: He et al., CVPR 2016. Pretrained weights from `keras.applications`.
- AI assistance: Claude (Anthropic) was used to help write and debug this code and to run the
  preliminary experiments (declared in the interim report, as the course guidelines require).
