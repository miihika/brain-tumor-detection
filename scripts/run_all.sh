#!/usr/bin/env bash
# Final-phase pipeline, in order (CPU times on a 2-core machine in brackets).
# Notebook 01 must have been run first (dataset download + data/binary + split manifest).
set -e
export KERAS_BACKEND=torch
python -c "from src.cropping import build_cropped_dataset; from src import config; build_cropped_dataset().to_csv(config.DATASET_INFO_DIR/'crop_boxes.csv', index=False)"   # [1 min]
# near-duplicate check + leak-free split: notebook 07 with RERUN=True          [5 min]
python scripts/run_extract_features.py        # cached feature maps, incl. 2 augmented views  [~3 h]
python scripts/run_mlp.py                     # MLP grid + 3 seeds x 3 protocols            [~40 min]
python scripts/run_heads.py final original_cropped   # frozen CNN heads                     [~20 min]
python scripts/run_finetune.py 42,7,2026      # fine-tuning, 3 models x 3 seeds              [~6 h]
python scripts/run_interim_protocol.py        # P0 ablation (extracts uncropped features)   [~1.5 h]
# then run notebooks 04-12 to regenerate tables and figures
