"""Frozen-backbone heads (VGG16, ResNet50, hybrid) on the cropped protocols: grid search on the
final validation split, then 3 seeds per protocol; saves head weights for fine-tuning."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports

import os, sys; os.environ.setdefault("KERAS_BACKEND","torch")
import torch; torch.set_num_threads(int(os.environ.get("NT","1")))
from src import experiments as E, config
from src.evaluation import save_json
protocols = sys.argv[1:] or ["final","original_cropped"]
chosen = {}
mf = E.manifest_for("final")
(config.DATA_DIR/"models").mkdir(parents=True, exist_ok=True)
for name in ["vgg16","resnet50","hybrid"]:
    X = E.cnn_features(name, "final")
    cfg, grid = E.grid_search(name, X, mf, E.HEAD_GRID)
    grid.to_csv(config.RESULTS_DIR/"final"/f"grid_{name}.csv", index=False)
    chosen[name] = cfg; print(name, "chosen", cfg, flush=True); print(grid.to_string(), flush=True)
    for p in protocols:
        mp = E.manifest_for(p); Xp = E.cnn_features(name, p)
        recs, nets = E.run_seeds(name, "frozen", p, Xp, mp, cfg, keep_models=True)
        if p == "final":
            for s, net in nets.items():
                net.save_weights(config.DATA_DIR/"models"/f"final_{name}_frozen_s{s}.weights.h5")
        print(p, name, "val bal", E.summarise(recs,"val")["balanced_accuracy"], "test bal", E.summarise(recs,"test")["balanced_accuracy"], flush=True)
save_json(chosen, "chosen_heads.json")
print("HEADS DONE")
