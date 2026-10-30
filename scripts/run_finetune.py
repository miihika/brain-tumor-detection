"""Fine-tune the upper block(s) of VGG16 / ResNet50 / hybrid on the final protocol."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports

import os, sys, json; os.environ.setdefault("KERAS_BACKEND","torch")
import numpy as np, keras, torch
from src import experiments as E, config, training
from src import finetune as F
seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [42]
which = sys.argv[2].split(",") if len(sys.argv) > 2 else ["resnet50","vgg16","hybrid"]
m = E.manifest_for("final"); cw = E.class_weight(m)
chosen = json.loads((config.RESULTS_DIR/"final"/"chosen_heads.json").read_text())
for seed in seeds:
    for name in which:
        tag = f"final_{name}_finetuned_s{seed}"
        if (E.RUN_DIR/f"{tag}.json").exists(): print("skip", tag); continue
        keras.utils.set_random_seed(seed)
        names = ["vgg16","resnet50"] if name == "hybrid" else [name]
        cfg = chosen[name]
        head = F.build_head(sum(E.models.BACKBONES[n][2] for n in names), cfg["units"], cfg["dropout"])
        head.load_weights(config.DATA_DIR/"models"/f"final_{name}_frozen_s{seed}.weights.h5")
        model, uppers = F.build_finetune_model(names, head, trainable=True)
        tr = F.CachedViews(names, m, "train", class_weight=cw, train=True, seed=seed)
        va = F.CachedViews(names, m, "val"); te = F.CachedViews(names, m, "test")
        p0 = F.predict(model, va); print(tag, "start val", training.compute_metrics(va.y, p0)["balanced_accuracy"], flush=True)
        hist = F.finetune(model, tr, va)
        model.save_weights(config.DATA_DIR/"models"/f"{tag}.weights.h5")
        rec = {"model": name, "variant": "finetuned", "protocol": "final", "seed": seed,
               "config": {**cfg, "ft_lr": F.FT_LR, "ft_max_epochs": F.FT_MAX_EPOCHS, "ft_patience": F.FT_PATIENCE,
                          "unfrozen": {"vgg16": "block5 (3 conv)", "resnet50": "conv5 stage (3 blocks, BN frozen)"}},
               "epochs_run": len(hist.history["loss"]), "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
               "train_seconds": round(hist.train_seconds, 1),
               "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
               "history": {k: [round(float(v), 4) for v in vals] for k, vals in hist.history.items()}}
        for split, ds in [("val", va), ("test", te)]:
            prob = F.predict(model, ds); rec[f"{split}_metrics"] = training.compute_metrics(ds.y, prob)
            np.save(E.RUN_DIR/f"final_{name}_finetuned_s{seed}_{split}_prob.npy", prob)
        (E.RUN_DIR/f"{tag}.json").write_text(json.dumps(rec, indent=2))
        print(tag, "val", rec["val_metrics"]["balanced_accuracy"], "test", rec["test_metrics"]["balanced_accuracy"], rec["train_seconds"], flush=True)
print("FT DONE")
