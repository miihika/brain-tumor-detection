"""P0 (interim protocol: original split, uncropped) for the frozen CNNs, using the chosen head configs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports

import os, json; os.environ.setdefault("KERAS_BACKEND","torch")
from src import experiments as E, config
chosen=json.loads((config.RESULTS_DIR/"final"/"chosen_heads.json").read_text())
for name in ["vgg16","resnet50","hybrid"]:
    m=E.manifest_for("original_uncropped"); X=E.cnn_features(name,"original_uncropped")
    recs=E.run_seeds(name,"frozen","original_uncropped",X,m,chosen[name])
    print(name, E.summarise(recs,"val")["accuracy"], E.summarise(recs,"test")["balanced_accuracy"], flush=True)
print("P0 DONE")
