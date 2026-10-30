"""MLP baseline: grid search on the final validation split, then 3 seeds under each protocol.
Run from the repository root: python scripts/run_mlp.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, so `src` imports
import os; os.environ.setdefault("KERAS_BACKEND","torch")
import torch; torch.set_num_threads(1)
from src import experiments as E
from src.evaluation import save_json
m=E.manifest_for("final"); X=E.mlp_arrays("final"); E.mlp_arrays("original_uncropped")
cfg,grid=E.grid_search("mlp",X,m,E.MLP_GRID); print(grid.to_string()); print("chosen",cfg)
grid.to_csv(E.config.RESULTS_DIR/"final"/"grid_mlp.csv",index=False); save_json({"mlp":{k:(list(v) if isinstance(v,tuple) else v) for k,v in cfg.items()}},"chosen_mlp.json")
for p in E.PROTOCOLS:
    mp=E.manifest_for(p); Xp=E.mlp_arrays(p)
    recs=E.run_seeds("mlp","frozen" if False else "trained",p,Xp,mp,cfg)
    print(p, E.summarise(recs,"val")["balanced_accuracy"], E.summarise(recs,"test")["balanced_accuracy"])
print("MLP DONE")
