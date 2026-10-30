"""Near-duplicate check across train / val / test (final phase).

Exact duplicates (same MD5) were removed in the interim stage. This module looks for
*near* duplicates: re-saved, resized, flipped or rotated copies of the same slice.
Each brain-cropped image gets a 64-bit perceptual hash (pHash) for each of its 8
flips/rotations; two images are near-duplicates if any pair of hashes differs in
<= MAX_BITS bits. Pairs that cross splits are reported, since they would let the
model see (a version of) a validation/test image during training.
"""
import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from . import config

MAX_BITS = 8


def dihedral(img):
    out = []
    for k in range(4):
        r = np.rot90(img, k)
        out += [r, np.fliplr(r)]
    return out


def phash_table(crop_df):
    hashes = []
    for p in crop_df["cropped_path"]:
        g = cv2.imread(str(config.ROOT / p), cv2.IMREAD_GRAYSCALE)
        hashes.append([imagehash.phash(Image.fromarray(np.ascontiguousarray(v))).hash.flatten()
                       for v in dihedral(g)])
    return np.array(hashes, dtype=bool)          # (n, 8, 64)


def find_pairs(H, max_bits=MAX_BITS, block=512):
    """All pairs (i < j) whose identity hash of i is within max_bits of any transform of j."""
    n = len(H)
    base = H[:, 0, :].astype(np.uint8)            # (n, 64)
    allv = H.astype(np.uint8)                     # (n, 8, 64)
    pairs = []
    for s in range(0, n, block):
        a = base[s:s + block]                     # (b, 64)
        # Hamming distance = popcount(xor); use dot products on {0,1}
        d = np.full((len(a), n), 64, dtype=np.int16)
        for t in range(8):
            v = allv[:, t, :]
            ham = (a.sum(1)[:, None] + v.sum(1)[None, :] - 2 * a.astype(np.int16) @ v.T.astype(np.int16))
            d = np.minimum(d, ham)
        ii, jj = np.nonzero(d <= max_bits)
        for i, j in zip(ii + s, jj):
            if i < j:
                pairs.append((int(i), int(j), int(d[i - s, j])))
    return pd.DataFrame(pairs, columns=["i", "j", "bits"])


def cross_split_report(crop_df, pairs):
    df = crop_df.reset_index(drop=True)
    p = pairs.copy()
    p["split_i"], p["split_j"] = df.loc[p.i, "split"].values, df.loc[p.j, "split"].values
    p["label_i"], p["label_j"] = df.loc[p.i, "label"].values, df.loc[p.j, "label"].values
    p["file_i"], p["file_j"] = df.loc[p.i, "cropped_path"].values, df.loc[p.j, "cropped_path"].values
    p["cross"] = p.split_i != p.split_j
    return p


# ------------------------------------------------------------------ groups + leak-free split
CORR_THRESHOLD = 0.97          # pixel correlation (64x64, best flip/rotation) that confirms a pHash match


def confirm_pairs(crop_df, pairs, size=64):
    """Adds 'corr': Pearson correlation of the two 64x64 crops under the best of the 8
    flips/rotations. pHash alone over-matches similar-looking slices; this confirms them."""
    small = [cv2.resize(cv2.imread(str(config.ROOT / f), cv2.IMREAD_GRAYSCALE), (size, size),
                        interpolation=cv2.INTER_AREA).astype(np.float32) for f in crop_df["cropped_path"]]

    def z(a):
        a = a - a.mean()
        return a / (np.linalg.norm(a) + 1e-6)

    Z = [[z(v) for v in dihedral(s)] for s in small]
    pairs = pairs.copy()
    pairs["corr"] = [max(float((Z[i][0] * Z[j][t]).sum()) for t in range(8))
                     for i, j in zip(pairs["i"], pairs["j"])]
    return pairs


def near_duplicate_groups(n, confirmed, threshold=CORR_THRESHOLD):
    """Connected components of the confirmed near-duplicate graph -> one group id per image."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    q = confirmed[confirmed["corr"] >= threshold]
    _, labels = connected_components(coo_matrix((np.ones(len(q)), (q["i"], q["j"])), shape=(n, n)),
                                     directed=False)
    return labels


def leak_free_split(crop_df, groups, seed=config.SEED, n_folds=13, test_folds=3, val_folds=2):
    """Group-aware split: every near-duplicate group goes entirely to one of train/val/test.
    StratifiedGroupKFold keeps the 4 source classes in proportion; 3/13 folds -> test (~23%),
    2/13 -> validation (~15%), the rest -> train (~62%), close to the original proportions."""
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.empty(len(crop_df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(crop_df, crop_df["source_class"], groups)):
        fold[idx] = k
    split = np.where(fold < test_folds, "test", np.where(fold < test_folds + val_folds, "val", "train"))
    out = crop_df.copy()
    out["group"], out["split_original"], out["split"] = groups, crop_df["split"], split
    return out
