"""Brain-region cropping (final phase).

Why: in this dataset the no-tumor images mostly come from a different source
(Br35H) than the tumor images, and they differ in file size, field of view and
black border. Only 1.0% of no-tumor files are 512x512 against 93.1% of tumor
files, so a model can score well by recognising the *source* instead of the tumor.

What: every slice is cropped to the bounding box of the head, padded with black
to a square, and saved as a fixed 256x256 grayscale PNG. After this step every
image has the same size, the same framing (head touching the borders of the
square) and the same file format, so file-size and border cues are removed
before any model sees the data.

Steps per image:
  1. grayscale, 5x5 Gaussian blur
  2. threshold at 10% of the image's 99th-percentile intensity (robust to bright text/markers)
  3. morphological open + close to remove specks and fill gaps
  4. keep the largest connected component (the head) -> bounding box
  5. add a 2% margin, crop, pad to square (black), resize to 256x256 (area interpolation)
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import config

CROP_SIZE = 256


def head_bbox(gray):
    """(x0, y0, x1, y1) bounding box of the largest bright component; whole image if none."""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = max(10.0, 0.10 * float(np.percentile(blur, 99)))
    mask = (blur > thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return 0, 0, w, h
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh = stats[i, :4]
    if bw * bh < 0.05 * w * h:            # implausibly small -> keep the whole image
        return 0, 0, w, h
    m = int(round(0.02 * max(bw, bh)))
    return max(0, x - m), max(0, y - m), min(w, x + bw + m), min(h, y + bh + m)


def crop_to_square(gray, size=CROP_SIZE):
    x0, y0, x1, y1 = head_bbox(gray)
    crop = gray[y0:y1, x0:x1]
    ch, cw = crop.shape
    side = max(ch, cw)
    canvas = np.zeros((side, side), dtype=crop.dtype)
    oy, ox = (side - ch) // 2, (side - cw) // 2
    canvas[oy:oy + ch, ox:ox + cw] = crop
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA), (x0, y0, x1, y1)


def build_cropped_dataset(manifest_csv=config.SPLIT_MANIFEST, out_dir=config.CROPPED_DIR):
    """Writes data/cropped/{split}/{label}/<same file name>.png and returns a table of
    bounding boxes (used later to check what source cues survive the crop)."""
    m = pd.read_csv(manifest_csv)
    rows = []
    for r in m.itertuples():
        src = config.ROOT / r.binary_path.replace("\\", "/")
        gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        out, (x0, y0, x1, y1) = crop_to_square(gray)
        dst = Path(out_dir) / r.split / r.label / (Path(src).stem + ".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), out)
        rows.append(dict(binary_path=r.binary_path, cropped_path=str(dst.relative_to(config.ROOT)),
                         split=r.split, label=r.label, source_class=r.source_class,
                         width=r.width, height=r.height, bx0=x0, by0=y0, bx1=x1, by1=y1))
    return pd.DataFrame(rows)
