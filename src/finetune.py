"""Final phase: fine-tuning the last block of VGG16 / ResNet50 (and both, in the hybrid).

Model = frozen lower part  ->  trainable upper block  ->  GAP  ->  [concat]  ->  head
          (cached, see features.extract_intermediate)

  VGG16    upper = block5_conv1..3 + block5_pool      (7.08 M weights, all trainable)
  ResNet50 upper = conv5_block1..3                     (14.98 M weights; BatchNorm layers kept
                                                         frozen in inference mode, the usual
                                                         practice when fine-tuning on small data)
  Hybrid   both uppers on the same image, pooled features concatenated (512 + 2048 = 2560)

Training: two stages (gradual unfreezing). Stage 1 = the frozen-backbone head from
05/06 (trained on pooled features). Stage 2 = copy that head on top of the trainable
upper block(s) and train everything with Adam at a low learning rate (1e-5), so the
pretrained filters are adjusted, not overwritten. Training-time augmentation: each
epoch, every training image is shown as one of 3 cached views (original or one of two
random augmentations: rotation +-15 deg, shifts 10%, zoom 10%, horizontal flip).
"""
import time

import keras
import numpy as np
from keras import layers

from . import config
from .features import INTER_SHAPE, SPLIT_POINT, N_AUG_VIEWS, open_intermediate
from .models import BACKBONES

LAST_CONV = {"vgg16": "block5_conv3", "resnet50": "conv5_block3_out"}
FT_LR = 1e-5
FT_MAX_EPOCHS = 12
FT_PATIENCE = 3


def build_upper(name, trainable=True):
    constructor, _, _ = BACKBONES[name]
    base = constructor(weights="imagenet", include_top=False, pooling="avg",
                       input_shape=(*config.CNN_IMG_SIZE, 3))
    upper = keras.Model(base.get_layer(SPLIT_POINT[name]).output, base.output, name=f"{name}_upper")
    # same layers (shared weights), stopping at the last conv layer -- used by Grad-CAM
    upper.conv_model = keras.Model(base.get_layer(SPLIT_POINT[name]).output,
                                   base.get_layer(LAST_CONV[name]).output, name=f"{name}_upper_conv")
    upper.trainable = trainable
    if trainable:
        for layer in upper.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False          # keep BN statistics fixed (inference mode)
    return upper


def build_head(input_dim, units=config.DENSE_UNITS, dropout=config.DROPOUT, name="head"):
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(units, activation="relu")(inputs)
    x = layers.Dropout(dropout)(x)
    return keras.Model(inputs, layers.Dense(2, activation="softmax")(x), name=name)


def build_finetune_model(names, head, trainable=True):
    """names = ['vgg16'] | ['resnet50'] | ['vgg16', 'resnet50']; inputs are the cached maps."""
    inputs = [keras.Input(shape=INTER_SHAPE[n], name=f"{n}_maps") for n in names]
    uppers = [build_upper(n, trainable) for n in names]
    pooled = [u(x) for u, x in zip(uppers, inputs)]
    x = layers.Concatenate(name="feature_fusion")(pooled) if len(pooled) > 1 else pooled[0]
    model = keras.Model(inputs if len(inputs) > 1 else inputs[0], head(x), name="_".join(names) + "_ft")
    return model, uppers


class CachedViews(keras.utils.PyDataset):
    """Batches of cached feature maps. For training, each epoch draws one of the
    (1 + N_AUG_VIEWS) cached views per image; for evaluation only the original view."""

    def __init__(self, names, manifest, split, class_weight=None, train=False, seed=0, batch_size=config.BATCH_SIZE):
        super().__init__(workers=1, use_multiprocessing=False)
        n_all = len(manifest)
        self.names, self.train, self.bs = names, train, batch_size
        self.rows = np.flatnonzero((manifest["split"] == split).values)            # rows in the orig memmap
        self.y = (manifest["label"].values[self.rows] == "tumor").astype(int)
        self.orig = {n: open_intermediate(n, "orig", n_all) for n in names}
        self.aug = {}
        if train:
            n_train = int((manifest["split"] == "train").sum())
            self.aug = {n: [open_intermediate(n, f"aug{v}", n_train) for v in range(N_AUG_VIEWS)] for n in names}
        self.cw = class_weight
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.rows) / self.bs))

    def on_epoch_end(self):
        n = len(self.rows)
        if self.train:
            self.order = self.rng.permutation(n)
            self.view = self.rng.integers(0, 1 + N_AUG_VIEWS, size=n)               # 0 = original
        else:
            self.order, self.view = np.arange(n), np.zeros(n, dtype=int)

    def __getitem__(self, i):
        pos = np.sort(self.order[i * self.bs:(i + 1) * self.bs])                     # sorted = faster memmap reads
        xs = []
        for n in self.names:
            x = np.empty((len(pos), *INTER_SHAPE[n]), dtype=np.float32)
            for k, p in enumerate(pos):
                v = self.view[p]
                x[k] = self.orig[n][self.rows[p]] if v == 0 else self.aug[n][v - 1][p]
            xs.append(x)
        y = keras.utils.to_categorical(self.y[pos], 2)
        x = tuple(xs) if len(xs) > 1 else xs[0]
        if self.cw is None:
            return x, y
        return x, y, np.array([self.cw[int(t)] for t in self.y[pos]], dtype=np.float32)


def pooled_features(name, manifest, batch=64):
    """Frozen (ImageNet) pooled features for every image = upper block run on the cache.
    Identical to running the full frozen backbone with global average pooling."""
    path = config.FEATURE_DIR / f"{name}_pooled_cropped.npy"
    if path.exists():
        return np.load(path)
    upper = build_upper(name, trainable=False)
    maps = open_intermediate(name, "orig", len(manifest))
    out = np.concatenate([keras.ops.convert_to_numpy(upper(np.asarray(maps[s:s + batch], dtype=np.float32),
                                                           training=False))
                          for s in range(0, len(manifest), batch)])
    np.save(path, out)
    return out


def finetune(model, train_ds, val_ds, lr=FT_LR, epochs=FT_MAX_EPOCHS, patience=FT_PATIENCE, verbose=2):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[stop], verbose=verbose)
    history.train_seconds = time.time() - start
    return history


def predict(model, ds):
    probs = [model.predict_on_batch(ds[i][0]) for i in range(len(ds))]
    return np.concatenate([keras.ops.convert_to_numpy(p) for p in probs])[:, 1]
