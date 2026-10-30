"""Grad-CAM (Selvaraju et al., 2017) for the fine-tuned CNNs.

For a model  maps -> upper block -> GAP -> [concat] -> head, we split the upper block
at its last convolutional layer (VGG16: block5_conv3, 14x14; ResNet50:
conv5_block3_out, 7x7), take the gradient of P(tumor) with respect to that layer's
activations, average it over space to get one weight per channel, and form the
ReLU of the weighted channel sum. The map is upsampled to 224x224 and overlaid on
the cropped slice. For the hybrid, one map per branch is produced.
"""
import cv2
import keras
import numpy as np

from .finetune import LAST_CONV


def split_upper(name, upper):
    """(maps -> last-conv activations) model, and the remaining tail as a function.
    The tail after the last conv layer has no weights: VGG16 = 2x2 max-pool (block5_pool)
    then global average pooling; ResNet50 = global average pooling."""
    conv_model = upper.conv_model                      # built in finetune.build_upper, shares weights

    def tail(a, training=False):
        if name == "vgg16":
            a = keras.ops.max_pool(a, pool_size=2, strides=2)
        return keras.ops.mean(a, axis=(1, 2))
    return conv_model, tail


def gradcam(names, uppers, head, maps_list):
    """names/uppers as in build_finetune_model; maps_list = one (n, 14, 14, C) array per branch.
    Returns P(tumor) and a list of CAMs (n, h, w), one per branch, each scaled to [0, 1]."""
    import torch
    parts = [split_upper(n, u) for n, u in zip(names, uppers)]
    acts = []
    with torch.no_grad():
        for (lower, _), maps in zip(parts, maps_list):
            acts.append(lower(np.asarray(maps, dtype=np.float32), training=False).detach())
    acts = [a.clone().requires_grad_(True) for a in acts]
    with torch.enable_grad():
        pooled = [tail(a, training=False) for (_, tail), a in zip(parts, acts)]
        x = torch.cat(pooled, dim=1) if len(pooled) > 1 else pooled[0]
        p = head(x, training=False)[:, 1]
        grads = torch.autograd.grad(p.sum(), acts)
    cams = []
    for a, g in zip(acts, grads):
        w = g.mean(dim=(1, 2), keepdim=True)                         # channel weights
        cam = torch.relu((w * a).sum(dim=-1)).detach().cpu().numpy()  # (n, h, w)
        cam /= cam.reshape(len(cam), -1).max(axis=1)[:, None, None] + 1e-8
        cams.append(cam)
    return p.detach().cpu().numpy(), cams


def overlay(gray224, cam, alpha=0.45):
    """Heat-map overlay (RGB uint8) of a [0, 1] CAM on a 224x224 grayscale image."""
    heat = cv2.resize(cam.astype(np.float32), gray224.shape[::-1], interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(heat, 0, 1)), cv2.COLORMAP_JET)[..., ::-1]
    base = np.repeat(gray224[..., None], 3, axis=2).astype(np.float32)
    return np.uint8((1 - alpha) * base + alpha * heat)
