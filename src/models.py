"""The four models compared in the project.

  1. MLP baseline         - fully connected network on flattened 64x64 grayscale pixels
  2. VGG16                - ImageNet-pretrained VGG16 (frozen) + classifier head
  3. ResNet50             - ImageNet-pretrained ResNet50 (frozen) + the same head
  4. Hybrid VGG16-ResNet50 - both backbones on the same image, pooled features
                             concatenated, then the same head

All models end in Dense(2, softmax) and are trained with categorical cross-entropy,
so the output layer and loss are identical across the comparison.

Interim stage (frozen backbones): a frozen backbone gives the same features for an
image every epoch, so we compute them once (src/features.py) and train only the head
on the cached features. assemble_cnn() wires backbone(s) + head into the full
end-to-end Keras model, which is what gets fine-tuned in the final stage.
"""
import keras
from keras import layers
from keras.applications import ResNet50, VGG16, resnet50, vgg16

from . import config

BACKBONES = {
    # name: (constructor, preprocess_input, pooled feature size)
    "vgg16": (VGG16, vgg16.preprocess_input, 512),
    "resnet50": (ResNet50, resnet50.preprocess_input, 2048),
}


def build_mlp(input_shape=(*config.MLP_IMG_SIZE, 1)):
    inputs = keras.Input(shape=input_shape, name="mri_64x64_gray")
    x = layers.Flatten()(inputs)                       # 64*64 = 4096 pixel features
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="mlp_baseline")


def build_backbone(name):
    """Pretrained convolutional base without its ImageNet classifier.
    pooling='avg' = global average pooling of the last feature maps."""
    constructor, _, _ = BACKBONES[name]
    base = constructor(weights="imagenet", include_top=False, pooling="avg",
                       input_shape=(*config.CNN_IMG_SIZE, 3), name=f"{name}_base")
    base.trainable = False                             # frozen at the interim stage
    return base


def build_head(input_dim, name):
    """Classifier head shared by VGG16, ResNet50 and the hybrid: Dense -> Dropout -> softmax."""
    inputs = keras.Input(shape=(input_dim,), name="pooled_features")
    x = layers.Dense(config.DENSE_UNITS, activation="relu")(inputs)
    x = layers.Dropout(config.DROPOUT)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    return keras.Model(inputs, outputs, name=name)


def head_input_dim(backbone_names):
    return sum(BACKBONES[n][2] for n in backbone_names)


def assemble_cnn(backbone_names, head):
    """Full end-to-end model: image -> backbone(s) -> [concatenate] -> head.

    ["vgg16"] or ["resnet50"] gives the single-backbone models; ["vgg16", "resnet50"]
    gives the hybrid. Input is a 224x224x3 image already passed through
    preprocess_input (VGG16 and ResNet50 in Keras use the same 'caffe' ImageNet
    normalisation, so one preprocessed image feeds both backbones).
    """
    inputs = keras.Input(shape=(*config.CNN_IMG_SIZE, 3), name="mri_224x224")
    pooled = [build_backbone(n)(inputs) for n in backbone_names]
    x = layers.Concatenate(name="feature_fusion")(pooled) if len(pooled) > 1 else pooled[0]
    outputs = head(x)
    return keras.Model(inputs, outputs, name="_".join(backbone_names) + "_full")
