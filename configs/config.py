"""Configuration parameters for model training and data processing

Contains paths to data directories, model architecture settings, hyperparameters,
and augmentation options used throughout training pipeline.
"""

import os
from pathlib import Path

# data paths
DATA_ROOT = "/net/projects/cmap/data"
KC_SHAPE_ROOT = str(Path(DATA_ROOT) / "kane-county-data")
KC_IMAGE_ROOT = str(Path(DATA_ROOT) / "KC-images")
KC_RIVER_ROOT = str(Path(DATA_ROOT) / "KC-river-images")
USE_NIR = False
# KC_DEM_ROOT = str(Path(KC_SHAPE_ROOT) / "KC_DEM_2017")
KC_DEM_ROOT = None
KC_MASK_ROOT = str(Path(DATA_ROOT) / "KC-masks/separate-masks")
OUTPUT_ROOT = str(Path("/net/projects/cmap/workspaces/") / f"{os.environ['USER']}")

# model selection
MODEL = "deeplabv3+"
BACKBONE = "resnet101"
# check backbone, mean, and std when setting weights
WEIGHTS = True

DROPOUT = 0.0

# model hyperparams
# mean/std of imagenet for pretrained model
DATASET_MEAN = [0.485, 0.456, 0.406]  # RGB only
DATASET_STD = [0.229, 0.224, 0.225]  # RGB only

# mean/std of NAIP data + DEM
# DATASET_MEAN = [
#     0.328,  # R
#     0.420,  # G
#     0.418,  # B
#     0.547,  # NIR (optional)
#     0.0,    # DEM (optional)
# ]
# DATASET_STD = [
#     0.30,  # R
#     0.25,  # G
#     0.25,  # B
#     0.36,  # NIR (optional)
#     1.0,   # DEM (optional)
# ]
BATCH_SIZE = 16
PATCH_SIZE = 512
NUM_CLASSES = 5  # predicting 4 classes + background
LEARNING_RATE = 1e-5
NUM_WORKERS = 8
EPOCHS = 4
IGNORE_INDEX = 0  # index in images to ignore for jaccard index
LOSS_FUNCTION = "JaccardLoss"  # JaccardLoss, DiceLoss, TverskyLoss, LovaszLoss
PATIENCE = 5
THRESHOLD = 0.01
WEIGHT_DECAY = 0
REGULARIZATION_TYPE = None
REGULARIZATION_WEIGHT = 1.0e-5
GRADIENT_CLIPPING = False
CLIP_VALUE = 1.0

# data augmentation
SPATIAL_AUG_INDICES = [
    0,  # HorizontalFlip
    1,  # VerticalFlip
    2,  # Rotate
    3,  # Affine
    4,  # Elastic
    5,  # Perspective
    6,  # ResizedCrop
]

# only applied to images-- not masks
IMAGE_AUG_INDICES = [
    0,  # Contrast
    1,  # Brightness
    2,  # Gaussian Noise
    3,  # Gaussian Blur0
    # 4,  # Plasma Brightness
    # 5,  # Saturation
    # 6,  # Channel Shuffle
    # 7,  # Gamma
]

# Augmentation
ROTATION_DEGREES = 360
COLOR_CONTRAST = 0.1
COLOR_BRIGHTNESS = 0.1
RESIZED_CROP_SIZE = (PATCH_SIZE, PATCH_SIZE)
GAUSSIAN_NOISE_PROB = 0.5  # tuned
GAUSSIAN_NOISE_STD = 0.05
GAUSSIAN_BLUR_SIGMA = (0.3, 0.4)
GAUSSIAN_BLUR_KERNEL = (7, 7)  # tuned
PLASMA_ROUGHNESS = (0.0, 0.2)
PLASMA_BRIGHTESS = (0.1, 0.3)
SATURATION_LIMIT = 0.3  # tuned
SHADOW_INTENSITY = (-0.05, 0.0)
SHADE_QUANTITY = (0.0, 0.05)
GAMMA = (0.8, 1.2)


SPATIAL_AUG_MODE = "all"  # all or random
COLOR_AUG_MODE = "all"  # all or random

# KaneCounty data
KC_SHAPE_FILENAME = "KC_StormwaterDataJan2024.gdb.zip"
KC_LAYER = 4
KC_LABELS = {
    "BACKGROUND": 0,
    "POND": 1,
    "WETLAND": 2,
    "DRY BOTTOM - TURF": 3,
    "DRY BOTTOM - MESIC PRAIRIE": 4,
}

# River data
RD_SHAPE_FILE = "Kane_Co_Open_Water_Layer.zip"
RD_LAYER = 1
RD_LABELS = {
    "BACKGROUND": 0,
    "STREAM/RIVER": 5,
}

USE_RIVERDATASET = True  # change to True if training w/ RiverDataset

# for wandb
WANDB_API = ""
