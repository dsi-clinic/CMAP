import os

# data paths
DATA_ROOT = "/net/projects/cmap/data"
KC_SHAPE_ROOT = os.path.join(DATA_ROOT, "kane-county-data")
KC_IMAGE_ROOT = os.path.join(DATA_ROOT, "KC-images")
KC_RIVER_ROOT = os.path.join(DATA_ROOT, "KC-river-images")
KC_MASK_ROOT = os.path.join(DATA_ROOT, "KC-masks/separate-masks")
OUTPUT_ROOT = f"/net/projects/cmap/workspaces/{os.environ['USER']}"

# model selection
MODEL = "deeplabv3+"
BACKBONE = "resnet50"
# check backbone, mean, and std when setting weights
WEIGHTS = True

# model hyperparams
DATASET_MEAN = [
    0.3281668683529412,
    0.4208941459215686,
    0.4187784871764706,
    0.5470313711372549,
]
DATASET_STD = [
    0.030595504117647058,
    0.02581302749019608,
    0.025523325960784313,
    0.03643713776470588,
]
BATCH_SIZE = 16
PATCH_SIZE = 512
NUM_CLASSES = 5  # predicting 4 classes + background
LR = 1e-5
NUM_WORKERS = 8
EPOCHS = 30
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
    3,  # Gaussian Blur
    # 4,  # Plasma Brightness
    5,  # Saturation
    # 6,  # Channel Shuffle
    # 7,  # Gamma
]

# Augmentation
ROTATION_DEGREES = 360
COLOR_CONTRAST = 0.2
COLOR_BRIGHTNESS = 0.2
RESIZED_CROP_SIZE = (PATCH_SIZE, PATCH_SIZE)
GAUSSIAN_NOISE_PROB = 0.2
GAUSSIAN_NOISE_STD = 0.05
GAUSSIAN_BLUR_SIGMA = (0.1, 2.0)
PLASMA_ROUGHNESS = (0.0, 0.2)
PLASMA_BRIGHTESS = (0.1, 0.3)
SATURATION_LIMIT = 0.1
SHADOW_INTENSITY = (-0.05, 0.0)
SHADE_QUANTITY = (0.0, 0.05)
GAMMA = (0.8, 1.2)


SPATIAL_AUG_MODE = "all"  # all or random
COLOR_AUG_MODE = "random"  # all or random

# KaneCounty data
KC_SHAPE_FILENAME = "KC_StormwaterDataJan2024.gdb.zip"
KC_LAYER = 4
KC_LABEL_COL = "BasinType"
KC_LABELS = {
    "BACKGROUND": 0,
    "POND": 1,
    "WETLAND": 2,
    "DRY BOTTOM - TURF": 3,
    "DRY BOTTOM - MESIC PRAIRIE": 4,
}

# for wandb
WANDB_API = ""
