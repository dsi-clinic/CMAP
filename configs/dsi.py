import os

# data paths
DATA_ROOT = "/net/projects/cmap/data"
KC_SHAPE_ROOT = os.path.join(DATA_ROOT, "kane-county-data")
KC_IMAGE_ROOT = os.path.join(DATA_ROOT, "KC-images")
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
CONTEXT_SIZE = 512  # randomize? [1/2 PATCH_SIZE, PATCH_SIZE]
NUM_CLASSES = 5  # predicting 4 classes + background
LR = 1e-5
NUM_WORKERS = 8
EPOCHS = 30
IGNORE_INDEX = 0  # index in images to ignore for jaccard index
LOSS_FUNCTION = "JaccardLoss"  # JaccardLoss, DiceLoss, TverskyLoss, LovaszLoss
PATIENCE = 5
THRESHOLD = 0.01

# data augmentation
SPATIAL_AUG_INDICES = [
    0,  # HorizontalFlip
    1,  # VerticalFlip
    2,  # Rotate
    3,  # Affine
    4,  # Elastic
    5,  # Perspective
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
AUG_PARAMS = {
    "rotation_degrees": 360,
    "contrast_limit": 0.2,
    "brightness_limit": 0.2,
    "gaussian_noise_prob": 0.2,
    "gaussian_noise_std": 0.05,
    "gaussian_blur_sigma": (0.1, 2.0),
    "plasma_roughness": (0.0, 0.2),
    "saturation_limit": 0.1,
    "shadow_intensity": (-0.05, 0.0),
    "shade_quantity": (0.0, 0.05),
    "gamma": (0.8, 1.2),
}
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
