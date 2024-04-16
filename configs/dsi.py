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
NUM_CLASSES = 5  # predicting 4 classes + background
LR = 1e-5
NUM_WORKERS = 8
EPOCHS = 30
IGNORE_INDEX = 0  # index in images to ignore for jaccard index
LOSS_FUNCTION = "JaccardLoss"  # JaccardLoss, DiceLoss, TverskyLoss, LovaszLoss
PATIENCE = 5
THRESHOLD = 0.01

# KaneCounty data
KC_SHAPE_FILENAME = "KC_StormwaterDataJan2024.gdb.zip"
KC_LAYER = 4
KC_LABEL_COL = "BasinType"
KC_COLORS = {
    0: (0, 0, 0, 0),
    1: (215, 80, 48, 255),
    2: (49, 102, 80, 255),
    3: (239, 169, 74, 255),
    4: (100, 107, 99, 255),
}
KC_LABELS = {
    "BACKGROUND": 0,
    "POND": 1,
    "WETLAND": 2,
    "DRY BOTTOM - TURF": 3,
    "DRY BOTTOM - MESIC PRAIRIE": 4,
}
KC_LABELS_INVERSE = {v: k for k, v in KC_LABELS.items()}

# all colors and labels
ALL_COLORS = {
    0: (0, 0, 0, 0),
    1: (215, 80, 48, 255),
    2: (49, 102, 80, 255),
    3: (239, 169, 74, 255),
    4: (100, 107, 99, 255),
    5: (89, 53, 31, 255),
    6: (2, 86, 105, 255),
    7: (207, 211, 205, 255),
    8: (195, 88, 49, 255),
    9: (144, 70, 132, 255),
    10: (29, 51, 74, 255),
    11: (71, 64, 46, 255),
    12: (114, 20, 34, 255),
    13: (37, 40, 80, 255),
    14: (94, 33, 41, 255),
    15: (255, 255, 255, 255),
}
ALL_LABELS = {
    0: "BACKGROUND",
    1: "POND",
    2: "WETLAND",
    3: "DRY BOTTOM - TURF",
    4: "DRY BOTTOM - MESIC PRAIRIE",
    5: "DEPRESSIONAL STORAGE",
    6: "DRY BOTTOM - WOODED",
    7: "POND - EXTENDED DRY",
    8: "PICP PARKING LOT",
    9: "DRY BOTTOM - GRAVEL",
    10: "UNDERGROUND",
    11: "UNDERGROUND VAULT",
    12: "PICP ALLEY",
    13: "INFILTRATION TRENCH",
    14: "BIORETENTION",
    15: "UNKNOWN",
}
