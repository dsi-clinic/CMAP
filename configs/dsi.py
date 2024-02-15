import os

# data paths
DATA_ROOT = "/net/projects/cmap/data"
KC_SHAPE_ROOT = os.path.join(DATA_ROOT, "kane-county-data")
KC_IMAGE_ROOT = os.path.join(DATA_ROOT, "KC-images")
KC_MASK_ROOT = os.path.join(DATA_ROOT, "KC-masks/top-structures-masks")
OUTPUT_ROOT = "/net/projects/cmap/model-outputs"

# model selection
MODEL = None
BACKBONE = None
WEIGHTS = None

# model hyperparams
BATCH_SIZE = 16
PATCH_SIZE = 512
NUM_CLASSES = 5  # predicting 4 classes + background
LR = 1e-3
NUM_WORKERS = 8
EPOCHS = 5
IGNORE_INDEX = 0  # index in images to ignore for jaccard index
