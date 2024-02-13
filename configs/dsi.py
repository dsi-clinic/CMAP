import os

# data paths
DATA_ROOT = "/net/projects/cmap/data"
KC_SHAPE_ROOT = os.path.join(DATA_ROOT, "kane-county-data")
KC_IMAGE_ROOT = os.path.join(DATA_ROOT, "KC-images")
KC_MASK_ROOT = os.path.join(DATA_ROOT, "KC-masks/top-structures-masks")
MODEL_STATES_ROOT = "/net/projects/cmap/model-states"

# model hyperparams
BATCH_SIZE = 64
PATCH_SIZE = 256
NUM_CLASSES = 5  # predicting 4 classes + background - is this correct?
LR = 1e-3
NUM_WORKERS = 2
# index in images to ignore for jaccard index
IGNORE_INDEX = 0
