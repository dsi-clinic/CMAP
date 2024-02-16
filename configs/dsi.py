import os

<<<<<<< HEAD
DATA_DIR = "/net/projects/cmap/data"
KC_SHAPE_DIR = os.path.join(DATA_DIR, "kane-county-data")
KC_IMAGE_DIR = os.path.join(DATA_DIR, "KC-images")
KC_MASK_DIR = os.path.join(DATA_DIR, "KC-masks/top-structures-masks")
=======
# data paths
DATA_ROOT = "/net/projects/cmap/data"
KC_SHAPE_ROOT = os.path.join(DATA_ROOT, "kane-county-data")
KC_IMAGE_ROOT = os.path.join(DATA_ROOT, "KC-images")
KC_MASK_ROOT = os.path.join(DATA_ROOT, "KC-masks/top-structures-masks")
OUTPUT_ROOT = "/net/projects/cmap/model-outputs"

# model selection
MODEL = "unet"
BACKBONE = "resnet50"
WEIGHTS = True

# model hyperparams
BATCH_SIZE = 32
PATCH_SIZE = 512
NUM_CLASSES = 5  # predicting 4 classes + background
LR = 1e-3
NUM_WORKERS = 8
EPOCHS = 11
IGNORE_INDEX = 0  # index in images to ignore for jaccard index
>>>>>>> 03641fb4f3bbba470cde955ad91cbfba656c5d0a
