import os

# data paths
DATA_ROOT = "/net/projects/cmap/data"
KC_SHAPE_ROOT = os.path.join(DATA_ROOT, "kane-county-data")
KC_IMAGE_ROOT = os.path.join(DATA_ROOT, "KC-images")
KC_MASK_ROOT = os.path.join(DATA_ROOT, "KC-masks/top-structures-masks")
<<<<<<< HEAD
OUTPUT_ROOT = "/net/projects/cmap/tamami/weights"
=======
OUTPUT_ROOT = "/net/projects/cmap/tamami/loss_fn"
>>>>>>> loss_fn_testing

# model selection
MODEL = "unet"
BACKBONE = "resnet50"
EXTRA_CLASS = False
WEIGHTS = True

# model hyperparams
BATCH_SIZE = 32
PATCH_SIZE = 256
NUM_CLASSES = 5  # predicting 4 classes + background
LR = 1e-3
NUM_WORKERS = 2
EPOCHS = 1
IGNORE_INDEX = 0  # index in images to ignore for jaccard index
LOSS_FUNCTION = "JaccardLoss"  # JaccardLoss, DiceLoss, TverskyLoss, LovaszLoss
