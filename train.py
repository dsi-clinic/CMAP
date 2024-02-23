"""
To run: from repo directory (2024-winter-cmap)
> python train.py configs.<config> [--experiment_name <name>]
    [--aug_type <aug>] [--split <split>]
"""

import argparse
import datetime
import importlib.util
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, DefaultDict, Tuple

import kornia.augmentation as K
import torch
from kornia.augmentation.container import AugmentationSequential
from segmentation_models_pytorch.losses import JaccardLoss
from torch.nn.modules import Module
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchgeo.datasets import NAIP, random_bbox_splitting, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler
from torchmetrics import Metric
from torchmetrics.classification import MulticlassJaccardIndex

from data.kc import KaneCounty
from utils.model import SegmentationModel
from utils.plot import plot_from_tensors

# import config and experiment name from runtime args
parser = argparse.ArgumentParser(
    description="Train a segmentation model to predict stormwater storage "
    + "and green infrastructure."
)
parser.add_argument("config", type=str, help="Path to the configuration file")
parser.add_argument(
    "--experiment_name",
    type=str,
    help="Name of experiment",
    default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)

# Current potential aug_type args: "all", "default", "plasma", "gauss"
aug_types = "'all', 'default', 'plasma', 'gauss'"
parser.add_argument(
    "--aug_type",
    type=str,
    help="Type of augmentation; potential inputs are: {aug_types}",
    default="default",
)

parser.add_argument(
    "--split",
    type=str,
    help="Ratio of split; enter the size of the train split as an int out of 100",
    default="80",
)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location(args.config)
config = importlib.import_module(args.config)

# if no experiment name provided, set to timestamp
exp_name = args.experiment_name
if exp_name is None:
    exp_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
aug_type = args.aug_type
if aug_type is None:
    aug_type = "default"
split = float(int(args.split) / 100)
if split is None:
    split = 0.80

# set output path and exit run if path already exists
out_root = os.path.join(config.OUTPUT_ROOT, exp_name)
os.makedirs(out_root, exist_ok=False)

# create directory for output images
train_images_root = os.path.join(out_root, "train-images")
test_image_root = os.path.join(out_root, "test-images")
os.mkdir(train_images_root)
os.mkdir(test_image_root)

# open tensorboard writer
writer = SummaryWriter(out_root)

# copy training script and config to output directory
shutil.copy(Path(__file__).resolve(), out_root)
shutil.copy(Path(config.__file__).resolve(), out_root)

# Set up logging
log_filename = os.path.join(out_root, "training_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout),
    ],
)

# build dataset
naip = NAIP(config.KC_IMAGE_ROOT)
kc = KaneCounty(config.KC_MASK_ROOT)
dataset = naip & kc

train_dataset, test_dataset = random_bbox_splitting(dataset, [split, 1 - split])

train_sampler = RandomBatchGeoSampler(
    dataset=train_dataset,
    size=config.PATCH_SIZE,
    batch_size=config.BATCH_SIZE,
)
test_sampler = GridGeoSampler(
    dataset=test_dataset, size=config.PATCH_SIZE, stride=config.PATCH_SIZE
)

# create dataloaders (must use batch_sampler)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_sampler=train_sampler,
    collate_fn=stack_samples,
    num_workers=config.NUM_WORKERS,
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=config.BATCH_SIZE,
    sampler=test_sampler,
    collate_fn=stack_samples,
    num_workers=config.NUM_WORKERS,
)

# get device for training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logging.info(f"Using {device} device")

# create the model
model = SegmentationModel(
    model=config.MODEL,
    backbone=config.BACKBONE,
    num_classes=config.NUM_CLASSES,
    weights=config.WEIGHTS,
).model.to(device)
logging.info(model)

# set the loss function, metrics, and optimizer
loss_fn = JaccardLoss(mode="multiclass", classes=config.NUM_CLASSES)
train_jaccard = MulticlassJaccardIndex(
    num_classes=config.NUM_CLASSES,
    ignore_index=config.IGNORE_INDEX,
    average="micro",
).to(device)
test_jaccard = MulticlassJaccardIndex(
    num_classes=config.NUM_CLASSES,
    ignore_index=config.IGNORE_INDEX,
    average="micro",
).to(device)
optimizer = AdamW(model.parameters(), lr=config.LR)

# Various augmentation definitions
default_aug = AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomRotation(degrees=360, align_corners=True),
    data_keys=["image", "mask"],
    keepdim=True,
)
plasma_aug = AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomPlasmaShadow(
        roughness=(0.1, 0.7),
        shade_intensity=(-1.0, 0.0),
        shade_quantity=(0.0, 1.0),
        keepdim=True,
    ),
    K.RandomRotation(degrees=360, align_corners=True),
    data_keys=["image", "mask"],
    keepdim=True,
)
gauss_aug = AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.25),
    K.RandomRotation(degrees=360, align_corners=True),
    data_keys=["image", "mask"],
    keepdim=True,
)
all_aug = AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomPlasmaShadow(
        roughness=(0.1, 0.7),
        shade_intensity=(-1.0, 0.0),
        shade_quantity=(0.0, 1.0),
        keepdim=True,
    ),
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.25),
    K.RandomRotation(degrees=360, align_corners=True),
    data_keys=["image", "mask"],
    keepdim=True,
)

# Choose the proper augmentation format
if aug_type == "plasma":
    aug = plasma_aug
elif aug_type == "gauss":
    aug = gauss_aug
elif aug_type == "all":
    aug = all_aug
else:
    aug = default_aug

mean = config.DATASET_MEAN
std = config.DATASET_STD
scale_mean = torch.tensor(0.0)
scale_std = torch.tensor(255.0)

normalize = K.Normalize(mean=mean, std=std)
scale = K.Normalize(mean=scale_mean, std=scale_std)


def train_setup(
    sample: DefaultDict[str, Any], epoch: int, batch: int
) -> Tuple[torch.Tensor]:
    # send img and mask to device; convert y to float tensor for augmentation
    X = sample["image"].to(device)
    y = sample["mask"].type(torch.float32).to(device)

    # scale both img and mask to range of [0, 1] (req'd for augmentations)
    X = scale(X)

    # augment img and mask with same augmentations
    X_aug, y_aug = aug(X, y)

    # denormalize mask to reset to index tensor (req'd for loss func)
    y = y_aug.type(torch.int64)

    # remove channel dim from y (req'd for loss func)
    y_squeezed = y[:, 0, :, :].squeeze()

    # plot first batch
    if batch == 0:
        save_dir = os.path.join(train_images_root, f"epoch-{epoch}")
        os.mkdir(save_dir)
        for i in range(config.BATCH_SIZE):
            plot_tensors = {
                "image": X[i].cpu(),
                "mask": sample["mask"][i],
                "augmented_image": X_aug[i].cpu(),
                "augmented_mask": y[i].cpu(),
            }
            sample_fname = os.path.join(
                save_dir, f"train_sample-{epoch}.{i}.png"
            )
            plot_from_tensors(
                plot_tensors,
                sample_fname,
                "grid",
                KaneCounty.colors,
                KaneCounty.labels,
                sample["bbox"][i],
            )

    return normalize(X_aug), y_squeezed


def train(
    dataloader: DataLoader,
    model: Module,
    loss_fn: Module,
    jaccard: Metric,
    optimizer: Optimizer,
    epoch: int,
):
    num_batches = len(dataloader)
    model.train()
    jaccard.reset()
    train_loss = 0
    for batch, sample in enumerate(dataloader):
        X, y = train_setup(sample, epoch, batch)
        # The following comments provide pseudocode to theoretically filter tiles
        # The problem is that X here is a batch, not a specific image, so it won't work
        # Ideally, we filter before sending the dataset to the dataloader.
        # total_pixels = X.size
        # label_count = torch.sum(X != 0)
        # percentage_cover = (label_count / total_pixels) * 100

        # Filter patches based on weight criteria
        # if percentage_cover <= 1:
        # Skip this sample if weight criteria is not met
        #    continue

        # compute prediction error
        outputs = model(X)
        loss = loss_fn(outputs, y)

        # update jaccard index
        preds = outputs.argmax(dim=1)
        jaccard.update(preds, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1)
            logging.info(f"loss: {loss:>7f}  [{current:>5d}/{num_batches:>5d}]")
    train_loss /= num_batches
    final_jaccard = jaccard.compute()

    # Need to rename scalars?
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("IoU/train", final_jaccard, epoch)
    logging.info(f"Jaccard Index: {final_jaccard}")


def test(
    dataloader: DataLoader,
    model: Module,
    loss_fn: Module,
    jaccard: Metric,
    epoch: int,
):
    num_batches = len(dataloader)
    model.eval()
    jaccard.reset()
    test_loss = 0
    with torch.no_grad():
        for batch, sample in enumerate(dataloader):
            X = sample["image"].to(device)
            X_scaled = scale(X)
            X = normalize(X_scaled)
            y = sample["mask"].to(device)
            y_squeezed = y[:, 0, :, :].squeeze()

            # compute prediction error
            outputs = model(X)
            loss = loss_fn(outputs, y_squeezed)

            # update metric
            preds = outputs.argmax(dim=1)
            jaccard.update(preds, y_squeezed)

            # add test loss to rolling total
            test_loss += loss.item()

            # plot first batch
            if batch == 0:
                save_dir = os.path.join(test_image_root, f"epoch-{epoch}")
                os.mkdir(save_dir)
                for i in range(config.BATCH_SIZE):
                    plot_tensors = {
                        "image": X_scaled[i].cpu(),
                        "ground_truth": sample["mask"][i],
                        "prediction": preds[i].cpu(),
                    }
                    sample_fname = os.path.join(
                        save_dir, f"test_sample-{epoch}.{i}.png"
                    )
                    plot_from_tensors(
                        plot_tensors,
                        sample_fname,
                        "row",
                        KaneCounty.colors,
                        KaneCounty.labels,
                        sample["bbox"][i],
                    )
    test_loss /= num_batches
    final_jaccard = jaccard.compute()
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("IoU/test", final_jaccard, epoch)
    logging.info(
        f"Test Error: \n Jaccard index: {final_jaccard:>7f}, "
        + f"Avg loss: {test_loss:>7f} \n"
    )

    # Now returns test_loss such that it can be compared against previous losses
    return test_loss


# How much the loss needs to drop to reset a plateau
threshold = 0.01

# How many epochs loss needs to plateau before terminating
patience = 5

# Beginning loss
best_loss = None

# How long it's been plateauing
plateau_count = 0

for t in range(config.EPOCHS):
    logging.info(f"Epoch {t + 1}\n-------------------------------")
    # train(train_dataloader, model, loss_fn, train_jaccard, optimizer, t + 1)
    test_loss = test(test_dataloader, model, loss_fn, test_jaccard, t + 1)

    # Checks for plateau
    if best_loss is None:
        best_loss = test_loss
    elif test_loss < best_loss - threshold:
        best_loss = test_loss
        plateau_count = 0
    # Potentially add another if clause to plateau check
    # such that if test_loss jumps up again, plateau resets?
    else:
        plateau_count += 1
        if plateau_count >= patience:
            logging.info(
                f"Loss Plateau: {t} epochs, reached patience of {patience}"
            )
            break
print("Done!")
writer.close()

torch.save(model.state_dict(), os.path.join(out_root, "model.pth"))
logging.info(f"Saved PyTorch Model State to {out_root}")
