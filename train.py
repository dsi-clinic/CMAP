"""
To run: from repo directory (2024-winter-cmap)
> python -m train configs.<config> [--experiment_name <name>]
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

# import albumentations as A
import kornia.augmentation as K

# import numpy as np
# from albumentations.pytorch import ToTensorV2
import torch
from kornia.augmentation.container import AugmentationSequential
from torch.nn.modules import Module
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchgeo.datasets import NAIP, BoundingBox, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler
from torchmetrics import Metric
from torchmetrics.classification import MulticlassJaccardIndex

from data.kc import KaneCounty

# project imports
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
args = parser.parse_args()
spec = importlib.util.spec_from_file_location(args.config)
config = importlib.import_module(args.config)

# if no experiment name provided, set to timestamp
exp_name = args.experiment_name
if exp_name is None:
    exp_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

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

# train/test split
roi = dataset.bounds
midx = roi.minx + (roi.maxx - roi.minx) / 2
midy = roi.miny + (roi.maxy - roi.miny) / 2

# random batch sampler for training, grid sampler for testing
train_roi = BoundingBox(roi.minx, midx, roi.miny, roi.maxy, roi.mint, roi.maxt)
train_sampler = RandomBatchGeoSampler(
    dataset=dataset,
    size=config.PATCH_SIZE,
    batch_size=config.BATCH_SIZE,
    roi=train_roi,
)
test_roi = BoundingBox(midx, roi.maxx, roi.miny, roi.maxy, roi.mint, roi.maxt)
test_sampler = GridGeoSampler(
    dataset, size=config.PATCH_SIZE, stride=config.PATCH_SIZE, roi=test_roi
)

# create dataloaders (must use batch_sampler)
train_dataloader = DataLoader(
    dataset,
    batch_sampler=train_sampler,
    collate_fn=stack_samples,
    num_workers=config.NUM_WORKERS,
)
test_dataloader = DataLoader(
    dataset,
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
loss_fn_class = getattr(
    importlib.import_module("segmentation_models_pytorch.losses"),
    config.LOSS_FUNCTION,
)
# Initialize the loss function with the required parameters
loss_fn = loss_fn_class(mode="multiclass")

# IoU metric
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

# def get_train_augmentation_pipeline():
#     """
#     Extend the augmentation pipeline for aerial image segmentation with additional
#     transformations to simulate various environmental conditions and viewing angles.

#     Returns:
#         A callable augmentation pipeline that applies the defined transformations.
#     """
#     # Define the extended augmentation pipeline
#     augmentation_pipeline = A.Compose(
#         [
#             # Rotate the image by up to 180 degrees, without a preferred direction
#             A.Rotate(limit=180, p=0.5, border_mode=0),
#             # Horizontal and vertical flipping
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             # Random scaling
#             A.RandomScale(scale_limit=0.1, p=0.5),
#             # Brightness and contrast adjustments
#             A.RandomBrightnessContrast(
#                 brightness_limit=0.2, contrast_limit=0.2, p=0.5
#             ),
#             # Slight Gaussian blur to mimic atmospheric effects
#             A.GaussianBlur(blur_limit=(3, 3), p=0.2),
#             # Normalize the image (ensure to adjust mean and std as per your dataset)
#             A.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0
#             ),
#             # Convert image and mask to PyTorch tensors
#             ToTensorV2(),
#         ],
#         additional_targets={"mask": "image"},
#     )  # Apply the same transform to both image and mask

#     return augmentation_pipeline


# def get_test_augmentation_pipeline():
#     """
#     Extend the augmentation pipeline for aerial image segmentation with additional
#     transformations to simulate various environmental conditions and viewing angles.

#     Returns:
#         A callable augmentation pipeline that applies the defined transformations.
#     """
#     # Define the extended augmentation pipeline
#     augmentation_pipeline = A.Compose(
#         [
#             # Normalize the image (ensure to adjust mean and std as per your dataset)
#             A.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0
#             ),
#             # Convert image and mask to PyTorch tensors
#             ToTensorV2(),
#         ],
#         additional_targets={"mask": "image"},
#     )  # Apply the same transform to both image and mask

#     return augmentation_pipeline


# train_augmentation_pipeline = get_train_augmentation_pipeline()
# test_augmentation_pipeline = get_test_augmentation_pipeline()

mean = torch.tensor(0.0)
std = torch.tensor(255.0)
normalize = K.Normalize(mean=mean, std=std)
denormalize = K.Denormalize(mean=mean, std=std)
aug = AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    # K.RandomPlasmaShadow(
    # roughness=(0.1, 0.7),
    # shade_intensity=(-1.0, 0.0),
    # shade_quantity=(0.0, 1.0),
    # keepdim=True,
    # ),
    # K.RandomGaussianBlur(
    # kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.25
    # ),
    K.RandomRotation(degrees=360, align_corners=True),
    data_keys=["image", "mask"],
    keepdim=True,
)


def add_extra_channel(image_tensor, source_channel=0):
    # Assuming 'image_tensor' is a PyTorch tensor with shape
    # (batch_size, num_channels, height, width)

    # Select the source channel to duplicate
    original_channel = image_tensor[:, source_channel:source_channel + 1, :, :]

    # Generate copy of selected channel
    extra_channel = original_channel.clone()

    # Concatenate the extra channel to the original image along the second
    # dimension (channel dimension)
    augmented_tensor = torch.cat((image_tensor, extra_channel), dim=1)

    return augmented_tensor

def train_setup(
    sample: DefaultDict[str, Any], epoch: int, batch: int
) -> Tuple[torch.Tensor]:
    
    samp_image = sample["image"]
    samp_mask = sample["mask"]
    # add an extra channel to the images and masks
    if samp_image.size(1) != model.in_channels:
        for i in range(model.in_channels - samp_image.size(1)):
            samp_image = add_extra_channel(samp_image)
            samp_mask = add_extra_channel(samp_mask)

    # send img and mask to device; convert y to float tensor for augmentation
    X = samp_image.to(device)
    y = samp_mask.type(torch.float32).to(device)

    # normalize both img and mask to range of [0, 1] (req'd for augmentations)
    X, y = normalize(X), normalize(y)

    # augment img and mask with same augmentations
    X, y = aug(X, y)

    # denormalize mask to reset to index tensor (req'd for loss func)
    y = denormalize(y).type(torch.int64)

    # remove channel dim from y (req'd for loss func)
    y_squeezed = (y[:, 0, :, :].squeeze())

    # # plot first batch
    # if batch == 0:
    #     save_dir = os.path.join(train_images_root, f"epoch-{epoch}")
    #     os.mkdir(save_dir)
    #     for i in range(config.BATCH_SIZE):
    #         plot_tensors = {
    #             "image": sample["image"][i],
    #             "mask": sample["mask"][i],
    #             "augmented_image": denormalize(X)[i].cpu(),
    #             "augmented_mask": y[i].cpu(),
    #         }
    #         sample_fname = os.path.join(
    #             save_dir, f"train_sample-{epoch}.{i}.png"
    #         )
    #         plot_from_tensors(plot_tensors, sample_fname, "grid")

    return X, y_squeezed


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
        # images = sample["image"]
        # masks = sample["mask"]

        # # Convert PyTorch tensors to numpy arrays for Albumentations
        # images_np = images.cpu().numpy().astype(np.float32)
        # masks_np = masks.cpu().numpy().astype(np.float32)

        # # Apply augmentations
        # augmented = train_augmentation_pipeline(image=images_np, mask=masks_np)
        # augmented_image, augmented_mask = augmented["image"], augmented["mask"]

        # # Convert numpy arrays back to PyTorch tensors
        # X = torch.from_numpy(augmented_image).to(device)
        # y = torch.from_numpy(augmented_mask).to(device)

        # # Assuming your mask has a channel dimension that needs to be squeezed
        # y_squeezed = y.squeeze(1)

        X, y = train_setup(sample, epoch, batch)

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
            # images = sample["image"]
            # masks = sample["mask"]

            # # Convert PyTorch tensors to numpy arrays for Albumentations
            # images_np = images.cpu().numpy().astype(np.float32)
            # masks_np = masks.cpu().numpy().astype(np.float32)

            # # Apply augmentations
            # augmented = train_augmentation_pipeline(
            #     image=images_np, mask=masks_np
            # )
            # augmented_image, augmented_mask = (
            #     augmented["image"],
            #     augmented["mask"],
            # )

            # # Convert numpy arrays back to PyTorch tensors
            # X = torch.from_numpy(augmented_image).to(device)
            # y = torch.from_numpy(augmented_mask).to(device)

            # # Assuming your mask has a channel dimension that needs to be squeezed
            # y_squeezed = y.squeeze(1)

            samp_image = sample["image"]
            samp_mask = sample["mask"]
            # add an extra channel to the images and masks
            if samp_image.size(1) != model.in_channels:
                for i in range(model.in_channels - samp_image.size(1)):
                    samp_image = add_extra_channel(samp_image)
                    samp_mask = add_extra_channel(samp_mask)
            X = (samp_image.to(device))
            X = normalize(X)
            y = (samp_mask.to(device))
            y_squeezed = (y[:, 0, :, :].squeeze())

            # compute prediction error
            outputs = model(X)
            loss = loss_fn(outputs, y_squeezed)

            # update metric
            preds = outputs.argmax(dim=1)
            jaccard.update(preds, y_squeezed)

            # add test loss to rolling total
            test_loss += loss.item()

            # # plot first batch
            # if batch == 0:
            #     save_dir = os.path.join(test_image_root, f"epoch-{epoch}")
            #     os.mkdir(save_dir)
            #     for i in range(config.BATCH_SIZE):
            #         plot_tensors = {
            #             "image": sample["image"][i],
            #             "ground_truth": sample["mask"][i],
            #             "inference": preds[i].cpu(),
            #         }
            #         sample_fname = os.path.join(
            #             save_dir, f"test_sample-{epoch}.{i}.png"
            #         )
            #         plot_from_tensors(plot_tensors, sample_fname, "row")
    test_loss /= num_batches
    final_jaccard = jaccard.compute()
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("IoU/test", final_jaccard, epoch)
    logging.info(
        f"Test Error: \n Jaccard index: {final_jaccard:>7f}, "
        + f"Avg loss: {test_loss:>7f} \n"
    )


for t in range(config.EPOCHS):
    logging.info(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, train_jaccard, optimizer, t + 1)
    test(test_dataloader, model, loss_fn, test_jaccard, t + 1)
logging.info("Done!")
writer.close()


torch.save(model.state_dict(), os.path.join(out_root, "model.pth"))
logging.info(f"Saved PyTorch Model State to {out_root}")
