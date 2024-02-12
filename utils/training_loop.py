import argparse
import importlib.util
import os

import albumentations as A
import numpy as np

# import kornia as K
import torch
from albumentations.pytorch import ToTensorV2

# from kornia.augmentation.container import AugmentationSequential
from segmentation_models_pytorch.losses import JaccardLoss
from torch.nn.modules import Module
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torchgeo.datasets import NAIP, BoundingBox, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler
from torchmetrics import Metric
from torchmetrics.classification import MulticlassJaccardIndex

# project imports
from . import repo_root
from .model import SegmentationModel

# from .plot_sample import plot_training_sample

# import KaneCounty dataset class
spec = importlib.util.spec_from_file_location(
    "kane_county", os.path.join(repo_root, "data", "kane_county.py")
)
kane_county = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kane_county)
KaneCounty = kane_county.KaneCounty

# import config from runtime args
parser = argparse.ArgumentParser(
    description="Train a segmentation model to predict stormwater storage "
    + "and green infrastructure."
)
parser.add_argument("config", type=str, help="Path to the configuration file")
args = parser.parse_args()
spec = importlib.util.spec_from_file_location(args.config)
config = importlib.import_module(args.config)

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
print(f"Using {device} device")

# create the model
model = SegmentationModel(num_classes=config.NUM_CLASSES).model.to(device)
print(model)

# set the loss function, metrics, and optimizer
loss_fn = JaccardLoss(mode="multiclass", classes=config.NUM_CLASSES)
train_metric = MulticlassJaccardIndex(
    num_classes=config.NUM_CLASSES,
    ignore_index=config.IGNORE_INDEX,
    average="micro",
).to(device)
test_metric = MulticlassJaccardIndex(
    num_classes=config.NUM_CLASSES,
    ignore_index=config.IGNORE_INDEX,
    average="micro",
).to(device)
optimizer = AdamW(model.parameters(), lr=config.LR)


def get_train_augmentation_pipeline():
    """
    Extend the augmentation pipeline for aerial image segmentation with additional
    transformations to simulate various environmental conditions and viewing angles.

    Returns:
        A callable augmentation pipeline that applies the defined transformations.
    """
    # Define the extended augmentation pipeline
    augmentation_pipeline = A.Compose(
        [
            # Rotate the image by up to 180 degrees, without a preferred direction
            A.Rotate(limit=180, p=0.5, border_mode=0),
            # Horizontal and vertical flipping
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # Random scaling
            A.RandomScale(scale_limit=0.1, p=0.5),
            # Brightness and contrast adjustments
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            # Slight Gaussian blur to mimic atmospheric effects
            A.GaussianBlur(blur_limit=(3, 3), p=0.2),
            # Normalize the image (ensure to adjust mean and std as per your dataset)
            A.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0
            ),
            # Convert image and mask to PyTorch tensors
            ToTensorV2(),
        ],
        additional_targets={"mask": "image"},
    )  # Apply the same transform to both image and mask

    return augmentation_pipeline


def get_test_augmentation_pipeline():
    """
    Extend the augmentation pipeline for aerial image segmentation with additional
    transformations to simulate various environmental conditions and viewing angles.

    Returns:
        A callable augmentation pipeline that applies the defined transformations.
    """
    # Define the extended augmentation pipeline
    augmentation_pipeline = A.Compose(
        [
            # Normalize the image (ensure to adjust mean and std as per your dataset)
            A.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0
            ),
            # Convert image and mask to PyTorch tensors
            ToTensorV2(),
        ],
        additional_targets={"mask": "image"},
    )  # Apply the same transform to both image and mask

    return augmentation_pipeline


train_augmentation_pipeline = get_train_augmentation_pipeline()
test_augmentation_pipeline = get_test_augmentation_pipeline()


# TODO: add transforms
def train(
    dataloader: DataLoader,
    model: Module,
    loss_fn: Module,
    metric: Metric,
    optimizer: Optimizer,
):
    num_batches = len(dataloader)
    model.train()

    for batch, sample in enumerate(dataloader):
        images = sample["image"]
        masks = sample["mask"]

        # Convert PyTorch tensors to numpy arrays for Albumentations
        images_np = images.cpu().numpy().astype(np.float32)
        masks_np = masks.cpu().numpy().astype(np.float32)

        # Apply augmentations
        augmented = train_augmentation_pipeline(image=images_np, mask=masks_np)
        augmented_image, augmented_mask = augmented["image"], augmented["mask"]

        # Convert numpy arrays back to PyTorch tensors
        X = torch.from_numpy(augmented_image).to(device)
        y = torch.from_numpy(augmented_mask).to(device)

        # Assuming your mask has a channel dimension that needs to be squeezed
        y_squeezed = y.squeeze(1)

        # compute prediction error
        outputs = model(X)
        loss = loss_fn(outputs, y_squeezed)

        # update metric
        preds = outputs.argmax(dim=1)
        metric(preds, y_squeezed)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # if batch < 5:
        #    plot_training_sample(sample[0])

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{num_batches:>5d}]")
    final_metric = metric.compute()
    print(f"Jaccard Index: {final_metric}")


def test(dataloader, model, loss_fn, metric):
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for sample in dataloader:
            images = sample["image"]
            masks = sample["mask"]

            # Convert PyTorch tensors to numpy arrays for Albumentations
            images_np = images.cpu().numpy().astype(np.float32)
            masks_np = masks.cpu().numpy().astype(np.float32)

            # Apply augmentations
            augmented = train_augmentation_pipeline(
                image=images_np, mask=masks_np
            )
            augmented_image, augmented_mask = (
                augmented["image"],
                augmented["mask"],
            )

            # Convert numpy arrays back to PyTorch tensors
            X = torch.from_numpy(augmented_image).to(device)
            y = torch.from_numpy(augmented_mask).to(device)

            # Assuming your mask has a channel dimension that needs to be squeezed
            y_squeezed = y.squeeze(1)

            # compute prediction error
            outputs = model(X)
            loss = loss_fn(outputs, y_squeezed)

            # update metric
            preds = outputs.argmax(dim=1)
            metric(preds, y_squeezed)

            # add test loss to rolling total
            test_loss += loss.item()
    test_loss /= num_batches
    final_metric = metric.compute()
    print(
        f"Test Error: \n Jaccard index: {final_metric:>7f}, "
        + "Avg loss: {test_loss:>7f} \n"
    )


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, train_metric, optimizer)
    test(test_dataloader, model, loss_fn, test_metric)
print("Done!")


torch.save(
    model.state_dict(), os.path.join(config.MODEL_STATES_ROOT, "model.pth")
)
print(f"Saved PyTorch Model State to {config.MODEL_STATES_ROOT}")
