import argparse
import importlib.util
import os

import kornia as K
import torch
from kornia.augmentation.container import AugmentationSequential
from segmentation_models_pytorch.losses import JaccardLoss
from torch.nn.modules import Module
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torchgeo.datasets import NAIP, BoundingBox, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler
from torchmetrics import Metric
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.transforms import *


# project imports
from . import repo_root
from .model import SegmentationModel

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
        X = sample["image"].to(device)
        y = sample["mask"].to(device)

        y_squeezed = y[:, 0, :, :].squeeze()

        # Apply transformations
        transforms = [
            Normalize(mean=[0.485, 0.456, 0.406, 0.427], std=[0.229, 0.224, 0.225, 0.227]),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=[90, 270]),
            RandomResizedCrop(size=(256, 256), scale=(0.08, 1.0)),
            GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        ]

        # Apply transformations to input image
        for transform in transforms:
            X = transform(X)

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

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{num_batches:>5d}]")
    final_metric = metric.compute()
    print(f"Jaccard Index: {final_metric}")


def test(dataloader, model, loss_fn, metric):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for sample in dataloader:
            X = sample["image"].to(device)
            y = sample["mask"].to(device)
            y_squeezed = y[:, 0, :, :].squeeze()

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
