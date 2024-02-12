import importlib.util
import os

import torch

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
# TODO: figure out how to import as package
from . import repo_root
from .model import SegmentationModel

spec = importlib.util.spec_from_file_location(
    "kane_county", os.path.join(repo_root, "data", "kane_county.py")
)
kane_county = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kane_county)
KaneCounty = kane_county.KaneCounty

# TODO: figure out how to import config from config_loader
spec = importlib.util.spec_from_file_location(
    "configs", os.path.join(repo_root, "configs", "dsi.py")
)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


# hyperparameters
batch_size = 64
patch_size = 256
num_classes = 5  # predicting 4 classes + background
lr = 1e-3

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
    dataset=dataset, size=patch_size, batch_size=batch_size, roi=train_roi
)
test_roi = BoundingBox(midx, roi.maxx, roi.miny, roi.maxy, roi.mint, roi.maxt)
test_sampler = GridGeoSampler(
    dataset, size=patch_size, stride=patch_size, roi=test_roi
)

# create dataloaders (must use batch_sampler)
train_dataloader = DataLoader(
    dataset, batch_sampler=train_sampler, collate_fn=stack_samples
)
test_dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=test_sampler,
    collate_fn=stack_samples,
    num_workers = 2,
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
model = SegmentationModel(num_classes=num_classes).model.to(device)
print(model)

# set the loss function and optimizer
loss_fn = JaccardLoss(mode="multiclass", classes=num_classes)
train_metric = MulticlassJaccardIndex(
    num_classes=num_classes,
    ignore_index=0,
    average="micro",
).to(device)
test_metric = MulticlassJaccardIndex(
    num_classes=num_classes,
    ignore_index=0,
    average="micro",
).to(device)
optimizer = AdamW(model.parameters(), lr=lr)


# TODO: transforms
def train(
    dataloader: DataLoader, model: Module, loss_fn: Module, metric: Metric, optimizer: Optimizer
):
    size = len(dataloader.dataset)  # TODO: what is this?
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

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            loss, current = loss.item(), (batch + 1)
            print(
                f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]"
            )  # TODO: not correct currently


def test(dataloader, model, loss_fn, metric):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
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
    test_loss /= num_batches  # TODO: not working
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


# TODO: update with the correct path
torch.save(model.state_dict(), "/home/rubensteinm/2024-winter-cmap/output/model.pth")
print(
    "Saved PyTorch Model State to /home/rubensteinm/2024-winter-cmap/output/model.pth"
)
