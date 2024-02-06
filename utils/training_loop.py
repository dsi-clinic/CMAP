import importlib.util
import os

import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import JaccardLoss
from torch.nn.modules import Module
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torchgeo.datasets import NAIP, BoundingBox, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler

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
test_sampler = GridGeoSampler(dataset, size=patch_size, stride=patch_size, roi=test_roi)

# create dataloaders (must use batch_sampler)
train_dataloader = DataLoader(
    dataset, batch_sampler=train_sampler, collate_fn=stack_samples
)
test_dataloader = DataLoader(
    dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=stack_samples
)

# get device for training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")

# create the model
model = SegmentationModel(num_classes=num_classes).model.to(device)
print(model)

# set the loss function and optimizer
# loss_fn = JaccardLoss(mode="multiclass", classes=num_classes)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = AdamW(model.parameters(), lr=lr)


# TODO: transforms
def train(dataloader: DataLoader, model: Module, loss_fn: Module, optimizer: Optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, sample in enumerate(dataloader):
        X = sample["image"].to(device)
        y = sample["mask"].to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y[:, 0, :, :])

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            loss, current = loss.item(), (batch + 1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for sample in dataloader:
            X = sample["image"].to(device)
            y = sample["mask"].to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y[:, 0, :, :]).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# TODO: update with the correct path
torch.save(model.state_dict(), "/home/sjne/2024-winter-cmap/output/model.pth")
print("Saved PyTorch Model State to /home/sjne/2024-winter-cmap/output/model.pth")
