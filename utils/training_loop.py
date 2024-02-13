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
from torch.utils.tensorboard import SummaryWriter

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

# prepare tensorboard for logging
writer = SummaryWriter()

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
    train_loss = 0
    for batch, sample in enumerate(dataloader):
        X = sample["image"].to(device)
        y = sample["mask"].to(device)
        y_squeezed = y[:, 0, :, :].squeeze()
        """
        # Apply transformations
        X = (
            K.enhance.normalize_min_max(X, min_val=-1.0, max_val=1.0),
        )  # Normalize intensity
        transforms = AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
            K.augmentation.RandomPlasmaShadow(
                roughness=(0.1, 0.7),
                shade_intensity=(-1.0, 0.0),
                shade_quantity=(0.0, 1.0),
                keepdim=True,
            ),
            K.augmentation.RandomGaussianBlur(
                kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.25
            ),
            K.augmentation.RandomResizedCrop(
                size=(config.PATCH_SIZE, config.PATCH_SIZE),
                scale=(0.08, 1.0),
                p=0.25,
            ),
            K.augmentation.RandomRotation(degrees=(0., 360.)),
            data_keys=["image"],
            keepdim=True,
        ).to(device)
        X = transforms(X)
        """

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
            train_loss += loss
    final_metric = metric.compute() # is this doing the same thing as line below?
    print(f"Jaccard Index: {final_metric}")
    train_loss /= num_batches
    return train_loss

def test(dataloader, model, loss_fn, metric):
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
    return test_loss, final_metric


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss_train = train(train_dataloader, model, loss_fn, train_metric, optimizer)
    writer.add_scalar('Loss/train', loss_train, t)
    loss_test = test(test_dataloader, model, loss_fn, test_metric)[0]
    iou_test = test(test_dataloader, model, loss_fn, test_metric)[1]
    writer.add_scalar('Loss/test', loss_test, t)
    writer.add_scalar('IoU/test', iou_test, t)
    writer.close()
print("Done!")

# visualize results with TensorBoard on cmd line
"""
pip install tensorboard
tensorboard --logdir=runs
"""

torch.save(
    model.state_dict(), os.path.join(config.MODEL_STATES_ROOT, "model.pth")
)
print(f"Saved PyTorch Model State to {config.MODEL_STATES_ROOT}")
