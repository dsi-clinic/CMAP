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
from torch.nn.modules import Module
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchgeo.datasets import NAIP, random_bbox_splitting, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler
from torchmetrics import Metric
from torchmetrics.classification import MulticlassJaccardIndex

from data.kc import KaneCounty
from data.dem import KaneDEM
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
dem = KaneDEM(config.KC_DEM_ROOT)
dataset = naip & kc & dem

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


def copy_first_entry(a_list: list) -> list:
    """
    Copies the first entry in a list and appends it to the end.

    Parameters
    ----------
    a_list : list
        The list to modify

    Returns
    -------
    list
        The modified list
    """
    first_entry = a_list[0]

    # Append the copy to the end
    a_list.append(first_entry)

    return a_list


mean = config.DATASET_MEAN
std = config.DATASET_STD
# add copies of first entry to DATASET_MEAN and DATASET_STD
# to match data in_channels
if len(mean) != model.in_channels:
    for _ in range(model.in_channels - len(mean)):
        mean = copy_first_entry(mean)
        std = copy_first_entry(std)

scale_mean = torch.tensor(0.0)
scale_std = torch.tensor(255.0)
normalize = K.Normalize(mean=mean, std=std)
scale = K.Normalize(mean=scale_mean, std=scale_std)


def add_extra_channel(
    image_tensor: torch.Tensor, source_channel: int = 0
) -> torch.Tensor:
    """
    Adds an additional channel to an image by copying an existing channel.

    Parameters
    ----------
    image_tensor : torch.Tensor
        A tensor containing image data. Expected shape is
        (batch, channels, h, w)

    source_channel : int
        The index of the channel to be copied

    Returns
    -------
    torch.Tensor
        A modified tensor with added channels
    """
    # Select the source channel to duplicate
    original_channel = image_tensor[
        :, source_channel : source_channel + 1, :, :
    ]

    # Generate copy of selected channel
    extra_channel = original_channel.clone()

    # Concatenate the extra channel to the original image along the second
    # dimension (channel dimension)
    augmented_tensor = torch.cat((image_tensor, extra_channel), dim=1)

    return augmented_tensor


def train_setup(
    sample: DefaultDict[str, Any], epoch: int, batch: int
) -> Tuple[torch.Tensor]:
    """
    Sets up for the training step by sending images and masks to device,
    applying augmentations, and saving training sample images.

    Parameters
    ----------
    sample : DefaultDict[str, Any]
        A dataloader sample containing image, mask, and bbox data

    epoch : int
        The current epoch

    batch : int
        The current batch

    Returns
    -------
    Tuple[torch.Tensor]
        Augmented image and mask tensors to be used in the train step
    """
    samp_image = sample["image"]
    samp_mask = sample["mask"]
    # add extra channel(s) to the images and masks
    if samp_image.size(1) != model.in_channels:
        for _ in range(model.in_channels - samp_image.size(1)):
            samp_image = add_extra_channel(samp_image)
            samp_mask = add_extra_channel(samp_mask)

    # send img and mask to device; convert y to float tensor for augmentation
    X = samp_image.to(device)
    y = samp_mask.type(torch.float32).to(device)

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
                "mask": samp_mask[i],
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
) -> None:
    """
    Executes a training step for the model

    Parameters
    ----------
    dataloder : DataLoader
        Dataloader for the training data

    model : Module
        A PyTorch model

    loss_fn : Module
        A PyTorch loss function

    jaccard : Metric
        The metric to be used for evaluation, specifically the Jaccard Index

    optimizer : Optimizer
        The optimizer to be used for training

    epoch : int
        The current epoch
    """
    num_batches = len(dataloader)
    model.train()
    jaccard.reset()
    train_loss = 0
    for batch, sample in enumerate(dataloader):
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
    plateau_count: int,
) -> float:
    """
    Executes a testing step for the model and saves sample output images

    Parameters
    ----------
    dataloder : DataLoader
        Dataloader for the testing data

    model : Module
        A PyTorch model

    loss_fn : Module
        A PyTorch loss function

    jaccard : Metric
        The metric to be used for evaluation, specifically the Jaccard Index

    epoch : int
        The current epoch

    plateau_count : int
        The number of epochs the loss has been plateauing

    Returns
    -------
    float
        The test loss for the epoch
    """
    num_batches = len(dataloader)
    model.eval()
    jaccard.reset()
    test_loss = 0
    with torch.no_grad():
        for batch, sample in enumerate(dataloader):
            samp_image = sample["image"]
            samp_mask = sample["mask"]
            # add an extra channel to the images and masks
            if samp_image.size(1) != model.in_channels:
                for _i in range(model.in_channels - samp_image.size(1)):
                    samp_image = add_extra_channel(samp_image)
                    samp_mask = add_extra_channel(samp_mask)
            X = samp_image.to(device)
            X_scaled = scale(X)
            X = normalize(X_scaled)
            y = samp_mask.to(device)
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
            if batch == 0 or (
                plateau_count == config.PATIENCE - 1 and batch < 10
            ):
                save_dir = os.path.join(test_image_root, f"epoch-{epoch}")
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                for i in range(config.BATCH_SIZE):
                    plot_tensors = {
                        "image": X_scaled[i].cpu(),
                        "ground_truth": samp_mask[i],
                        "prediction": preds[i].cpu(),
                    }
                    sample_fname = os.path.join(
                        save_dir, f"test_sample-{epoch}.{batch}.{i}.png"
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
threshold = config.THRESHOLD

# How many epochs loss needs to plateau before terminating
patience = config.PATIENCE

# Beginning loss
best_loss = None

# How long it's been plateauing
plateau_count = 0

for t in range(config.EPOCHS):
    logging.info(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, train_jaccard, optimizer, t + 1)
    test_loss = test(
        test_dataloader, model, loss_fn, test_jaccard, t + 1, plateau_count
    )

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
