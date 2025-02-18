"""Train a segmentation model.

To run: from repo directory (2024-winter-cmap)
> python train.py configs.<config> [--experiment_name <name>]
    [--split <split>] [--tune]  [--num_trials <num>] [--dem] [--filled_dem]
"""

import argparse
import datetime
import importlib.util
import logging
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import kornia.augmentation as K
import torch
import wandb
from torch.nn.modules import Module
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchgeo.datasets import NAIP, random_bbox_assignment, stack_samples
from torchmetrics.classification import MulticlassJaccardIndex

from data.dem import KaneDEM
from data.kc import KaneCounty
from data.rd import RiverDataset
from data.sampler import BalancedGridGeoSampler, BalancedRandomBatchGeoSampler
from model import SegmentationModel
from utils.plot import find_labels_in_ground_truth, plot_from_tensors
from utils.transforms import apply_augs, create_augmentation_pipelines

MODEL_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def arg_parsing(argument):
    """Parsing arguments passed in from command line"""
    # if no experiment name provided, set to timestamp
    exp_name_arg = argument.experiment_name
    if exp_name_arg is None:
        exp_name_arg = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    split_arg = float(int(argument.split) / 100)
    # tuning with wandb
    wandb_tune = argument.tune
    num_trials_arg = int(argument.num_trials)
    # Inclusion of DEM
    dem_arg = argument.dem
    filled_dem_arg = argument.filled_dem

    return exp_name_arg, split_arg, wandb_tune, num_trials_arg, dem_arg, filled_dem_arg


def check_gpu_availability():
    """Check if a GPU is available and exit if no GPU is found."""
    if not torch.cuda.is_available():
        print(
            "WARNING: No GPU is available on this node."
            "Please ensure this script is run on a compute node with GPU access."
        )
        sys.exit(1)


def writer_prep(exp_n, trial_num, wandb_tune):
    """Preparing writers and logging for each training trial

    Args:
        exp_n: experiment name
        trial_num: current trial number
        wandb_tune: whether tuning with wandb
    """
    # set output path and exit run if path already exists
    exp_trial_name = f"{exp_n}_trial_{trial_num}"
    out_root = Path(config.OUTPUT_ROOT) / exp_trial_name
    if wandb_tune:
        Path.mkdir(out_root, exist_ok=True, parents=True)
        Path.mkdir(out_root, parents=True, exist_ok=True)
    else:
        Path.mkdir(out_root, exist_ok=True, parents=True)

    # create directory for output images
    train_images_root = Path(out_root) / "train-images"
    test_images_root = Path(out_root) / "test-images"

    try:
        Path.mkdir(train_images_root, parents=True, exist_ok=True)
        Path.mkdir(test_images_root, parents=True, exist_ok=True)

    except FileExistsError:
        shutil.rmtree(train_images_root)
        shutil.rmtree(test_images_root)
        Path.mkdir(train_images_root)
        Path.mkdir(test_images_root)

    # open tensorboard writer
    writer = SummaryWriter(out_root)

    # copy training script and config to output directory
    shutil.copy(Path(__file__).resolve(), out_root)
    shutil.copy(Path(config.__file__).resolve(), out_root)

    # Set up logging
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_filename = Path(out_root) / "training_log.txt"
    file_handler = logging.FileHandler(log_filename)
    stream_handler = logging.StreamHandler(sys.stdout)

    # log format
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return train_images_root, test_images_root, out_root, writer, logger


def initialize_dataset(config):
    """Load and merge NAIP, KaneCounty, and optional DEM data."""
    if not config.USE_NIR:

        class RGBOnlyNAIP(NAIP):
            def __getitem__(self, query):
                sample = super().__getitem__(query)
                sample["image"] = sample["image"][:3]  # Keep only RGB channels
                return sample

        naip_dataset = RGBOnlyNAIP(config.KC_IMAGE_ROOT)
        config.DATASET_MEAN = config.DATASET_MEAN[:3]
        config.DATASET_STD = config.DATASET_STD[:3]
    else:
        naip_dataset = NAIP(config.KC_IMAGE_ROOT)

    shape_path = Path(config.KC_SHAPE_ROOT) / config.KC_SHAPE_FILENAME
    dataset_config = (
        config.KC_LAYER,
        config.KC_LABELS,
        config.PATCH_SIZE,
        naip_dataset.crs,
        naip_dataset.res,
    )
    kc_dataset = KaneCounty(shape_path, dataset_config)

    if dem_include:
        dem = KaneDEM(config.KC_DEM_ROOT, config)
        naip_dataset = naip_dataset & dem
        if not config.USE_NIR:
            config.DATASET_MEAN.append(0.0)  # DEM mean
            config.DATASET_STD.append(1.0)  # DEM std
        print("naip and dem loaded")
        if filled_dem_include:
            filled_dem = KaneDEM(config.KC_DEM_ROOT, config, use_filled=True)
            naip_dataset = naip_dataset & filled_dem
            print("naip and dem and filled dem loaded")
    if config.USE_RIVERDATASET:
        naip_dataset = NAIP(config.KC_IMAGE_ROOT)
        rd_shape_path = Path(config.KC_SHAPE_ROOT) / config.RD_SHAPE_FILE

        config.NUM_CLASSES = 6  # predicting 5 classes + background

        rd_config = (
            config.RD_LABELS,
            config.PATCH_SIZE,
            naip_dataset.crs,
            naip_dataset.res,
        )

        combined_dataset = RiverDataset(rd_shape_path, rd_config, kc=True)

        if dem_include:
            dem = KaneDEM(config.KC_DEM_ROOT)
            naip_dataset = naip_dataset & dem
            print("naip and dem loaded")
        if filled_dem_include:
            filled_dem = KaneDEM(config.KC_DEM_ROOT, config, use_filled=True)
            naip_dataset = naip_dataset & filled_dem
            print("naip and filled dem loaded")

        return naip_dataset, combined_dataset

    else:  # this is the default; uses KC only
        naip_dataset = NAIP(config.KC_IMAGE_ROOT)
        shape_path = Path(config.KC_SHAPE_ROOT) / config.KC_SHAPE_FILENAME
        dataset_config = (
            config.KC_LAYER,
            config.KC_LABELS,
            config.PATCH_SIZE,
            naip_dataset.crs,
            naip_dataset.res,
        )
        kc_dataset = KaneCounty(shape_path, dataset_config)

        if dem_include:
            dem = KaneDEM(config.KC_DEM_ROOT, config)
            naip_dataset = naip_dataset & dem
            print("naip and dem loaded")
        if filled_dem_include:
            filled_dem = KaneDEM(config.KC_DEM_ROOT, config, use_filled=True)
            naip_dataset = naip_dataset & filled_dem
            print("naip and filled dem loaded")

        return naip_dataset, kc_dataset


def build_dataset(naip_set, split_rate):
    """Randomly split and load data to be the test and train sets

    Returns train dataloader and test dataloader
    """
    # record generator seed
    seed = random.SystemRandom().randint(0, sys.maxsize)
    logging.info("Dataset random split seed: %d", seed)
    generator = torch.Generator().manual_seed(seed)

    # split the dataset
    train_portion, test_portion = random_bbox_assignment(
        naip_set, [split_rate, 1 - split_rate], generator
    )
    train_dataset = train_portion & kc
    test_dataset = test_portion & kc

    train_sampler = BalancedRandomBatchGeoSampler(
        config={
            "dataset": train_dataset,
            "size": config.PATCH_SIZE,
            "batch_size": config.BATCH_SIZE,
        }
    )
    test_sampler = BalancedGridGeoSampler(
        config={
            "dataset": test_dataset,
            "size": config.PATCH_SIZE,
            "stride": config.PATCH_SIZE,
        }
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
    return train_dataloader, test_dataloader


def regularization_loss(model, reg_type, weight):
    """Calculate the regularization loss for the model parameters.

    Args:
        model: The PyTorch model for which to calculate the regularization loss.
        reg_type: The type of regularization, either "l1" or "l2".
        weight: The weight or strength of the regularization term.

    Returns:
    - float: The calculated regularization loss.
    """
    reg_loss = 0.0
    if reg_type == "l1":
        for param in model.parameters():
            reg_loss += torch.sum(torch.abs(param))
    elif reg_type == "l2":
        for param in model.parameters():
            reg_loss += torch.sum(param**2)
    return weight * reg_loss


def compute_loss(model, mask, y, loss_fn, reg_config):
    """Compute the total loss optionally the regularization loss.

    Args:
        model: The PyTorch model for which to compute the loss.
        mask: The input mask tensor.
        y: The target tensor.
        loss_fn: The loss function to use for computing the base loss.
        reg_config: a tuple of
            reg_type: The type of regularization, either "l1" or "l2".
            reg_weight: The weight or strength of the regularization term.

    Returns:
    - torch.Tensor: The total loss as a PyTorch tensor.
    """
    base_loss = loss_fn(mask, y)
    reg_type, reg_weight = reg_config
    if reg_type is not None:
        reg_loss = regularization_loss(model, reg_type, reg_weight)
        base_loss += reg_loss
    return base_loss


def create_model():
    """Setting up training model, loss function and measuring metrics

    Returns:
        tuple: A tuple containing:
            - model: The PyTorch model instance.
            - loss_fn: The loss function to use for training.
            - train_jaccard: The metric to measure Jaccard index on the training set.
            - test_jaccard: The metric to measure Jaccard index on the test set.
            - jaccard_per_class: The metric to measure Jaccard index per class.
            - optimizer: The optimizer for training the model.
    """
    # create the model
    model_configs = {
        "model": config.MODEL,
        "backbone": config.BACKBONE,
        "num_classes": config.NUM_CLASSES,
        "weights": config.WEIGHTS,
        "dropout": config.DROPOUT,
    }

    model = SegmentationModel(model_configs).model.to(MODEL_DEVICE)
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
    ).to(MODEL_DEVICE)
    test_jaccard = MulticlassJaccardIndex(
        num_classes=config.NUM_CLASSES,
        ignore_index=config.IGNORE_INDEX,
        average="micro",
    ).to(MODEL_DEVICE)
    jaccard_per_class = MulticlassJaccardIndex(
        num_classes=config.NUM_CLASSES,
        ignore_index=config.IGNORE_INDEX,
        average=None,
    ).to(MODEL_DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    return (
        model,
        loss_fn,
        train_jaccard,
        test_jaccard,
        jaccard_per_class,
        optimizer,
    )


def add_extra_channel(
    image_tensor: torch.Tensor, source_channel: int = 0
) -> torch.Tensor:
    """Adds an additional channel to an image by copying an existing channel.

    Args:
        image_tensor : A tensor containing image data. Expected shape is
            (batch, channels, h, w)
        source_channel : The index of the channel to be copied

    Returns:
        torch.Tensor: A modified tensor with added channels
    """
    # Select the source channel to duplicate
    original_channel = image_tensor[:, source_channel : source_channel + 1, :, :]

    # Generate copy of selected channel
    extra_channel = original_channel.clone()

    # Concatenate the extra channel to the original image along the second
    # dimension (channel dimension)
    augmented_tensor = torch.cat((image_tensor, extra_channel), dim=1)

    return augmented_tensor


def add_extra_channels(image, model):
    """Add extra channels to the image if necessary."""
    while image.size(1) < model.in_channels:
        image = add_extra_channel(image)
    return image


def apply_augmentations(
    dataset, spatial_augs, color_augs, spatial_aug_mode, color_aug_mode
):
    """Apply augmentations to the image and mask."""
    x_og, y_og = dataset
    aug_config = (spatial_augs, color_augs, spatial_aug_mode, color_aug_mode)
    x_aug, y_aug = apply_augs(aug_config, x_og, y_og)
    y_aug = y_aug.type(torch.int64)  # Convert mask to int64 for loss function
    y_squeezed = y_aug.squeeze()  # Remove channel dim from mask
    return x_aug, y_squeezed


def save_training_images(epoch, train_images_root, x, samp_mask, x_aug, y_aug, sample):
    """Save training sample images."""
    save_dir = Path(train_images_root) / f"epoch-{epoch}"
    Path.mkdir(save_dir, exist_ok=True)

    # Denormalize augmented images for plotting
    data_mean = config.DATASET_MEAN
    data_std = config.DATASET_STD
    if len(data_mean) < x_aug.size(1):
        data_mean = data_mean + [data_mean[0]] * (x_aug.size(1) - len(data_mean))
        data_std = data_std + [data_std[0]] * (x_aug.size(1) - len(data_std))

    mean = torch.tensor(data_mean).view(-1, 1, 1)
    std = torch.tensor(data_std).view(-1, 1, 1)
    x_aug_denorm = x_aug * std.to(x_aug.device) + mean.to(x_aug.device)

    for i in range(config.BATCH_SIZE):
        plot_tensors = {
            "RGB image": x[i][0:3, :, :].cpu() / 255.0,
            "augmented RGB image": x_aug_denorm[i][0:3, :, :].cpu().clip(0, 1),
            "mask": samp_mask[i],
            "augmented mask": y_aug[i].cpu(),
        }

        # Add NIR if enabled
        if config.USE_NIR:
            nir_idx = 3  # NIR is always after RGB
            plot_tensors.update(
                {
                    "NIR": x[i][nir_idx : nir_idx + 1, :, :].cpu() / 255.0,
                    "augmented NIR": x_aug_denorm[i][nir_idx : nir_idx + 1, :, :]
                    .cpu()
                    .clip(0, 1),
                }
            )

        if dem_include or filled_dem_include:
            base_idx = 3 if not config.USE_NIR else 4  # Adjust for NIR presence

            if dem_include:
                plot_tensors.update(
                    {
                        "DEM": x[i][-1, :, :].cpu() / 255.0,
                        "augmented DEM": x_aug_denorm[i][-1, :, :]
                        .cpu()
                        .clip(0, 1),
                    }
                )
                base_idx += 1  # Move index forward

            if filled_dem_include:
                plot_tensors.update(
                    {
                        "Difference DEM": x[i][-1, :, :].cpu() / 255.0,
                        "Augmented Difference DEM": x_aug_denorm[i][
                            -1, :, :
                        ]
                        .cpu()
                        .clip(0, 1),
                    }
                )

        sample_fname = Path(save_dir) / f"train_sample-{epoch}.{i}.png"
        plot_from_tensors(
            plot_tensors,
            sample_fname,
            kc.colors,
            kc.labels_inverse,
            sample["bbox"][i],
        )


def log_channel_stats(tensor: torch.Tensor, name: str, logger: logging.Logger):
    """Log statistics for each channel of the input tensor."""
    for i in range(tensor.size(1)):
        channel = tensor[:, i]
        logger.info(
            f"{name} channel {i} - min: {channel.min().item():.3f}, "
            f"max: {channel.max().item():.3f}, mean: {channel.mean().item():.3f}, "
            f"std: {channel.std().item():.3f}"
        )


def train_setup(
    sample: defaultdict[str, Any],
    train_config,
    aug_config,
    model,
) -> tuple[torch.Tensor]:
    """Setup for training: sends images to device and applies augmentations."""
    epoch, batch, train_images_root = train_config
    spatial_aug_mode, color_aug_mode, spatial_augs, color_augs = aug_config

    samp_image = sample["image"]
    samp_mask = sample["mask"]

    # Add extra channels to image if necessary
    samp_image = add_extra_channels(samp_image, model)

    # Send image and mask to device; convert mask to float tensor for augmentation
    x = samp_image.to(MODEL_DEVICE)
    y = samp_mask.type(torch.float32).to(MODEL_DEVICE)

    if batch == 0:  # Log stats for first batch only
        log_channel_stats(x, "raw input", logging.getLogger())

    # Scale to [0,1]
    x = x / 255.0

    if batch == 0:  # Log stats for first batch only
        log_channel_stats(x, "scaled input", logging.getLogger())

    # Extend mean/std if needed
    data_mean = config.DATASET_MEAN  # ImageNet mean
    data_std = config.DATASET_STD  # ImageNet std
    if len(data_mean) < model.in_channels:
        data_mean = data_mean + [data_mean[0]] * (model.in_channels - len(data_mean))
        data_std = data_std + [data_std[0]] * (model.in_channels - len(data_std))

    # Normalize using ImageNet statistics
    normalize = K.Normalize(mean=data_mean, std=data_std)
    x_norm = normalize(x)

    if batch == 0:  # Log stats for first batch only
        log_channel_stats(x_norm, "normalized input", logging.getLogger())

    img_data = (x_norm, y)
    # Apply augmentations
    x_aug, y_squeezed = apply_augmentations(
        img_data, spatial_augs, color_augs, spatial_aug_mode, color_aug_mode
    )

    if batch == 0:  # Log stats for first batch only
        log_channel_stats(x_aug, "augmented input", logging.getLogger())

    # Save training sample images if first batch
    if batch == 0:
        save_training_images(
            epoch,
            train_images_root,
            samp_image,  # Save original images
            samp_mask,
            x_aug,
            y_squeezed,
            sample,
        )

    return x_aug.to(MODEL_DEVICE), y_squeezed.to(MODEL_DEVICE)


def train_epoch(
    dataloader,
    model,
    train_config,
    aug_config,
    writer,
    args,
    wandb_tune,
) -> None:
    """Executes a training step for the model

    Args:
        dataloader: The data loader containing the training data.
        model: The PyTorch model to be trained.
        train_config: a tuple of
            - loss_fn: The loss function to be used for training.
            - jaccard: The metric to measure Jaccard index during training.
            - optimizer: The optimizer to be used for updating model parameters.
            - epoch: The current epoch number.
            - train_images_root: The root directory for saving training sample images.
        aug_config: a tuple of
            - spatial_augs: The sequence of spatial augmentations.
            - color_augs: The sequence of color augmentations.
            - spatial_aug_mode: The mode for spatial augmentations.
            - color_aug_mode: The mode for color augmentations.
        writer: The TensorBoard writer for logging training metrics.
        args: Additional arguments for debugging or special training conditions.
        args: Additional arguments for debugging or special training conditions.
        wandb_tune: whether tuning with wandb
    """
    loss_fn, jaccard, optimizer, epoch, train_images_root = train_config
    spatial_augs, color_augs, spatial_aug_mode, color_aug_mode = aug_config

    num_batches = len(dataloader)
    model.train()
    jaccard.reset()
    train_loss = 0
    for batch, sample in enumerate(dataloader):
        train_config = (epoch, batch, train_images_root)
        aug_config = (
            spatial_aug_mode,
            color_aug_mode,
            spatial_augs,
            color_augs,
        )
        x, y = train_setup(
            sample,
            train_config,
            aug_config,
            model,
        )
        x = x.to(MODEL_DEVICE)
        y = y.to(MODEL_DEVICE)

        # compute prediction error
        outputs = model(x)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = compute_loss(
            model,
            outputs,
            y,
            loss_fn,
            (config.REGULARIZATION_TYPE, config.REGULARIZATION_WEIGHT),
        )

        # update jaccard index
        preds = outputs.argmax(dim=1)
        jaccard.update(preds, y)

        # backpropagation
        loss.backward()

        # Gradient clipping
        if config.GRADIENT_CLIPPING:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_VALUE)

        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1)
            logging.info(f"loss: {loss:7.7f}  [{current:5d}/{num_batches:5d}]")

        # Break after the first batch in debug mode
        if args.debug and batch == 0:
            print("Debug mode: Exiting training loop after first batch.")
            break
    
    train_loss /= num_batches
    final_jaccard = jaccard.compute()

    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("IoU/train", final_jaccard, epoch)
    logging.info("Train Jaccard index: %.4f", final_jaccard)

    if wandb_tune:
        wandb.log(
            {
                "train_loss": train_loss,
                "train_jaccard": final_jaccard.item(),
                "epoch": epoch,
                "batch": batch,
            }
        )

    return final_jaccard


def test(
    dataloader: DataLoader,
    model: Module,
    test_config,
    writer,
    wandb_tune: bool,
    num_examples: int = 10,
) -> float:
    """Executes a testing step for the model and saves sample output images.

    Args:
        dataloader: Dataloader for the testing data.
        model: A PyTorch model.
        test_config: A tuple containing:
            - loss_fn: A PyTorch loss function.
            - jaccard: The metric to be used for evaluation, specifically the
                    Jaccard Index.
            - epoch: The current epoch.
            - plateau_count: The number of epochs the loss has been plateauing.
            - test_image_root: The root directory for saving test images.
            - writer: The TensorBoard writer for logging test metrics.
            - num_classes: The number of labels to predict.
            - jaccard_per_class: The metric to calculate Jaccard index per class.
        writer: The TensorBoard writer for logging test metrics.
        wandb_tune: whether tune with wandb
        num_examples: The number of examples to save.

    Returns:
        float: The test loss for the epoch.
    """
    (
        loss_fn,
        jaccard,
        epoch,
        plateau_count,
        test_image_root,
        writer,
        num_classes,
        jaccard_per_class,
    ) = test_config
    num_batches = len(dataloader)
    model.eval()
    jaccard.reset()
    jaccard_per_class.reset()
    test_loss = 0
    with torch.no_grad():
        for batch, sample in enumerate(dataloader):
            samp_image = sample["image"]
            samp_mask = sample["mask"]

            # add an extra channel to the images and masks
            if samp_image.size(1) != model.in_channels:
                samp_image = add_extra_channel(samp_image)

            x = samp_image.to(MODEL_DEVICE)

            if batch == 0:  # Log stats for first batch only
                log_channel_stats(x, "test raw input", logging.getLogger())

            # Scale to [0,1] before normalization
            x = x / 255.0

            if batch == 0:  # Log stats for first batch only
                log_channel_stats(x, "test scaled input", logging.getLogger())

            # Extend mean/std if needed
            data_mean = config.DATASET_MEAN
            data_std = config.DATASET_STD
            if len(data_mean) < model.in_channels:
                data_mean = data_mean + [data_mean[0]] * (
                    model.in_channels - len(data_mean)
                )
                data_std = data_std + [data_std[0]] * (
                    model.in_channels - len(data_std)
                )

            # Normalize
            normalize = K.Normalize(mean=data_mean, std=data_std)
            x = normalize(x)

            if batch == 0:  # Log stats for first batch only
                log_channel_stats(x, "test normalized input", logging.getLogger())

            y = samp_mask.to(MODEL_DEVICE)
            if y.size(0) == 1:
                y_squeezed = y
            else:
                y_squeezed = y.squeeze()

            # compute prediction error
            outputs = model(x)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = loss_fn(outputs, y_squeezed)

            # update metric
            preds = outputs.argmax(dim=1)
            jaccard.update(preds, y_squeezed)

            # update Jaccard per class metric
            jaccard_per_class.forward(preds, y_squeezed)

            # add test loss to rolling total
            test_loss += loss.item()

            # plot first batch
            if batch == 0 or (
                plateau_count == config.PATIENCE - 1 and batch < num_examples
            ):
                epoch_dir = Path(test_image_root) / f"epoch-{epoch}"
                if not Path.exists(epoch_dir):
                    Path.mkdir(epoch_dir)

                # Denormalize for plotting
                data_mean = config.DATASET_MEAN
                data_std = config.DATASET_STD
                if len(data_mean) < x.size(1):  # Extend mean/std if needed
                    data_mean = data_mean + [data_mean[0]] * (
                        x.size(1) - len(data_mean)
                    )
                    data_std = data_std + [data_std[0]] * (x.size(1) - len(data_std))

                mean = torch.tensor(data_mean).view(-1, 1, 1)
                std = torch.tensor(data_std).view(-1, 1, 1)
                x_denorm = x * std.to(x.device) + mean.to(x.device)

                for i in range(config.BATCH_SIZE):
                    plot_tensors = {
                        "RGB image": x_denorm[i][0:3, :, :].cpu().clip(0, 1),
                        "ground truth": samp_mask[i],
                        "prediction": preds[i].cpu(),
                    }

                    # Add DEM if enabled
                    base_idx = 3 if not config.USE_NIR else 4  # Adjust for NIR presence

                    if dem_include:
                        plot_tensors["DEM"] = (
                            x_denorm[i][-1, :, :].cpu().clip(0, 1)
                        )
                        base_idx += 1  # Move to the next channel index

                    if filled_dem_include:
                        plot_tensors["Filled DEM"] = (
                            x_denorm[i][-1, :, :].cpu().clip(0, 1)
                        )

                    ground_truth = samp_mask[i]
                    label_ids = find_labels_in_ground_truth(ground_truth)

                    for label_id in label_ids:
                        label_name = kc.labels_inverse.get(label_id, "UNKNOWN")
                        save_dir = Path(epoch_dir) / label_name
                        if not Path.exists(save_dir):
                            Path.mkdir(save_dir)
                        sample_fname = (
                            Path(save_dir) / f"test_sample-{epoch}.{batch}.{i}.png"
                        )
                        plot_from_tensors(
                            plot_tensors,
                            sample_fname,
                            kc.colors,
                            kc.labels_inverse,
                            sample["bbox"][i],
                        )
    test_loss /= num_batches
    final_jaccard = jaccard.compute()
    final_jaccard_per_class = jaccard_per_class.compute()
    writer.add_scalar("loss/test", test_loss, epoch)
    writer.add_scalar("IoU/test", final_jaccard, epoch)
    logger = logging.getLogger()
    logger.info("Test error:")
    logger.info(f"Jaccard index: {final_jaccard:.3f}")
    logger.info(f"Test avg loss: {test_loss:.3f}")

    if wandb_tune:
        wandb.log(
            {
                "test_loss": test_loss,
                "test_jaccard": final_jaccard.item(),
                "epoch": epoch,
            }
        )

    # Access the labels and their names
    _labels = {}
    for label_name, label_id in kc.labels.items():
        _labels[label_id] = label_name
        if len(_labels) == num_classes:
            break

    for i, label_name in _labels.items():
        logger.info(f"IoU for {label_name}: {final_jaccard_per_class[i]:.3f}")

    return test_loss, final_jaccard


def train(
    model: Module,
    train_test_config,
    aug_config,
    path_config: tuple[str, str, str],
    writer: SummaryWriter,
    wandb_tune: bool,
    args,
    epoch,
) -> tuple[float, float]:
    """Train a deep learning model using the specified configuration and parameters.

    Args:
        model: The deep learning model to be trained.
        train_test_config: A tuple containing:
                - train_dataloader: DataLoader for training dataset.
                - train_jaccard: Function to calculate Jaccard index for training.
                - test_jaccard: Function to calculate Jaccard index for test.
                - test_dataloader: DataLoader for test dataset.
                - loss_fn: Loss function used for training and testing.
                - optimizer: Optimization algorithm used for training.
                - jaccard_per_class: Function to calculate Jaccard index per class.
        aug_config: A tuple containing:
                - spatial_augs: Spatial augmentations applied during training.
                - color_augs: Color augmentations applied during training.
        path_config: A tuple containing:
                - out_root: Root directory for saving the trained model.
                - train_images_root: Root directory for training images.
                - test_image_root: Root directory for test images.
        writer: The writer object for logging training progress.
        wandb_tune: Whether running hyperparameter tuning with wandb.
        args: Additional arguments for debugging or special training conditions.
        epoch: The configuration for the number of epochs.

    Returns:
        Tuple[float, float]: A tuple containing the Jaccard index for the last
             epoch of training and for the test dataset.
    """
    (
        train_dataloader,
        train_jaccard,
        test_jaccard,
        test_dataloader,
        loss_fn,
        optimizer,
        jaccard_per_class,
    ) = train_test_config
    (
        out_root,
        train_images_root,
        test_image_root,
    ) = path_config
    (
        spatial_augs,
        color_augs,
    ) = aug_config

    # How much the loss needs to drop to reset a plateau
    threshold = config.THRESHOLD

    # How many epochs loss needs to plateau before terminating
    patience = config.PATIENCE

    # Beginning loss
    best_loss = None

    # How long it's been plateauing
    plateau_count = 0

    # How many classes we're predicting
    num_classes = config.NUM_CLASSES

    # reducing number of epoch in debugging or hyperparameter tuning
    if args.debug:
        epoch = 1
    elif wandb_tune:
        epoch = 10
    else:
        epoch = config.EPOCHS

    for t in range(epoch):
        if t == 0:
            test_config = (
                loss_fn,
                test_jaccard,
                t,
                plateau_count,
                test_image_root,
                writer,
                num_classes,
                jaccard_per_class,
            )
            test_loss, t_jaccard = test(
                test_dataloader,
                model,
                test_config,
                writer,
                args,
            )
            print(f"untrained loss {test_loss:.3f}, jaccard {t_jaccard:.3f}")

        logging.info(f"Epoch {t + 1}\n-------------------------------")
        train_config = (
            loss_fn,
            train_jaccard,
            optimizer,
            t + 1,
            train_images_root,
        )
        aug_config = (
            spatial_augs,
            color_augs,
            config.SPATIAL_AUG_MODE,
            config.COLOR_AUG_MODE,
        )
        epoch_jaccard = train_epoch(
            train_dataloader,
            model,
            train_config,
            aug_config,
            writer,
            args,
            wandb_tune,
        )

        test_config = (
            loss_fn,
            test_jaccard,
            t + 1,
            plateau_count,
            test_image_root,
            writer,
            num_classes,
            jaccard_per_class,
        )
        test_loss, t_jaccard = test(
            test_dataloader,
            model,
            test_config,
            writer,
            wandb_tune,
        )
        # Checks for plateau
        if best_loss is None:
            best_loss = test_loss
        elif test_loss < best_loss - threshold:
            best_loss = test_loss
            plateau_count = 0
        else:
            plateau_count += 1
            if plateau_count >= patience:
                logging.info(
                    f"Loss Plateau: {t} epochs, reached patience of {patience}"
                )

        if wandb_tune:
            wandb.log(
                {
                    "epoch": t + 1,
                    "train_jaccard": epoch_jaccard.item(),
                    "test_jaccard": t_jaccard.item(),
                    "test_loss": test_loss,
                }
            )

        # Break after the first iteration in debug mode
        if args.debug and t == 0:
            print("Debug mode: Skipping the rest of the training loop")
            break

    print("Done!")

    torch.save(model.state_dict(), Path(out_root) / "model.pth")
    logging.info("Saved PyTorch Model State to %s", out_root)

    return epoch_jaccard, t_jaccard


def one_trial(exp_n, num, wandb_tune, naip_set, split_rate, args):
    """Runing a single trial of training

    Input:
        exp_n: experiment name
        num: current number of trial
        wandb_tune: whether tuning with wandb
    """
    (
        train_images_root,
        test_image_root,
        out_root,
        writer,
        logger,
    ) = writer_prep(exp_n, num, wandb_tune)
    # Set 'epoch' based on debug mode
    if args.debug:
        epoch = 1
    else:
        epoch = config.EPOCHS
    # randomly splitting the data at every trial
    train_dataloader, test_dataloader = build_dataset(naip_set, split_rate)
    (
        model,
        loss_fn,
        train_jaccard,
        test_jaccard,
        jaccard_per_class,
        optimizer,
    ) = create_model()
    spatial_augs, color_augs = create_augmentation_pipelines(
        config,
        config.SPATIAL_AUG_INDICES,
        config.IMAGE_AUG_INDICES,
    )
    logging.info("Trial %d\n====================================", num + 1)
    train_test_config = (
        train_dataloader,
        train_jaccard,
        test_jaccard,
        test_dataloader,
        loss_fn,
        optimizer,
        jaccard_per_class,
    )
    aug_config = (
        spatial_augs,
        color_augs,
    )
    path_config = (
        out_root,
        train_images_root,
        test_image_root,
    )

    train_iou, test_iou = train(
        model,
        train_test_config,
        aug_config,
        path_config,
        writer,
        wandb_tune,
        args,
        epoch,
    )
    writer.close()
    logger.handlers.clear()
    return train_iou, test_iou


if __name__ == "__main__":
    # Check GPU availability; if GPU available, run on compute node, else exit
    check_gpu_availability()

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
    parser.add_argument(
        "--split",
        type=str,
        help="Ratio of split; enter the size of the train split as an int out of 100",
        default="80",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Whether to apply hyperparameter tuning with wandb; enter True or False",
        default=False,
    )
    parser.add_argument(
        "--num_trials",
        type=str,
        help="Please enter the number of trial for each train",
        default="1",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode", default=False
    )
    parser.add_argument(
        "--dem",
        action="store_true",
        help="Include Bare Earth DEM in model",
        default=False,
    )
    parser.add_argument(
        "--filled_dem",
        action="store_true",
        help="Include Filled Bare Earth DEM file in model",
        default=False,
    )

    args = parser.parse_args()

    config = importlib.import_module(args.config)

    # enable debug mode
    if args.debug:
        epoch = 1
    else:
        epoch = config.EPOCHS
    # Enable debug mode in config
    config.DEBUG_MODE = args.debug

    exp_name, split, wandb_tune, num_trials, dem_include, filled_dem_include = (
        arg_parsing(args)
    )

    logging.info("Using %s device", MODEL_DEVICE)

    naip, kc = initialize_dataset(config)

    def run_trials():
        """Running training for multiple trials"""
        # Extract config dictionary from module

        if wandb_tune:
            wandb.init(project="CMAP")
            print("wandb taken over config")
        else:
            # Initialize wandb with default configuration but disable logging
            wandb.init(project="CMAP", config=config, mode="disabled")

        train_ious = []
        test_ious = []

        for num in range(num_trials):
            train_iou, test_iou = one_trial(
                exp_name, num, wandb_tune, naip, split, args
            )
            train_ious.append(float(train_iou))
            test_ious.append(float(test_iou))

        test_average = mean(test_ious)
        train_average = mean(train_ious)
        test_std = 0
        train_std = 0
        if num_trials > 1:
            test_std = stdev(test_ious)
            train_std = stdev(train_ious)

        print(
            f"""
            Training result: {train_ious},
            average: {train_average:.3f}, standard deviation: {train_std:.3f}"""
        )
        print(
            f"""
            Test result: {test_ious},
            average: {test_average:.3f}, standard deviation:{test_std:.3f}"""
        )

        if wandb_tune:
            wandb.run.summary["average_test_jaccard_index"] = test_average
            wandb.finish()

    run_trials()
