"""Train a segmentation model.

To run: from repo directory (2024-winter-cmap)
> python train.py configs.<config> [--experiment_name <name>]
    [--split <split>] [--tune]  [--num_trials <num>]
"""

import argparse
import datetime
import importlib.util
import logging
import random
import shutil
import sys
import time  # Add time module for timing
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import kornia.augmentation as K
import torch
import torch.multiprocessing as mp
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


def get_model_device():
    """Returns model device for interpretation by multiprocessing"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


MODEL_DEVICE = get_model_device()


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

    return exp_name_arg, split_arg, wandb_tune, num_trials_arg


def check_gpu_availability():
    """Check if a GPU is available and exit if no GPU is found."""
    if not torch.cuda.is_available():
        print(
            "WARNING: No GPU is available on this node."
            "Please ensure this script is run on a compute node with GPU access."
        )
        sys.exit(1)


def writer_prep(exp_n, trial_num, wandb_tune, config):
    """Preparing writers and logging for each training trial

    Args:
        exp_n: STR experiment name
        trial_num: INT current trial number
        wandb_tune: BOOL whether tuning with wandb
        config (module): Configuration object containing OUTPUT_ROOT, etc.
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


class RGBOnlyNAIP(NAIP):
    """Generates RGB Dataset"""

    def __getitem__(self, query):
        """Returns sample from dataset"""
        sample = super().__getitem__(query)
        sample["image"] = sample["image"][:3]  # Keep only RGB channels
        return sample


def initialize_dataset(config):
    """Load and merge NAIP, KaneCounty, and optional DEM data."""
    if not config.USE_NIR:
        naip_dataset = RGBOnlyNAIP(config.KC_IMAGE_ROOT)

    else:
        naip_dataset = NAIP(config.KC_IMAGE_ROOT)

    # Load Difference DEM if specified
    if config.USE_DIFFDEM:
        dem = KaneDEM(config.KC_DEM_ROOT, config, use_difference=True)
        naip_dataset = naip_dataset & dem

        print("difference dem loaded")

    # Load Base DEM if specified
    elif config.USE_BASEDEM:
        dem = KaneDEM(config.KC_DEM_ROOT, config)
        naip_dataset = naip_dataset & dem

        print("base dem loaded")

    # Load appropriate label dataset
    if config.USE_RIVERDATASET:
        rd_shape_path = Path(config.KC_SHAPE_ROOT) / config.RD_SHAPE_FILE
        label_dataset = RiverDataset(
            patch_size=config.PATCH_SIZE,
            crs=naip_dataset.crs,
            res=naip_dataset.res,
            path=rd_shape_path,
            kc=True,
        )
        print("river dataset loaded")
    else:
        # Default: use Kane County dataset
        kc_shape_path = Path(config.KC_SHAPE_ROOT) / config.KC_SHAPE_FILENAME
        label_dataset = KaneCounty(
            layers=config.KC_LAYER,
            labels=config.KC_LABELS,
            patch_size=config.PATCH_SIZE,
            dest_crs=naip_dataset.crs,
            res=naip_dataset.res,
            path=kc_shape_path,
            balance_classes=False,
        )
        print("kc dataset loaded")
    return naip_dataset, label_dataset


def build_dataloaders(images, labels, split_rate, config):
    """Randomly split and load data to be the test and train sets

    Args:
    images (Dataset): The images dataset
    labels (Dataset): The labels dataset
    split_rate (float): Ratio of data in training set (e.g., 0.8 for 80%)
    config (module): Configuration object containing PATCH_SIZE, BATCH_SIZE, NUM_WORKERS.

    Returns tuple (train dataloader, test dataloader)
    """
    # record generator seed
    seed = random.SystemRandom().randint(0, sys.maxsize)
    logging.info("Dataset random split seed: %d", seed)
    generator = torch.Generator().manual_seed(seed)
    logging.info(f"Initial NAIP dataset size: {len(images)}")

    # split the dataset
    train_portion, test_portion = random_bbox_assignment(
        images, [split_rate, 1 - split_rate], generator
    )
    train_dataset = train_portion & labels
    test_dataset = test_portion & labels

    logging.info("After intersection:")
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")

    # Check if datasets are empty before creating samplers
    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty after intersection!")
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty after intersection!")

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

    # Log sampler lengths
    logging.info(f"Train sampler length: {len(train_sampler)}")
    logging.info(f"Test sampler length: {len(test_sampler)}")

    # Add prefetching and persistent workers
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        collate_fn=stack_samples,
        num_workers=config.NUM_WORKERS,
        prefetch_factor=2,  # Add prefetching
        persistent_workers=True
        if config.NUM_WORKERS > 0
        else False,  # Keep workers alive between epochs
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=test_sampler,
        collate_fn=stack_samples,
        num_workers=config.NUM_WORKERS,
        prefetch_factor=2,  # Add prefetching
        persistent_workers=True
        if config.NUM_WORKERS > 0
        else False,  # Keep workers alive between epochs
    )

    logging.info(f"Train dataloader length: {len(train_dataloader)}")
    logging.info(f"Test dataloader length: {len(test_dataloader)}")

    return train_dataloader, test_dataloader


def regularization_loss(model, reg_type, reg_weight):
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
    return reg_weight * reg_loss


def compute_loss(
        model,
        mask,
        y,
        loss_fn,
        reg_type: str, 
        reg_weight: float,
):
    """Compute the total loss optionally the regularization loss.

    Args:
        model: The PyTorch model for which to compute the loss.
        mask: The input mask tensor.
        y: The target tensor.
        loss_fn: The loss function to use for computing the base loss.
        reg_type: The type of regularization, either "l1" or "l2".
        reg_weight: The weight or strength of the regularization term.

    Returns:
    - torch.Tensor: The total loss as a PyTorch tensor.
    """
    base_loss = loss_fn(mask, y)
    if reg_type is not None:
        reg_loss = regularization_loss(model, reg_type, reg_weight)
        base_loss += reg_loss
    return base_loss


def create_model(
    config,
    num_classes,
    device="cpu",
    debug=False,
):
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
    # calculate input channels based on config
    in_channels = 3  # base RGB channels
    if config.USE_NIR:
        in_channels += 1  # add NIR channel
    if config.USE_DIFFDEM or config.USE_BASEDEM:
        in_channels += 1  # add DEM channel

    model_config = {
        "model": config.MODEL,
        "backbone": config.BACKBONE,
        "num_classes": num_classes,
        "weights": config.WEIGHTS,
        "dropout": config.DROPOUT,
        "in_channels": in_channels,
    }

    model = SegmentationModel(model_config).model.to(device)
    if not debug:
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
        num_classes=num_classes,
        ignore_index=config.IGNORE_INDEX,
        average="micro",
    ).to(device)
    test_jaccard = MulticlassJaccardIndex(
        num_classes=num_classes,
        ignore_index=config.IGNORE_INDEX,
        average="micro",
    ).to(device)
    jaccard_per_class = MulticlassJaccardIndex(
        num_classes=num_classes,
        ignore_index=config.IGNORE_INDEX,
        average=None,
    ).to(device)

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
    dataset,
    spatial_augs,
    color_augs,
    dem_augs,
    spatial_aug_mode,
    color_aug_mode,
    dem_aug_mode,
):
    """Apply augmentations to the image and mask."""
    x_og, y_og = dataset
    aug_config = (
        spatial_augs,
        color_augs,
        dem_augs,
        spatial_aug_mode,
        color_aug_mode,
        dem_aug_mode,
    )
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

        # Add DEM if enabled
        if config.USE_DIFFDEM:
            plot_tensors.update(
                {
                    "Difference DEM": x[i][3, :, :].cpu() / 255.0,
                    "Augmented Difference DEM": x_aug_denorm[i][3, :, :].cpu() / 255.0,
                }
            )

        elif config.USE_BASEDEM:
            plot_tensors.update(
                {
                    "Base DEM": x[i][3, :, :].cpu() / 255.0,
                    "Augmented Base DEM": x_aug_denorm[i][3, :, :].cpu() / 255.0,
                }
            )

        sample_fname = Path(save_dir) / f"train_sample-{epoch}.{i}.png"
        plot_from_tensors(
            plot_tensors,
            sample_fname,
            labels.colors,
            labels.labels_inverse,
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


def log_per_class_iou_tensor(
    writer: SummaryWriter,
    class_labels,
    per_class_iou_tensor: torch.Tensor,
    prefix: str,
    epoch: int,
):
    """Logs per-class IoU values to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter.
        class_labels: Iterable of class labels
        per_class_iou_tensor: Tensor containing per-class IoU values.
        prefix: String prefix for the metric key (e.g., "IoU/train" or "IoU/test").
        epoch: Current epoch number.
    """
    class_labels = {
        label_id: label_name for label_name, label_id in class_labels
    }  # kc.labels.items()
    for i in sorted(class_labels.keys()):
        writer.add_scalar(
            f"{prefix}/{class_labels[i]}", per_class_iou_tensor[i].item(), epoch
        )


def normalize_channels(x):
    """Normalize all channels in a batch efficiently.

    Args:
        x: Input tensor of shape [batch, channels, height, width]

    Returns:
        Normalized tensor with values between 0 and 1
    """
    # find min and max per channel across batch
    min_vals = (
        x.min(dim=0, keepdim=True)[0]
        .min(dim=2, keepdim=True)[0]
        .min(dim=3, keepdim=True)[0]
    )
    max_vals = (
        x.max(dim=0, keepdim=True)[0]
        .max(dim=2, keepdim=True)[0]
        .max(dim=3, keepdim=True)[0]
    )
    return (x - min_vals) / (
        max_vals - min_vals + 1e-8
    )  # add epsilon to avoid division by zero


def train_setup(
    sample: defaultdict[str, Any],
    train_config,
    aug_config,
    model,
) -> tuple[torch.Tensor]:
    """Setup for training: sends images to device and applies augmentations."""
    epoch, batch, train_images_root = train_config
    (
        spatial_augs,
        color_augs,
        dem_augs,
        spatial_aug_mode,
        color_aug_mode,
        dem_aug_mode,
    ) = aug_config

    samp_image = sample["image"]
    samp_mask = sample["mask"]

    # Add extra channels to image if necessary
    samp_image = add_extra_channels(samp_image, model)

    # Send image and mask to device; convert mask to float tensor for augmentation
    x = samp_image.to(MODEL_DEVICE)
    y = samp_mask.type(torch.float32).to(MODEL_DEVICE)

    if batch == 0:  # Log stats for first batch only
        log_channel_stats(x, "raw input", logging.getLogger())

    x = normalize_channels(x)

    if batch == 0:  # Log stats for first batch only
        log_channel_stats(x, "scaled input", logging.getLogger())

    # Extend mean/std dynamically if needed
    data_mean = config.DATASET_MEAN  # ImageNet mean
    data_std = config.DATASET_STD  # ImageNet std
    if len(data_mean) < model.in_channels:
        missing_channels = model.in_channels - len(data_mean)
        computed_means = torch.mean(x[:, len(data_mean) :], dim=[0, 2, 3]).tolist()
        data_mean = data_mean + computed_means[:missing_channels]
    if len(data_std) < model.in_channels:
        missing_channels = model.in_channels - len(data_std)
        computed_stds = torch.std(x[:, len(data_std) :], dim=[0, 2, 3]).tolist()
        data_std = data_std + computed_stds[:missing_channels]

    # Normalize using ImageNet statistics
    normalize = K.Normalize(mean=data_mean, std=data_std)
    x_norm = normalize(x)

    if batch == 0:  # Log stats for first batch only
        log_channel_stats(x_norm, "normalized input", logging.getLogger())

    img_data = (x_norm, y)
    # Apply augmentations
    x_aug, y_squeezed = apply_augmentations(
        img_data,
        spatial_augs,
        color_augs,
        dem_augs,
        spatial_aug_mode,
        color_aug_mode,
        dem_aug_mode,
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
            - dem_augs: The sequence of DEM augmentations.
            - spatial_aug_mode: The mode for spatial augmentations.
            - color_aug_mode: The mode for color augmentations.
            - dem_aug_mode: The mode for DEM augmentations.
        writer: The TensorBoard writer for logging training metrics.
        args: Additional arguments for debugging or special training conditions.
        args: Additional arguments for debugging or special training conditions.
        wandb_tune: whether tuning with wandb
    """
    # Add timing for dataloader initialization
    dataloader_start = time.time()

    loss_fn, jaccard, optimizer, epoch, train_images_root, num_classes = train_config
    (
        spatial_augs,
        color_augs,
        dem_augs,
        spatial_aug_mode,
        color_aug_mode,
        dem_aug_mode,
    ) = aug_config

    # start timing for this epoch
    epoch_start_time = time.time()

    # Log dataloader initialization time
    dataloader_time = epoch_start_time - dataloader_start
    logging.info(f"Dataloader initialization took {dataloader_time:.2f}s")

    num_batches = len(dataloader)

    class_area_counts = {i: 0 for i in range(num_classes)}
    total_pixels = 0

    model.train()
    jaccard.reset()

    train_jaccard_per_class = MulticlassJaccardIndex(
        num_classes=num_classes,
        ignore_index=config.IGNORE_INDEX,
        average=None,
    ).to(MODEL_DEVICE)

    train_loss = 0
    iteration_start_time = time.time()
    for batch, sample in enumerate(dataloader):
        train_config = (epoch, batch, train_images_root)
        aug_config = (
            spatial_augs,
            color_augs,
            dem_augs,
            spatial_aug_mode,
            color_aug_mode,
            dem_aug_mode,
        )
        x, y = train_setup(
            sample,
            train_config,
            aug_config,
            model,
        )
        x = x.to(MODEL_DEVICE)
        y = y.to(MODEL_DEVICE)

        mask_int = y.long()
        batch_pixels = mask_int.numel()
        total_pixels += batch_pixels
        for i in range(num_classes):
            class_area_counts[i] += (mask_int == i).sum().item()

        # Break after the first batch in debug mode
        if args.debug and batch == 0:
            print("Debug mode: Exiting training loop after first batch.")
            break

        # compute prediction error
        outputs = model(x)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = compute_loss(
            model,
            outputs,
            y,
            loss_fn,
            config.REGULARIZATION_TYPE,
            config.REGULARIZATION_WEIGHT,
        )

        # update jaccard index
        preds = outputs.argmax(dim=1)
        jaccard.update(preds, y)
        train_jaccard_per_class.update(preds, y)

        # backpropagation
        loss.backward()

        # Gradient clipping
        if config.GRADIENT_CLIPPING:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_VALUE)

        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        if batch % 100 == 0:
            current_time = time.time()
            iteration_time = current_time - iteration_start_time
            loss, current = loss.item(), (batch + 1)
            avg_time_per_iteration = (
                iteration_time / 100 if batch > 0 else iteration_time
            )
            logging.info(
                f"loss: {loss:7.7f}  [{current:5d}/{num_batches:5d}] time: {avg_time_per_iteration:.1f}s/it"
            )
            iteration_start_time = current_time

    # calculate epoch duration
    epoch_duration = time.time() - epoch_start_time

    train_loss /= num_batches
    final_jaccard = jaccard.compute()

    class_area_percentages = {
        i: (class_area_counts[i] / total_pixels * 100) for i in range(num_classes)
    }
    logging.info(f"Per-class area percentages for epoch {epoch}:")
    for class_id, percentage in class_area_percentages.items():
        class_name = labels.labels.get(class_id, f"Class {class_id}")
        logging.info(f"  {class_name}: {percentage:.2f}%")

    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("IoU/train", final_jaccard, epoch)
    writer.add_scalar(
        "time/epoch_duration", epoch_duration, epoch
    )  # log epoch duration

    final_train_iou = train_jaccard_per_class.compute()
    log_per_class_iou_tensor(
        writer, labels.labels.items(), final_train_iou, "IoU/train", epoch
    )

    logging.info("Train Jaccard index: %.4f", final_jaccard)
    logging.info(
        f"Epoch {epoch} completed in {epoch_duration:.0f} seconds"
    )  # log epoch duration

    if wandb_tune:
        wandb.log(
            {
                "train_loss": train_loss,
                "train_jaccard": final_jaccard.item(),
                "epoch": epoch,
                "batch": batch,
                "epoch_duration": epoch_duration,  # log epoch duration to wandb
            }
        )

    return final_jaccard


def test(
    dataloader: DataLoader,
    model: Module,
    test_config,
    writer,
    wandb_tune: bool,
    labels,
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
        labels: The labels for the dataset.
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

            # Scale all channels to 0 to 1 using vectorized operation
            x = normalize_channels(x)

            if batch == 0:  # Log stats for first batch only
                log_channel_stats(x, "test scaled input", logging.getLogger())

            # Extend mean/std dynamically if needed
            data_mean = config.DATASET_MEAN
            data_std = config.DATASET_STD
            if len(data_mean) < model.in_channels:
                missing_channels = model.in_channels - len(data_mean)
                computed_means = torch.mean(
                    x[:, len(data_mean) :], dim=[0, 2, 3]
                ).tolist()
                data_mean = data_mean + computed_means[:missing_channels]
            if len(data_std) < model.in_channels:
                missing_channels = model.in_channels - len(data_std)
                computed_stds = torch.std(x[:, len(data_std) :], dim=[0, 2, 3]).tolist()
                data_std = data_std + computed_stds[:missing_channels]

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
                    # Add DEM if enabled
                    if config.USE_DIFFDEM:
                        plot_tensors["Difference DEM"] = (
                            x_denorm[i][3, :, :].cpu().clip(0, 1)
                        )
                    if config.USE_BASEDEM:
                        plot_tensors["Base DEM"] = x_denorm[i][3, :, :].cpu().clip(0, 1)

                    ground_truth = samp_mask[i]
                    label_ids = find_labels_in_ground_truth(ground_truth)

                    for label_id in label_ids:
                        label_name = labels.labels_inverse.get(label_id, "UNKNOWN")
                        save_dir = Path(epoch_dir) / label_name.replace("/", "-")
                        if not Path.exists(save_dir):
                            Path.mkdir(save_dir)
                        sample_fname = (
                            Path(save_dir) / f"test_sample-{epoch}.{batch}.{i}.png"
                        )
                        plot_from_tensors(
                            plot_tensors,
                            sample_fname,
                            labels.colors,
                            labels.labels_inverse,
                            sample["bbox"][i],
                        )
    test_loss /= num_batches
    final_jaccard = jaccard.compute()
    final_jaccard_per_class = jaccard_per_class.compute()
    writer.add_scalar("loss/test", test_loss, epoch)
    writer.add_scalar("IoU/test", final_jaccard, epoch)

    log_per_class_iou_tensor(
        writer, labels.labels.items(), final_jaccard_per_class, "IoU/test", epoch
    )

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
    for label_name, label_id in labels.labels.items():
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
    labels,
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
                - dem_augs: DEM augmentations applied during training.
        path_config: A tuple containing:
                - out_root: Root directory for saving the trained model.
                - train_images_root: Root directory for training images.
                - test_image_root: Root directory for test images.
        writer: The writer object for logging training progress.
        wandb_tune: Whether running hyperparameter tuning with wandb.
        args: Additional arguments for debugging or special training conditions.
        epoch: The configuration for the number of epochs.
        labels: The labels for the dataset.

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
        dem_augs,
        config.SPATIAL_AUG_MODE,
        config.COLOR_AUG_MODE,
        config.DEM_AUG_MODE,
    ) = aug_config

    # How much the loss needs to drop to reset a plateau
    threshold = config.THRESHOLD

    # How many epochs loss needs to plateau before terminating
    patience = config.PATIENCE

    # Beginning loss
    best_loss = None

    # How long it's been plateauing
    plateau_count = 0

    # reducing number of epoch in debugging or hyperparameter tuning
    if args.debug:
        epoch = 1
    elif wandb_tune:
        epoch = 10
    else:
        epoch = config.EPOCHS

    # track total training time
    total_training_start = time.time()

    for t in range(epoch):
        if t == 0:
            test_config = (
                loss_fn,
                test_jaccard,
                t,
                plateau_count,
                test_image_root,
                writer,
                len(labels.labels),
                jaccard_per_class,
            )
            test_loss, t_jaccard = test(
                test_dataloader,
                model,
                test_config,
                writer,
                args,
                labels,
            )
            print(f"untrained loss {test_loss:.3f}, jaccard {t_jaccard:.3f}")

        logging.info(f"Epoch {t + 1}\n-------------------------------")
        train_config = (
            loss_fn,
            train_jaccard,
            optimizer,
            t + 1,
            train_images_root,
            len(labels.labels),
        )
        aug_config = (
            spatial_augs,
            color_augs,
            dem_augs,
            config.SPATIAL_AUG_MODE,
            config.COLOR_AUG_MODE,
            config.DEM_AUG_MODE,
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
            len(labels),
            jaccard_per_class,
        )
        test_loss, t_jaccard = test(
            test_dataloader,
            model,
            test_config,
            writer,
            wandb_tune,
            labels,
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

    # calculate and log total training time
    total_training_time = time.time() - total_training_start
    logging.info(f"Total training completed in {total_training_time:.2f} seconds")
    writer.add_scalar("time/total_training_time", total_training_time, 0)

    if wandb_tune:
        wandb.log({"total_training_time": total_training_time})

    print("Done!")

    torch.save(model.state_dict(), Path(out_root) / "model.pth")
    logging.info("Saved PyTorch Model State to %s", out_root)

    return epoch_jaccard, t_jaccard


def one_trial(exp_n, num, wandb_tune, images, labels, split_rate, args):
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
    ) = writer_prep(exp_n, num, wandb_tune, config)
    # Set 'epoch' based on debug mode
    if args.debug:
        epoch = 1
    else:
        epoch = config.EPOCHS
    # randomly splitting the data at every trial
    train_dataloader, test_dataloader = build_dataloaders(
        images, labels, split_rate, config
    )
    (
        model,
        loss_fn,
        train_jaccard,
        test_jaccard,
        jaccard_per_class,
        optimizer,
    ) = create_model(
        config,
        len(labels.labels),
        device=MODEL_DEVICE,
        debug=args.debug,
    )
    spatial_augs, color_augs, dem_augs = create_augmentation_pipelines(
        config,
        config.SPATIAL_AUG_INDICES,
        config.IMAGE_AUG_INDICES,
        config.DEM_AUG_INDICES,
    )
    print("In one_trial(), DEM augs are:", config.DEM_AUG_INDICES)
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
        dem_augs,
        config.SPATIAL_AUG_MODE,
        config.COLOR_AUG_MODE,
        config.DEM_AUG_MODE,
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
        labels,
    )
    writer.close()
    logger.handlers.clear()
    return train_iou, test_iou


def run_trials(trial_id, gpu_id, args_dict, split, exp_name, wandb_tune, num_trials):
    """Running training for multiple trials"""
    torch.cuda.set_device(gpu_id)

    global config, images, labels
    config = importlib.import_module(args_dict["config"])
    print("Multiprocessing:", config.MULTIPROCESSING)
    images, labels = initialize_dataset(config)

    args = Namespace(**args_dict)

    if wandb_tune:
        wandb.init(project="CMAP")
        print("wandb taken over config")
    else:
        # Initialize wandb with default configuration but disable logging
        wandb.init(project="CMAP", config=config, mode="disabled")

    train_ious = []
    test_ious = []

    if config.MULTIPROCESSING:
        train_iou, test_iou = one_trial(
            exp_name, trial_id, wandb_tune, images, labels, split, args
        )
        train_ious.append(round(float(train_iou), 3))
        test_ious.append(round(float(test_iou), 3))
    else:
        for num in range(num_trials):
            train_iou, test_iou = one_trial(
                exp_name, num, wandb_tune, images, labels, split, args
            )
            train_ious.append(round(float(train_iou), 3))
            test_ious.append(round(float(test_iou), 3))

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


if __name__ == "__main__":
    # Check GPU availability; if GPU available, run on compute node, else exit
    mp.set_start_method("spawn", force=True)

    check_gpu_availability()
    num_gpus = torch.cuda.device_count()
    print("The number of GPUs available is:", num_gpus)

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

    args = parser.parse_args()

    config = importlib.import_module(args.config)

    args_dict = vars(args)  # make argparse Namespace pickle-safe
    print(args_dict)
    exp_name, split, wandb_tune, num_trials = arg_parsing(args)
    num_trials = int(num_trials)

    logging.info("Using %s device", MODEL_DEVICE)

    if config.MULTIPROCESSING:
        processes = []
        for trial_id in range(num_trials):
            gpu_id = trial_id % num_gpus
            p = mp.Process(
                target=run_trials,
                args=(
                    trial_id,
                    gpu_id,
                    args_dict,
                    split,
                    exp_name,
                    wandb_tune,
                    num_trials,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        run_trials(0, 0, args_dict, split, exp_name, wandb_tune, num_trials)
