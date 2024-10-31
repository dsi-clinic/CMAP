"""
To run: from repo directory (2024-winter-cmap)
> python train.py configs.<config> [--experiment_name <name>]
    [--split <split>] [--tune]  [--num_trials <num>]
"""

import argparse
import datetime
import importlib.util
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, DefaultDict, Tuple

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
from data.sampler import BalancedGridGeoSampler, BalancedRandomBatchGeoSampler
from model import SegmentationModel
from utils.plot import find_labels_in_ground_truth, plot_from_tensors
from utils.transforms import apply_augs, create_augmentation_pipelines

MODEL_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

def initialize_wandb():
    """
    Initializes wandb with sweep configuration.
    """
    wandb.init(config = wandb.config)
    return config

def arg_parsing(argument):
    """
    Parsing arguments passed in from command line
    """
    # if no experiment name provided, set to timestamp
    exp_name_arg = argument.experiment_name
    if exp_name_arg is None:
        exp_name_arg = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'

    split_arg = float(int(argument.split) / 100)
    # tuning with wandb
    wandb_tune_arg = argument.tune
    num_trials_arg = int(argument.num_trials)

    return exp_name_arg, split_arg, wandb_tune_arg, num_trials_arg


def writer_prep(exp_n, trial_num, wandb_t):
    """
    Preparing writers and logging for each training trial
    Args:
        trial_num: current trial number
    """
    # set output path and exit run if path already exists
    exp_trial_name = f"{exp_n}_trial_{trial_num}"
    out_root = os.path.join(config.OUTPUT_ROOT, exp_trial_name)
    if wandb_t:
        os.makedirs(out_root, exist_ok=True)
    else:
        os.makedirs(out_root, exist_ok=False)

    # create directory for output images
    train_images_root = os.path.join(out_root, "train-images")
    test_images_root = os.path.join(out_root, "test-images")

    try:
        os.mkdir(train_images_root)
        os.mkdir(test_images_root)

    except FileExistsError:
        shutil.rmtree(train_images_root)
        shutil.rmtree(test_images_root)
        os.mkdir(train_images_root)
        os.mkdir(test_images_root)

    # open tensorboard writer
    writer = SummaryWriter(out_root)

    # copy training script and config to output directory
    shutil.copy(Path(__file__).resolve(), out_root)
    shutil.copy(Path(config.__file__).resolve(), out_root)

    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_filename = os.path.join(out_root, "training_log.txt")
    file_handler = logging.FileHandler(log_filename)
    stream_handler = logging.StreamHandler(sys.stdout)

    # log format
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return train_images_root, test_images_root, out_root, writer, logger


def initialize_dataset():
    """
    Initialize the dataset by loading NAIP and KaneCounty data.
    This function loads NAIP (National Agriculture Imagery Program)
    data and KaneCounty shapefile data. Optionally, if DEM
    (Digital Elevation Model) data is provided, it is also loaded
    and merged with NAIP data.

    Args:
        config: settings in the configuration file

    Returns:
        tuple: A tuple containing the loaded NAIP and KaneCounty
            datasets.
            The first element is the NAIP dataset, and the
            second element is the KaneCounty dataset.
    """
    naip_dataset = NAIP(config.KC_IMAGE_ROOT)

    shape_path = os.path.join(config.KC_SHAPE_ROOT, config.KC_SHAPE_FILENAME)
    dataset_config = (
        config.KC_LAYER,
        config.KC_LABELS,
        config.PATCH_SIZE,
        naip_dataset.crs,
        naip_dataset.res,
    )
    kc_dataset = KaneCounty(shape_path, dataset_config)

    if config.KC_DEM_ROOT is not None:
        dem = KaneDEM(config.KC_DEM_ROOT)
        naip_dataset = naip_dataset & dem
        print("naip and dem loaded")

    return naip_dataset, kc_dataset


def build_dataset(naip_set, split_rate):
    """
    Randomly split and load data to be the test and train sets
    Returns train dataloader and test dataloader
    """
    # record generator seed
    seed = random.randint(0, sys.maxsize)
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
            "patch_size": wandb.config.patch_size,
            "batch_size": wandb.config.batch_size,
        }
    )
    test_sampler = BalancedGridGeoSampler(
        config={
            "dataset": test_dataset,
            "size": wandb.config.patch_size,
            "stride": wandb.config.patch_size,
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
    """
    Calculate the regularization loss for the model parameters.
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
    """
    Compute the total loss optionally the regularization loss.

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
    """
    Setting up training model, loss function and measuring metrics

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
        model.parameters(), lr=wandb.config.learning_rate, weight_decay=config.WEIGHT_DECAY
    )

    return (
        model,
        loss_fn,
        train_jaccard,
        test_jaccard,
        jaccard_per_class,
        optimizer,
    )


def copy_first_entry(a_list: list) -> list:
    """
    Copies the first entry in a list and appends it to the end.

    Args:
        a_list: The list to modify

    Returns:
        list: The modified list
    """
    first_entry = a_list[0]

    # Append the copy to the end
    a_list.append(first_entry)

    return a_list


def normalize_func(model):
    """
    Create normalization functions for input data to a given model.

    This function generates normalization functions based on the mean
    and standard deviation specified in the configuration. If the
    number of channels in the model input does not match the length of
    the mean and standard deviation lists, it replicates the first entry
    of each list to match the number of input channels.

    Args:
        model: The model for which the normalization functions are created.

    Returns:
        tuple: A tuple containing two normalization functions.
               The first function normalizes input data using the specified
               mean and standard deviation.
               The second function scales input data to a range between 0 and 255.
    """
    data_mean = config.DATASET_MEAN
    data_std = config.DATASET_STD
    # add copies of first entry to DATASET_MEAN and DATASET_STD
    # to match data in_channels
    if len(data_mean) != model.in_channels:
        for _ in range(model.in_channels - len(data_mean)):
            data_mean = copy_first_entry(data_mean)
            data_std = copy_first_entry(data_std)

    scale_mean = torch.tensor(0.0)
    scale_std = torch.tensor(255.0)
    normalize = K.Normalize(mean=data_mean, std=data_std)
    scale = K.Normalize(mean=scale_mean, std=scale_std)
    return normalize, scale


def add_extra_channel(
    image_tensor: torch.Tensor, source_channel: int = 0
) -> torch.Tensor:
    """
    Adds an additional channel to an image by copying an existing channel.

    Args:
        image_tensor : A tensor containing image data. Expected shape is
            (batch, channels, h, w)
        source_channel : The index of the channel to be copied

    Returns:
        torch.Tensor: A modified tensor with added channels
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


def normalize_and_scale(sample_image, model):
    """
    Normalize and scale the sample image.
    """
    normalize, scale = normalize_func(model)
    scaled_image = scale(sample_image)
    return scaled_image, normalize


def add_extra_channels(image, model):
    """
    Add extra channels to the image if necessary.
    """
    while image.size(1) < model.in_channels:
        image = add_extra_channel(image)
    return image


def apply_augmentations(
    dataset, spatial_augs, color_augs, spatial_aug_mode, color_aug_mode
):
    """
    Apply augmentations to the image and mask.
    """
    x_og, y_og = dataset
    aug_config = (spatial_augs, color_augs, spatial_aug_mode, color_aug_mode)
    x_aug, y_aug = apply_augs(aug_config, x_og, y_og)
    y_aug = y_aug.type(torch.int64)  # Convert mask to int64 for loss function
    y_squeezed = y_aug.squeeze()  # Remove channel dim from mask
    return x_aug, y_squeezed


def save_training_images(
    epoch, train_images_root, x, samp_mask, x_aug, y_aug, sample
):
    """
    Save training sample images.
    """
    save_dir = os.path.join(
        train_images_root,
        f"epoch-{epoch}",
    )
    os.makedirs(save_dir, exist_ok=True)

    for i in range(config.BATCH_SIZE):
        plot_tensors = {
            "RGB Image": x[i].cpu(),
            "Mask": samp_mask[i],
            "Augmented_RGBImage": x_aug[i].cpu(),
            "Augmented_Mask": y_aug[i].cpu(),
        }
        sample_fname = os.path.join(save_dir, f"train_sample-{epoch}.{i}.png")
        plot_from_tensors(
            plot_tensors,
            sample_fname,
            kc.colors,
            kc.labels_inverse,
            sample["bbox"][i],
        )


def train_setup(
    sample: DefaultDict[str, Any],
    train_config,
    aug_config,
    model,
) -> Tuple[torch.Tensor]:
    """
    Sets up for the training step by sending images and masks to device,
    applying augmentations, and saving training sample images.

    Args:
        sample: A dataloader sample containing image, mask, and bbox data.
        train_config: a tuple of
            - epoch: The current epoch.
            - batch: The current batch.
            - train_images_root: The root path for saving training sample images.
        aug_config: a tuple of
            - spatial_aug_mode: The mode for spatial augmentations.
            - color_aug_mode: The mode for color augmentations.
            - spatial_augs: The sequence of spatial augmentations.
            - color_augs: The sequence of color augmentations.
        model: The PyTorch model instance.

    Returns:
        A tuple of augmented image and mask tensors to be used in the train step
    """
    epoch, batch, train_images_root = train_config
    spatial_aug_mode, color_aug_mode, spatial_augs, color_augs = aug_config

    samp_image = sample["image"]
    samp_mask = sample["mask"]

    # Add extra channels to image if necessary
    samp_image = add_extra_channels(samp_image, model)

    # Send image and mask to device; convert mask to float tensor for augmentation
    x = samp_image.to(MODEL_DEVICE)
    y = samp_mask.type(torch.float32).to(MODEL_DEVICE)

    # Normalize and scale image
    x_scaled, normalize = normalize_and_scale(x, model)

    img_data = (x_scaled, y)
    # Apply augmentations
    x_aug, y_squeezed = apply_augmentations(
        img_data, spatial_augs, color_augs, spatial_aug_mode, color_aug_mode
    )

    # Save training sample images if first batch
    if batch == 0:
        save_training_images(
            epoch,
            train_images_root,
            x,
            samp_mask,
            x_aug,
            y_squeezed,
            sample,
        )

    return normalize(x_aug), y_squeezed


def train_epoch(
    dataloader,
    model,
    train_config,
    aug_config,
    writer,
) -> None:
    """
    Executes a training step for the model

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

        # compute prediction error
        outputs = model(x)
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
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.CLIP_VALUE
            )

        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1)
            logging.info("loss: %7.7f  [%5d/%5d]", loss, current, num_batches)
    train_loss /= num_batches
    final_jaccard = jaccard.compute()

    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("IoU/train", final_jaccard, epoch)
    logging.info("Train Jaccard index: %.4f", final_jaccard)
    return final_jaccard


def test(
    dataloader: DataLoader,
    model: Module,
    test_config,
    writer,
) -> float:
    """
    Executes a testing step for the model and saves sample output images.

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
                for _ in range(model.in_channels - samp_image.size(1)):
                    samp_image = add_extra_channel(samp_image)
            x = samp_image.to(MODEL_DEVICE)
            normalize, scale = normalize_func(model)
            x_scaled = scale(x)
            x = normalize(x_scaled)
            y = samp_mask.to(MODEL_DEVICE)
            if y.size(0) == 1:
                y_squeezed = y
            else:
                y_squeezed = y.squeeze()

            # compute prediction error
            outputs = model(x)
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
                plateau_count == config.PATIENCE - 1 and batch < 10
            ):
                epoch_dir = os.path.join(test_image_root, f"epoch-{epoch}")
                if not os.path.exists(epoch_dir):
                    os.mkdir(epoch_dir)
                for i in range(config.BATCH_SIZE):
                    plot_tensors = {
                        "RGB Image": x_scaled[i].cpu(),
                        "ground_truth": samp_mask[i],
                        "prediction": preds[i].cpu(),
                    }
                    ground_truth = samp_mask[i]
                    label_ids = find_labels_in_ground_truth(ground_truth)

                    for label_id in label_ids:
                        label_name = kc.labels_inverse.get(label_id, "UNKNOWN")
                        save_dir = os.path.join(epoch_dir, label_name)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        sample_fname = os.path.join(
                            save_dir, f"test_sample-{epoch}.{batch}.{i}.png"
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
    logging.info(
        "\nTest error: \n Jaccard index: %4f, \nTest avg loss: %4f \n",
        final_jaccard,
        test_loss,
    )

    # Access the labels and their names
    _labels = {}
    for label_name, label_id in kc.labels.items():
        _labels[label_id] = label_name
        if len(_labels) == num_classes:
            break

    for i, label_name in _labels.items():
        logging.info(
            "IoU for %s: %f \n", label_name, final_jaccard_per_class[i]
        )

    # Now returns test_loss such that it can be compared against previous losses
    return test_loss, final_jaccard


def train(
    model: Module,
    train_test_config,
    aug_config,
    path_config: Tuple[str, str, str],
    writer: SummaryWriter,
    wandb_t,
) -> Tuple[float, float]:
    """
    Train a deep learning model using the specified configuration and parameters.

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

    # reducing number of epoch in hyperparameter tuning
    if wandb_t:
        epoch_config = 10

    else:
        epoch_config = config.EPOCHS

    for t in range(epoch_config):
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
            )
            print(f"untrained loss {test_loss:.3f}, jaccard {t_jaccard:.3f}")

        logging.info("Epoch %d\n-------------------------------", t + 1)
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
                    "Loss Plateau: %d epochs, reached patience of %d",
                    t,
                    patience,
                )
                break

    print("Done!")

    torch.save(model.state_dict(), os.path.join(out_root, "model.pth"))
    logging.info("Saved PyTorch Model State to %s", out_root)

    return epoch_jaccard, t_jaccard


def one_trial(exp_n, num, wandb_t, naip_set, split_rate):
    """
    Runing a single trial of training
    Input:
        exp_n: experiment name
        num: current number of trial
        wandb_t: whether tuning with wandb
    """
    (
        train_images_root,
        test_image_root,
        out_root,
        writer,
        logger,
    ) = writer_prep(exp_n, num, wandb_t)
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
    )
    writer.close()
    logger.handlers.clear()
    return train_iou, test_iou


if __name__ == "__main__":
    # import config and experiment name from runtime args
    parser = argparse.ArgumentParser(
        description="Train a segmentation model to predict stormwater storage "
        + "and green infrastructure."
    )
    parser.add_argument(
        "config", type=str, help="Path to the configuration file"
    )
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

    args = parser.parse_args()
    config = importlib.import_module(args.config)
    exp_name, split, wandb_tune, num_trials = arg_parsing(args)

    logging.info("Using %s device", MODEL_DEVICE)

    naip, kc = initialize_dataset()

    def run_trials():
        """
        Running training for multiple trials
        """

        if wandb_tune:
            run = wandb.init(project="cmap_train")
            vars(args).update(run.config)
            print("wandb taken over config")

        train_ious = []
        test_ious = []

        for num in range(num_trials):
            train_iou, test_iou = one_trial(
                exp_name, num, wandb_tune, naip, split
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
            run.log({"average_test_jaccard_index": test_average})
            wandb.finish()

    run_trials()

if __name__ == "__main__":
    # Ensure wandb is logged in
    wandb.login()

    # Define the sweep configuration
    sweep_config = 'sweep_config.yml'

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="CMAP") 

    # Run the sweep agent
    wandb.agent(sweep_id, function=run_trials, count=2)