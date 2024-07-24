"""
This module provides functions for image augmentation and processing.

Functions:
- separate_channels(image, rgb_indices): Separates specified RGB channels
from other channels in an image tensor.
- combine_channels(rgb, other_channels, rgb_mask, original_shape):
Recombines the RGB and other channels after augmentation.
- create_augmentation_pipelines(config, spatial_aug_indices, color_aug_indices):
Creates lists of spatial and color augmentations based on provided indices
and parameters.
- apply_augs(spatial_transforms, color_transforms, image, mask, spatial_mode,
color_mode, rgb_channels=None): Applies spatial and color augmentations to an image
and its corresponding mask.

Parameters:
- image (torch.Tensor): The input image tensor.
- mask (torch.Tensor): The corresponding mask tensor.
- spatial_aug_indices (list): Indices to select spatial augmentations.
- color_aug_indices (list): Indices to select color augmentations for RGB channels.
- spatial_mode (str): Augmentation mode for spatial augmentations -
'random' for random augmentations or 'all' for all.
- color_mode (str): Augmentation mode for color augmentations -
'random' for random augmentations or 'all' for all.
- rgb_channels (list): Indices of RGB channels in the image tensor.
"""

import random

import kornia.augmentation as K
import torch


def separate_channels(image, rgb_indices):
    """Separate specified RGB channels from other channels."""
    rgb_mask = torch.zeros(
        image.shape[1], dtype=torch.bool, device=image.device
    )
    rgb_mask[rgb_indices] = True

    rgb = image[:, rgb_mask, :, :]  # Extract RGB channels
    other_channels = image[:, ~rgb_mask, :, :]  # Extract non-RGB channels
    return rgb, other_channels


def combine_channels(rgb, other_channels, rgb_mask, original_shape):
    """Recombine the RGB and other channels after augmentation."""
    combined = torch.empty(original_shape, dtype=rgb.dtype, device=rgb.device)
    combined[:, rgb_mask, :, :] = rgb
    combined[:, ~rgb_mask, :, :] = other_channels
    return combined


def create_augmentation_pipelines(
    config, spatial_aug_indices, color_aug_indices
):
    """
    Create lists of spatial and color augmentations based on provided indices
    and parameters.

    Parameters:
        spatial_aug_indices (list): Indices to select spatial augmentations.
        color_aug_indices (list): Indices to select color augs for RGB channels.

    Returns:
        tuple(list): List of spatial and color augmentation objects.
    """
    # Define all possible spatial augmentations (applied to image and mask)
    all_spatial_transforms = [
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomRotation(degrees=config.ROTATION_DEGREES, p=0.5),
        K.RandomAffine(
            degrees=45, translate=(0.0625, 0.0625), scale=(0.9, 1.1), p=0.5
        ),
        K.RandomElasticTransform(
            kernel_size=(63, 63), sigma=(32.0, 32.0), alpha=(1.0, 1.0), p=0.5
        ),
        K.RandomPerspective(distortion_scale=0.5, p=0.5),
        K.RandomResizedCrop(size=config.RESIZED_CROP_SIZE),
    ]

    # Define all possible color augmentations
    # (applied only to the RGB channels of the image)
    all_color_transforms = [
        K.RandomContrast(contrast=config.COLOR_CONTRAST, p=0.5),
        K.RandomBrightness(brightness=config.COLOR_BRIGHTNESS, p=0.5),
        # Introduce a lower bound to the noise to make it softer
        K.RandomGaussianNoise(
            mean=0.0,
            std=config.GAUSSIAN_NOISE_STD,
            p=config.GAUSSIAN_NOISE_PROB,
        ),
        K.RandomGaussianBlur(
            kernel_size=config.GAUSSIAN_BLUR_KERNEL,
            sigma=config.GAUSSIAN_BLUR_SIGMA,
            p=0.5,
        ),
        K.RandomPlasmaBrightness(roughness=config.PLASMA_BRIGHTESS, p=0.5),
        K.RandomPlasmaShadow(
            roughness=config.PLASMA_ROUGHNESS,
            shade_intensity=config.SHADOW_INTENSITY,
            shade_quantity=config.SHADE_QUANTITY,
            p=0.5,
        ),
        K.RandomSaturation(saturation=config.SATURATION_LIMIT, p=0.5),
        K.RandomChannelShuffle(p=0.5),
        K.RandomGamma(gamma=config.GAMMA, p=0.5),
    ]

    # Select the specific augmentations for this pipeline based on the given indices
    selected_spatial_transforms = [
        all_spatial_transforms[i] for i in spatial_aug_indices
    ]
    selected_color_transforms = [
        all_color_transforms[i] for i in color_aug_indices
    ]

    return selected_spatial_transforms, selected_color_transforms


def apply_augs(
    aug_config,
    image,
    mask,
    rgb_channels=None,
):
    """
    Apply spatial and color augs to an image and its corresponding mask.

    Parameters:
        spatial_transforms (list): List of spatial augmentations to apply.
        color_transforms (list): List of color augmentations for RGB channels.
        image (torch.Tensor): The input image tensor.
        mask (torch.Tensor): The corresponding mask tensor.
        rgb_channels (list): Indices of RGB channels in the image tensor.

    Returns:
        torch.Tensor: The augmented image.
        torch.Tensor: The augmented mask, spatially transformed
                      in sync with the image.
    """
    if rgb_channels is None:
        rgb_channels = [0, 1, 2]

    # Create a boolean mask for identifying RGB channels
    rgb_mask = torch.zeros(
        image.shape[1], dtype=torch.bool, device=image.device
    )
    rgb_mask[rgb_channels] = True

    spatial_transforms, color_transforms, spatial_mode, color_mode = aug_config

    # Apply spatial augmentations to the image and mask
    augmented_image, augmented_mask = get_spatial_augmentation(
        spatial_transforms, spatial_mode, image, mask
    )

    # Separate RGB channels for color augmentation
    rgb_only, non_rgb = separate_channels(augmented_image, rgb_channels)

    # Apply color augmentations only to the RGB channels
    augmented_rgb = get_augmented_rgb(color_transforms, color_mode, rgb_only)

    # Recombine RGB and non-RGB channels
    fully_augmented_image = combine_channels(
        augmented_rgb, non_rgb, rgb_mask, image.shape
    )

    # Generate a mask of non-padded areas in the augmented image
    # and ensure color augmentations do not affect zero-padded areas
    fully_augmented_image *= augmented_image.any(dim=1, keepdim=True)

    return fully_augmented_image, augmented_mask


def get_spatial_augmentation(spatial_transforms, mode, image, mask):
    """
    Return the image and mask after spatial augmentation

    Parameters:
        spatial_transforms (list): List of spatial augmentations to apply.
        mode (str): Augmentation mode - 'random' for random augmentations
                    or 'all' for all.
        image (torch.Tensor): The input image tensor.
        mask (torch.Tensor): The corresponding mask tensor.
    """
    # Randomly select augmentations if modes are specified
    if mode == 'random':
        spatial_augmentations = random.sample(
            spatial_transforms, k=random.randint(1, len(spatial_transforms))
        )
    else:
        spatial_augmentations = spatial_transforms

    # Apply spatial augmentations to the image and mask
    spatial_aug_pipeline = K.AugmentationSequential(
        *spatial_augmentations, data_keys=["image", "mask"], same_on_batch=False
    )

    # Apply spatial augmentations to the image and mask
    return spatial_aug_pipeline(image, mask)


def get_augmented_rgb(color_transforms, mode, rgb_only):
    """
    Return the RGB channels after color augmentation

    Parameters:
        color_transforms (list): List of color augmentations to apply.
        mode (str): Augmentation mode - 'random' for random augmentations
                    or 'all' for all.
        rgb_only: The RGB channels to apply color augmentations to
    """
    if mode == 'random':
        color_augmentations = random.sample(
            color_transforms, k=random.randint(1, len(color_transforms))
        )
    else:
        color_augmentations = color_transforms

    # Apply color augmentations only to the RGB channels
    color_aug_pipeline = K.AugmentationSequential(
        *color_augmentations, data_keys=["image"], same_on_batch=False
    )

    return color_aug_pipeline(rgb_only)
