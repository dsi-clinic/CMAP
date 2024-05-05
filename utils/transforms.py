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
    aug_params, spatial_aug_indices, color_aug_indices
):
    """
    Create lists of spatial and color augmentations based on provided indices
    and parameters.

    Parameters:
        aug_params (dict): Dictionary with parameters for each augmentation.
        spatial_aug_indices (list): Indices to select spatial augmentations.
        color_aug_indices (list): Indices to select color augs for RGB channels.

    Returns:
        tuple(list): List of spatial and color augmentation objects.
    """
    # Define all possible spatial augmentations (applied to image and mask)
    all_spatial_transforms = [
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomRotation(degrees=aug_params["rotation_degrees"], p=0.5),
        K.RandomAffine(
            degrees=45, translate=(0.0625, 0.0625), scale=(0.9, 1.1), p=0.5
        ),
        K.RandomElasticTransform(
            kernel_size=(63, 63), sigma=(32.0, 32.0), alpha=(1.0, 1.0), p=0.5
        ),
        K.RandomPerspective(distortion_scale=0.5, p=0.5),
        K.RandomResizedCrop(
            size=aug_params["resized_crop_size"], scale=(0.08, 1.0)
        ),
    ]

    # Define all possible color augmentations
    # (applied only to the RGB channels of the image)
    all_color_transforms = [
        K.RandomContrast(contrast=aug_params.get("contrast_limit", 0.2), p=0.5),
        K.RandomBrightness(
            brightness=aug_params.get("brightness_limit", 0.2), p=0.5
        ),
        # Introduce a lower bound to the noise to make it softer
        K.RandomGaussianNoise(
            mean=0.0,
            std=aug_params.get("gaussian_noise_std", 0.1),
            p=aug_params.get("gaussian_noise_prob", 0.2),
        ),
        K.RandomGaussianBlur(
            kernel_size=(3, 3),
            sigma=aug_params.get("gaussian_blur_sigma", (0.1, 2.0)),
            p=0.5,
        ),
        K.RandomPlasmaBrightness(
            roughness=aug_params.get("plasma_roughness", (0.1, 0.3)), p=0.5
        ),
        K.RandomPlasmaShadow(
            roughness=aug_params.get("plasma_roughness", (0.1, 0.3)),
            shade_intensity=aug_params.get("shadow_intensity", (-0.2, 0.0)),
            shade_quantity=aug_params.get("shade_quantity", (0.0, 0.2)),
            p=0.5,
        ),
        K.RandomSaturation(
            saturation=aug_params.get("saturation_limit", 0.2), p=0.5
        ),
        K.RandomChannelShuffle(p=0.5),
        K.RandomGamma(gamma=aug_params.get("gamma", (0.5, 1.5)), p=0.5),
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
    spatial_transforms,
    color_transforms,
    image,
    mask,
    spatial_mode,
    color_mode,
    rgb_channels=None,
):
    """
    Apply spatial and color augs to an image and its corresponding mask.

    Parameters:
        spatial_transforms (list): List of spatial augmentations to apply.
        color_transforms (list): List of color augmentations for RGB channels.
        image (torch.Tensor): The input image tensor.
        mask (torch.Tensor): The corresponding mask tensor.
        mode (str): Augmentation mode - 'random' for random augmentations
                    or 'all' for all.
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

    # Random mode: pick random number and set of augmentations to apply
    if spatial_mode == "random":
        spatial_augmentations = random.sample(
            spatial_transforms, k=random.randint(1, len(spatial_transforms))
        )
    else:
        spatial_augmentations = spatial_transforms

    if color_mode == "random":
        color_augmentations = random.sample(
            color_transforms, k=random.randint(1, len(color_transforms))
        )
    else:
        color_augmentations = color_transforms

    # Create a pipeline for spatial augmentations and apply them to the image and mask
    spatial_aug_pipeline = K.AugmentationSequential(
        *spatial_augmentations, data_keys=["image", "mask"], same_on_batch=False
    )
    augmented_image, augmented_mask = spatial_aug_pipeline(image, mask)

    # Generate a mask of non-padded areas in the augmented image
    non_padded_area_mask = augmented_image.any(dim=1, keepdim=True)

    # Separate RGB channels for color augmentation
    rgb_only, non_rgb = separate_channels(augmented_image, rgb_channels)

    # Apply color augmentations only to the RGB channels
    color_aug_pipeline = K.AugmentationSequential(
        *color_augmentations, data_keys=["image"], same_on_batch=False
    )
    augmented_rgb = color_aug_pipeline(rgb_only)

    # Recombine RGB and non-RGB channels
    fully_augmented_image = combine_channels(
        augmented_rgb, non_rgb, rgb_mask, image.shape
    )

    # Ensure color augmentations do not affect zero-padded areas
    fully_augmented_image *= non_padded_area_mask

    return fully_augmented_image, augmented_mask
