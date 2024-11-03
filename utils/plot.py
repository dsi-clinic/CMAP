"""
Module: plot.py

This module provides utility functions for working with image data, including
building colormaps, plotting images, and analyzing ground truth masks.

Dependencies:
- matplotlib.patches as mpatches
- matplotlib.pyplot as plt
- numpy as np
- torch
- ListedColormap from matplotlib.colors
- Tensor from torch
- Dict from typing
- BoundingBox from torchgeo.datasets.utils

Functions:
- build_cmap(colors: Dict[int, tuple]) -> ListedColormap:
    Build a ListedColormap object from a dictionary.

- plot_from_tensors(
    sample: Dict[str, Tensor],
    save_path: str,
    colors: Dict[int, tuple] = None,
    labels: Dict[int, str] = None,
    coords: BoundingBox = None,
) -> None:
    Plots a sample from the training dataset and saves to provided file path.

- determine_dominant_label(ground_truth: Tensor) -> int:
    Determines the most common label ID from a ground truth mask tensor.

- find_labels_in_ground_truth(ground_truth: Tensor) -> List[int]:
    Finds all unique label IDs from a ground truth mask tensor.

- create_outline(
        mask: torch.Tensor,
        iterations: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    Creates an outline for the given ground truth mask tensor.
"""

from typing import Dict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_dilation
from skimage import measure
from torch import Tensor
from torchgeo.datasets.utils import BoundingBox


def build_cmap(colors: Dict[int, tuple]):
    """
    Build a ListedColormap object from a dictionary.

    Parameters
    ----------
    colors : Dict[int, tuple]
        A dictionary containing a color mapping
            keys : indices
            values : (r, g, b)
    """
    cmap_list = []
    for i in colors.keys():
        cmap_list.append(
            (
                colors[i][0] / 255.0,
                colors[i][1] / 255.0,
                colors[i][2] / 255.0,
            )
        )
    cmap = ListedColormap(cmap_list)
    return cmap


def plot_from_tensors(
    sample: Dict[str, Tensor],
    save_path: str,
    colors: Dict[int, tuple] = None,
    labels: Dict[int, str] = None,
    coords: BoundingBox = None,
):
    """
    Plots a sample from the training dataset and saves to provided file path.

    Parameters
    ----------
    sample : dict
        A sample from the training dataset containing a dict of names for
        images and tensors of image data. Each tensor must only contain data
        for a single image

    save_path : str
        The path to save the plot to

    colors : Dict[int, tuple]
        A dictionary containing a color mapping for masks
            keys : mask indices
            values : (r, g, b)

    labels : Dict[int, str]
        A dictionary containing a label mapping for masks
            keys : mask indices
            values : labels

    coords : torchgeo.datasets.utils.BoundingBox
        The x, y, t coords for the sample taken from a dataloader sample's
        bbox key
    """
    # Create the colormap
    cmap = build_cmap(colors) if colors is not None else "viridis"

    # Determine the layout and create subplots
    fig, axs = plt.subplots(
        len(sample) // 2 + len(sample) % 2, min(len(sample), 2), figsize=(8, 8)
    )
    axs = np.array(axs).reshape(-1)

    # Plot each input tensor and gather unique labels
    unique_labels = Tensor()

    for i, (name, tensor) in enumerate(sample.items()):
        ax = axs[i]

        if "image" in name.lower():
            # Plot prediction image
            ax.imshow(tensor[0:3, :, :].permute(1, 2, 0))
        elif "vs" in name.lower():
            # Plot the combined image of ground truth outline and prediction
            ax.imshow(tensor.permute(1, 2, 0), cmap=cmap)
        else:
            unique = (
                tensor[0].unique()
                if len(tensor.shape) != 2
                else tensor.unique()
            )
            ax.imshow(
                tensor[0] if len(tensor.shape) != 2 else tensor,
                cmap=cmap,
                vmin=0,
                vmax=len(cmap.colors) - 1,
                interpolation="none",
            )
            unique_labels = torch.cat((unique, unique_labels))

        ax.set_title(name.replace("_", " "))
        ax.axis("off")

    # Create the legend if labels were provided
    if labels is not None and colors is not None:
        unique_labels = unique_labels.unique().type(torch.int).tolist()
        patches = [
            mpatches.Patch(color=cmap.colors[i], label=labels[i])
            for i in unique_labels
        ]

        fig.legend(
            handles=patches,
            bbox_to_anchor=(1.05, 0.5),
            loc="center left",
            borderaxespad=0.0,
        )

    # Add bounding box coords to the plot if provided
    if coords is not None:
        fig.text(0, 0, s=coords, fontsize=10, color="gray")

    # Save the figure
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def determine_dominant_label(ground_truth: Tensor) -> int:
    """
    Determines the most common label ID from a ground truth mask tensor.

    Parameters
    ----------
    ground_truth : Tensor
        The ground truth mask tensor, which should contain label indices.

    Returns
    -------
    int
        The ID of the most common label in the ground truth.
    """
    unique, counts = ground_truth.unique(return_counts=True)
    # Remove the background label '0' from consideration if present
    if 0 in unique:
        background_index = (unique == 0).nonzero(as_tuple=True)[0].item()
        unique = torch.cat(
            [unique[:background_index], unique[background_index + 1 :]]
        )
        counts = torch.cat(
            [counts[:background_index], counts[background_index + 1 :]]
        )

    if (
        counts.numel() == 0
    ):  # Check if there are no labels other than the background
        return 15  # Return ID for 'UNKNOWN'

    most_common_index = counts.argmax()
    most_common_label_id = unique[most_common_index].item()
    return most_common_label_id


def find_labels_in_ground_truth(ground_truth: Tensor):
    """
    Finds all unique label IDs from a ground truth mask tensor.

    Parameters
    ----------
    ground_truth : Tensor
        The ground truth mask tensor, which should contain label indices.

    Returns
    -------
    List[int]
        A list of the unique label IDs in the ground truth.
    """
    unique = ground_truth.unique()
    # Remove the background label '0' from consideration if present
    if 0 in unique:
        unique = unique[unique != 0]

    return unique.tolist() if unique.numel() > 0 else [15]


def create_outline(
    mask: torch.Tensor,
    iterations: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates an outline for the given ground truth mask tensor.

    Args:
        mask : Tensor
            The ground truth mask tensor.
        iterations : int
            Number of dilation iterations for the outline thickness.

    Returns
    -------
    Tuple [torch.Tensor, torch.Tensor]
        1. The outline mask with preserved label information.
        2. The binary outline mask indicating outline positions.
    """
    # Convert to numpy for processing
    mask_np = mask.cpu().numpy()

    # Initialize outline masks
    outline_labels = np.zeros_like(mask_np)
    outline_binary = np.zeros_like(mask_np)

    # Get unique labels (excluding background if 0 is present)
    unique_labels = np.unique(mask_np)
    if 0 in unique_labels:
        unique_labels = unique_labels[unique_labels != 0]

    # Process each label separately
    for label in unique_labels:
        # Create binary mask for this label
        label_mask = mask_np == label

        # Find contours for this label
        contours = measure.find_contours(label_mask, level=0.5)

        # Draw contours for this label
        label_outline = np.zeros_like(mask_np)
        for contour in contours:
            contour = np.round(contour).astype(int)
            label_outline[contour[:, 0], contour[:, 1]] = 1

        # Dilate the outline
        dilated_outline = binary_dilation(label_outline, iterations=iterations)

        # Add this label's outline to both masks
        outline_labels[dilated_outline] = label
        outline_binary[dilated_outline] = 1

    # Convert back to tensors
    return (
        torch.tensor(outline_labels, dtype=torch.float32).unsqueeze(0),
        torch.tensor(outline_binary, dtype=torch.float32).unsqueeze(0),
    )


def combine_images(
    outline: tuple[torch.Tensor, torch.Tensor],
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    colors: Dict[int, tuple],
    outline_alpha: float = 1.0,
    pred_alpha: float = 0.5,
) -> torch.Tensor:
    """
    Combines the outline of the ground truth with the prediction image

    Args:
        outline: tuple
            A tuple containing (outline_labels, outline_binary).
        ground_truth: Tensor
            The ground truth mask tensor
        prediction: Tensor
            The prediction mask tensor
        colors: Dict
            A dictionary containing a color mapping for masks.
            keys : mask indices
            values : (r, g, b)
        outline_alpha: float
            The transparency factor for the outline (0 to 1).
        pred_alpha: float
            The transparency factor for the prediction (0 to 1).

    Returns
    -------
    A tensor representing the combined image
    """
    outline_labels, outline_binary = outline

    if prediction.dim() == 2:
        prediction = prediction.unsqueeze(0)

    if ground_truth is None:
        ground_truth = prediction

    if ground_truth.dim() == 2:
        ground_truth = ground_truth.unsqueeze(0)

    # Create color versions of both the prediction and outline
    color_prediction = torch.ones(3, *prediction.shape[1:])
    color_outline = torch.ones(3, *prediction.shape[1:])

    # Create masks for colored regions (non-background)
    # pred_colored_mask = prediction[0] != 0

    # Color the prediction
    for label_id, color in colors.items():
        if label_id == 0:
            continue
        mask = prediction[0] == label_id
        if mask.any():
            for c in range(3):
                color_val = color[c] / 255.0
                # Blend with white background using alpha
                color_prediction[c][mask] = color_val * pred_alpha + (
                    1 - pred_alpha
                )

    # Color the outline using the outline labels
    outline_mask = outline_binary[0] > 0
    for label_id, color in colors.items():
        if label_id == 0:
            continue
        mask = (outline_labels[0] == label_id) & outline_mask
        if mask.any():
            for c in range(3):
                color_val = color[c] / 255.0
                # Blend with white background using alpha
                color_outline[c][mask] = color_val * outline_alpha + (
                    1 - outline_alpha
                )

    # Combine the images
    combined_image = torch.where(
        outline_mask.unsqueeze(0), color_outline, color_prediction
    )

    # Ensure all values are between 0 and 1
    combined_image = torch.clamp(combined_image, 0, 1)

    return combined_image
