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
    mode: str = "row",
    colors: Dict[int, tuple] = None,
    labels: Dict[int, str] = None,
    coords: BoundingBox = None
) -> None:
    Plots a sample from the training dataset and saves to provided file path.

- determine_dominant_label(ground_truth: Tensor) -> int:
    Determines the most common label ID from a ground truth mask tensor.

- find_labels_in_ground_truth(ground_truth: Tensor) -> List[int]:
    Finds all unique label IDs from a ground truth mask tensor.
"""

from typing import Dict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from skimage import measure
from scipy.ndimage import binary_dilation
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
) -> torch.Tensor:
    """
    Create an outline for the given mask.

    Args:
        mask (torch.Tensor): The ground truth mask tensor.

    Returns:
        torch.Tensor: A binary tensor representing the outline.
    """
    # Convert to numpy for contour detection
    mask_np = mask.cpu().numpy()

    # Detect contours of the binary mask
    contours = measure.find_contours(mask_np, level=0.5)

    # Create an empty outline mask with the same shape, then add contours
    outline = np.zeros_like(mask_np)  # Updated this line
    for contour in contours:
        contour = np.round(contour).astype(int)
        outline[contour[:, 0], contour[:, 1]] = 1

    outline = binary_dilation(outline, iterations=iterations)

    # Convert outline back to a tensor
    return torch.tensor(outline, dtype=torch.float32).unsqueeze(0)

def combine_images(
        outline: torch.Tensor, 
        prediction: torch.Tensor, 
        colors: Dict[int, tuple],
        alpha: float = 0.5,
) -> torch.Tensor:
    """
    Combines an outline image with a prediction image.

    Args:
        outline: A binary tensor representing the outline (shape: [H, W]).
        prediction: A tensor representing the predicted segmentation (shape: [H, W]).
        colors: A dictionary mapping class indices to RGB tuples.
        alpha: The transparency factor for the outline overlay. (0 <= alpha <= 1)

    Returns:
        A tensor representing the combined image.
    """
    # Ensure the outline is in the right shape
    if outline.dim() == 2:  # If the outline is 2D, add a channel dimension
        outline = outline.unsqueeze(0)  # Shape: [1, H, W]

    # If prediction is also 2D, add a channel dimension
    if prediction.dim() == 2:
        prediction = prediction.unsqueeze(0)  # Shape: [1, H, W]

    # Make sure both have the same shape
    assert outline.shape == prediction.shape, "Outline and prediction must have the same dimensions"

    # Create a color version of the prediction image
    color_prediction = torch.zeros(3, *prediction.shape[1:])  # Shape: [3, H, W]
    
    # Loop over each class and assign colors
    for label_id, color in colors.items():
        color_prediction[0][prediction[0] == label_id] = color[0] / 255.0  # Red channel
        color_prediction[1][prediction[0] == label_id] = color[1] / 255.0  # Green channel
        color_prediction[2][prediction[0] == label_id] = color[2] / 255.0  # Blue channel

    # Create a color version of the outline for visualization (e.g., red)
    outline_color = torch.zeros_like(color_prediction)  # Shape: [3, H, W]
    outline_color[0] = outline  # Red channel
    outline_color[1] = outline * 0  # Green channel
    outline_color[2] = outline * 0  # Blue channel

    # Combine the images with transparency
    combined_image = (1 - alpha) * color_prediction + alpha * outline_color

    return combined_image

