from typing import Dict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
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
    mode: str = "row",
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

    mode : str
        Either 'grid' or 'row' for orientation of images on plot

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
    # create the colormap
    if colors is not None:
        cmap = build_cmap(colors)
    else:
        cmap = "viridis"

    # Set the figure dimensions and create subplots accordingly
    if mode == "grid":
        n_rows = int(round(len(sample.keys()) / 2))
        n_cols = 2 if len(sample.keys()) > 1 else 1
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8, 8))
        # Ensure axs is always an array, even if it's 1x1
        axs = np.array(axs).reshape(-1)
    elif mode == "row":
        fig, axs = plt.subplots(1, len(sample.keys()), figsize=(12, 4))
        # Ensure axs is always an array, even if it's 1x1
        axs = np.array(axs).reshape(-1)
    else:
        raise ValueError("Invalid mode")

    # Plot each input tensor
    unique_labels = Tensor()
    for i, (name, tensor) in enumerate(sample.items()):
        ax = axs[i]

        if "image" in name:
            # Handle RGB image tensors by ignoring the NIR channel
            img = tensor[0:3, :, :].permute(1, 2, 0)
            ax.imshow(img)
        elif "dem" in name:
            img = tensor[4, :, :].permute(1,2,0)
            ax.imshow(img, cmap='gray')
        elif "nir" in name:
            img = tensor[3, :, :].permute(1, 2, 0)
            ax.imshow(img, cmap='gray')
        else:
            # Get the unique labels present in the mask
            if len(tensor.shape) == 2:
                unique = tensor.unique()
                ax.imshow(
                    tensor,
                    cmap=cmap,
                    vmin=0,
                    vmax=len(cmap.colors) - 1,
                    interpolation="none",
                )
            else:
                unique = tensor[0].unique()
                ax.imshow(
                    tensor[0],
                    cmap=cmap,
                    vmin=0,
                    vmax=len(cmap.colors) - 1,
                    interpolation="none",
                )
            unique_labels = torch.cat((unique, unique_labels)).unique()
        ax.set_title(name.replace("_", " "))
        ax.axis("off")

    # Create the legend if labels were provided
    if labels is not None and colors is not None:
        unique_labels = unique_labels.type(torch.int).tolist()
        patches = []
        for i in unique_labels:
            patches.append(
                mpatches.Patch(color=cmap.colors[i], label=labels[i])
            )

        fig.legend(
            handles=patches,
            bbox_to_anchor=(1.05, 0.5),
            loc="center left",
            borderaxespad=0.0,
        )

    # If bounding box coords were provided, add them to the plot
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
