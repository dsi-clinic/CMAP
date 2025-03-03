"""Module: plot.py

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

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from matplotlib.colors import ListedColormap
from torch import Tensor
from torchgeo.datasets.utils import BoundingBox


def build_cmap(colors: dict[int, tuple]):
    """Build a ListedColormap object from a dictionary.

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
    sample: dict[str, Tensor],
    save_path: str,
    colors: dict[int, tuple] = None,
    labels: dict[int, str] = None,
    coords: BoundingBox = None,
):
    """Plot sample from training dataset and save to file

    Parameters
    ----------
    sample : dict
        sample from training dataset containing dict of names for
        images and tensors of image data. each tensor must only contain data
        for a single image

    save_path : str
        path to save plot to

    colors : Dict[int, tuple]
        color mapping for masks
            keys : mask indices
            values : (r, g, b)

    labels : Dict[int, str]
        label mapping for masks
            keys : mask indices
            values : labels

    coords : torchgeo.datasets.utils.BoundingBox
        x, y, t coords for sample taken from dataloader sample's bbox key
    """
    # Create the colormap
    cmap = build_cmap(colors) if colors is not None else "viridis"
    min_dims = 2

    # Determine the layout and create subplots
    nrows = len(sample) // 2 + len(sample) % 2
    ncols = min(len(sample), 2)
    print(
        f"plot_from_tensors: creating {nrows}x{ncols} subplot grid for {len(sample)} tensors"
    )
    fig, axs = plt.subplots(nrows, ncols, figsize=(8, 8))
    axs = np.array(axs).reshape(-1)

    # Plot each input tensor and gather unique labels
    unique_labels = Tensor()
    for i, (name, tensor) in enumerate(sample.items()):
        ax = axs[i]

        if "image" in name.lower():
            ax.imshow(rearrange(tensor, "c h w -> h w c"))
        elif "dem" in name.lower() or "nir" in name.lower():
            # Squeeze out the channel dimension for DEM/NIR visualization
            ax.imshow(tensor.squeeze(0), cmap="viridis")
        else:
            unique = tensor.unique() if tensor.ndim > min_dims else tensor.unique()
            ax.imshow(
                tensor[0] if tensor.ndim > min_dims else tensor,
                cmap=cmap,
                vmin=0,
                vmax=len(cmap.colors) - 1 if isinstance(cmap, ListedColormap) else None,
                interpolation="none",
            )
            unique_labels = torch.cat((unique, unique_labels))

        ax.set_title(name.replace("_", " "))
        ax.axis("off")

    # Create the legend if labels were provided
    if labels is not None and colors is not None:
        unique_labels = unique_labels.unique().type(torch.int).tolist()
        patches = [
            mpatches.Patch(color=cmap.colors[i], label=labels[label])
            for i, label in enumerate(unique_labels)
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
    """Determines the most common label ID from a ground truth mask tensor.

    Parameters
    ----------
    ground_truth : Tensor
        The ground truth mask tensor, which should contain label indices.

    Returns:
    -------
    int
        The ID of the most common label in the ground truth.
    """
    unique, counts = ground_truth.unique(return_counts=True)
    # Remove the background label '0' from consideration if present
    if 0 in unique:
        background_index = (unique == 0).nonzero(as_tuple=True)[0].item()
        unique = torch.cat([unique[:background_index], unique[background_index + 1 :]])
        counts = torch.cat([counts[:background_index], counts[background_index + 1 :]])

    if counts.numel() == 0:  # Check if there are no labels other than the background
        return 15  # Return ID for 'UNKNOWN'

    most_common_index = counts.argmax()
    most_common_label_id = unique[most_common_index].item()
    return most_common_label_id


def find_labels_in_ground_truth(ground_truth: Tensor):
    """Finds all unique label IDs from a ground truth mask tensor.

    Parameters
    ----------
    ground_truth : Tensor
        The ground truth mask tensor, which should contain label indices.

    Returns:
    -------
    List[int]
        A list of the unique label IDs in the ground truth.
    """
    unique = ground_truth.unique()
    # Remove the background label '0' from consideration if present
    if 0 in unique:
        unique = unique[unique != 0]

    return unique.tolist() if unique.numel() > 0 else [15]
