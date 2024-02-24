from typing import Dict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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

    # set the figure dimensions
    if mode == "grid":
        fig, axs = plt.subplots(
            int(round(len(sample.keys()) / 2)), 2, figsize=(8, 8)
        )
    elif mode == "row":
        fig, axs = plt.subplots(1, len(sample.keys()), figsize=(12, 4))
    else:
        raise ValueError("Invalid mode")

    # plot each input tensor
    unique_labels = Tensor()
    for i, (name, tensor) in enumerate(sample.items()):
        if mode == "grid":
            ax = axs[i // 2][i % 2]
        else:
            ax = axs[i]

        if "image" in name:
            img = tensor[0:3, :, :].permute(1, 2, 0)
            ax.imshow(img)
        else:
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
        ax.set_title(name)
        ax.axis("off")

    # create the legend if labels were provided
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

    if coords is not None:
        fig.text(0, 0, s=coords, fontsize=10, color="gray")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
