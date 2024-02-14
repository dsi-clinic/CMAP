import os
from typing import Any, DefaultDict, Dict

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


def plot_from_sample(sample: DefaultDict[str, Any], save_root: str):
    """
    Plots a sample from the training dataset and saves to provided directory.

    Parameters
    ----------
    sample : defaultdict
        A sample from the training dataset, generated by DataLoader

    save_root : str
        The directory to save the plot to
    """
    # extract the mask and image from the sample
    mask = sample["mask"].numpy().astype(np.uint8).squeeze()
    image = sample["image"].numpy().astype(np.uint8).squeeze()

    _, ax = plt.subplots(1, 2, figsize=(10, 5))

    # plot the mask
    ax[0].imshow(mask[1], extent=sample["bbox"][0][:4])
    ax[0].set_title("Mask")
    ax[0].axis("off")

    # plot the image
    ax[1].imshow(image[1], extent=sample["bbox"][0][:4], cmap="gray")
    ax[1].set_title("NAIP Chip")
    ax[1].axis("off")

    # TODO: change this to a more descriptive name
    fname = os.path.join(save_root, "sample.png")
    plt.savefig(fname)


def plot_from_tensors(sample: Dict[str, Tensor], save_path: str):
    """
    Plots a sample from the training dataset and saves to provided file path.

    Parameters
    ----------
    sample : dict
        A sample from the training dataset containing a dict of names for
        images and tensors of image data

    save_path : str
        The path to save the plot to
    """
    _, axs = plt.subplots(
        int(round(len(sample.keys()) / 2)), 2, figsize=(10, 10)
    )

    for i, (name, tensor) in enumerate(sample.items()):
        ax = axs[i // 2][i % 2]
        if len(tensor.shape) == 2:
            ax.imshow(tensor)
        elif len(tensor.shape) == 3:
            ax.imshow(tensor[0])
        else:
            raise ValueError(
                "Expected tensor with dimensions h x w or c x h x w"
            )
        ax.set_title(name)
        ax.axis("off")

    plt.savefig(save_path)