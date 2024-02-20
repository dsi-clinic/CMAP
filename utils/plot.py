from typing import Dict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
from torch import Tensor

colors = {
    0: (0, 0, 0, 0),
    1: (215, 80, 48, 255),
    2: (49, 102, 80, 255),
    3: (239, 169, 74, 255),
    4: (100, 107, 99, 255),
    5: (89, 53, 31, 255),
    6: (2, 86, 105, 255),
    7: (207, 211, 205, 255),
    8: (195, 88, 49, 255),
    9: (144, 70, 132, 255),
    10: (29, 51, 74, 255),
    11: (71, 64, 46, 255),
    12: (114, 20, 34, 255),
    13: (37, 40, 80, 255),
    14: (94, 33, 41, 255),
    15: (255, 255, 255, 255),
}


def plot_from_tensors(
    sample: Dict[str, Tensor],
    save_path: str,
    mode: str = "grid",
    colors: Dict[int, tuple] = colors,
    labels: Dict[int, str] = None,
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
    """
    # create the colormap
    cmap_list = []
    for i in range(len(colors)):
        cmap_list.append(
            (
                colors[i][0] / 255.0,
                colors[i][1] / 255.0,
                colors[i][2] / 255.0,
            )
        )
    cmap = ListedColormap(cmap_list)

    # set the figure dimensions
    if mode == "grid":
        _, axs = plt.subplots(
            int(round(len(sample.keys()) / 2)), 2, figsize=(10, 8)
        )
    elif mode == "row":
        _, axs = plt.subplots(1, len(sample.keys()), figsize=(14, 4))
    else:
        raise ValueError("Invalid mode")

    # plot each input tensor
    labels_present = Tensor()
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
                lab = tensor.unique()
                ax.imshow(tensor, cmap=cmap, interpolation="none")
            else:
                lab = tensor[0].unique()
                ax.imshow(tensor[0], cmap=cmap, interpolation="none")
            labels_present = torch.cat((lab, labels_present)).unique()
        ax.set_title(name)
        ax.axis("off")

    # create the legend if labels were provided
    if labels:
        labels_present = labels_present.type(torch.int).tolist()
        print(labels_present)
        patches = []
        for i in labels_present:
            patches.append(mpatches.Patch(color=cmap_list[i], label=labels[i]))

        plt.legend(
            handles=patches,
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0.0,
        )
    plt.savefig(save_path)
    plt.close()
