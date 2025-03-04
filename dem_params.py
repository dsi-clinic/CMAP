"""Calculates a sample mean and standard deviation for DEM data"""

import itertools
import os
from pathlib import Path

# import cv2
import numpy as np
import torch
from einops import rearrange
from torchgeo.datasets import NAIP

from configs import config
from data.dem import KaneDEM
from data.sampler import BalancedGridGeoSampler

# data paths
DATA_ROOT = "/net/projects/cmap/data"
KC_SHAPE_ROOT = str(Path(DATA_ROOT) / "kane-county-data")
KC_IMAGE_ROOT = str(Path(DATA_ROOT) / "KC-images")
KC_RIVER_ROOT = str(Path(DATA_ROOT) / "KC-river-images")
# KC_DEM_ROOT = None
KC_DEM_ROOT = str(Path(KC_SHAPE_ROOT) / "KC_DEM_2017")
KC_MASK_ROOT = str(Path(DATA_ROOT) / "KC-masks/separate-masks")
OUTPUT_ROOT = str(Path("/net/projects/cmap/workspaces/") / f"{os.environ['USER']}")


def calculate_image_stats(dem_image):
    """Calculate the mean and standard deviation of an image.

    Parameters
    ----------
    file_path : str
        Path
        to the image file

    Returns:
    -------
    Tuple[np.ndarray, np.ndarray]
        Mean and standard deviation of the image across channels
    """
    img = rearrange(dem_image, "h w c -> c h w")
    mean = torch.mean(img, axis=(1, 2))
    std = torch.std(img, axis=(1, 2))
    return mean, std


def main():
    """Calculate the mean and standard deviation of a DEM file."""
    means = []
    stds = []
    naip_dataset = NAIP(KC_IMAGE_ROOT)
    dem = KaneDEM(
        KC_DEM_ROOT,
        config=config,
        crs=naip_dataset.crs,
        res=naip_dataset.res,
        use_filled=True,
    )
    sampler = BalancedGridGeoSampler(
        config={"dataset": naip_dataset, "size": 256, "stride": 256}
    )
    bounding_boxes = itertools.islice(sampler, 1000)
    i = 0
    for bounding_box in bounding_boxes:
        i += 1
        dem_piece = dem[bounding_box]["image"]
        mean, std = calculate_image_stats(dem_piece)
        means.append(mean)
        stds.append(std)
        if i % 10 == 0:
            print(i)
            print(np.mean(means))
            print(np.mean(stds))
    print("final results")
    print(np.mean(means))
    print(np.mean(stds))


if __name__ == "__main__":
    main()
