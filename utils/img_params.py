"""
To run:
> python img_params.py <directory_path>
"""

import argparse
import os
from typing import NoReturn, Tuple

import cv2
import numpy as np
from einops import rearrange


def calculate_image_stats(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and standard deviation of an image.

    Parameters
    ----------
    file_path : str
        Path to the image file

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Mean and standard deviation of the image across channels
    """
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    img = rearrange(img, "h w c -> c h w")
    mean = np.mean(img, axis=(1, 2))
    std = np.std(img, axis=(1, 2))
    return mean, std


def main(root: str) -> NoReturn:
    """
    Calculate and print the mean and standard deviation of all images in a
    directory.

    Parameters
    ----------
    root : str
        Path to the directory containing images.
    """
    means = []
    stds = []

    for filename in os.listdir(root):
        file_path = os.path.join(root, filename)
        mean, std = calculate_image_stats(file_path)
        means.append(mean)
        stds.append(std)

    means = np.array(means)
    stds = np.array(stds)
    overall_mean = np.mean(means, axis=0)
    overall_std = np.mean(std, axis=0)
    print(f"Overall mean: \n{overall_mean}")
    print(f"Overall std: \n{overall_std}")

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            description="Calculate image statistics (mean/std) for all images "
            + "in a directory."
        )
        parser.add_argument(
            "directory_path",
            type=str,
            help="Path to the directory containing images.",
        )
        args = parser.parse_args()
        main(args.directory_path)
