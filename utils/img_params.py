import os

import cv2
import numpy as np


def params(fp):
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    img = img.transpose(2, 0, 1)
    mean = np.mean(img, axis=(1, 2))
    std = np.std(img, axis=(1, 2))
    del img
    return mean, std


def get_img_params(root):
    means = []
    stds = []

    for fn in os.listdir(root):
        fp = os.path.join(root, fn)
        mean, std = params(fp)
        means.append(mean)
        stds.append(std)

    means = np.array(means)
    stds = np.array(stds)
    print(np.mean(means, axis=0))
    print(np.std(stds, axis=0))
