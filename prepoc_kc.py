"""
To run: from repo directory (2024-winter-cmap)
> python preproc_kc.py configs.dsi --action all
"""

import argparse
import importlib.util
import os

from utils.get_naip_images import get_images, get_river_images

parser = argparse.ArgumentParser(
    description="Preprocess Kane County data to get corresponding NAIP images"
)
parser.add_argument("config", type=str, help="Path to the configuration file")
parser.add_argument(
    "--action",
    type=str,
    default="all",
    choices=["all", "images", "river"],
    help="Specify the action to perform",
)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location(args.config)
config = importlib.import_module(args.config)


def get_kane_county_images() -> None:
    """
    Retrieves NAIP images that have any intersection with shapes in the Kane
    County stormwater structures dataset (layer 4 of
    KC_StormwaterDataJan2024.gdb.zip)
    """
    data_fpath = os.path.join(
        config.KC_SHAPE_ROOT, "KC_StormwaterDataJan2024.gdb.zip"
    )
    layer = 4
    save_dir = config.KC_IMAGE_ROOT
    get_images("image", data_fpath, save_dir, layer)


def get_kane_county_river_images() -> None:
    """
    Retrieves NAIP images that have any intersection with shapes in the Kane
    County open water dataset, specifically to the river and stream features.
    """
    data_fpath = os.path.join(
        config.KC_SHAPE_ROOT, "Kane_Co_Open_Water_Layer.zip"
    )
    save_dir = config.KC_RIVER_ROOT
    get_river_images("image", data_fpath, save_dir)


# Determine which action to perform based on user input
if args.action == "all":
    get_kane_county_images()
    get_kane_county_river_images()

elif args.action == "images":
    get_kane_county_images()

elif args.action == "river":
    get_kane_county_river_images()
