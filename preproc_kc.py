"""
To run: from repo directory (2024-winter-cmap)
> python preproc_kc.py configs.<config>
"""

import argparse
import importlib.util
import os

import geopandas as gpd

from utils.create_shape_mask import create_mask
from utils.get_naip_images import get_images

parser = argparse.ArgumentParser(
    description="Preprocess Kane County data to generate masks and get "
    + "corresponding NAIP images"
)
parser.add_argument("config", type=str, help="Path to the configuration file")
args = parser.parse_args()
spec = importlib.util.spec_from_file_location(args.config)
config = importlib.import_module(args.config)


def create_kane_county_masks() -> None:
    """
    Creates masks for the Kane County stormwater structures dataset (layer 4
    of KC_StormwaterDataJan2024.gdb.zip)
    """
    shape_fpath = os.path.join(
        config.KC_SHAPE_DIR, "KC_StormwaterDataJan2024.gdb.zip"
    )
    layer = 4
    gdf = gpd.read_file(shape_fpath, layer=layer)
    labels = {
        "POND": 1,
        "WETLAND": 2,
        "DRY BOTTOM - TURF": 3,
        "DRY BOTTOM - MESIC PRAIRIE": 4,
    }

    img_root = config.KC_IMAGE_DIR

    for img_fname in os.listdir(img_root):
        try:
            img_fpath = os.path.join(img_root, img_fname)
            create_mask(
                img_fpath, gdf, config.KC_MASK_DIR, "BasinType", labels=labels
            )
        except Exception:
            continue


def get_kane_county_images() -> None:
    """
    Retrieves NAIP images that have any intersection with shapes in the Kane
    County stormwater structures dataset (layer 4 of
    KC_StormwaterDataJan2024.gdb.zip)
    """
    data_fpath = os.path.join(
        config.KC_SHAPE_DIR, "KC_StormwaterDataJan2024.gdb.zip"
    )
    layer = 4
    save_dir = config.KC_IMAGE_DIR
    get_images("image", data_fpath, save_dir, layer)
