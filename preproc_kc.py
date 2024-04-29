"""
To run: from repo directory (2024-winter-cmap)
> python preproc_kc.py configs.<config>
"""

import argparse
import importlib.util
import os

from utils.get_naip_images import get_images

parser = argparse.ArgumentParser(
    description="Preprocess Kane County data to get corresponding NAIP images"
)
parser.add_argument("config", type=str, help="Path to the configuration file")
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
