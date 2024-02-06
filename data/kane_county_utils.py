import importlib.util
import os
from typing import NoReturn

import geopandas as gpd

from . import repo_root

spec = importlib.util.spec_from_file_location(
    "config", os.path.join(repo_root, "configs", "dsi.py")
)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

spec = importlib.util.spec_from_file_location(
    "get_naip_images", os.path.join(repo_root, "utils", "get_naip_images.py")
)
get_naip_images = importlib.util.module_from_spec(spec)
spec.loader.exec_module(get_naip_images)
get_images = get_naip_images.get_images

spec = importlib.util.spec_from_file_location(
    "create_shape_mask",
    os.path.join(repo_root, "utils", "create_shape_mask.py"),
)
create_shape_mask = importlib.util.module_from_spec(spec)
spec.loader.exec_module(create_shape_mask)
create_mask = create_shape_mask.create_mask


def create_kane_county_masks() -> NoReturn:
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


def get_kane_county_images() -> NoReturn:
    data_fpath = os.path.join(
        config.KC_SHAPE_DIR, "KC_StormwaterDataJan2024.gdb.zip"
    )
    layer = 4
    save_dir = config.KC_IMAGE_DIR
    get_images("image", data_fpath, save_dir, layer)


create_kane_county_masks()
