import os
from typing import List, NoReturn

import fiona
import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio.mask import mask
from shapely.geometry import box, shape


def create_mask(
    img_fpath: str, shape_fpath: str, save_dir: str, layer: int = None
) -> NoReturn:
    """
    Creates a mask for shapefile features that intersect with a given image.

    Parameters
    ----------
    img_fpath : str
        Path to image

    shape_fpath : str
        Path to shapefile

    save_dir : str
        Path to save directory

    layer : int
        Layer of shapefile to use
    """
    # open image and get bounding box
    img = rasterio.open(img_fpath)
    bbox = box(*img.bounds)

    # extract intersecting shapes
    shapes = get_intersecting_shapes(bbox, img.crs, shape_fpath, layer)

    # create mask and stack into 3-channel image
    out_img, out_transform = mask(img, shapes, crop=False)
    out_img_unique = np.where(out_img != 0, 255, out_img)
    msk = np.dstack(
        (out_img_unique[0], out_img_unique[1], out_img_unique[2])
    ).astype("uint8")
    msk = msk.transpose(2, 0, 1)

    # set output metadata and filename
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    out_tif = os.path.join(
        save_dir, os.path.basename(img_fpath).replace(".tif", "_mask.tif")
    )

    out_meta = img.meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "dtype": "uint8",
            "height": out_img.shape[1],
            "width": out_img.shape[2],
            "transform": out_transform,
            "count": 3,
            "crs": img.crs,
        }
    )

    # write mask to file
    with rasterio.open(out_tif, "w", **out_meta) as dest:
        dest.write(msk.astype("uint8"))


def get_intersecting_shapes(
    bbox: box, crs: str, shape_fpath: str, layer: int = None
) -> List[shapely.Geometry]:
    """
    Returns a list of shapes in a shapefile that intersect with a given bounding box.

    Parameters
    ----------
    bbox : shapely.geometry.box
        The bounding box to check for intersections

    crs : str
        The CRS of the bounding box

    shape_fpath : str
        The path to the shapefile

    Returns
    -------
    List[shapely.geometry.Geometry]
        A list of shapes that intersect with the bounding box
    """
    if fiona.listlayers(shape_fpath) is None:
        gdf = gpd.read_file(shape_fpath)
    elif layer is not None:
        gdf = gpd.read_file(shape_fpath, layer=layer)
    else:
        raise ValueError("No layer specified")

    gdf = gdf.to_crs(crs)
    shapes = [
        row["geometry"]
        for _, row in gdf.iterrows()
        if shapely.intersection(shape(row["geometry"]), bbox)
    ]

    return shapes
