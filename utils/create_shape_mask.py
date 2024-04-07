import os
from typing import Dict, List

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from einops import rearrange
from rasterio.mask import mask
from shapely.geometry import box


def create_mask(
    img_fpath: str,
    gdf: gpd.GeoDataFrame,
    save_dir: str,
    label_col: str,
    labels: Dict[str, int],
) -> None:
    """
    Creates a mask for shapefile features that intersect with a given image.

    Parameters
    ----------
    img_fpath : str
        Path to image

    gdf : geopandas.GeoDataFrame
        A geopandas dataframe containing geometries

    save_dir : str
        Path to save directory

    label_col : str
        The column containing the labels of the shapes

    labels : Dict[str, int]
        A dict containing the labels of the shapes and their corresponding values
    """

    # open image and get bounding box
    with rasterio.open(img_fpath) as src:
        bbox = box(*src.bounds)
        crs = src.crs

        # extract intersecting shapes
        shapes = get_intersecting_shapes(bbox, crs, gdf, label_col, set(labels.keys()))

        # create empty mask and copy metadata from image
        output = np.zeros((src.count, src.height, src.width), dtype=np.uint8)
        out_meta = src.meta.copy()

        # create mask for each label and combine into one mask
        for label in shapes:
            out_img, out_transform = mask(src, shapes[label], crop=False)
            out_img_unique = np.where(out_img != 0, labels[label], out_img)
            output = np.maximum(output, out_img_unique)

    # take one layer because all layers are the same
    msk = np.array([output[0]]).astype("uint8")

    # set output filename and metadata
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    out_tif = os.path.join(save_dir, f"mask_{os.path.basename(img_fpath)}")
    out_meta.update(
        {
            "driver": "GTiff",
            "dtype": "uint8",
            "height": output.shape[1],
            "width": output.shape[2],
            "transform": out_transform,
            "count": 1,
            "crs": crs,
        }
    )

    # write mask to file
    with rasterio.open(out_tif, "w", **out_meta) as dest:
        dest.write(msk.astype("uint8"))


def get_intersecting_shapes(
    bbox: shapely.Polygon,
    crs: str,
    gdf: gpd.GeoDataFrame,
    label_col: str,
    labels: List[str],
) -> Dict[str, List[shapely.Geometry]]:
    """
    Returns a dict containing a list of shapes that intersect with a given bounding
    box for each key in labels.

    Parameters
    ----------
    bbox : shapely.Polygon
        The bounding box to check for intersections

    crs : str
        The CRS of the bounding box

    gdf : geopandas.GeoDataFrame
        A geopandas dataframe containing geometries

    label_col : str
        The column containing the labels of the shapes

    labels : Set[str]
        The labels of the shapes to extract from the gdf

    Returns
    -------
    Dict[str, List[shapely.Geometry]]
        A dict with label as key and a list of shapes as value
    """
    gdf.to_crs(crs, inplace=True)
    intersecting = gdf[gdf["geometry"].intersects(bbox)]

    shapes = {}
    for _, row in intersecting.iterrows():
        if row[label_col] in labels:
            if row[label_col] in shapes:
                shapes[row[label_col]].append(row["geometry"])
            else:
                shapes[row[label_col]] = [row["geometry"]]

    return shapes
