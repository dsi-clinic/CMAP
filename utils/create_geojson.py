import os
from typing import Dict, List, NoReturn

import geopandas as gpd
import rasterio
import shapely
import json
from shapely.geometry import box


def create_geojson(
    img_fpath: str,
    gdf: gpd.GeoDataFrame,
    save_dir: str,
    label_col: str,
    labels: Dict[str, int]
) -> NoReturn:
    """
    Creates a GeoJSON file containing polygons for shapefile features that intersect with a given image.

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
        intersecting_shapes = get_intersecting_shapes(
            bbox, crs, gdf, label_col, set(labels.keys())
        )

    # write intersecting shapes to GeoJSON files
    for label, shapes in intersecting_shapes.items():
        feature_collection = {
            "type": "FeatureCollection",
            "features": []
        }
        for shape in shapes:
            feature = {
                "type": "Feature",
                "properties": {
                    label_col: label,
                    "label_value": labels[label]
                },
                "geometry": shapely.geometry.mapping(shape)
            }
            feature_collection["features"].append(feature)

        # set output filename
        out_geojson = os.path.join(save_dir, f"{label}_features.geojson")

        # write GeoJSON to file
        with open(out_geojson, "w") as f:
            json.dump(feature_collection, f)

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
