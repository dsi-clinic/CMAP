"""NAIP image retrieval from shapefiles.

Retrieves NAIP imagery from Planetary Computer's STAC catalog based on shapefile geometries.
Handles coordinate transformations and downloads images overlapping specified areas.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import fiona
import geopandas as gpd
import planetary_computer
import pystac_client
import requests
from fiona import transform
from shapely.geometry import box, shape


def get_geometry(fpath: str, layer: int = None) -> Iterator[dict[str, Any]]:
    """Read in a shapefile and yield each feature as a dictionary with id and geometry.

    Parameters
    ----------
    fpath : str
        Path to shapefile

    layer : int
        Index of the layer in the shapefile to read in

    Yields:
    ------
    dict
        Dictionary with id and geometry
    """
    with fiona.open(fpath, layer=layer) as shapefile:
        for feature in iter(shapefile):
            # transform the geometry to EPSG:4326 (standard lat/long)
            geom = transform.transform_geom(
                shapefile.crs,
                "EPSG:4326",
                feature["geometry"],
                antimeridian_cutting=True,
            )
            yield {"id": feature.id, "geometry": geom}


def get_river_geometry(fpath: str):
    """Read in a river shapefile and yield each feature as a dictionary.

    Parameters
    ----------
    fpath : str
        Path to shapefile

    layer : int
        Optional layer index of the shapefile to read in

    Yields:
    ------
    dict
        Dictionary with id and geometry
    """
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(fpath)

    # Transform the GeoDataFrame to EPSG:4326
    gdf = gdf.to_crs(epsg=4326)

    # Filter by FCODE to only include stream and river
    gdf = gdf[gdf["FCODE"] == "STREAM/RIVER"]

    # Yield each feature as a dictionary
    for index, row in gdf.iterrows():
        yield {"id": index, "geometry": row.geometry}


def get_catalog() -> pystac_client.Client:
    """Get the planetary computer catalog.

    Returns:
    -------
    pystac_client.Client
        Pystac client for accessing the planetary computer catalog
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    return catalog


def get_token() -> str:
    """Get a token for accessing naip images on the planetary computer catalog.

    Returns:
    -------
    str
        Token for accessing naip images on the planetary computer catalog
    """
    res = requests.get(
        "https://planetarycomputer.microsoft.com/api/sas/v1/token/naip",
        timeout=10,
    )
    return res.json()["token"]


def area_of_overlap(
    image_geom: dict[str, Any], area_of_interest: dict[str, Any]
) -> float:
    """Calculate the area of overlap between two geometries based based on coordinates.

    Used specifically for calculating the percentage overlap between an image and
    area of interest.

    Parameters
    ----------
    image_geom : dict
        Geometry of the image. Dictionary contains geometry type and coordinates

    area_of_interest : dict
        Geometry of the area of interest. Dictionary contains geometry type and
        coordinates

    Returns:
    -------
    float
        Percentage overlap between the image and area of interest, with area of
        interest as the denominator
    """
    target_area = shape(area_of_interest).area
    overlap_area = shape(image_geom).intersection(shape(area_of_interest)).area
    return overlap_area / target_area


def split_geometry(geometry, max_size=0.01):
    """Splits the geometry into smaller parts using a grid overlay method.

    Parameters:
    - geometry: The geometry to split (shapely.geometry).
    - max_size: The maximum size of each part (in degrees, for geographic coordinates).

    Returns:
    - A list of smaller geometries.
    """
    minx, miny, maxx, maxy = geometry.bounds
    x_range = maxx - minx
    y_range = maxy - miny
    x_divisions = int(x_range / max_size) + 1
    y_divisions = int(y_range / max_size) + 1

    # Create a grid of bounding boxes (as shapely geometries)
    grid = []
    for i in range(x_divisions):
        for j in range(y_divisions):
            grid.append(
                box(
                    minx + i * max_size,
                    miny + j * max_size,
                    minx + (i + 1) * max_size,
                    miny + (j + 1) * max_size,
                )
            )

    # Intersect the grid with the original geometry to create the parts
    parts = [geometry.intersection(b) for b in grid if geometry.intersects(b)]
    return parts


def get_image_url(
    geometry: dict[str, Any],
    catalog: pystac_client.Client,
    date_range: str,
    img_type: str,
) -> str:
    """Retrieves NAIP image url corresponding to an area of interest.

    If an area of interest covers more than one image, the url for the image with
    the most overlap is retrieved.

    Parameters
    ----------
    geometry : fiona.model.Geometry
        Similar to a GeoJSON, contains geometry data for shape from shapefile
        corresponding to area of interest

    catalog : pystac_client.Client
        Planetary computer catalog to search for the image

    date_range : str
        Date range to search for imagery in the format "YYYY-MM-DD/YYYY-MM-DD"

    img_type : str
        Type of image to retrieve. Options are "rendered_preview" and "image"
            "rendered_preview" is a preview image in png format
            "image" is the full image in GeoTIFF format

    Returns:
    -------
    str
        URL of the image corresponding to the area of interest
    """
    if img_type not in ["rendered_preview", "image"]:
        raise ValueError("img_type must be either 'rendered_preview' or 'image'")

    search = catalog.search(
        collections=["naip"], intersects=geometry, datetime=date_range
    )
    items = search.item_collection()

    if len(items) == 0:
        print("No images found for this geometry")
        return None

    if geometry["type"] != "Point" and len(items) > 1:
        max_overlap = sorted(
            items,
            key=lambda x: area_of_overlap(x.geometry, geometry),
            reverse=True,
        )[0]
        return max_overlap.assets[img_type].href.split("?", 1)[0]
    return items[0].assets[img_type].href.split("?", 1)[0]


def get_image_urls(
    geometry: dict[str, Any],
    catalog: pystac_client.Client,
    date_range: str,
    img_type: str,
) -> list:
    """Retrieves NAIP image urls corresponding to an area of interest.

    Parameters
    ----------
    geometry : fiona.model.Geometry
        Similar to a GeoJSON, contains geometry data for shape from shapefile
        corresponding to area of interest

    catalog : pystac_client.Client
        Planetary computer catalog to search for the image

    date_range : str
        Date range to search for imagery in the format "YYYY-MM-DD/YYYY-MM-DD"

    img_type : str
        Type of image to retrieve. Options are "rendered_preview" and "image"

    Returns:
    -------
    list
        List of URLs of the images corresponding to the area of interest
    """
    if img_type not in ["rendered_preview", "image"]:
        raise ValueError("img_type must be either 'rendered_preview' or 'image'")

    # Search for NAIP images that intersect with the geometry
    search = catalog.search(
        collections=["naip"], intersects=geometry, datetime=date_range
    )
    items = search.item_collection()

    if len(items) == 0:
        print("No images found for this geometry")
        return []

    # Collect all URLs of intersecting images
    urls = [item.assets[img_type].href.split("?", 1)[0] for item in items]
    return urls


def download_image(
    url: str, save_dir: str, img_type: str, image_id: str = None
) -> None:
    """Downloads images from specified URL.

    Parameters
    ----------
    url : str
        URL of the image to be downloaded

    save_dir : str
        Directory to save the image to

    img_type : str
        Type of image to retrieve. Options are "rendered_preview" and "image"
            "rendered_preview" is a preview image in png format
            "image" is the full image in GeoTIFF format

    id: str
        Name/ID of the image to be used as file name
    """
    # create the save path
    if img_type == "image":
        if image_id is None:
            img_name = url.split("?", 1)[0].split("/")
        else:
            img_name = image_id + ".tif"
    else:
        if image_id is None:
            img_name = url.split("?", 1)[1].split("&")[1][5:]
        else:
            img_name = image_id + ".png"
    save_fpath = Path(save_dir) / img_name

    if not Path.exists(save_dir):
        Path.mkdir(save_dir)

    # retrieve the image and write it to disk; if the request fails, print the error
    res = requests.get(url, stream=True, timeout=10)
    failure_code = 200
    if res.status_code != failure_code:
        print(f"Failed to download image: {res.text}. Status code: {res.status_code}")
    else:
        print(f"Writing image {img_name} to disk")
        with Path.open(save_fpath, "wb") as file:
            for chunk in res.iter_content(chunk_size=8192):
                file.write(chunk)


def get_images(img_type: str, data_fpath: str, save_dir: str, layer: int = None):
    """Retrieves NAIP images overlapping the geometries in a shapefile.

    Parameters
    ----------
    img_type : str
        Type of image to retrieve. Options are "rendered_preview" and "image"
            "rendered_preview" is a preview image in png format
            "image" is the full image in GeoTIFF format

    data_fpath : str
        Path to shapefile containing geometries to retrieve images for

    save_dir : str
        Directory to save the images to

    layer : int
        Index of the layer in the shapefile to get images for
    """
    geometry_stream = get_geometry(data_fpath, layer)
    catalog = get_catalog()

    urls = set()
    for geometry in geometry_stream:
        try:
            url = get_image_url(
                geometry["geometry"], catalog, "2021-01-01/2024-01-01", img_type
            )
            if url and url not in urls:
                urls.add(url)
        except requests.RequestException as req_error:
            print(f"Network error retrieving image url: {str(req_error)}")
    for url in urls:
        url = url + "?" + get_token()
        download_image(url, save_dir, img_type)


def get_river_images(img_type: str, data_fpath: str, save_dir: str):
    """Retrieves NAIP images overlapping the geometries in a shapefile.

    Parameters
    ----------
    img_type : str
        Type of image to retrieve. Options are "rendered_preview" and "image"
            "rendered_preview" is a preview image in png format
            "image" is the full image in GeoTIFF format

    data_fpath : str
        Path to shapefile containing geometries to retrieve images for

    save_dir : str
        Directory to save the images to

    layer : int
        Index of the layer in the shapefile to get images for
    """
    geometry_stream = get_river_geometry(data_fpath)
    catalog = get_catalog()

    urls = set()
    for geometry in geometry_stream:
        try:
            url_list = get_image_urls(
                geometry["geometry"], catalog, "2021-01-01/2024-01-01", "image"
            )
            urls.update(url_list)
        except requests.RequestException as req_error:
            print(f"Network error while retrieving URLs: {str(req_error)}")
            parts = split_geometry(geometry["geometry"])
            for part in parts:
                part_url_list = get_image_urls(
                    part, catalog, "2021-01-01/2024-01-01", "image"
                )
                urls.update(part_url_list)
    for url in urls:
        url = url + "?" + get_token()
        download_image(url, save_dir, img_type)
