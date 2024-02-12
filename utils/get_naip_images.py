import os
from typing import Any, Dict, Iterator, NoReturn

import fiona
import planetary_computer
import pystac_client
import requests
from constants import DATA_DIR
from shapely.geometry import shape


def get_geometry(fpath: str) -> Iterator[Dict[str, Any]]:
    """
    Read in a shapefile and yield each feature as a dictionary with id and geometry.

    Parameters
    ----------
    fpath : str
        Path to shapefile

    Yields
    ------
    dict
        Dictionary with id and geometry
    """
    with fiona.open(fpath) as sf:
        for feature in iter(sf):
            yield {"id": feature.id, "geometry": feature["geometry"]}


def get_catalog() -> pystac_client.Client:
    """
    Get the planetary computer catalog.

    Returns
    -------
    pystac_client.Client
        Pystac client for accessing the planetary computer catalog
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    return catalog


def area_of_overlap(
    image_geom: Dict[str, Any], area_of_interest: Dict[str, Any]
) -> float:
    """
    Calculates the area of overlap between two geometries based based on coordinates.
    Used specifically for calculating the percentage overlap between an image and
    area of interest.

    Parameters
    ----------
    image_geom : dict
        Geometry of the image. Dictionary contains geometry type and coordinates

    area_of_interest : dict
        Geometry of the area of interest. Dictionary contains geometry type and
        coordinates

    Returns
    -------
    float
        Percentage overlap between the image and area of interest, with area of
        interest as the denominator
    """
    target_area = shape(area_of_interest).area
    overlap_area = shape(image_geom).intersection(shape(area_of_interest)).area
    return overlap_area / target_area


def get_image_url(
    geometry: fiona.model.Geometry,
    catalog: pystac_client.Client,
    date_range: str,
    img_type: str,
) -> str:
    """
    Retrieves NAIP image url corresponding to an area of interest.
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

    Returns
    -------
    str
        URL of the image corresponding to the area of interest
    """
    if img_type not in ["rendered_preview", "image"]:
        raise ValueError(
            "img_type must be either 'rendered_preview' or 'image'"
        )

    # Search for NAIP images that intersect with the geometry
    search = catalog.search(
        collections=["naip"], intersects=geometry, datetime=date_range
    )
    items = search.item_collection()

    # if area of interest is not a point and more than one image is found,
    # get the image with the most overlap
    if geometry.type != "Point" and len(items) > 1:
        max_overlap = sorted(
            items,
            key=lambda x: area_of_overlap(x.geometry, dict(geometry)),
            reverse=True,
        )[0]
        url = max_overlap.assets[img_type].href
    # else just take the most recent image
    else:
        url = items[0].assets[img_type].href

    return url


def download_image(url: str, id: str, save_dir: str, img_type: str) -> NoReturn:
    """
    Downloads images from specified URL.

    Parameters
    ----------
    url : str
        URL of the image to be downloaded

    id: str
        Name/ID of the image to be used as file name

    save_dir : str
        Directory to save the image to

    img_type : str
        Type of image to retrieve. Options are "rendered_preview" and "image"
            "rendered_preview" is a preview image in png format
            "image" is the full image in GeoTIFF format
    """
    # create the save path
    if img_type == "image":
        img_name = id + ".tif"
    else:
        img_name = id + ".png"
    save_fpath = os.path.join(save_dir, img_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # retrieve the image and write it to disk
    res = requests.get(url, stream=True)
    if res.status_code != 200:
        print(
            f"Failed to download image: {res.text}. Status code: {res.status_code}"
        )
    else:
        print(f"Writing image {img_name} to disk")

    with open(save_fpath, "wb") as f:
        for chunk in res.iter_content(chunk_size=8192):
            f.write(chunk)


def main(img_type: str, data_fpath: str, save_dir: str):
    geometry_stream = get_geometry(data_fpath)
    catalog = get_catalog()

    urls = set()
    for geometry in geometry_stream:
        geom_id = geometry["id"]
        url = get_image_url(
            geometry["geometry"], catalog, "2021-01-01/2024-01-01", img_type
        )
        if url not in urls:
            urls.add(url)
            download_image(url, geom_id, save_dir, img_type)


if __name__ == "__main__":
    data_fpath = os.path.join(DATA_DIR, "GIBI_2021_shapefiles/GIBI_All.shp")
    save_dir = os.path.join(DATA_DIR, "GIBI-images")
    main("image", data_fpath, save_dir)