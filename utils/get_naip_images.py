import fiona
import os
import pystac_client
import planetary_computer
import requests

from shapely.geometry import shape
from typing import Iterator, NoReturn, Any, Dict

from constants import DATA_DIR


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


def get_image(
    geometry: Dict[str, Any], date_range: str, img_type: str, save_dir: str
) -> NoReturn:
    """
    Retrieves NAIP imagery corresponding to an area of interest and writes it to disk.
    If an area of interest covers more than one image, the image with the most overlap
    is retrieved.

    Parameters
    ----------
    geometry : dict
        Dictionary containing the id and geometry of the area of interest

    date_range : str
        Date range to search for imagery in the format "YYYY-MM-DD/YYYY-MM-DD"

    img_type : str
        Type of image to retrieve. Options are "rendered_preview" and "image"
            "rendered_preview" is a preview image in png format
            "image" is the full image in GeoTIFF format

    save_dir : str
        Directory to save the image to
    """
    if img_type not in ["rendered_preview", "image"]:
        raise ValueError("img_type must be either 'rendered_preview' or 'image'")

    # access the planetary computer catalog and retrieve NAIP images that
    # intersect with the geometry
    id = geometry["id"]
    area_of_interest = geometry["geometry"]

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["naip"], intersects=area_of_interest, datetime=date_range
    )
    items = search.item_collection()

    # if area of interest is not a point and more than one image is found,
    # get the image with the most overlap
    if area_of_interest.type != "Point" and len(items) > 1:
        max_overlap = sorted(
            items,
            key=lambda x: area_of_overlap(x.geometry, dict(area_of_interest)),
            reverse=True,
        )[0]
        url = max_overlap.assets[img_type].href
    # else just take the most recent image
    else:
        url = items[0].assets[img_type].href

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
        f"Failed to download image: {res.text}. Status code: {res.status_code}"
    else:
        print(f"Writing image {img_name} to disk")

    with open(save_fpath, "wb") as f:
        for chunk in res.iter_content(chunk_size=8192):
            f.write(chunk)


def main():
    data_fpath = os.path.join(DATA_DIR, "GIBI_2021_shapefiles/GIBI_All.shp")
    geometry_stream = get_geometry(data_fpath)
    save_dir = os.path.join(DATA_DIR, "GIBI-images")

    for geometry in geometry_stream:
        get_image(
            geometry,
            "2021-01-01/2024-01-01",
            "image",
            save_dir,
        )


if __name__ == "__main__":
    main()
