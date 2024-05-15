import os
import fiona
import geopandas as gpd
import planetary_computer
import pystac_client
import requests
from shapely.geometry import box


def get_geometry(fpath: str):
    """
    Read in a shapefile and yield each feature as a dictionary with id and geometry, transformed to EPSG:4326.

    Parameters
    ----------
    fpath : str
        Path to shapefile

    layer : int
        Optional layer index of the shapefile to read in

    Yields
    ------
    dict
        Dictionary with id and geometry
    """
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(fpath)

    # Transform the GeoDataFrame to EPSG:4326
    gdf = gdf.to_crs(epsg=4326)

    # Filter by FCODE to only include stream and river
    gdf = gdf[gdf['FCODE'] == 'STREAM/RIVER']

    # Yield each feature as a dictionary
    for index, row in gdf.iterrows():
        yield {'id': index, 'geometry': row.geometry}


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


def get_token() -> str:
    """
    Get a token for accessing naip images on the planetary computer catalog.

    Returns
    -------
    str
        Token for accessing naip images on the planetary computer catalog
    """
    res = requests.get(
        "https://planetarycomputer.microsoft.com/api/sas/v1/token/naip"
    )
    return res.json()["token"]


def split_geometry(geometry, max_size=0.01):
    """
    Splits the geometry into smaller parts using a grid overlay method.

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
            grid.append(box(minx + i * max_size, miny + j * max_size, 
                            minx + (i + 1) * max_size, miny + (j + 1) * max_size))

    # Intersect the grid with the original geometry to create the parts
    parts = [geometry.intersection(b) for b in grid if geometry.intersects(b)]
    return parts


def get_image_urls(
    geometry: fiona.model.Geometry,
    catalog: pystac_client.Client,
    date_range: str,
    img_type: str
) -> list:
    """
    Retrieves NAIP image urls corresponding to an area of interest.

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

    Returns
    -------
    list
        List of URLs of the images corresponding to the area of interest
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

    if len(items) == 0:
        print("No images found for this geometry")
        return []

    # Collect all URLs of intersecting images
    urls = [item.assets[img_type].href.split("?", 1)[0] for item in items]
    return urls


def download_image(
    url: str, save_dir: str, img_type: str, id: str = None
) -> None:
    """
    Downloads images from specified URL.

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
        if id is None:
            img_name = url.split("?", 1)[0].split("/")[-1]
        else:
            img_name = id + ".tif"
    else:
        if id is None:
            img_name = url.split("?", 1)[1].split("&")[1][5:]
        else:
            img_name = id + ".png"
    save_fpath = os.path.join(save_dir, img_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # retrieve the image and write it to disk; if the request fails, print the error
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


def get_river_images(
    img_type: str, data_fpath: str, save_dir: str
):
    """
    Retrieves NAIP images overlapping the geometries in a shapefile.

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
    geometry_stream = get_geometry(data_fpath)
    catalog = get_catalog()

    urls = set()
    for geometry in geometry_stream:
        try:
            url_list = get_image_urls(
                geometry["geometry"], catalog, "2021-01-01/2024-01-01", "image"
            )
            urls.update(url_list) # this takes out every single element in the list
        except Exception as e:
            print("Attempting to split geometry and retry...")
            parts = split_geometry(geometry["geometry"])
            for part in parts:
                try:
                    part_url_list = get_image_urls(part, catalog, "2021-01-01/2024-01-01", "image")
                    urls.update(part_url_list)
                except Exception as e:
                    print(f"Failed request for part: {str(e)}")
    for url in urls:
        url = url + "?" + get_token()
        download_image(url, save_dir, img_type)
