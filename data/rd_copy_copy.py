"""
This module provides a custom PyTorch GeoDataset for working with vector data
representing river labels or features in the River images dataset. The vector data is
stored as shapes in a GeoDatabase file, and this module allows for retrieving
samples of labels or features as masks or rasterized images within specified
bounding boxes.
"""

import math
import sys

import geopandas as gpd
import numpy as np
import rasterio
import torch

# from rtree import index  # or any spatial index you are using
from shapely.geometry import box
from torchgeo.datasets import BoundingBox, GeoDataset

# from tqdm import tqdm


class RiverDataset(GeoDataset):
    """Vector dataset for river labels stored as shapes in GeoDatabase."""

    all_bands = ["Label"]
    is_image = False

    # all colors and labels
    all_colors = {
        0: (0, 0, 0, 0),
        1: (255, 255, 0, 255),
    }

    all_labels = {0: "UNKNOWN", 1: "STREAM/RIVER"}

    def __init__(self, path: str, configs) -> None:
        """Initialize a new river dataset instance.

        Args:
            path: directory to the file to load
            configs: a tuple containing
                layer: specifying layer of GPKG
                labels: a dictionary containing a label mapping for masks
                patch_size: the patch size used for the model
                dest_crs: the coordinate reference system (CRS) to convert to
                res: resolution of the dataset in units of CRS

        Raises:
            FileNotFoundError: if no files are found in path
        """
        super().__init__()

        labels, patch_size, dest_crs, res = configs
        gdf = self._load_and_prepare_data(path, dest_crs)
        self.gdf = gdf

        # Debug prints
        print(f"Configs received: {configs}")
        print(f"Type of labels: {type(labels)}, Content: {labels}")

        context_size = math.ceil(patch_size / 2 * res)
        self.context_size = context_size
        self._crs = dest_crs
        self._res = res

        self._populate_index(path, gdf, context_size, patch_size)
        self.labels = labels
        self.colors = {i: self.all_colors[i] for i in labels.values()}
        self.labels_inverse = {v: k for k, v in labels.items()}

        # Debug print
        print(f"Initializing RiverDataset with configs: {configs}")

    def _load_and_prepare_data(self, path, dest_crs):
        """Load and prepare the GeoDataFrame.

        Args:
            path: directory to the file to load
            layer: specifying layer of GPKG
            labels: a dictionary containing a label mapping for masks
            dest_crs: the coordinate reference system (CRS) to convert to

        Returns:
            gdf: A GeoDataFrame filtered and converted to the target CRS
        """

        # Read the shapefile into a GeoDataFrame
        gdf = gpd.read_file(path)

        # debug print
        print("Initial GeoDataFrame loaded:")
        print(gdf.head())

        # Debug print: Check unique values in FCODE
        print(f"Unique FCODE values: {gdf['FCODE'].unique()}")

        # Filter by FCODE to only include stream and river
        gdf = gdf[gdf["FCODE"] == "STREAM/RIVER"]

        # debug print
        print("GeoDataFrame after filtering by FCODE:")
        print(gdf.head())

        # Transform the GeoDataFrame to dest_crs
        gdf = gdf.to_crs(dest_crs)
        return gdf

    def _populate_index(self, path, gdf, context_size, patch_size):
        """Populate spatial index with chips intersecting geometries in gdf."""

        chip_size = 0.005
        mint, maxt = 0, sys.maxsize

        self.bounding_boxes = []
        i = 0  # initialize chip index counter

        # get the total bounds for the entire gdf
        minx, miny, maxx, maxy = gdf.total_bounds
        x = minx

        # iterate over the x-axis within the bounds of the gdf
        while x < maxx:
            y = miny

            # iterate over the y-axis within the bounds of the gdf
            while y < maxy:
                # create a rectangular chip
                chip = box(x, y, x + chip_size, y + chip_size)
                coords = (x, y, x + chip_size, y + chip_size, mint, maxt)

                # find intersecting geometries in gdf
                intersecting_rows = gdf[gdf.intersects(chip)]
                for _, row in intersecting_rows.iterrows():
                    # insert only intersecting chips with their corresponding row data
                    self.index.insert(i, coords, row)
                    i += 1  # increment the global index for each chip
                    print(i, coords, row)

                y += chip_size  # move to the next chip in the y-direction

            x += chip_size  # move to the next chip in the x-direction

    def __getitem__(self, query: BoundingBox):
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        objs = [hit.object for hit in hits]

        if not objs:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        shapes = []
        # print("objs in __getitem__:", len(objs))
        for obj in objs:
            shape = obj["geometry"]
            label = self.labels[obj["FCODE"]]
            shapes.append((shape, label))
        # print("len of shapes in __getitem__:", len(shapes))

        width = (query.maxx - query.minx) / self._res
        height = (query.maxy - query.miny) / self._res
        # print("x range in bounds:", width)
        # print("query in __getitem__:", query)
        # print("width in __getitem__:", width)
        # print("height in __getitem__:", height)
        transform = rasterio.transform.from_bounds(
            query.minx, query.miny, query.maxx, query.maxy, width, height
        )
        if shapes and min((round(height), round(width))) != 0:
            # print("found features")
            # print("shapes in __getitem__:", shapes)
            masks = rasterio.features.rasterize(
                shapes,
                out_shape=(round(height), round(width)),
                transform=transform,
            )
            # print("sum of masks in __getitem__:", np.sum(masks))
            # print("mask shape in __getitem__:", masks.shape)
            # print("unique entries in mask in __getitem__:", np.unique(masks))
        else:
            # print("no features found in this query")
            masks = np.zeros((round(height), round(width)), dtype=np.uint8)

        sample = {
            "mask": torch.Tensor(masks).long(),
            "crs": self.crs,
            "bbox": query,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __getlabels__(self):
        """
        Returns the labels of the dataset.
        """
        return self.labels
