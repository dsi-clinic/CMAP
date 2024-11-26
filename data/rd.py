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

from pathlib import Path


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
        # Fix the dictionary comprehension to map keys to colors
        self.colors = {label_value: self.all_colors[label_value] for label_value in labels.values()}
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
        print("Initial River GeoDataFrame loaded:")
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
        print("gdf complete")
        
        return gdf

    def _populate_index(
        self,
        path,
        gdf,
        context_size,
        patch_size,
        reference_crs=4326,
        target_chip_size=0.005,
    ):
        """Populate spatial index with proportional chips based on CRS bounds."""

        mint, maxt = 0, sys.maxsize

        self.bounding_boxes = []
        i = 0  # initialize chip index counter

        from pyproj import CRS, Transformer
        from tqdm import tqdm

        # Get the CRS of the GeoDataFrame
        gdf_crs = gdf.crs
        print(f"GeoDataFrame CRS: {gdf_crs}")

        # Set up transformations if necessary
        if gdf_crs.to_epsg() != reference_crs:
            print(f"Transforming bounds to reference CRS {reference_crs}...")
            transformer = Transformer.from_crs(
                gdf_crs, CRS.from_epsg(reference_crs), always_xy=True
            )
            ref_minx, ref_miny, ref_maxx, ref_maxy = (
                transformer.transform_bounds(*gdf.total_bounds)
            )
        else:
            ref_minx, ref_miny, ref_maxx, ref_maxy = gdf.total_bounds

        print(
            f"CRS bounds: minx={ref_minx}, miny={ref_miny}, \
                maxx={ref_maxx}, maxy={ref_maxy}"
        )

        # Calculate the proportion of chip size to the reference CRS bounds
        ref_x_extent = ref_maxx - ref_minx
        proportional_factor = (
            target_chip_size / ref_x_extent
        )  # Use x_extent for consistency

        print(f"Proportional factor: {proportional_factor}")

        # Transform bounds back to target CRS if needed
        if gdf_crs.to_epsg() != reference_crs:
            print(f"Transforming bounds back to target CRS {gdf_crs}...")
            transformer = Transformer.from_crs(
                CRS.from_epsg(reference_crs), gdf_crs, always_xy=True
            )
            minx, miny, maxx, maxy = transformer.transform_bounds(
                ref_minx, ref_miny, ref_maxx, ref_maxy
            )
        else:
            minx, miny, maxx, maxy = gdf.total_bounds

        print(
            f"Target CRS bounds: minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}"
        )

        # Calculate the proportional chip size for the target CRS
        x_extent = maxx - minx
        y_extent = maxy - miny
        chip_size_x = x_extent * proportional_factor
        chip_size_y = y_extent * proportional_factor

        print(
            f"Chip sizes: chip_size_x={chip_size_x}, chip_size_y={chip_size_y}"
        )

        # Calculate total number of iterations for progress bar
        total_iterations = (x_extent / chip_size_x) * (y_extent / chip_size_y)

        # Iterate over the x-axis within the bounds of the gdf
        for x in tqdm(
            np.arange(minx, maxx, chip_size_x),
            desc="Processing x-axis",
            total=int(total_iterations),
        ):
            # Iterate over the y-axis within the bounds of the gdf
            for y in np.arange(miny, maxy, chip_size_y):
                # Create a rectangular chip
                chip = box(x, y, x + chip_size_x, y + chip_size_y)
                coords = (x, y, x + chip_size_x, y + chip_size_y, mint, maxt)

                # Find intersecting geometries in gdf
                intersecting_rows = gdf[gdf.intersects(chip)]
                for _, row in intersecting_rows.iterrows():
                    # Insert only intersecting chips with their corresponding row data
                    self.index.insert(i, coords, row[["FCODE", "geometry"]])
                    i += 1  # Increment the global index for each chip

        print(f"Total chips inserted: {i}")

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
