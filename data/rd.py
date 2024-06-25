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
from torchgeo.datasets import BoundingBox, GeoDataset


class RiverDataset(GeoDataset):
    """Vector dataset for river labels stored as shapes in GeoDatabase."""

    all_bands = ["Label"]
    is_image = False

    # all colors and labels
    all_colors = {
        0: (255, 255, 0, 255),
        1: (255, 255, 255, 255),
    }

    all_labels = {
        0: "STREAM/RIVER",
        1: "UNKNOWN",
    }

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

        layer, labels, patch_size, dest_crs, res = configs
        gdf = self._load_and_prepare_data(path, layer, labels, dest_crs)
        self.gdf = gdf

        context_size = math.ceil(patch_size / 2 * res)
        self.context_size = context_size
        self._crs = dest_crs
        self._res = res

        self._populate_index(path, gdf, context_size)
        self.labels = labels
        self.colors = {i: self.all_colors.get(i, (0, 0, 0, 0)) for i in labels.values()}
        self.labels_inverse = {v: k for k, v in labels.items()}

    def _load_and_prepare_data(self, path, layer, labels, dest_crs):
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
        gdf = gpd.read_file(path, layer = layer)

        # Filter by FCODE to only include stream and river
        gdf = gdf[gdf["FCODE"] == "STREAM/RIVER"]
        gdf = gdf[gdf["BasinType"].isin(labels.keys())]

        # Transform the GeoDataFrame to dest_crs
        gdf = gdf.to_crs(dest_crs)
        return gdf
    
    def _populate_index(self, path, gdf, context_size):
        """Populate the spatial index with data from the GeoDataFrame.

        Args:
            gdf: GeoDataFrame containing the data
            context_size: size of the context around shapes for sampling
        """
        i = 0
        for _, row in gdf.iterrows():
            minx, miny, maxx, maxy = row["geometry"].bounds
            mint, maxt = 0, sys.maxsize
            coords = (
                minx - context_size,
                maxx + context_size,
                miny - context_size,
                maxy + context_size,
                mint,
                maxt,
            )
            self.index.insert(i, coords, row)
            i += 1
        if i == 0:
            msg = f"No {self.__class__.__name__} data was found in `path='{path}'`"
            raise FileNotFoundError(msg)
        
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
        for obj in objs:
            shape = obj["geometry"]
            label = self.labels[obj["BasinType"]]
            shapes.append((shape, label))

        width = (query.maxx - query.minx) / self._res
        height = (query.maxy - query.miny) / self._res
        transform = rasterio.transform.from_bounds(
            query.minx, query.miny, query.maxx, query.maxy, width, height
        )
        if shapes and min((round(height), round(width))) != 0:
            masks = rasterio.features.rasterize(
                shapes,
                out_shape=(round(height), round(width)),
                transform=transform,
            )
        else:
            # If no features are found in this query, return an empty mask
            # with the default fill value and dtype used by rasterize
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
