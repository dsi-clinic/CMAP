"""Custom PyTorch GeoDataset for GIBI point data.

This module reads point features from a shapefile and rasterizes them to create masks.
Since the data consists of points (not polygons), each point is mapped directly to a pixel,
which may cause slight positional offsets without buffering.
"""

import math
import sys

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import rasterize
from torchgeo.datasets import BoundingBox, GeoDataset


class GIBIDataset(GeoDataset):
    """Custom vector dataset for GIBI point-based data."""

    def __init__(self, path: str, configs) -> None:
        """Initialize a new GIBIDataset instance.

        Args:
            path: directory to the shapefile to load
            configs: a tuple containing
                layer: layer to read (for formats like GPKG)
                labels: a dictionary containing label mappings for masks
                patch_size: size of patches used for sampling
                dest_crs: the CRS to convert to
                res: resolution of the dataset
        """
        super().__init__()

        layer, labels, patch_size, dest_crs, res = configs

        # load, filter geospatial data from file, convert it to the target CRS
        self.gdf = self._load_and_prepare_data(path, layer, labels, dest_crs)
        self.context_size = math.ceil(patch_size / 2 * res)
        self._crs = dest_crs
        self._res = res

        self._populate_index(path, self.gdf, self.context_size)
        # store dictionary of labels, mapping feature types to integer labels used in the output mask
        self.labels = labels

    def _load_and_prepare_data(self, path, layer, labels, dest_crs):
        """Load and filter the GeoDataFrame."""
        if layer:  # Only use 'layer' if it's provided
            gdf = gpd.read_file(path, layer=layer)
        else:
            gdf = gpd.read_file(path)

        gdf = gdf[["GI_Type", "geometry"]]
        gdf = gdf[gdf["GI_Type"].isin(labels.keys())]
        gdf = gdf.to_crs(dest_crs)
        return gdf

    def _populate_index(self, path, gdf, context_size):
        """Populate spatial index with the filtered data."""
        i = 0
        for _, row in gdf.iterrows():
            minx, miny, maxx, maxy = row["geometry"].bounds
            coords = (
                minx - context_size,
                maxx + context_size,
                miny - context_size,
                maxy + context_size,
                0,
                sys.maxsize,
            )
            self.index.insert(i, coords, row)
            i += 1

        if i == 0:
            raise FileNotFoundError(f"No data found in `path='{path}'`")

    def __getitem__(self, query: BoundingBox):
        """Retrieve point data and generate a rasterized mask."""
        hits = list(self.index.intersection(tuple(query), objects=True))
        objs = [hit.object for hit in hits]

        if not objs:
            raise IndexError(f"Query {query} not found in index.")

        shapes = []
        for obj in objs:
            shape = obj["geometry"]
            label = self.labels.get(obj["GI_Type"], 0)
            shapes.append((shape, label))

        width = (query.maxx - query.minx) / self._res
        height = (query.maxy - query.miny) / self._res
        transform = rasterio.transform.from_bounds(
            query.minx, query.miny, query.maxx, query.maxy, width, height
        )

        masks = rasterize(
            shapes,
            out_shape=(round(height), round(width)),
            transform=transform,
            dtype=np.uint8,
        )

        sample = {
            "mask": torch.tensor(masks).long(),
            "crs": self.crs,
            "bbox": query,
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __getlabels__(self):
        """Return labels used in the dataset."""
        return self.labels
