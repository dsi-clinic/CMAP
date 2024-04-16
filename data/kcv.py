import sys
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from torchgeo.datasets import BoundingBox, GeoDataset


class KaneCounty(GeoDataset):
    """Abstract base class for :class:`GeoDataset` stored as vector files."""

    all_bands = ["Label"]
    is_image = False

    def __init__(
        self,
        path: str,
        layer: int,
        label_col: str,
        labels: dict[int, str],
        dest_crs: Optional[CRS] = None,
        res: float = 0.0001,
    ) -> None:
        """Initialize a new KaneCounty dataset instance.

        Args:
            path: directory to the file to load
            layer: specifying layer of GPKG
            label_col: name of the dataset property that has the label to be
                rasterized into the mask
            labels: a dictionary containing a label mapping for masks
            dest_crs: the coordinate reference system (CRS) to convert to
            res: resolution of the dataset in units of CRS

        Raises:
            FileNotFoundError: if no files are found in path
        """
        super().__init__()

        # read from the file, filter by label, and convert crs
        gdf = gpd.read_file(path, layer=layer)[[label_col, "geometry"]]
        gdf = gdf[gdf[label_col].isin(labels.keys())]
        gdf = gdf.to_crs(dest_crs)
        self.gdf = gdf

        # Populate the dataset index
        i = 0
        for _, row in gdf.iterrows():
            minx, miny, maxx, maxy = row["geometry"].bounds
            mint, maxt = 0, sys.maxsize
            coords = (minx, maxx, miny, maxy, mint, maxt)
            self.index.insert(i, coords, row)
            i += 1
        if i == 0:
            msg = f"No {self.__class__.__name__} data was found in `path='{path}'`"
            raise FileNotFoundError(msg)

        self.label_col = label_col
        self.labels = labels
        self._crs = dest_crs
        self._res = res

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
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
            label = self.labels[obj[self.label_col]]
            shapes.append((shape, label))

        width = (query.maxx - query.minx) / self.res
        height = (query.maxy - query.miny) / self.res
        transform = rasterio.transform.from_bounds(
            query.minx, query.miny, query.maxx, query.maxy, width, height
        )
        if shapes:
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
            "mask": torch.tensor(masks).long(),
            "crs": self.crs,
            "bbox": query,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
