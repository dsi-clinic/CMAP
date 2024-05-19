import math
import sys
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from torchgeo.datasets import BoundingBox, GeoDataset


class KaneCounty(GeoDataset):
    """Vector ataset for Kane County labels stored as shapes in GeoDatabase."""

    all_bands = ["Label"]
    is_image = False

    # all colors and labels
    all_colors = {
        0: (0, 0, 0, 0),
        1: (215, 80, 48, 255),
        2: (49, 102, 80, 255),
        3: (239, 169, 74, 255),
        4: (100, 107, 99, 255),
        5: (89, 53, 31, 255),
        6: (2, 86, 105, 255),
        7: (207, 211, 205, 255),
        8: (195, 88, 49, 255),
        9: (144, 70, 132, 255),
        10: (29, 51, 74, 255),
        11: (71, 64, 46, 255),
        12: (114, 20, 34, 255),
        13: (37, 40, 80, 255),
        14: (94, 33, 41, 255),
        15: (255, 255, 255, 255),
    }
    all_labels = {
        0: "BACKGROUND",
        1: "POND",
        2: "WETLAND",
        3: "DRY BOTTOM - TURF",
        4: "DRY BOTTOM - MESIC PRAIRIE",
        5: "DEPRESSIONAL STORAGE",
        6: "DRY BOTTOM - WOODED",
        7: "POND - EXTENDED DRY",
        8: "PICP PARKING LOT",
        9: "DRY BOTTOM - GRAVEL",
        10: "UNDERGROUND",
        11: "UNDERGROUND VAULT",
        12: "PICP ALLEY",
        13: "INFILTRATION TRENCH",
        14: "BIORETENTION",
        15: "UNKNOWN",
    }

    def __init__(
        self,
        path: str,
        layer: int,
        label_col: str,
        labels: dict[int, str],
        patch_size: int,
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
            patch_size: the patch size used for the model
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

        # preserve context around shapes and ensure small shapes can be sampled
        context_size = math.ceil(patch_size / 2 * res)

        # Populate the dataset index
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

        self.label_col = label_col
        self.labels = labels
        self.colors = {i: self.all_colors[i] for i in labels.values()}
        self.labels_inverse = {v: k for k, v in labels.items()}
        self.context_size = context_size
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
            "mask": torch.tensor(masks).long(),
            "crs": self.crs,
            "bbox": query,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
