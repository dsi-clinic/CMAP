"""This module provides a custom PyTorch GeoDataset for working with vector data.

The vector data represents labels or features in Kane County, Illinois. The data is
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

    def __init__(self, path: str, configs, balance_classes: bool = False) -> None:
        """Initialize a new KaneCounty dataset instance.

        Args:
            path: directory to the file to load
            configs: a tuple containing
                layer: specifying layer of GPKG
                labels: a dictionary containing a label mapping for masks
                patch_size: the patch size used for the model
                dest_crs: the coordinate reference system (CRS) to convert to
                res: resolution of the dataset in units of CRS
            balance_classes: whether to balance classes by repeating underrepresented ones

        Raises:
            FileNotFoundError: if no files are found in path
        """
        super().__init__()

        layer, labels, patch_size, dest_crs, res = configs

        gdf = self._load_and_prepare_data(path, layer, labels, dest_crs)
        self.gdf = gdf

        context_size = math.ceil(patch_size / 2 * res)
        print(f"context_size: {context_size}")
        print(f"patch_size: {patch_size}")
        print(f"res: {res}")
        self.context_size = context_size
        self._crs = dest_crs
        self._res = res

        self._populate_index(path, gdf, context_size, balance_classes)
        self.labels = labels
        self.colors = {i: self.all_colors[i] for i in labels.values()}
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
        gdf = gpd.read_file(path, layer=layer)[["BasinType", "geometry"]]
        gdf = gdf[gdf["BasinType"].isin(labels.keys())]
        gdf = gdf.to_crs(dest_crs)
        return gdf

    def _populate_index(self, path, gdf, context_size, balance_classes=False):
        """Populate the spatial index with data from the GeoDataFrame.

        Args:
            path: directory to the file to load
            gdf: GeoDataFrame containing the data
            context_size: size of the context around shapes for sampling
            balance_classes: whether to balance classes by repeating underrepresented ones
        """
        # get counts of each class
        class_counts = gdf["BasinType"].value_counts()
        total_samples = len(gdf)

        # log before-balancing distribution
        print("Class distribution before balancing:")
        for cls, count in class_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {cls}: {count} samples ({percentage:.2f}%)")

        if balance_classes:
            max_count = class_counts.max()
            class_multipliers = {
                cls: max(1, int(max_count / count))
                for cls, count in class_counts.items()
            }

            # calculate post-balancing counts and distribution
            balanced_counts = {
                cls: count * class_multipliers[cls]
                for cls, count in class_counts.items()
            }
            total_balanced = sum(balanced_counts.values())

            print("\nClass distribution after balancing:")
            for cls, count in balanced_counts.items():
                percentage = (count / total_balanced) * 100
                print(f"  {cls}: {count} samples ({percentage:.2f}%)")

            print(f"\nClass balancing multipliers: {class_multipliers}")

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

            # insert this item once normally
            self.index.insert(i, coords, row)
            i += 1

            # if balancing is enabled, repeat underrepresented classes
            if balance_classes:
                basin_type = row["BasinType"]
                # repeat this item (multiplier-1) more times
                for _ in range(class_multipliers[basin_type] - 1):
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
        """Returns the labels of the dataset."""
        return self.labels
