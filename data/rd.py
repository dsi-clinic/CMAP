"""This module provides a custom PyTorch GeoDataset for working with vector data.

The vector data represents river labels or features in the River images dataset.
It is stored as shapes in a GeoDatabase file, and this module allows for retrieving
samples of labels or features as masks or rasterized images within specified
bounding boxes. The KC gdf is added to the RD gdf and chips are generated using
this combined gdf.
"""

import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from shapely.geometry import box
from torchgeo.datasets import BoundingBox, GeoDataset

from configs.config import (
    KC_LABELS,
    KC_LAYER,
    KC_SHAPE_FILENAME,
    KC_SHAPE_ROOT,
)
from data.kc import KaneCounty

# Add the parent directory (contains both 'configs' and 'data') to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))


"""
This module provides a custom PyTorch GeoDataset for working with vector data
representing labels or features in Kane County, Illinois. The vector data is
stored as shapes in a GeoDatabase file, and this module allows for retrieving
samples of labels or features as masks or rasterized images within specified
bounding boxes.
"""


# Load the RiverDataset class
# This will also contain the KC class; chips will be generated using the combined KC and RD gdfs


class RiverDataset(GeoDataset):
    """Vector dataset for river labels stored as shapes in GeoDatabase."""

    all_bands = ["Label"]
    is_image = False

    all_colors = {
        0: (0, 0, 0, 0),
        1: (255, 255, 0, 255),
    }

    def __init__(self, path: str, rd_configs, kc: False) -> None:
        """Initialize a new river dataset instance.

        Args:
            path: directory to the file to load
            rd_configs: a tuple containing
                layer: specifying layer of GPKG
                labels: a dictionary containing a label mapping for masks
                patch_size: the patch size used for the model
                dest_crs: the coordinate reference system (CRS) to convert to
                res: resolution of the dataset in units of CRS
            kc: a boolean to include the KC dataset or not; default is True

        Raises:
            FileNotFoundError: if no files are found in path
        """
        super().__init__()

        patch_size, dest_crs, res = rd_configs
        gdf = self._load_and_prepare_data(path, dest_crs)
        self.gdf = gdf

        box_size = math.ceil(patch_size / 2 * res) * 2
        self.box_size = box_size
        self._crs = dest_crs
        self._res = res

        self.labels = {"BACKGROUND": 0, "STREAM/RIVER": 1}
        self.colors = {
            label_value: self.all_colors[label_value]
            for label_value in self.labels.values()
        }
        # Get the last index used by RD population
        last_rd_idx = self._populate_index(self.gdf, box_size=box_size)

        if kc:
            kc_shape_path = Path(KC_SHAPE_ROOT) / KC_SHAPE_FILENAME
            kc_config = (KC_LAYER, KC_LABELS, patch_size, dest_crs, res)
            kc_dataset = KaneCounty(kc_shape_path, kc_config)
            print(f"river dataset crs {self.crs}")
            print(f"KC crs {kc_dataset.crs}")

            # Merge indices and labels
            current_idx = last_rd_idx  # Start KC IDs after RD IDs
            max_kc = 5000
            kc_added_count = 0

            for item in kc_dataset.index.intersection(
                kc_dataset.index.bounds,
                objects=True,
            ):
                # Convert KC object to same format as RD objects
                obj_dict = {
                    "BasinType": item.object["BasinType"],
                    "geometry": item.object["geometry"],
                }
                self.index.insert(
                    current_idx,  # Use unique index ID
                    item.bounds,
                    [obj_dict],  # wrap in list to match RD format
                )
                # print(f"KC inserting ID {current_idx} coords {item.bounds}") # Optional: keep for debugging
                current_idx += 1
                kc_added_count += 1
                if kc_added_count >= max_kc:  # Check against number added, not index
                    break

            print(f"Added {kc_added_count} KC index entries.")

            # Preserve label ordering when combining
            rd_labels = (
                self.labels.copy()
            )  # Make a copy to avoid modifying the original
            combined_labels = {}
            next_idx = 0

            # First add RD labels with their original indices
            for label, idx in rd_labels.items():
                combined_labels[label] = idx  # Keep original indices
                next_idx = max(next_idx, idx + 1)

            # Then add KC labels that aren't already present
            for label in KC_LABELS.keys():
                if label not in combined_labels:
                    combined_labels[label] = next_idx
                    next_idx += 1

            self.labels = combined_labels

            # Preserve color mapping with correct indices
            combined_colors = {}
            for label, idx in combined_labels.items():
                if label in rd_labels:
                    # Use RD color with the correct index
                    combined_colors[idx] = self.all_colors[rd_labels[label]]
                else:
                    # Use KC color with the correct index
                    combined_colors[idx] = kc_dataset.all_colors[KC_LABELS[label]]

            self.colors = combined_colors

        self.labels_inverse = {v: k for k, v in self.labels.items()}
        print(f"Initialized RiverDataset with configs: {rd_configs}")

    def _load_and_prepare_data(self, path, dest_crs):
        """Load and prepare the GeoDataFrame.

        Args:
            path: directory to the file to load
            dest_crs: the coordinate reference system (CRS) to convert to

        Returns:
            gdf: A GeoDataFrame filtered and converted to the target CRS
        """
        # Read the shapefile into a GeoDataFrame
        gdf = gpd.read_file(path)

        # Transform the GeoDataFrame to dest_crs
        gdf = gdf.to_crs(dest_crs)
        gdf = gdf[gdf["FCODE"] == "STREAM/RIVER"]
        gdf["BasinType"] = gdf["FCODE"]

        return gdf

    # FIXME: reducing total_points for faster debugging, need to restore to 900
    def _populate_index(self, gdf, total_points=176, box_size=150):
        """Populate spatial index with random bounding boxes across the dataset extent."""
        import time

        import numpy as np

        start = time.time()
        mint, maxt = 0, sys.maxsize
        half_box = box_size / 2

        # Get the total bounds of the GeoDataFrame
        total_bounds = gdf.total_bounds
        minx, miny, maxx, maxy = total_bounds

        print(f"Total bounds: {total_bounds}")
        print(f"Generating {total_points} random query boxes of size {box_size}...")

        i = 0  # query box counter
        added_count = 0
        max_attempts_factor = 5  # Try more times to find boxes with intersections
        max_attempts = total_points * max_attempts_factor
        attempts = 0

        while added_count < total_points and attempts < max_attempts:
            attempts += 1
            # Generate a random center point within the total bounds
            center_x = np.random.uniform(minx + half_box, maxx - half_box)
            center_y = np.random.uniform(miny + half_box, maxy - half_box)

            # Create the bounding box coordinates
            query_minx = center_x - half_box
            query_maxx = center_x + half_box
            query_miny = center_y - half_box
            query_maxy = center_y + half_box
            coords = (query_minx, query_maxx, query_miny, query_maxy, mint, maxt)

            # Create a shapely box for intersection checking
            query_box = box(query_minx, query_miny, query_maxx, query_maxy)

            # Find geometries intersecting this random box
            # Use spatial index for potentially faster intersection
            possible_matches_index = list(gdf.sindex.intersection(query_box.bounds))
            possible_matches = gdf.iloc[possible_matches_index]
            intersecting_rows = possible_matches[possible_matches.intersects(query_box)]

            # Only add to index if the box intersects with at least one geometry
            if not intersecting_rows.empty:
                self.index.insert(
                    i,
                    coords,
                    intersecting_rows[["BasinType", "geometry"]].to_dict("records"),
                )
                i += 1
                added_count += 1

            if attempts % (total_points // 10) == 0:
                print(
                    f"Attempt {attempts}/{max_attempts}, Added {added_count}/{total_points} boxes."
                )

        if added_count < total_points:
            print(
                f"Warning: Only generated {added_count}/{total_points} intersecting query boxes after {attempts} attempts."
            )
        else:
            print(f"Successfully generated {added_count} intersecting query boxes.")

        print(f"Total points inserted into index: {i}")
        print(f"Time taken for index population: {time.time() - start:.0f} seconds")

        return i  # Return the final index count

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
        if not hits:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        shapes = []
        for hit in hits:
            # hit.object is now a list of dictionaries
            for obj in hit.object:
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
        """Returns the labels of the dataset"""
        return self.labels
