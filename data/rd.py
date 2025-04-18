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

        # Store kc flag for use later
        self.kc = kc

        # Initialize containers for index data
        kc_index_data = []
        num_kc_samples_to_add = 0
        kc_labels_present = set()

        if self.kc:
            print("DEBUG: kc=True, preparing KC dataset...")
            # Load KC data first to determine the number of samples
            kc_shape_path = Path(KC_SHAPE_ROOT) / KC_SHAPE_FILENAME
            kc_config = (KC_LAYER, KC_LABELS, patch_size, dest_crs, res)
            try:
                kc_dataset = KaneCounty(kc_shape_path, kc_config)
                print(f"DEBUG: KC dataset loaded. CRS: {kc_dataset.crs}")

                # FIXME: THIS IS JUST FOR DEBUGGING
                # FOR REAL TRAINING DO NOT TRUNCATE KC
                max_kc = 100  # Maximum KC samples to consider
                print(
                    f"DEBUG: Iterating through KC index to collect up to {max_kc} samples..."
                )

                # Iterate through KC index to gather data without inserting yet
                kc_items_processed = 0
                for item in kc_dataset.index.intersection(
                    kc_dataset.index.bounds, objects=True
                ):
                    kc_items_processed += 1
                    obj_dict = {
                        "BasinType": item.object["BasinType"],
                        "geometry": item.object["geometry"],
                    }
                    kc_index_data.append(
                        (item.bounds, [obj_dict])
                    )  # Store bounds and object list
                    kc_labels_present.add(item.object["BasinType"])
                    num_kc_samples_to_add += 1
                    if num_kc_samples_to_add >= max_kc:
                        print(f"DEBUG: Reached max_kc limit ({max_kc}).")
                        break
                print(
                    f"DEBUG: Collected {num_kc_samples_to_add} KC samples from {kc_items_processed} KC index items."
                )

            except FileNotFoundError:
                print("WARN: KC shapefile not found. Proceeding without KC data.")
                self.kc = False  # Ensure kc flag is false if data isn't loaded

        # Determine target number of RD samples
        rd_target_points = (
            num_kc_samples_to_add if self.kc else 176
        )  # Default if not kc
        # TODO: Consider restoring the default 900 or making it configurable
        print(f"DEBUG: Target number of RD samples set to: {rd_target_points}")

        # Populate RD index
        print("DEBUG: Populating RD index...")
        actual_rd_samples_added = self._populate_index(
            self.gdf, total_points=rd_target_points, box_size=box_size
        )
        print(
            f"DEBUG: RD index population finished. Added {actual_rd_samples_added} RD samples."
        )

        # Now, add KC samples to the index, starting after RD samples
        if self.kc:
            print(
                f"DEBUG: Adding {num_kc_samples_to_add} collected KC samples to index, starting at ID {actual_rd_samples_added}..."
            )
            kc_current_idx = actual_rd_samples_added
            for bounds, obj_list in kc_index_data:
                self.index.insert(kc_current_idx, bounds, obj_list)
                kc_current_idx += 1
            print(
                f"DEBUG: Finished adding KC samples. Final index ID: {kc_current_idx - 1}"
            )

            # --- Label and Color Merging ---
            print("DEBUG: Merging RD and KC labels and colors...")
            # Start with base RD labels
            rd_labels = {"BACKGROUND": 0, "STREAM/RIVER": 1}
            combined_labels = rd_labels.copy()
            next_idx = max(combined_labels.values()) + 1

            # Add KC labels that aren't already present (using the KC_LABELS mapping)
            for (
                label_name,
                kc_original_idx,
            ) in KC_LABELS.items():  # Iterate through the canonical KC_LABELS
                if (
                    label_name not in combined_labels
                    and label_name in kc_labels_present
                ):
                    combined_labels[label_name] = next_idx
                    print(
                        f"  DEBUG: Adding new label '{label_name}' with index {next_idx}"
                    )
                    next_idx += 1
                elif label_name in combined_labels:
                    print(f"  DEBUG: Label '{label_name}' already exists.")
                # else: label from KC_LABELS not present in the actual loaded KC data

            self.labels = combined_labels
            print(f"DEBUG: Final merged labels: {self.labels}")

            # Preserve color mapping with correct *final* indices
            combined_colors = {}
            # Base RD colors
            combined_colors[0] = self.all_colors[0]  # BACKGROUND
            combined_colors[1] = self.all_colors[1]  # STREAM/RIVER

            # KC colors (using kc_dataset.all_colors and the final combined_labels index)
            for label_name, final_idx in self.labels.items():
                if label_name in KC_LABELS:  # Check if it's a KC label
                    kc_original_idx = KC_LABELS[label_name]
                    combined_colors[final_idx] = kc_dataset.all_colors[kc_original_idx]
                    print(
                        f"  DEBUG: Assigning color for '{label_name}' (final index {final_idx}) from KC color {kc_original_idx}"
                    )

            self.colors = combined_colors
            print(f"DEBUG: Final merged colors: {self.colors}")
            # --- End Label and Color Merging ---

        self.labels_inverse = {v: k for k, v in self.labels.items()}
        print(
            f"DEBUG: Initialized RiverDataset. Final index size: {len(self.index)}. Final index bounds: {self.index.bounds}"
        )
        print(
            f"DEBUG: Initialized RiverDataset with configs: {rd_configs}, kc={self.kc}"
        )

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

        print(f"DEBUG [_populate_index]: Total bounds: {total_bounds}")
        print(
            f"DEBUG [_populate_index]: Generating {total_points} random query boxes of size {box_size}..."
        )

        i = 0  # query box counter
        added_count = 0
        max_attempts_factor = 20  # Increased factor
        max_attempts = total_points * max_attempts_factor
        attempts = 0
        intersect_fail_count = 0

        print(f"DEBUG [_populate_index]: Max attempts set to {max_attempts}")

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
            else:
                intersect_fail_count += 1

            # More frequent logging
            if attempts % (max_attempts // 20) == 0:
                print(
                    f"DEBUG [_populate_index]: Attempt {attempts}/{max_attempts}, Added {added_count}/{total_points}, Intersection fails: {intersect_fail_count}"
                )

        if added_count < total_points:
            print(
                f"Warning: Only generated {added_count}/{total_points} intersecting query boxes after {attempts} attempts."
            )
        else:
            print(
                f"DEBUG [_populate_index]: Successfully generated {added_count} intersecting query boxes."
            )

        print(f"DEBUG [_populate_index]: Total points inserted into index: {i}")
        print(
            f"DEBUG [_populate_index]: Time taken for index population: {time.time() - start:.0f} seconds"
        )

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
        if self.kc:
            print(f"DEBUG [__getitem__ kc=True] Query: {query}")

        for hit in hits:
            # hit.object is now a list of dictionaries
            for obj in hit.object:
                try:
                    shape = obj["geometry"]
                    basin_type = obj["BasinType"]
                    label = self.labels[basin_type]  # Use merged labels
                    shapes.append((shape, label))
                    if self.kc:
                        print(
                            f"  DEBUG [__getitem__ kc=True]: Obj BasinType: {basin_type}, Assigned Label: {label}"
                        )
                except KeyError:
                    print(
                        f"  WARN [__getitem__]: BasinType '{obj.get('BasinType', 'N/A')}' not found in self.labels: {self.labels}. Skipping object."
                    )
                    continue

        if self.kc:
            print(
                f"DEBUG [__getitem__ kc=True]: Final shapes for rasterization: {[(s[1], type(s[0])) for s in shapes]}"
            )  # Print label and geom type

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
