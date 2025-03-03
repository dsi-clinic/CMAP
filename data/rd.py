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
from shapely.geometry import MultiPoint, Point, box
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
        self._populate_index(self.gdf, box_size=box_size)

        if kc:
            kc_shape_path = Path(KC_SHAPE_ROOT) / KC_SHAPE_FILENAME
            kc_config = (KC_LAYER, KC_LABELS, patch_size, dest_crs, res)
            kc_dataset = KaneCounty(kc_shape_path, kc_config)
            print(f"river dataset crs {self.crs}")
            print(f"KC crs {kc_dataset.crs}")

            # Merge indices and labels
            i = 0
            max_kc = 10

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
                    item.id,
                    item.bounds,
                    [obj_dict],  # wrap in list to match RD format
                )
                print(f"KC inserting coords {item.bounds}")
                # FIXME debug hack for fewer KC
                i += 1
                if i > max_kc:
                    break

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

    def _populate_index(self, gdf, total_points=177, box_size=150):
        """Populate spatial index with random points within river polygons."""
        import time

        import numpy as np
        from tqdm import tqdm

        start = time.time()
        mint, maxt = 0, sys.maxsize
        half_box = box_size / 2

        # calculate total area and points per geometry
        total_area = gdf["geometry"].area.sum()
        if len(gdf) > total_points:
            raise ValueError(
                f"total_points ({total_points}) must be >= number of geometries ({len(gdf)})"
            )

        # allocate points proportionally with minimum 1 point per geometry
        areas = gdf["geometry"].area
        relative_points = areas / total_area * (total_points - len(gdf))
        points_per_geom = np.maximum(1, np.floor(relative_points))

        print(f"\nInitial allocation: {points_per_geom.sum():.0f} points")

        # adjust allocation if over total_points
        while points_per_geom.sum() > total_points:
            max_idx = points_per_geom.argmax()
            points_per_geom[max_idx] -= 1
            points_per_geom = np.maximum(1, points_per_geom)

        print(f"After adjustment: {points_per_geom.sum():.0f} points")

        i = 0  # point counter
        # iterate through each river polygon
        for idx, row in tqdm(gdf.iterrows(), desc="Processing rivers", total=len(gdf)):
            geom = row["geometry"]
            n_points = int(points_per_geom[idx])

            points_added = 0
            attempts = 0
            max_attempts = 1
            valid_points = []

            while points_added < n_points and attempts < max_attempts:
                # generate many points at once using bounds
                minx, miny, maxx, maxy = geom.bounds
                factor = 100.0
                remaining_points = n_points - points_added
                batch_size = int(remaining_points * factor)

                # Time point generation
                t0 = time.time()
                points_x = np.random.uniform(minx, maxx, size=batch_size)
                points_y = np.random.uniform(miny, maxy, size=batch_size)
                points = MultiPoint([Point(x, y) for x, y in zip(points_x, points_y)])
                t1 = time.time()
                point_gen_time = t1 - t0

                # Time point filtering
                t0 = time.time()
                new_valid_points = [p for p in points.geoms if geom.contains(p)]
                t1 = time.time()
                point_filter_time = t1 - t0
                valid_points.extend(new_valid_points)

                print(f"\nGeometry {idx}:")
                print(f"- Target points: {n_points}")
                print(f"- Generated points: {batch_size}")
                print(f"- Valid points in batch: {len(new_valid_points)}")
                print(f"- Point generation time: {point_gen_time:.3f}s")
                print(f"- Point filtering time: {point_filter_time:.3f}s")
                print(f"- Points/sec filtered: {batch_size/point_filter_time:.0f}")

                # Time box creation and index insertion
                t0 = time.time()
                box_count = 0
                for point in valid_points[points_added:]:
                    if points_added >= n_points:
                        break

                    x, y = point.x, point.y
                    bbox = box(x - half_box, y - half_box, x + half_box, y + half_box)
                    coords = (
                        x - half_box,
                        x + half_box,
                        y - half_box,
                        y + half_box,
                        mint,
                        maxt,
                    )

                    # find all geometries that intersect with this box
                    t2 = time.time()
                    intersecting_rows = gdf[gdf.intersects(bbox)]
                    t3 = time.time()

                    if not intersecting_rows.empty:
                        self.index.insert(
                            i,
                            coords,
                            intersecting_rows[["BasinType", "geometry"]].to_dict(
                                "records"
                            ),
                        )
                        i += 1
                        points_added += 1
                        box_count += 1

                t1 = time.time()
                box_time = t1 - t0
                intersect_time = t3 - t2

                print(f"- Box creation & insertion time: {box_time:.3f}s")
                print(f"- Intersection check time: {intersect_time:.3f}s")
                print(f"- Boxes created: {box_count}")
                if box_count > 0:
                    print(f"- Time per box: {box_time/box_count:.3f}s")

                attempts += 1

                # break early if we have enough points
                if points_added >= n_points:
                    break

            if points_added < n_points:
                print(
                    f"Warning: Only added {points_added}/{n_points} points for geometry {idx} after {attempts} attempts"
                )

        print(f"\nTotal points inserted: {i}")
        print(f"Target points: {total_points}")
        print(f"Time taken: {time.time() - start:.0f} seconds")

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
