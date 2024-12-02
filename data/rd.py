"""
This module provides a custom PyTorch GeoDataset for working with vector data
representing river labels or features in the River images dataset. The vector data is
stored as shapes in a GeoDatabase file, and this module allows for retrieving
samples of labels or features as masks or rasterized images within specified
bounding boxes.
The KC gdf is added to the RD gdf and chips are generated using this combined
gdf. To train with river data, use this file and not kc.py.
"""

import math
import sys

import geopandas as gpd
import numpy as np
import rasterio
import torch
import pandas as pd
from pathlib import Path


# Add the parent directory (contains both 'configs' and 'data') to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import from configs
from configs.config import KC_SHAPE_ROOT, KC_SHAPE_FILENAME, KC_LAYER, KC_LABELS


# from rtree import index  # or any spatial index you are using
from shapely.geometry import box
from torchgeo.datasets import BoundingBox, GeoDataset

# from tqdm import tqdm



"""
This module provides a custom PyTorch GeoDataset for working with vector data
representing labels or features in Kane County, Illinois. The vector data is
stored as shapes in a GeoDatabase file, and this module allows for retrieving
samples of labels or features as masks or rasterized images within specified
bounding boxes.
"""



# Load the Kane County Dataset first
# The KC gdf will later be combined with the RD gdf

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
        5: (255, 255, 0, 255), # this corresponds to RIVER/STREAM objects
        6: (89, 53, 31, 255),
        7: (2, 86, 105, 255),
        8: (207, 211, 205, 255),
        9: (195, 88, 49, 255),
        10: (144, 70, 132, 255),
        11: (29, 51, 74, 255),
        12: (71, 64, 46, 255),
        13: (114, 20, 34, 255),
        14: (37, 40, 80, 255),
        15: (94, 33, 41, 255),
        16: (255, 255, 255, 255),
       
    }
    all_labels = {
        0: "BACKGROUND",
        1: "POND",
        2: "WETLAND",
        3: "DRY BOTTOM - TURF",
        4: "DRY BOTTOM - MESIC PRAIRIE",
        #5: "STREAM/RIVER" # this line is not necessary as it is added again later
        6: "DEPRESSIONAL STORAGE",
        7: "DRY BOTTOM - WOODED",
        8: "POND - EXTENDED DRY",
        9: "PICP PARKING LOT",
        10: "DRY BOTTOM - GRAVEL",
        11: "UNDERGROUND",
        12: "UNDERGROUND VAULT",
        13: "PICP ALLEY",
        14: "INFILTRATION TRENCH",
        15: "BIORETENTION",
        16: "UNKNOWN",
    }

    def __init__(self, path: str, kc_configs) -> None:
        """Initialize a new KaneCounty dataset instance.

        Args:
            path: directory to the file to load
            kc_configs: a tuple containing
                layer: specifying layer of GPKG
                labels: a dictionary containing a label mapping for masks
                patch_size: the patch size used for the model
                dest_crs: the coordinate reference system (CRS) to convert to
                res: resolution of the dataset in units of CRS

        Raises:
            FileNotFoundError: if no files are found in path
        """
        super().__init__()

        layer, labels, patch_size, dest_crs, res = kc_configs

        gdf = self._load_and_prepare_data(path, layer, labels, dest_crs)
        self.gdf = gdf

        context_size = math.ceil(patch_size / 2 * res)
        self.context_size = context_size
        self._crs = dest_crs
        self._res = res

        #self._populate_index(path, gdf, context_size)
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

        # debug print
        print("Initial Kane County GeoDataFrame loaded:")
        print(gdf.head())

        gdf = gdf[gdf["BasinType"].isin(labels.keys())]
        gdf = gdf.to_crs(dest_crs)

        # debug print
        print("Kane countys filtered gdf")
        print(gdf.head())

        return gdf

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




# Load the RiverDataset class
# This will also contain the KC class; chips will be generated using the combined KC and RD gdfs

class RiverDataset(GeoDataset):
    """Vector dataset for river labels stored as shapes in GeoDatabase."""

    all_bands = ["Label"]
    is_image = False

    all_colors = {
        0: (0, 0, 0, 0),
        5: (255, 255, 0, 255),
    }

    all_labels = {0: "UNKNOWN", 5: "STREAM/RIVER"}

    def __init__(self, path: str, rd_configs, kc: True) -> None:
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

        labels, patch_size, dest_crs, res = rd_configs
        gdf = self._load_and_prepare_data(path, dest_crs)
        self.gdf = gdf

        # Debug prints
        print(f"Configs received: {rd_configs}")
        print(f"Type of labels: {type(labels)}, Content: {labels}")
        
        # Initialize the KaneCounty dataset if kc is True
        if kc:
            kc_shape_path = Path(KC_SHAPE_ROOT) / KC_SHAPE_FILENAME
            print("Loaded KC shape path")
            
            kc_config = (
                KC_LAYER,
                KC_LABELS,
                patch_size,
                dest_crs,
                res,
            )
            print("Loaded KC config")
            
            # Create the KaneCounty dataset instance
            kc_dataset = KaneCounty(kc_shape_path, kc_config)
            print("Initialized KC dataset")

            # Combine the River dataset and KaneCounty dataset gdfs
            self.gdf = pd.concat([self.gdf, kc_dataset.gdf], ignore_index=True)
            self.gdf['BasinType'] = self.gdf['BasinType'].fillna(self.gdf['FCODE'])

            print(f"Combined GeoDataFrame shape: {self.gdf.shape}")
        
            KC_LABELS["STREAM/RIVER"] = 5 # add the river labels to the existing KC labels
            labels = KC_LABELS
            
            self.gdf = self.gdf[self.gdf["BasinType"].isin(labels.keys())] # only extract KC and RD objects
            
            kc_dataset.colors[5] = (255, 255, 0, 255) # add the river colors to the existing KC colors
            self.all_colors = kc_dataset.colors

        
        context_size = math.ceil(patch_size / 2 * res)
        self.context_size = context_size
        self._crs = dest_crs
        self._res = res

        self._populate_index(path, self.gdf, context_size, patch_size)
        self.labels = labels
        self.colors = {label_value: self.all_colors[label_value] for label_value in labels.values()}
        self.labels_inverse = {v: k for k, v in labels.items()}

        # Debug print
        print(f"Initializing RiverDataset with configs: {rd_configs}")

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

        # Debug print
        print("Initial River GeoDataFrame loaded:")
        print(gdf.head())

        # Debug print
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
        reference_crs=4326, # this is the original CRS
        target_chip_size=0.005,
    ):
        """Populate spatial index with proportional chips based on CRS bounds."""

        mint, maxt = 0, sys.maxsize

        self.bounding_boxes = []
        i = 0  # initialize chip index counter

        from pyproj import CRS, Transformer
        from tqdm import tqdm

        # Get the CRS of the GeoDataFrame
        gdf_crs = gdf.crs # extract the NAIP CRS
        print(f"GeoDataFrame CRS: {gdf_crs}")

        # Set up transformations 
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
                    self.index.insert(i, coords, row[["BasinType", "geometry"]]) # replace FCODE with BasinType
                    i += 1  # Increment the global index for each chip

        print(f"Total chips inserted: {i}") # chips and polygons are many-to-many 

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
