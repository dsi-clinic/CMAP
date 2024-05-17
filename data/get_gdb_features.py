"""
This code is for listing the features and names contained within a gdb file
To run: from repo directory (2024-winter-cmap)
> python data/get_gdb_features.py configs.<config>
"""

import argparse
import importlib.util
import os

from osgeo import ogr

parser = argparse.ArgumentParser(
    description="Obtain names of features in .gdb data required for preprocessing"
)
args = parser.parse_args()
config = importlib.import_module(args.config)

gdb_path = os.path.join(config.KC_DEM_DIR, "Kane_DEM.gdb")


def list_feature_classes(gdb_path):
    """
    Given the path to a gdb file, prints the names of layers in file
    """
    driver = ogr.GetDriverByName("OpenFileGDB")
    gdb = driver.Open(gdb_path, 0)  # Change 0 to 1 for read-only access

    if gdb is None:
        print(f"Failed to open {gdb_path}")
        return

    print("Feature classes in the geodatabase:")
    for i in range(gdb.GetLayerCount()):
        layer = gdb.GetLayerByIndex(i)
        print(layer.GetName())

    gdb = None  # Close the geodatabase


list_feature_classes(gdb_path)
