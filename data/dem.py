""""
***************convert .gdb file to .tif file****************
* must have gdal installed
* first run 
    gdal_translate -of GTiff "input.gdb" "output.tif" -b <band-number>
in command line to convert .gdb file into .tif file.
* replace "input.gdb" and "output.tif" in above code with file paths to
input .gdb file and output .tif files
"""
import argparse
import importlib.util
import os

from osgeo import gdal, ogr
# https://gis.stackexchange.com/questions/28966/python-gdal-package-missing-header-file-when-installing-via-pip

parser = argparse.ArgumentParser(
    description="Preprocess DEM data in .tif format to generate a raster layer"
)
args = parser.parse_args()
config = importlib.import_module(args.config)


"""
def create_dem_mask() -> None:

    # Define paths
    input_gdb = os.path.join(
        config.KC_DEM_DIR, "Kane_DEM.gdb"
    )
    input_feature_class = "name_of_your_feature_class"
    output_tif = "output_path.tif"

    # Open the input geodatabase
    driver = ogr.GetDriverByName("OpenFileGDB")
    gdb = driver.Open(input_gdb, 0)  # Change 0 to 1 for read-only access

    # Get the input layer
    layer = gdb.GetLayerByName(input_feature_class)

    # Get extent and resolution
    x_min, x_max, y_min, y_max = layer.GetExtent()
    x_res = 100  # Define resolution in X direction
    y_res = 100  # Define resolution in Y direction

    # Create the output raster
    target_ds = gdal.GetDriverByName('GTiff').Create(output_tif, int((x_max - x_min) / x_res), int((y_max - y_min) / y_res), 1, gdal.GDT_Float32)

    # Set the projection
    target_ds.SetProjection(layer.GetSpatialRef().ExportToWkt())

    # Set the geotransform
    target_ds.SetGeoTransform((x_min, x_res, 0, y_max, 0, -y_res))

    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[0], options=["ATTRIBUTE=Elevation"])

    # Close datasets
    target_ds = None
    gdb = None

    print("Conversion completed successfully.")
"""

""""
***************define custom Raster Dataset .tif files**************
"""
import rasterio
from rasterio.merge import merge
from rasterio.plot import show

# Paths to the input raster files
input_files = ["/net/projects/cmap/data/kane-county-data/KC_DEM_2017/KaneDEM.tif"]

# Open each raster file and read its data
src_files_to_mosaic = []
for file in input_files:
    src = rasterio.open(file)
    src_files_to_mosaic.append(src)

# Merge the raster files
mosaic, out_trans = merge(src_files_to_mosaic)

# Metadata of the output raster dataset
out_meta = src.meta.copy()
out_meta.update({"driver": "GTiff",
                 "height": mosaic.shape[1],
                 "width": mosaic.shape[2],
                 "transform": out_trans})

# Output file path
output_file = "output_raster.tif"

# Write the merged raster to a new file
with rasterio.open(output_file, "w", **out_meta) as dest:
    dest.write(mosaic)

# Optionally, visualize the output raster
show(mosaic)
