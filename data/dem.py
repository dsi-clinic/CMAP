from osgeo import gdal
# https://gis.stackexchange.com/questions/28966/python-gdal-package-missing-header-file-when-installing-via-pip

# Path to your .adf file
adf_file = "path/to/your/file.adf"

# Open the .adf file
dataset = gdal.Open(adf_file)

# Check if the dataset was successfully opened
if dataset is None:
    print("Failed to open the dataset")
    exit(1)

# Get the raster band (assuming there is only one band)
band = dataset.GetRasterBand(1)

# Get the elevation data for each pixel
elevation_data = band.ReadAsArray()

# Close the dataset
dataset = None

# Print the elevation data for each pixel
print("Elevation data:")
print(elevation_data)

""""
***************convert .adf file to .tif file****************
"""
from osgeo import gdal

# Path to the input .adf file
input_adf = "input.adf"

# Path to the output .tif file
output_tif = "output.tif"

# Open the .adf file
dataset = gdal.Open(input_adf)

# Convert to GeoTIFF (.tif) format and save
gdal.Translate(output_tif, dataset, format="GTiff")

# Close the dataset
dataset = None

""""
***************create multi band raster dataset by merging .tif files**************
"""
import rasterio
from rasterio.merge import merge
from rasterio.plot import show

# Paths to the input raster files
input_files = ["band1.tif", "band2.tif", "band3.tif"]

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
