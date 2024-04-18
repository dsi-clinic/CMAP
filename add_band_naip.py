import argparse
import importlib.util
import os
import rasterio
import torch
import torchgeo
import numpy as np

from osgeo import gdal, osr
from torchgeo.datasets import GeoDataset, NAIP
from torch.utils.data import DataLoader
from torchvision import transforms

# import config and experiment name from runtime args
parser = argparse.ArgumentParser(
    description="Train a segmentation model to predict stormwater storage "
    + "and green infrastructure."
)
parser.add_argument("config", type=str, help="Path to the configuration file")
args = parser.parse_args()
config = importlib.import_module(args.config)

# Step 1: Load the raster dataset
naip_dataset = NAIP(config.KC_IMAGE_ROOT)

# Step 2: Load the data for the new band from a separate TIFF file
new_band_filepath = os.path.join(config.KC_DEM_ROOT, "KaneDEM.tif")
with rasterio.open(new_band_filepath) as src:
    new_band = src.read(1)

# Convert the new band to a PyTorch tensor
new_band = torch.from_numpy(new_band)

# Step 2: Load the data for the new band from a separate TIFF file
#new_band_dataset = gdal.Open(new_band_filepath)
#new_band = new_band_dataset.GetRasterBand(1).ReadAsArray()

# assert new_band.shape[1:] == naip_dataset[0][0].shape[1:], "New band dimensions do not match NAIP dataset dimensions"

# Loop through the NAIP dataset and add the new band to each sample
for i in range(len(naip_dataset)):
    naip_sample, _ = naip_dataset[i]  # NAIP sample and label
    naip_sample_with_new_band = torch.cat((naip_sample, new_band), dim=0)  # Concatenate the new band
    naip_dataset[i] = (naip_sample_with_new_band, _)  # Update the dataset with the sample containing the new band

# Save the modified dataset
naip_output_path = os.path.join(config.KC_IMAGE_ROOT, "modified_naip.tif")
torch.save(naip_dataset, naip_output_path)

""""
# function for converting the combined data as a GeoDataset
def save_as_geotiff(data, output_path, geotransform, projection):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get dimensions
    rows, cols, bands = data.shape

    # Create new GeoTIFF
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, cols, rows, bands, gdal.GDT_Float32)

    # Write data to bands
    for band_num in range(bands):
        out_ds.GetRasterBand(band_num + 1).WriteArray(data[:, :, band_num])

    # Set geotransform and projection
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)

    # Close dataset
    out_ds = None

# Define output directory for the combined datasets
output_dir = os.path.join(config.KC_DEM_ROOT, "KC_DEM_merged")

data_loader = DataLoader(raster_dataset, batch_size=1, shuffle=False)

for i, sample in enumerate(data_loader):
    combined_tensor_data = sample['image']  # Access the combined tensor data

    # Define output path for the combined dataset
    output_path = os.path.join(output_dir, f"sample_{i}.tif")

    # Save combined tensor data as a GeoTIFF
    save_as_geotiff(
        combined_tensor_data.numpy(),  # Convert tensor to numpy array
        output_path,
        naip_dataset.geotransform,  # Use geotransform from KaneCounty dataset
        naip_dataset.projection,  # Use projection from KaneCounty dataset
    )
"""