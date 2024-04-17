import argparse
import importlib.util
import os
import rasterio
import torch
import torchgeo
import numpy as np

from osgeo import gdal, osr
from torchgeo.datasets import GeoDataset
from torch.utils.data import DataLoader
from data.kc import KaneCounty
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
raster_dataset = KaneCounty(config.KC_MASK_ROOT)

# Step 2: Load the data for the new band from a separate TIFF file
new_band_filepath = os.path.join(config.KC_DEM_ROOT, "KaneDEM.tif")
new_band_dataset = gdal.Open(new_band_filepath)
new_band = new_band_dataset.GetRasterBand(1).ReadAsArray()

# Transform KaneCounty into tensor
class CombineBands:
    def __init__(self, new_band):
        self.new_band = new_band

    def __call__(self, sample):
        existing_bands = sample['image']
        combined_bands = torch.cat((existing_bands, self.new_band.unsqueeze(0)), dim=0)
        return {'image': combined_bands}

# Instantiate custom transform
combine_bands_transform = CombineBands(new_band)

# transform KaneCounty
raster_dataset.transform = transforms.Compose([combine_bands_transform, transforms.ToTensor()])

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
        raster_dataset.geotransform,  # Use geotransform from KaneCounty dataset
        raster_dataset.projection,  # Use projection from KaneCounty dataset
    )
