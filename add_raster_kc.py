import argparse
import importlib.util
import os
import rasterio
import torch
import torchgeo

from data.kc import KaneCounty

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
with rasterio.open('') as new_band_file:
    new_band_data = new_band_file.read(1)  # Assuming the new band is a single-band TIFF

# Step 3: Combine the existing bands with the new band
existing_data = raster_dataset.data  # Assuming the existing data is already loaded
# Assuming existing_data is a tensor of shape (num_bands, height, width)
new_data = torch.cat([existing_data, torch.from_numpy(new_band_data).unsqueeze(0)], dim=0)

# Step 4: Save or export the modified raster dataset
# You can save it as a new raster file
added_dem_band = os.path.join(config.KC_DEM_ROOT, "KC_DEM_merged.tif")
torchgeo.utils.write_raster(added_dem_band, new_data.numpy(), raster_dataset.transform, raster_dataset.crs, raster_dataset.profile)
