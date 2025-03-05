"""Convert .gdb file to .tif file.

Must have gdal installed. First run:
    gdal_translate -of GTiff "input.gdb" "output.tif" -b <band-number>
in command line to convert .gdb file into .tif file.

Replace "input.gdb" and "output.tif" in above code with file paths to
input a .gdb file and output a .tif file.
"""

from pathlib import Path

import numpy as np
import rasterio
import torch
from torchgeo.datasets import BoundingBox, RasterDataset


class KaneDEM(RasterDataset):
    """A dataset class for handling Kane County Digital Elevation Model (DEM) data.

    Attributes:
        filename_glob (str): A string representing the pattern
        for matching DEM file names.
    """

    filename_glob = "Kane2017BE.tif"

    def __init__(
        self,
        paths,
        config,
        crs=None,
        res=None,
        transforms=None,
        epsilon=1e-6,
        use_difference=False,
    ):
        """Initializes a KaneDEM instance.

        Args:
            paths: A list of paths to the DEM files.
            config: Configuration object containing PATCH_SIZE and USE_NIR.
            crs: The CRS of the DEM.
            res: The resolution of the DEM.
            transforms: The transforms to apply to the DEM.
            epsilon: A small value to prevent division by zero.
            use_difference: Changes input to filled DEM set
        """
        super().__init__(paths, crs=crs, res=res, transforms=transforms)
        self.patch_size = config.PATCH_SIZE
        self.use_nir = config.USE_NIR
        self.epsilon = epsilon

        # Use filled DEM file if specified
        if use_difference:
            self.filename_glob = "Kane2017BE_difference.tif"
        print(f"Using DEM file: {self.filename_glob}")

    def __getitem__(self, query: BoundingBox):
        """Retrieves a specific DEM sample from the dataset."""
        sample = dict.fromkeys(["image"])

        with rasterio.open(str(self.paths) / Path(self.filename_glob)) as src:
            bbox = (query.minx, query.miny, query.maxx, query.maxy)
            bbox = rasterio.warp.transform_bounds(self.crs, src.crs, *bbox)
            window = src.window(*bbox)

            dem_chunk = src.read(1, window=window).astype(np.float32)

            # Handle nodata values first
            mask = dem_chunk != src.nodata
            dem_chunk[~mask] = 0

            # Ensure the chunk is the expected size
            if dem_chunk.shape != (self.patch_size, self.patch_size):
                temp = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
                h, w = (
                    min(dem_chunk.shape[0], self.patch_size),
                    min(dem_chunk.shape[1], self.patch_size),
                )
                temp[:h, :w] = dem_chunk[:h, :w]
                dem_chunk = temp
                temp_mask = np.zeros_like(temp, dtype=bool)
                temp_mask[:h, :w] = mask[:h, :w]
                mask = temp_mask

            # Calculate statistics only on valid data
            if np.any(mask):
                valid_data = dem_chunk[mask]
                chunk_mean = np.nanmean(valid_data)

                # Calculate elevation differences only on valid data
                diff_x = np.diff(np.where(mask, dem_chunk, np.nan), axis=1)
                diff_y = np.diff(np.where(mask, dem_chunk, np.nan), axis=0)
                elevation_diffs = np.concatenate(
                    [
                        diff_x[~np.isnan(diff_x)].flatten(),
                        diff_y[~np.isnan(diff_y)].flatten(),
                    ]
                )

                chunk_std = (
                    np.nanstd(elevation_diffs) if len(elevation_diffs) > 0 else 1.0
                )

                # Normalize using chunk statistics, ensuring std is not zero
                if chunk_std > self.epsilon:
                    dem_chunk = np.where(mask, (dem_chunk - chunk_mean) / chunk_std, 0)
                else:
                    dem_chunk = np.where(mask, dem_chunk - chunk_mean, 0)

            # Add channel dimension to make shape [B, 1, H, W]
            sample["image"] = torch.from_numpy(dem_chunk[np.newaxis, ...]).float()
            return sample

    def __getallbands__(self):
        """Get all bands for this dataset."""
        return ["elevation"]
