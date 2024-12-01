"""Convert .gdb file to .tif file.

Must have gdal installed. First run:
    gdal_translate -of GTiff "input.gdb" "output.tif" -b <band-number>
in command line to convert .gdb file into .tif file.

Replace "input.gdb" and "output.tif" in above code with file paths to
input a .gdb file and output a .tif file.
"""

from pathlib import Path

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

    def __init__(self, paths, crs=None, res=None, transforms=None):
        """Initializes a KaneDEM instance.

        Args:
            paths (str or List[str]): Path(s) to the DEM data.
            crs (Optional[str]): Coordinate reference system (CRS) of the DEM data.
            res (Optional[float]): Spatial resolution of the DEM data.
            transforms (Optional[callable]): function/transform to apply to DEM data.

        Returns:
            None
        """
        super().__init__(paths, crs, res, transforms=transforms)
        self.all_bands = ["elevation"]  # Assuming single band for elevation

    def __getitem__(self, query: BoundingBox):
        """Retrieves a specific DEM sample from the dataset.

        Args:
            query: An index or query to retrieve the DEM sample.

        Returns:
            dict: A dictionary containing the elevation data.
        """
        # This method loads the DEM data similar to how other raster data is loaded
        sample = dict.fromkeys(["image"])
        with rasterio.open(str(self.paths) / Path(self.filename_glob)) as src:
            bbox = (query.minx, query.miny, query.maxx, query.maxy)
            bbox = rasterio.warp.transform_bounds(self.crs, src.crs, *bbox)
            window = src.window(*bbox)
            sample["image"] = src.read(window=window)
            # FIXME debugging hack
            # if sample["image"].shape != (1, 512, 512):
            #   print("sample image shape in dem: ", sample["image"].shape)
        sample["image"] = torch.from_numpy(sample["image"])
        return sample

    def __getallbands__(self):
        """Get all bands for this dataset."""
        return self.all_bands
