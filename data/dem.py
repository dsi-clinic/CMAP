""""
***************convert .gdb file to .tif file****************
* must have gdal installed
* first run 
    gdal_translate -of GTiff "input.gdb" "output.tif" -b <band-number>
in command line to convert .gdb file into .tif file.
* replace "input.gdb" and "output.tif" in above code with file paths to
input a .gdb file and output a .tif file
"""

from torchgeo.datasets import RasterDataset

class KaneDEM(RasterDataset):
    filename_glob = "*DEM.tif"

    def __init__(self, paths, crs=None, res=None, transforms=None):
        super().__init__(paths, crs, res, transforms=transforms)
        self.all_bands = ["elevation"]  # Assuming single band for elevation

    def __getitem__(self, query):
        # This method loads the DEM data similar to how other raster data is loaded
        sample = super().__getitem__(query)
        elevation = sample[
            "image"
        ]  # Assuming the elevation data is stored as 'image'
        return {"elevation": elevation}

    