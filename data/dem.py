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
    """
    A dataset class for handling Kane County Digital Elevation Model (DEM) data.

    Attributes:
        filename_glob (str): A string representing the pattern
        for matching DEM file names.
    """

    filename_glob = "*2017BE.tif"

    def __init__(self, paths, crs=None, res=None, transforms=None):
        """
        Initializes a KaneDEM instance.

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

    def __getitem__(self, query):
        """
        Retrieves a specific DEM sample from the dataset.

        Args:
            query: An index or query to retrieve the DEM sample.

        Returns:
            dict: A dictionary containing the elevation data.
        """
        # This method loads the DEM data similar to how other raster data is loaded
        sample = super().__getitem__(query)
        elevation = sample[
            "image"
        ]  # Assuming the elevation data is stored as 'image'
        return {"elevation": elevation}

    def __getallbands__(self):
        """
        Get all bands for this dataset.
        """
        return self.all_bands
