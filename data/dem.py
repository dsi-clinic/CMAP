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
    is_image = True
    separate_files = False
    all_bands = ["elevation"]

