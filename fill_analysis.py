from data.dem import KaneDEM
from pysheds.grid import Grid
import matplotlib.pyplot as plt

def run_fill_analysis():
    dem_data = KaneDEM(paths='path_to_converted_file.tif')
    elevation = dem_data[0]["elevation"]

run_fill_analysis()