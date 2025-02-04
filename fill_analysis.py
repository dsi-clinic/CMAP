"""Module performs fill analysis, finds the difference, and exports it"""

from pathlib import Path

import numpy as np
from pysheds.grid import Grid

import configs.config as config


def export_filled_dem(fill_dem, grid, output_path):
    """Exports difference DEM to file

    Args:
        fill_dem: DEM containing filled DEM
        grid: PySheds grid object
        output_path: file path to export the DEM data to
    """
    grid.to_raster(fill_dem, output_path)

    print(f"Difference Diff DEM has been exported to {output_path}")

    return

def normalize_diff_dem(diff_dem):
    """Normalizes difference DEM(WIP) - should be done in train.py?

    Args:
        diff_dem: DEM containing difference between filled and original DEM
    Returns:
        normalized: normalized diff_dem with mean=0 and stddev=1
    """
    # Consider local normalization as shown in Castillo et al.(2014)

    mean = np.mean(diff_dem) 
    std = np.std(diff_dem)
    
    if std == 0:  # Prevent division by zero
        return diff_dem - mean  # If no variation, just center to 0
    
    normalized = (diff_dem - mean) / std

    return normalized

def fill_analysis(tiff_path):
    """Loads DEM tiff file, performs fill analysis, and finds the difference

    Args:
        tiff_path: file path to DEM tiff file

    Returns:
        diff_dem: DEM containing difference between filled and original DEM
        grid: PySheds grid object
    """
    # Load Data
    grid = Grid.from_raster(tiff_path)
    dem = grid.read_raster(tiff_path).copy()

    # Fill Pits and Depressions in place to limit memory usage
    grid.fill_pits(dem, out=dem)
    grid.fill_depressions(dem, out=dem)

    # Should be done in train.py
    # # Find Difference Between Filled DEM and Original DEM
    # diff_dem = filled_dem - dem

    # # Normalize Difference DEM with Z-score normalization (mean = 0, stddev = 1)
    # normalized_diff_dem = normalize_diff_dem(diff_dem)

    return dem, grid


if __name__ == "__main__":
    dem_path = str(Path(config.KC_DEM_ROOT) / "Kane2017BE.tif")
    output_path = str(Path(config.KC_DEM_ROOT) / "Kane2017BE_fill_diff.tif")

    fill_dem, grid = fill_analysis(dem_path)

    export_filled_dem(fill_dem, grid, output_path)
