"""Module performs fill analysis, finds the difference, and exports it"""

from pathlib import Path

from pysheds.grid import Grid

import configs.config as config


def export_filled_dem(diff_dem, grid, output_path):
    """Exports difference DEM to file
    
    Args:
        diff_dem: DEM containing difference between filled and original DEM
        grid: PySheds grid object
        output_path: file path to export the DEM data to
    """
    grid.to_raster(diff_dem, output_path)

    print(f"Difference Diff DEM has been exported to {output_path}")

    return


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
    dem = grid.read_raster(tiff_path)

    # Fill Pits and Depressions
    pit_filled_dem = grid.fill_pits(dem)
    filled_dem = grid.fill_depressions(pit_filled_dem)

    # Find Difference Between Filled DEM and Original DEM
    diff_dem = filled_dem - dem

    return diff_dem, grid


if __name__ == "__main__":
    dem_path = str(Path(config.KC_DEM_ROOT) / "Kane2017BE.tif")
    output_path = str(Path(config.KC_DEM_ROOT) / "Kane2017BE_fill_diff.tif")

    diff_dem, grid = fill_analysis(dem_path)

    export_filled_dem(diff_dem, grid, output_path)
