"""Module performs fill analysis on baseline DEM and exports it

Run this prior to diff_dem_analysis.py to fill all depressions prior to finding difference.
diff_dem_analysis.py will take the filled DEM exported from here and the baseline DEM and find difference to emphasize depressions.
"""

from pathlib import Path

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

    print(f"Filled DEM has been exported to {output_path}")

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
    dem = grid.read_raster(tiff_path).copy()

    # Fill Pits and Depressions in place to limit memory usage
    grid.fill_pits(dem, out=dem)
    grid.fill_depressions(dem, out=dem)

    return dem, grid


if __name__ == "__main__":
    dem_path = str(Path(config.KC_DEM_ROOT) / "Kane2017BE.tif")
    output_path = str(Path(config.KC_DEM_ROOT) / "Kane2017BE_filled.tif")

    fill_dem, grid = fill_analysis(dem_path)

    export_filled_dem(fill_dem, grid, output_path)
