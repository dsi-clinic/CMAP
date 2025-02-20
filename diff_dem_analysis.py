"""Module finds difference between filled/unfilled DEMs and exports it"""

from pathlib import Path

import numpy as np
import rasterio

import configs.config as config

if __name__ == "__main__":
    dem_path = str(Path(config.KC_DEM_ROOT) / "Kane2017BE.tif")
    filled_path = str(Path(config.KC_DEM_ROOT) / "Kane2017BE_filled.tif")
    output_path = str(Path(config.KC_DEM_ROOT) / "Kane2017BE_difference.tif")

    with (
        rasterio.open(filled_path) as filled_src,
        rasterio.open(dem_path) as normal_src,
    ):
        profile = filled_src.profile
        profile.update(dtype=rasterio.float32, nodata=-9999)

        with rasterio.open(output_path, "w", **profile) as dst:
            for _, window in filled_src.block_windows(
                1
            ):  # Unpack tuple (block_id, window)
                filled_dem = filled_src.read(1, window=window)
                normal_dem = normal_src.read(1, window=window)

                dem_diff = np.where(
                    filled_dem != normal_dem, filled_dem - normal_dem, 0
                )

                dst.write(dem_diff.astype(np.float32), 1, window=window)

    print(f"Difference DEM saved as {output_path}")
