import geopandas as gpd
from shapely.geometry import box, Polygon, MultiPolygon
import numpy as np
from tqdm import tqdm

import os


def split_polygon_to_patches(geometry, patch_size):
    """Split a polygon geometry into smaller patches of given size.

    Args:
        geometry: Shapely geometry of the polygon.
        patch_size: Size of the patches in CRS units.

    Returns:
        List of smaller polygon patches.
    """
    minx, miny, maxx, maxy = geometry.bounds
    x_range = np.arange(minx, maxx, patch_size)
    y_range = np.arange(miny, maxy, patch_size)
    patches = []
    for x in x_range:
        for y in y_range:
            patch = box(x, y, x + patch_size, y + patch_size)
            if geometry.intersects(patch):
                intersection = geometry.intersection(patch)
                if isinstance(intersection, Polygon):
                    patches.append(intersection)
                #elif intersection.geom_type == 'MultiPolygon':
                    #patches.extend([poly for poly in intersection])
                elif isinstance(intersection, MultiPolygon):
                    patches.extend([poly for poly in intersection.geoms])

    
    return patches

def preprocess_shapefile(input_path, output_path, patch_size):
    """Preprocess the shapefile by splitting polygons into smaller patches.

    Args:
        input_path: Path to the input shapefile.
        output_path: Path to the output shapefile.
        patch_size: Size of the patches in CRS units.
    """
    # Load the input shapefile
    gdf = gpd.read_file(f'/vsizip/{input_path}')

    # List to store the new patches
    new_patches = []

    # Split each polygon into smaller patches
    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Splitting polygons"):
        geometry = row['geometry']
        patches = split_polygon_to_patches(geometry, patch_size)
        for patch in patches:
            new_row = row.copy()
            new_row['geometry'] = patch
            new_patches.append(new_row)

    # Create a new GeoDataFrame with the patches
    new_gdf = gpd.GeoDataFrame(new_patches, columns=gdf.columns, crs=gdf.crs)

    # Save the new shapefile
    new_gdf.to_file(output_path)

if __name__ == "__main__":
    input_shapefile = os.path.join("/net/projects/cmap/data/kane-county-data/Kane_Co_Open_Water_Layer.zip")
    output_shapefile = f"/net/projects/cmap/workspaces/Output_River_Shapefile"
    patch_size = 256  # Define your patch size here

    preprocess_shapefile(input_shapefile, output_shapefile, patch_size)