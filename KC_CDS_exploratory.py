import geopandas as gpd
import fiona

kaneCDS_file_path = "/net/projects/cmap/data/kane-county-data/CountywideDepressionalStorage.gdb.zip"

layers = fiona.listlayers(kaneCDS_file_path)

# Print the available layers
print("Available layers:", layers)

if layers:

    for selected_layer in layers:
        print("Selected layer:", selected_layer)
        
        # Load the GeoPandas DataFrame
        CDS_df = gpd.read_file(kaneCDS_file_path, layer=selected_layer)
        #print(CDS_df.head(3))
        """the WTR_Countywide_Stormwater_Storage__ATTACH layer doesn't have data 
        counts, so the below commented out line will flag an error"""
        print("Geometry Type:",type(CDS_df.geometry[0]))
        print("Length:", len(CDS_df.index))