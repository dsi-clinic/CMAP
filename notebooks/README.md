### Notebook directory

* `Image-Visualizations.ipynb` : This notebook demonstrates how to visualize images and shapefiles with geospatial 
metadata.
    * Satellite images sourced from NAIP via the planetary computer API. See the docs here: https://planetarycomputer.microsoft.com/dataset/naip
    * Green Infrastructure Baseline Inventory (GIBI) shapefile data sourced via https://www.metroplanning.org/work/project/23/subpage/8.  
    * Kane County Stormwater Storage infrastructure data provided by Kane County via CMAP box; to access this data in the future, contact [Rob Linke] (linkerobert@KaneCountyIL.gov) directly or through [Holly Hudson] (HHudson@cmap.illinois.gov)
    * More info on Kane County GIS can be found via https://gistech.countyofkane.org/gisims/kanemap/kanegis4_agox.html#.

* `DEM-Fill-Analysis.ipynb` : This notebook demonstrates how to visualize an overlay plot with the RGB aerial image and 'fill difference' DEM data.
    * Before using this notebook, change the filename_glob variable to "Kane2017BE_fill_diff.tif" in the KaneDEM class, which is located in /data/dem.py.