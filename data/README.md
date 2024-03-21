### Data directory

#### Kane County Stormwater Storage Infrastructure Data

This dataset contains geodatabase files with geospatial geometry data, including coordinates, shapes, and types for stormwater infrastructure.

Provided by Kane County via CMAP box; to access this data in the future, contact [Rob Linke](linkerobert@KaneCountyIL.gov) directly, or through [Holly Hudson](HHudson@cmap.illinois.gov)

#### National Agricultural Imagery Program (NAIP) Data

This dataset contains geotiff image files with aerial imagery of the United States and geospatial metadata.

Retrieved using the [Planetary Computer API](https://planetarycomputer.microsoft.com/dataset/naip).

OCM Partners, 2024: NAIP Digital Ortho Photo Image, https://www.fisheries.noaa.gov/inport/item/49502

#### Kane County Raster Dataset (`kc.py`)

This class defines a custom Raster Dataset for Kane County. It specifies file patterns and regex for file naming conventions and defines bands, colormap, and labels for different features present in the raster data. The dataset is designed to be used with TorchGeo for processing and analysis tasks related to Kane County's geospatial data. See TorchGeo's docs for the difference between a raster and vector dataset.

#### Kane County Vector Dataset (`kcv.py`)

This class defines a custom Vector Dataset for Kane County. It specifies file patterns and regex for file naming conventions, and defines colormap and labels for different features present in the vector data. The dataset is designed to be used with TorchGeo for processing and analysis tasks related to Kane County's geospatial data. See TorchGeo's docs for the difference between a raster and vector dataset.
