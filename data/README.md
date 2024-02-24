### Data

This directory contains information for use in this project. 

Please make sure to document each source file here.

# Kane County Stormwater Storage infrastructure data

Provided by Kane County via CMAP box; to access this data in the future, contact [Rob Linke] (linkerobert@KaneCountyIL.gov) directly, or through [Holly Hudson] (HHudson@cmap.illinois.gov)

# Kane County Data Retrieval and Processing

This script retrieves geospatial and image data for Kane County. It reads geospatial data files, defines labels for features, iterates over image files to create masks, handles exceptions, and optionally downloads images if not available locally. It prepares the data for further analysis and modeling tasks related to Kane County.

# Kane County Raster Dataset

This class defines a custom Raster Dataset for Kane County. It specifies file patterns and regex for file naming conventions, defines bands, colormap, and labels for different features present in the raster data. The dataset is designed to be used with TorchGeo for processing and analysis tasks related to Kane County's geospatial data.

