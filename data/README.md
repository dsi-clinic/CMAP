### Data directory

#### Kane County Stormwater Storage Infrastructure Data

This dataset contains geodatabase files with geospatial geometry data, including coordinates, shapes, and types for stormwater infrastructure.

Provided by Kane County via CMAP box; to access this data, contact [Rob Linke](linkerobert@KaneCountyIL.gov) directly, or through [Holly Hudson](HHudson@cmap.illinois.gov)

#### National Agricultural Imagery Program (NAIP) Data

This dataset contains geotiff image files with aerial imagery of the United States and geospatial metadata.

Retrieved using the [Planetary Computer API](https://planetarycomputer.microsoft.com/dataset/naip).

OCM Partners, 2024: NAIP Digital Ortho Photo Image, https://www.fisheries.noaa.gov/inport/item/49502

#### Kane County Vector Dataset (`kc.py`)

This class defines a custom Vector Dataset for Kane County. It specifies file patterns and regex for file naming conventions, and defines colormap and labels for different features present in the vector data. The dataset is designed to be used with TorchGeo for processing and analysis tasks related to Kane County's geospatial data. See TorchGeo's docs for the difference between a raster and vector dataset.

#### River Dataset (`rd.py`)

This class defines a custom Vector Dataset for both Kane County and the River Dataset. It specifies file patterns and regex for file naming conventions, and defines colormap and labels for different features present in the vector data. The dataset is designed to be used with TorchGeo for processing and analysis tasks related to Kane County's geospatial data. See TorchGeo's docs for the difference between a raster and vector dataset.

Additional information: The RiverDataset populates its index by generating chips that span Kane County, then finds the chips that intersect with polygons from the KaneCounty gdf and the RiverDataset gdf. In total, 7734 chips are inserted into the index, with each polygon object corresponding to at least one chip. 

#### Kane County DEM Dataset (`dem.py`)

This class defines the Digital Elevation Model data for Kane County. It specifies the regex for the file containing the DEM data, assuming that it is a single band file. Instructions for how to convert a geodatabase file into a tif file can be found at the top of the file. The conversion process requires the installation of gdal and is run on the command line.

#### Samplers for training and testing (`sampler.py)
This module provides custom samplers used for sampling patches or chips from geospatial datasets in a manner that avoids tries to avoid areas of missing label. These samplers are designed to handle cases where the bounding boxes in the dataset are smaller than the desired patch size, ensuring that both background and feature areas are adequately represented in the sampled data.
