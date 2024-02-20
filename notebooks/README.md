### Notebook directory

* `Image-Visualizations.ipynb` : This notebook demonstrates how to visualize images and shapefiles with geospatial 
metadata.
    * Satellite images sourced from NAIP via the planetary computer API. See the docs here: https://planetarycomputer.microsoft.com/dataset/naip
    * Green Infrastructure Baseline Inventory (GIBI) shapefile data sourced via https://www.metroplanning.org/work/project/23/subpage/8.  
    * Kane County Stormwater Storage infrastructure data provided by Kane County via CMAP

* `example_torchgeo_model.ipynb`: Image Classification with EuroSAT Dataset

This notebook demonstrates image classification using the EuroSAT dataset. It utilizes PyTorch Lightning for training and evaluation, including features like early stopping, model checkpointing, and TensorBoard logging. The notebook also includes data preprocessing with the EuroSAT100DataModule from TorchGeo. The goal is to train a ResNet18 model on satellite images to classify land cover types with high accuracy.

