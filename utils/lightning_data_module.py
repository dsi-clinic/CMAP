import importlib.util
import os

import torch

from torch.utils.data import DataLoader, random_split
from torchgeo.datasets import NAIP, BoundingBox, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler

import pytorch_lightning as pl
from typing import Optional

# project imports
from . import repo_root
from .model import SegmentationModel

spec = importlib.util.spec_from_file_location(
    "kane_county", os.path.join(repo_root, "data", "kane_county.py")
)
kane_county = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kane_county)
KaneCounty = kane_county.KaneCounty

spec = importlib.util.spec_from_file_location(
    "configs", os.path.join(repo_root, "configs", "dsi.py")
)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

KC_IMAGE_ROOT = "/net/projects/cmap/data/KC-images"
KC_MASK_ROOT = "/net/projects/cmap/dataKC-masks/top-structures-masks"

# build dataset
naip = NAIP(config.KC_IMAGE_ROOT)
kc = KaneCounty(config.KC_MASK_ROOT)
dataset = naip & kc

# train/test split
roi = dataset.bounds
midx = roi.minx + (roi.maxx - roi.minx) / 2
midy = roi.miny + (roi.maxy - roi.miny) / 2

# hyperparameters
batch_size = 64
patch_size = 256

# random batch sampler for training, grid sampler for testing
train_roi = BoundingBox(roi.minx, midx, roi.miny, roi.maxy, roi.mint, roi.maxt)
train_sampler = RandomBatchGeoSampler(
    dataset=dataset, size=patch_size, batch_size=batch_size, roi=train_roi
)
test_roi = BoundingBox(midx, roi.maxx, roi.miny, roi.maxy, roi.mint, roi.maxt)
test_sampler = GridGeoSampler(
    dataset, size=patch_size, stride=patch_size, roi=test_roi
)

class KCDataModule(pl.LightningDataModule):

    def __init__(self,):
        super(KCDataModule).__init__()
        
    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """
      
        # Assign train/val datasets for use in dataloaders
        # the stage is used in the Pytorch Lightning trainer method, which you can call as fit (training, evaluation) or test, also you can use it for predict, not implemented here
        
        if stage == "fit" or stage is None:
            train_set_full =  dataset
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = dataset
            
    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet. 
    def train_dataloader(self):
        return DataLoader(self.train, 
                          batch_size=batch_size,
                          num_workers=8, 
                          batch_sampler=train_sampler, 
                          collate_fn=stack_samples)

    def val_dataloader(self):
        return DataLoader(self.validate, 
                          batch_size=batch_size,
                          num_workers=8, 
                          batch_sampler=train_sampler, 
                          collate_fn=stack_samples)

    def test_dataloader(self):
        return DataLoader(self.test, 
                          batch_size=batch_size, 
                          num_workers=8,
                          batch_sampler=test_sampler,
                          collate_fn=stack_samples)