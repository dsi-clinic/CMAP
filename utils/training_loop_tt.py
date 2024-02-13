"""
Creating own DataModule: https://medium.com/@janwinkler91/pytorch-lightning-creating-my-first-custom-data-module-64a33f437356
Lightning DataModule Docs: https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/datamodules.html
MNIST DataModule Docs: https://pytorch-lightning.readthedocs.io/en/1.5.10/extensions/datamodules.html#
"""

import importlib.util
import os

import torch
import torch.nn as nn

# from segmentation_models_pytorch.losses import JaccardLoss
from torch.nn.modules import Module
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torchgeo.datasets import NAIP, BoundingBox, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler
from torchgeo.trainers import SemanticSegmentationTask

# imports for lightning
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# project imports
# TODO: figure out how to import as package
from . import repo_root
from .lightning_data_module import KCDataModule

# hyperparameters
num_classes = 5  # predicting 4 classes + background
lr = 1e-3

# get device for training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Pretrained weights
# can't find weights with 4 input channels
"""
weights = ResNet50_Weights.SENTINEL2_ALL_MOCO 
"""

# Lightning model
# Parameters
epochs = 5
fast_dev_run = False
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", dirpath=repo_root, save_top_k=1, save_last=True
)
early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10)
logger = TensorBoardLogger(save_dir=repo_root, name="lightning_logs", version=None)

task = SemanticSegmentationTask(
    model='unet',
    backbone='resnet50',
    weights=None, 
    in_channels=4, # NAIP has R, G, B, and NIR channels
    num_classes=num_classes, 
    num_filters=3, # only applicable when model="fcn"
    loss='jaccard', 
    lr=lr, 
    patience=10, 
    freeze_backbone=False, 
    freeze_decoder=False
)

# Training
trainer = Trainer(
    accelerator=device,
    callbacks=[checkpoint_callback, early_stopping_callback],
    fast_dev_run=fast_dev_run,
    log_every_n_steps=1,
    logger=logger,
    min_epochs=1,
    max_epochs=epochs
)

trainer.fit(model=task, datamodule=KCDataModule)

trainer.test(model=task, datamodule=KCDataModule)

# save the model status
"""
# TODO: update with the correct path
torch.save(task.state_dict(), "/home/tamamitamura/2024-winter-cmap/output/model.pth")
print(
    "Saved PyTorch Model State to /home/tamamitamura/2024-winter-cmap/output/model.pth"
)
"""
