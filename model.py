"""
Module: segmentation_model.py

This module provides a class for configuring and
using semantic segmentation models for image
analysis tasks. It includes support for popular
models such as UNet, DeepLabV3+, and FCN.

Dependencies:
- importlib
- os
- typing.Union
- segmentation_models_pytorch as smp
- timm
- torchgeo.models.FCN, torchgeo.models.get_weight
- torchgeo.trainers.utils
- torchvision.models._api.WeightsEnum

Usage:
To use this module, import it and create an instance of the
SegmentationModel class, providing the desired parameters.
Then, the configured model can be accessed via the 'model'
attribute of the instance.

Example:
    from segmentation_model import SegmentationModel

    # Create an instance of SegmentationModel
    segmentation_model = SegmentationModel(
        model='unet',
        backbone='resnet50',
        in_channels=5,
        num_classes=10
    )

    # Access the configured model
    model = segmentation_model.model
"""

import importlib
import os
from typing import Union

import segmentation_models_pytorch as smp
from torchgeo.models import FCN, get_weight
from torchgeo.trainers import utils
from torchvision.models._api import WeightsEnum


class SegmentationModel:
    """
    Code taken from torchgeo.models.SemanticSegmentationTask.configure_models

    A class for configuring and using semantic segmentation models
    for image analysis tasks.

    Attributes:
        model_type (str): The type of segmentation model to use,
        such as 'unet', 'deeplabv3+', or 'fcn'.
        backbone (str): The encoder to use, which is the classification model that
        will be used to extract features.
        in_channels (int): The number of input channels, i.e., the depth of the
        input image.
        num_classes (int): The number of classes to predict.
        weights (Union[WeightsEnum, str, bool]): The weights to use for the model.
        state_dict (dict): The state dictionary containing the model's weights.
        model: The configured segmentation model.

    """

    def __init__(
        self,
        model: str = "unet",
        backbone: str = "resnet50",
        in_channels: int = 5,
        num_classes: int = None,
    ):
        """
        Initialize an instance of the SegmentationModel class.

        Parameters:
            model (str): The type of segmentation model to use. Default is 'unet'.
            backbone (str): The encoder to use. Default is 'resnet50'.
            in_channels (int): The number of input channels. Default is 5.
            num_classes (int): The number of classes to predict. Default is None.
        """
        self.model_type = model
        self.backbone = backbone
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.weights = None
        self.state_dict = None
        self.model = None

    def set_weights(self, weights: Union[WeightsEnum, str, bool]):
        """
        Set the weights for the segmentation model.

        Parameters:
            weights (Union[WeightsEnum, str, bool]): The weights to use for the model.
                If True, uses imagenet weights. Can also accept a string path to
                a weights file, or a WeightsEnum with pretrained weights.
        """
        if weights and weights is not True:
            weights_module, weights_submodule = weights.split(".")
            imported_module = importlib.import_module("torchgeo.models")
            weights_class = getattr(imported_module, weights_module)
            weights_attribute = getattr(weights_class, weights_submodule)
            weights_meta = weights_attribute.meta
            weights_chans = weights_meta["in_chans"]
            del imported_module

            self.in_channels = max(self.in_channels, weights_chans)

            if (
                weights_module.split("_")[0].lower() != self.backbone
                and self.backbone != "vit_small_patch16_224"
            ):
                raise ValueError(
                    "Backbone for weights doesn't match model backbone."
                )

            if isinstance(weights_attribute, WeightsEnum):
                self.state_dict = weights_attribute.get_state_dict(
                    progress=True
                )
            elif os.path.exists(weights_attribute):
                _, self.state_dict = utils.extract_backbone(weights_attribute)
            else:
                self.state_dict = get_weight(weights_attribute).get_state_dict(
                    progress=True
                )

            self.weights = weights

    def configure_model(self):
        """
        Configure the segmentation model based on the provided parameters.
        """
        num_filters = 3 if self.model_type == "fcn" else None
        if self.model_type == "unet":
            self.model = smp.Unet(
                encoder_name=self.backbone,
                encoder_weights="swsl" if self.weights is True else None,
                in_channels=self.in_channels,
                classes=self.num_classes,
            )
        elif self.model_type == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.backbone,
                encoder_weights="imagenet" if self.weights is True else None,
                in_channels=self.in_channels,
                classes=self.num_classes,
            )
        elif self.model_type == "fcn":
            self.model = FCN(
                in_channels=self.in_channels,
                classes=self.num_classes,
                num_filters=num_filters,
            )
        else:
            raise ValueError(
                f"Model type '{self.model_type}' is not valid. "
                "Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

        if (
            self.weights
            and self.weights is not True
            and self.model_type != "test_weights"
        ):
            self.model.encoder.load_state_dict(self.state_dict)
        self.model.in_channels = self.in_channels
