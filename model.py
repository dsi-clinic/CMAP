"""This module provides a class for configuring and using segmentation models.

It includes utilities for selecting different architectures, backbones,
input channels, number of classes,
number of filters, and weights to initialize the model.
"""

import importlib
from pathlib import Path

import segmentation_models_pytorch as smp
from torchgeo.models import get_weight
from torchgeo.trainers import utils
from torchvision.models._api import WeightsEnum

from fcn import FCN


class SegmentationModel:
    """This class represents a segmentation model for image segmentation tasks.

    It allows configuring various aspects of the model architecture, such as
    the model type, backbone, number of input channels,
    number of classes to predict, number of filters (for FCN model), and
    weights to initialize the model.

    Attributes:
        None

    Methods:
        __init__: Initializes the SegmentationModel object with the provided
        parameters and configures the model accordingly.

    Exceptions:
        ValueError: Raised if the provided model type is not valid or if the
        backbone for weights does not match the model backbone.
    """

    def __init__(
        self,
        model_type: str,
        backbone: str,
        weights: bool,
        num_classes: int,
        in_channels: int,
        dropout: float = 0.3,
    ):
        """Initialize the SegmentationModel object with the provided model configuration.

        Parameters
        ----------
            model_type: The model type to use ('unet', 'deeplabv3+', and 'fcn')
            backbone: The encoder to use, which is the classification model
            that will be used to extract features. Options are listed on the
            smp docs.
            weights: Union[str, bool], The weights to use for the model.
            If True, uses imagenet weights. Can also accept a string path
            to a weights file, or a WeightsEnum with pretrained weights.
            num_classes: int, The number of classes to predict. Should match
            the number of classes in the mask.

        Returns:
        -------
        None
        """
        model = model_type
        self.backbone = backbone
        self.weights = weights
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout = dropout

        if model != "fcn":
            state_dict = None
            # set custom weights
            # Assuming config.WEIGHTS contains the desired value,
            # e.g., "ResNet50_Weights.LANDSAT_TM_TOA_MOCO"
            if self.weights and self.weights is not True:
                weights_module, weights_submodule = self.weights.split(".")
                weights_attribute = getattr(
                    importlib.import_module("torchgeo.models"),
                    weights_module + "." + weights_submodule,
                )
                weights_chans = weights_attribute.meta["in_chans"]

                self.in_channels = max(self.in_channels, weights_chans)

                weights_backbone = weights_module.split("_")[0]
                if (
                    weights_backbone.lower() != self.backbone
                    and self.backbone != "vit_small_patch16_224"
                ):
                    raise ValueError(
                        "Backbone for weights does not match model backbone."
                    )

                if isinstance(weights_attribute, WeightsEnum):
                    state_dict = weights_attribute.get_state_dict(progress=True)
                elif Path.exists(weights_attribute):
                    _, state_dict = utils.extract_backbone(weights_attribute)
                else:
                    state_dict = get_weight(weights_attribute).get_state_dict(
                        progress=True
                    )

            if model == "unet":
                self.model = smp.Unet(
                    encoder_name=self.backbone,
                    encoder_weights="swsl" if self.weights is True else None,
                    in_channels=self.in_channels,
                    classes=self.num_classes,
                    aux_params={
                        "classes": self.num_classes,
                        "dropout": self.dropout,
                    },
                )

            elif model == "deeplabv3+":
                self.model = smp.DeepLabV3Plus(
                    encoder_name=self.backbone,
                    encoder_weights=("imagenet" if self.weights is True else None),
                    in_channels=self.in_channels,
                    classes=self.num_classes,
                    aux_params={
                        "classes": self.num_classes,
                        "dropout": self.dropout,
                    },
                )

        elif model == "fcn":
            self.model = FCN(
                in_channels=self.in_channels,
                classes=self.num_classes,
                num_filters=3,
                dropout=self.dropout,
            )

        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                "Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )
        if self.weights and self.weights is not True:
            self.model.encoder.load_state_dict(state_dict)
        self.model.in_channels = self.in_channels

    def __getbackbone__(self):
        """Returns the backbone of the model"""
        return self.backbone

    def __getweights__(self):
        """Returns the weights of the model"""
        return self.weights

    def __getdropout__(self):
        """Returns the dropout rate of the model"""
        return self.dropout
