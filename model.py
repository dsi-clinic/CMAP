"""
This module provides a class for configuring and using segmentation models.

It includes utilities for selecting different architectures, backbones,
input channels, number of classes,
number of filters, and weights to initialize the model.

Attributes:
    segmentation_models_pytorch (module): A Python package containing implementations
    of various segmentation models.
    timm (module): A library for model architectures and pretrained weights from
    the PyTorch Image Models repository.
    torchgeo.models (module): Models provided by the TorchGeo library for
    geospatial data processing.
    torchgeo.trainers.utils (module): Utilities for training models in the
    TorchGeo library.
    torchvision.models._api (module): A module containing model definitions
    from the torchvision library.

Classes:
    SegmentationModel: A class for configuring and using segmentation models.

Functions:
    None

Exceptions:
    None
"""

import importlib
import os
import torch

import segmentation_models_pytorch as smp
from torchgeo.models import FCN, get_weight
from torchgeo.trainers import utils
from torchvision.models._api import WeightsEnum
from diffusers.src.diffusers.models.unets import UNet2DConditionModel
import torch.nn as nn

class SegmentationModel():
    """
    This class represents a segmentation model for image segmentation tasks.

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

    def __init__(self, model_config):
        """
        Initialize the SegmentationModel object with the provided model configuration.

        Parameters
        ----------
        model_config : dict
            A dictionary containing configuration parameters for the model.
            The dictionary should contain the following keys:
                - "model": str, The model type to use. Options are
                'unet', 'deeplabv3+', and 'fcn'.
                - "backbone": str, The encoder to use, which is the classification
                model that will be used to extract features. Options are
                listed on the smp docs.
                - "num_classes": int, The number of classes to predict. Should
                match the number of classes in the mask.
                - "weights": Union[str, bool], The weights to use for the model.
                If True, uses imagenet weights. Can also accept a string path
                to a weights file, or a WeightsEnum with pretrained weights.

        Returns
        -------
        None
        """
        model = model_config["model"]
        self.backbone = model_config["backbone"]
        self.num_classes = model_config["num_classes"]
        self.weights = model_config["weights"]
        self.in_channels = model_config.get("in_channels")
        self.model_path = model_config["model_path"]
        if self.in_channels is None:
            self.in_channels = 5

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
                elif os.path.exists(weights_attribute):
                    _, state_dict = utils.extract_backbone(weights_attribute)
                else:
                    state_dict = get_weight(weights_attribute).get_state_dict(
                        progress=True
                    )
            if model == "diffsat":
                from diffusers.src.diffusers import DiffusionPipeline
                # batch_size = x.shape[0]
                # timesteps = torch.zeros(batch_size, device=x.device)
                # encoder_hidden_states = torch.zeros(batch_size, 77, 768, device=x.device)
                self.model = UNet2DConditionModel.from_pretrained(
                    self.model_path,
                    subfolder="unet",
                    in_channels=self.in_channels,
                    out_channels=self.num_classes,
                    low_cpu_mem_usage=False,
                    ignore_mismatched_sizes=True,
                    use_safetensors=True
                    #encoder_name=self.backbone,
                    # encoder_weights="swsl" if self.weights is True else None,
                    #timestep=timesteps, encoder_hidden_states=encoder_hidden_states
                )


                
            elif model == "unet":
                self.model = smp.Unet(
                    encoder_name=self.backbone,
                    encoder_weights="swsl" if self.weights is True else None,
                    in_channels=self.in_channels,
                    classes=self.num_classes,
                )

            elif model == "deeplabv3+":
                self.model = smp.DeepLabV3Plus(
                    encoder_name=self.backbone,
                    encoder_weights=("imagenet" if self.weights is True else None),
                    in_channels=self.in_channels,
                    classes=self.num_classes,
                )

            elif model == "fcn":
                self.model = FCN(
                    in_channels=self.in_channels,
                    classes=self.num_classes,
                    num_filters=3,
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
        """
        returns the backbone of the model
        """
        return self.backbone

    def __getweights__(self):
        """
        returns the weights of the model
        """
        return self.weights
    
    # def forward(self, x):
    #     if self.model == 'diffsat':
    #         batch_size = x.shape[0]
    #         timesteps = torch.zeros(batch_size, device=x.device)
    #         encoder_hidden_states = torch.zeros(batch_size, 77, 768, device=x.device)
    #         return self.model(x, timestep=timesteps, encoder_hidden_states=encoder_hidden_states)
    #     else:
    #         return self.model(x)

