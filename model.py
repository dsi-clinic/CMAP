"""This module provides a class for configuring and using segmentation models.

It includes utilities for selecting different architectures, backbones,
input channels, number of classes,
number of filters, and weights to initialize the model.
"""

import segmentation_models_pytorch as smp
from torchgeo.models import get_model, get_weight

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
        model: str,
        backbone: str,
        num_classes: int,
        weights: str | bool,
        in_channels: int,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
    ):
        """Initialize the SegmentationModel object with the provided parameters.

        Parameters
        ----------
        model : str
            The model type to use. Options are 'unet', 'deeplabv3+', and 'fcn'.
        backbone : str
            The encoder to use, which is the classification model that will be used to extract features.
            Options are listed on the smp docs.
        num_classes : int
            The number of classes to predict. Should match the number of classes in the mask.
        weights : Union[str, bool]
            The weights to use for the model. If True, uses imagenet weights.
            Can also accept a string path to a weights file, or a WeightsEnum with pretrained weights.
        in_channels : int
            Number of input channels to the model.
        dropout : float, optional
            Dropout rate to use in the model, by default 0.3
        freeze_backbone : bool, optional
            Whether to freeze the backbone of the model, by default False
        freeze_decoder : bool, optional
            Whether to freeze the decoder of the model, by default False
        """
        self.model_type = model
        self.backbone = backbone.lower()
        self.num_classes = num_classes
        self.weights = weights
        self.in_channels = in_channels
        self.dropout = dropout
        self.freeze_backbone = freeze_backbone
        self.freeze_decoder = freeze_decoder

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

        if model != "fcn" and self.weights and self.weights is not True:
            weights = get_weight(self.weights)
            if weights.meta["model"].lower() != self.backbone:
                raise ValueError(
                    f"Backbone for weights does not match model backbone. Backbone for weights is "
                    f"{weights.meta['model']} and backbone for model is {self.backbone}"
                )
            encoder = get_model(weights.meta["model"])
            weights_chans = encoder.meta["in_chans"]
            encoder.load_state_dict(weights.get_state_dict(progress=True))
            self.model.encoder = encoder

            self.in_channels = max(self.in_channels, weights_chans)

        self.model.in_channels = self.in_channels

        if self.freeze_backbone and model in ["unet", "deeplabv3+"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        if self.freeze_decoder and model in ["unet", "deeplabv3+"]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    def __getbackbone__(self):
        """Returns the backbone of the model"""
        return self.backbone

    def __getweights__(self):
        """Returns the weights of the model"""
        return self.weights

    def __getdropout__(self):
        """Returns the dropout rate of the model"""
        return self.dropout
