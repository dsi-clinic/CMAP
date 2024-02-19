import importlib
import os
from typing import Union

import segmentation_models_pytorch as smp
import timm
from torchgeo.models import FCN, get_weight
from torchgeo.trainers import utils
from torchvision.models._api import WeightsEnum


class SegmentationModel:
    def __init__(
        self,
        model: str = "unet",
        backbone: str = "resnet50",
        in_channels: int = 4,
        num_classes: int = None,
        num_filters: int = 3,
        weights: Union[WeightsEnum, str, bool] = True,
    ):
        """
        Code taken from torchgeo.models.SemanticSegmentationTask.configure_models

        Configure the model with the provided params.

        Parameters
        ----------
        model : str
            The model type to use. Options are 'unet', 'deeplabv3+', and 'fcn'.

        backbone : str
            The encoder to use, which is the classification model that will be used to
            extract features. Options are listed on the smp docs:
            <https://smp.readthedocs.io/en/latest/encoders.html>

        in_channels : int
            The number of input channels, i.e. the depth of the input image. For NAIP
            data, this is 4.
            When using TorchGeo's pretrained weights, change appropriately and add
            channels to data.

        num_classes : int
            The number of classes to predict. Should match the number of classes in
            the mask.

        num_filters : int
            The number of filters to use in the model. Only used for the FCN model.

        weights : Union[str, bool]
            The weights to use for the model. If True, uses imagenet weights. Can also
            accept a string path to a weights file, or a WeightsEnum with pretrained
            weights. Options for pretrained weights are listed on the torchgeo docs:
            <https://torchgeo.readthedocs.io/en/stable/api/models.html#pretrained-weights>
        """
        if model == "unet":
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="ssl" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == "fcn":
            self.model = FCN(
                in_channels=in_channels,
                classes=num_classes,
                num_filters=num_filters,
            )
        elif model == "test_weights":
            self.model = timm.create_model(
                backbone, in_chans=in_channels, num_classes=num_classes
            )
        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                "Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

        # set custom weights
        if model != "fcn":
            if weights and weights is not True:
                if isinstance(weights, WeightsEnum):
                    weights = getattr(
                        importlib.import_module("ResNet50_Weights"),
                        weights,
                    )
                    state_dict = weights.get_state_dict(progress=True)
                elif os.path.exists(weights):
                    _, state_dict = utils.extract_backbone(weights)
                else:
                    state_dict = get_weight(weights).get_state_dict(
                        progress=True
                    )
                self.model.encoder.load_state_dict(state_dict)
