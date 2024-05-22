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
        in_channels: int = 5,
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
            When adding DEM data, this will be 5.
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
        if model != "fcn":
            # set custom weights
            # Assuming config.WEIGHTS contains the desired value,
            # e.g., "ResNet50_Weights.LANDSAT_TM_TOA_MOCO"
            if weights and weights is not True:
                weights_module, weights_submodule = weights.split(".")
                imported_module = importlib.import_module("torchgeo.models")
                # Get the attribute from the module
                weights_class = getattr(imported_module, weights_module)
                weights_attribute = getattr(weights_class, weights_submodule)
                weights_meta = weights_attribute.meta
                weights_chans = weights_meta["in_chans"]
                del imported_module

                if weights_chans > in_channels:
                    in_channels = weights_chans

                weights_backbone = weights_module.split("_")[0]
                if (
                    weights_backbone.lower() != backbone
                    and backbone != "vit_small_patch16_224"
                ):
                    raise ValueError(
                        f"Backbone for weights '{weights_backbone}' does not match {backbone}."  # noqa: B950
                    )

                if isinstance(weights_attribute, WeightsEnum):
                    state_dict = weights_attribute.get_state_dict(progress=True)
                elif os.path.exists(weights_attribute):
                    _, state_dict = utils.extract_backbone(weights_attribute)
                else:
                    state_dict = get_weight(weights_attribute).get_state_dict(
                        progress=True
                    )

            if model == "unet":
                self.model = smp.Unet(
                    encoder_name=backbone,
                    encoder_weights="swsl" if weights is True else None,
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
            if weights and weights is not True and model != "test_weights":
                self.model.encoder.load_state_dict(state_dict)
            self.model.in_channels = in_channels
