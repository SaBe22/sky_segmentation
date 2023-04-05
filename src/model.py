"""A module that defines a segmentation model class"""
from functools import cached_property

import torch
from torch import nn
from torchvision import models

SUPPORTED_MODELS: list[str] = [
    "deeplabv3_mobilenetv3",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
]

class SegmentationModel:
    """
    A class for image segmentation models using PyTorch.

    Args:
        model_name: Name of the segmentation model to use. Currently supported models are
            "deeplabv3_mobilenetv3", "deeplabv3_resnet50", and "deeplabv3_resnet101".
        pretrained_weights: Path to the pretrained weights to load or "DEFAULT" to use default
            weights. Default is "DEFAULT".
        num_classes: Number of classes to segment. Default is 1.
    """
    def __init__(
        self,
        model_name: str = "deeplabv3_mobilenetv3",
        pretrained_weights: str = "DEFAULT",
        num_classes: int = 1,
        device: str = torch.device("cpu")
    ) -> None:
        self.model_name = model_name
        self.pretrained_weights = pretrained_weights
        self._model = None
        self.num_classes = num_classes
        self.device = device

    @cached_property
    def model(self) -> torch.nn.Module:
        """
        Returns:
            The PyTorch segmentation model chosen by the attribute model_name.
        """
        if self.model_name not in SUPPORTED_MODELS:
            raise NotImplementedError(f"Model {self.model_name} is unknown please us one of "
                                      "the following model {SUPPORTED_MODELS}")

        if self.model_name == "deeplabv3_mobilenetv3":
            self._model = models.segmentation.deeplabv3_mobilenet_v3_large(
                weights=self.pretrained_weights)
        elif self.model_name == "deeplabv3_resnet50":
            self._model = models.segmentation.deeplabv3_resnet50(weights=self.pretrained_weights)
        elif self.model_name == "deeplabv3_resnet101":
            self._model = models.segmentation.deeplabv3_resnet101(weights=self.pretrained_weights)
        else: # Should not happen
            raise NotImplementedError(f"Model {self.model_name} is unknown please us one of "
                                      "the following model {SUPPORTED_MODELS}")

        # Adapt deeplabv3 architecture to segment num_classes number of classes

        # Adapt classifier
        self._model.classifier[-1] = nn.Conv2d(
            in_channels=self._model.classifier[-1].in_channels,
            out_channels=self.num_classes,
            kernel_size=self._model.classifier[-1].kernel_size,
            stride=self._model.classifier[-1].stride,
        )

       # Adapt auxiliary classifier
        self._model.aux_classifier[-1] = nn.Conv2d(
            in_channels=self._model.aux_classifier[-1].in_channels,
            out_channels=self.num_classes,
            kernel_size=self._model.aux_classifier[-1].kernel_size,
            stride=self._model.aux_classifier[-1].stride,
        )

        self._model = self._model.to(device=self.device)
        return self._model

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass of the segmentation model.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width].

        Returns:
            The output tensor of shape [batch_size, num_classes, height, width].
        """
        return self.model(x)["out"]

    def save(self, save_path: str):
        """
        Save the model to a file.

        Args:
            save_path: Path to the file to save the model to.
        """
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path: str):
        """
        Load the model from a file.

        Args:
            load_path: Path to the file to load the model from.
        """
        self.model.load_state_dict(torch.load(load_path))
