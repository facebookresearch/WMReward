from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import PIL.Image
import torch


class Preprocessor(ABC):
    """
    This abstract base class defines the interface for image preprocessors.

    Subclasses should implement the abstract methods `from_pretrained` and
    `__call__` to provide specific loading and preprocessing logic for their
    respective models.

    Args:
        model (`nn.Module`): The torch model to use.
    """

    def __init__(self, model):
        self.model = model

    def to(self, device):
        """
        Moves the underlying model to the specified device
        (e.g., CPU or GPU).

        Args:
            device (`torch.device`): The target device.

        Returns:
            `Preprocessor`: The preprocessor object itself (for method chaining).
        """
        self.model = self.model.to(device)
        return self

    @abstractmethod
    def from_pretrained(self):
        """
        This abstract method defines how the preprocessor loads pre-trained
        weights or configurations specific to the model it supports. Subclasses
        must implement this method to handle model-specific loading logic.

        This method might download pre-trained weights from a repository or
        load them from a local file depending on the model's requirements.
        """
        pass

    @abstractmethod
    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        resolution_scale: float = 1.0,
        invert: bool = True,
        return_type: str = "pil",
    ):
        """
        Preprocesses an image for use with the underlying model.

        Args:
            image (`Union[PIL.Image.Image, np.ndarray, torch.Tensor]`): Input image as PIL Image,
                NumPy array, or PyTorch tensor format.
            resolution_scale (`float`, optional, defaults to 1.0): Scale factor for image resolution during
            resolution_scale (`float`, *optional*, defaults to 1.0): Scale factor for image resolution during
                preprocessing and post-processing. Defaults to 1.0 for no scaling.
            invert (`bool`, *optional*, defaults to True): Inverts the generated image if True.
            return_type (`str`, *optional*, defaults to "pil"): The desired return type, either "pt" for PyTorch tensor,
                "np" for NumPy array, or "pil" for PIL image.

        Returns:
            `Union[PIL.Image.Image, torch.Tensor]`: The preprocessed image in the
                specified format.
        """
        pass
