import os
from typing import List, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation

from ...image_processor import ImageMixin
from ..preprocessor import Preprocessor


class DepthPreprocessor(Preprocessor, ImageMixin):
    """Preprocessor specifically designed for monocular depth estimation.

    This class inherits from both `Preprocessor` and `ImageMixin`. Please refer to each
    one to get more information.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path: Union[str, os.PathLike]):
        model = AutoModelForDepthEstimation.from_pretrained(pretrained_model_or_path)
        # requires_grad = False
        for param in model.parameters():
            param.requires_grad = False

        return cls(model)

    # @torch.inference_mode
    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor, List[PIL.Image.Image]],
        resolution_scale: float = 1.0,
        invert: bool = False,
        batch_size: int = 1,
        return_type: str = "pil",
    ):
        if not isinstance(image, torch.Tensor):
            image = self.convert_image_to_tensor(image).to(self.model.device)

        image, resolution_scale = self.scale_image(image, resolution_scale)

        processed_images = []

        for i in range(0, len(image), batch_size):
            batch = image[i : i + batch_size].to(self.model.device)

            predicted_depth = self.model(batch).predicted_depth

            # depth models returns only batch, height and width, so we add the channel
            predicted_depth = predicted_depth.unsqueeze(1)

            # models like depth anything can return a different size image
            if batch.shape[2] != predicted_depth.shape[2] or batch.shape[3] != predicted_depth.shape[3]:
                predicted_depth = F.interpolate(
                    predicted_depth, size=(batch.shape[2], batch.shape[3]), mode="bilinear", align_corners=False
                )

            if invert:
                predicted_depth = 255 - predicted_depth
            processed_images.append(predicted_depth.cpu())
        predicted_depth = torch.cat(processed_images, dim=0)

        if resolution_scale != 1.0:
            predicted_depth, _ = self.scale_image(predicted_depth, 1 / resolution_scale)

        predicted_depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
        predicted_depth = predicted_depth.clamp(0, 1)

        image = self.post_process_image(predicted_depth, return_type)

        return image
