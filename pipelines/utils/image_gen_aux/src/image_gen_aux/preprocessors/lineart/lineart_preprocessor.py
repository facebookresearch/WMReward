# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import List, Union

import numpy as np
import PIL.Image
import torch
from safetensors.torch import load_file

from ...image_processor import ImageMixin
from ...utils import SAFETENSORS_FILE_EXTENSION, get_model_path
from ..preprocessor import Preprocessor
from .model import Generator


class LineArtPreprocessor(Preprocessor, ImageMixin):
    """Preprocessor specifically designed for converting images to line art.

    This class inherits from both `Preprocessor` and `ImageMixin`. Please refer to each
    one to get more information.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path: Union[str, os.PathLike],
        filename: str = None,
        subfolder: str = None,
        weights_only: bool = True,
    ) -> Generator:
        model_path = get_model_path(pretrained_model_or_path, filename, subfolder)

        file_extension = os.path.basename(model_path).split(".")[-1]
        if file_extension == SAFETENSORS_FILE_EXTENSION:
            state_dict = load_file(model_path, device="cpu")
        else:
            state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=weights_only)

        model = Generator()
        model.load_state_dict(state_dict)

        model.to(dtype=torch.bfloat16)

        for param in model.parameters():
            param.requires_grad = False
        return cls(model)

    # @torch.inference_mode
    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor, List[PIL.Image.Image]],
        resolution_scale: float = 1.0,
        invert: bool = True,
        batch_size: int = 1,
        return_type: str = "pil",
    ):
        """Preprocesses an image and generates line art using the pre-trained model.

        Args:
            image (`Union[PIL.Image.Image, np.ndarray, torch.Tensor, List[PIL.Image.Image]]`): Input image as PIL Image,
                NumPy array, PyTorch tensor format or a list of PIL Images
            resolution_scale (`float`, optional, defaults to 1.0): Scale factor for image resolution during
                preprocessing and post-processing. Defaults to 1.0 for no scaling.
            invert (`bool`, *optional*, defaults to True): Inverts the generated image if True (white or black background).
            batch_size (`int`, *optional*, defaults to 1): The number of images to process in each batch.
            return_type (`str`, *optional*, defaults to "pil"): The desired return type, either "pt" for PyTorch tensor, "np" for NumPy array,
                or "pil" for PIL image.

        Returns:
            `Union[PIL.Image.Image, np.ndarray, torch.Tensor]`: The generated line art in the
                specified output format.
        """
        if not isinstance(image, torch.Tensor):
            image = self.convert_image_to_tensor(image).to(self.model.device)

        image, resolution_scale = self.scale_image(image, resolution_scale)

        processed_images = []

        for i in range(0, len(image), batch_size):
            batch = image[i : i + batch_size].to(self.model.device)
            lineart = self.model(batch)
            if invert:
                lineart = 255 - lineart
            processed_images.append(lineart.cpu())
        lineart = torch.cat(processed_images, dim=0)

        if resolution_scale != 1.0:
            lineart, _ = self.scale_image(lineart, 1 / resolution_scale)

        image = self.post_process_image(lineart, return_type)

        return image
