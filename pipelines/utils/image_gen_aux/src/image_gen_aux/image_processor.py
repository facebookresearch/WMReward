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

from typing import List, Union

import cv2
import numpy as np
import PIL.Image
import torch
from PIL import Image


class ImageMixin:
    """
    A mixin class for converting images between different formats: PIL, NumPy, and PyTorch tensors.

    This class provides methods to:
    - Convert a PIL image or a NumPy array to a PyTorch tensor.
    - Post-process a PyTorch tensor image and convert it to the specified return type.
    - Convert a PIL image or a list of PIL images to NumPy arrays.
    - Convert a NumPy image to a PyTorch tensor.
    - Convert a PyTorch tensor to a NumPy image.
    - Convert a NumPy image or a batch of images to a PIL image.
    """

    def convert_image_to_tensor(
        self, image: Union[PIL.Image.Image, np.ndarray, List[PIL.Image.Image]], normalize: bool = True
    ) -> torch.Tensor:
        """
        Convert a PIL image or a NumPy array to a PyTorch tensor.

        Args:
            image (Union[PIL.Image.Image, np.ndarray]): The input image, either as a PIL image, a NumPy array or a list of
                PIL images.

        Returns:
            torch.Tensor: The converted image as a PyTorch tensor.
        """
        if isinstance(image, (PIL.Image.Image, list)):
            # We expect that if it is a list, it only should contain pillow images
            if isinstance(image, list):
                for single_image in image:
                    if not isinstance(single_image, PIL.Image.Image):
                        raise ValueError("All images in the list must be Pillow images.")

            image = self.pil_to_numpy(image, normalize)

        return self.numpy_to_pt(image)

    def post_process_image(self, image: torch.Tensor, return_type: str):
        """
        Post-process a PyTorch tensor image and convert it to the specified return type.

        Args:
            image (torch.Tensor): The input image as a PyTorch tensor.
            return_type (str): The desired return type, either "pt" for PyTorch tensor, "np" for NumPy array, or "pil" for PIL image.

        Returns:
            Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]]: The post-processed image in the specified return type.
        """
        if return_type == "pt":
            return image

        image = self.pt_to_numpy(image)
        if return_type == "np":
            return image

        image = self.numpy_to_pil(image)
        return image

    @staticmethod
    def pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image], normalize: bool = True) -> np.ndarray:
        """
        Convert a PIL image or a list of PIL images to NumPy arrays.

        Args:
            images (Union[List[PIL.Image.Image], PIL.Image.Image]): The input image(s) as PIL image(s).

        Returns:
            np.ndarray: The converted image(s) as a NumPy array.
        """
        if not isinstance(images, list):
            images = [images]

        if normalize:
            images = [np.array(image).astype(np.float32) / 255.0 for image in images]
        else:
            images = [np.array(image).astype(np.float32) for image in images]

        images = np.stack(images, axis=0)

        return images

    @staticmethod
    def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
        """
        Convert a NumPy image to a PyTorch tensor.

        Args:
            images (np.ndarray): The input image(s) as a NumPy array.

        Returns:
            torch.Tensor: The converted image(s) as a PyTorch tensor.
        """
        if images.ndim == 3:
            images = images[..., None]
        images = torch.from_numpy(images.transpose(0, 3, 1, 2)).float()

        return images

    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy image.

        Args:
            images (torch.Tensor): The input image(s) as a PyTorch tensor.

        Returns:
            np.ndarray: The converted image(s) as a NumPy array.
        """
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        return images

    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> List[PIL.Image.Image]:
        """
        Convert a NumPy image or a batch of images to a PIL image.

        Args:
            images (np.ndarray): The input image(s) as a NumPy array.

        Returns:
            List[PIL.Image.Image]: The converted image(s) as PIL images.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @staticmethod
    def scale_image(image: torch.Tensor, scale: float, mutiple_factor: int = 8) -> torch.Tensor:
        """
        Scales an image while maintaining aspect ratio and ensuring dimensions are multiples of `multiple_factor`.

        Args:
            image (`torch.Tensor`): The input image tensor of shape (batch, channels, height, width).
            scale (`float`): The scaling factor applied to the image dimensions.
            multiple_factor (`int`, *optional*, defaults to 8): The factor by which the new dimensions should be divisible.

        Returns:
            `torch.Tensor`: The scaled image tensor.
        """

        if scale == 1.0:
            return image, scale

        _batch, _channels, height, width = image.shape

        # Calculate new dimensions while maintaining aspect ratio
        new_height = int(height * scale)
        new_width = int(width * scale)

        # Ensure new dimensions are multiples of mutiple_factor
        new_height = (new_height // mutiple_factor) * mutiple_factor
        new_width = (new_width // mutiple_factor) * mutiple_factor

        # if the final height and widht changed because of the multiple_factor, we need to set the scale too
        scale = new_height / height

        # Resize the image using the calculated dimensions
        resized_image = torch.nn.functional.interpolate(
            image, size=(new_height, new_width), mode="bilinear", align_corners=False
        )

        return resized_image, scale

    @staticmethod
    def resize_numpy_image(image: np.ndarray, scale: float, multiple_factor: int = 8) -> np.ndarray:
        """
        Resizes a NumPy image while maintaining aspect ratio and ensuring dimensions are multiples of `multiple_factor`.

        Args:
            image (`np.ndarray`): The input image array of shape (height, width, channels) or (height, width) for grayscale.
            scale (`float`): The scaling factor applied to the image dimensions.
            multiple_factor (`int`, *optional*, defaults to 8): The factor by which the new dimensions should be divisible.

        Returns:
            `np.ndarray`: The resized image array.
        """

        if scale == 1.0:
            return image, scale

        if len(image.shape) == 3:  # Single image without batch dimension
            image = np.expand_dims(image, axis=0)

        batch_size, height, width, channels = image.shape

        # Calculate new dimensions while maintaining aspect ratio
        new_height = int(height * scale)
        new_width = int(width * scale)

        # Ensure new dimensions are multiples of multiple_factor
        new_height = (new_height // multiple_factor) * multiple_factor
        new_width = (new_width // multiple_factor) * multiple_factor

        # if the final height and widht changed because of the multiple_factor, we need to set the scale too
        scale = new_height / height

        # Resize each image in the batch
        resized_images = []
        for i in range(batch_size):
            resized_image = cv2.resize(image[i], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            resized_images.append(resized_image)

        # Stack resized images back into a single array
        resized_images = np.stack(resized_images, axis=0)

        return resized_images, scale
