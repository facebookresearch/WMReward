from typing import List, Union

import cv2
import numpy as np
import PIL.Image

from ...image_processor import ImageMixin


class LineArtStandardPreprocessor(ImageMixin):
    """Preprocessor specifically designed for converting images to line art standard.

    This class inherits from both `Preprocessor` and `ImageMixin`. Please refer to each
    one to get more information.
    """

    def __call__(
        self,
        image: Union[PIL.Image.Image, np.ndarray, List[PIL.Image.Image]],
        resolution_scale: float = 1.0,
        invert: bool = False,
        gaussian_sigma=6.0,
        intensity_threshold=8,
        return_type: str = "pil",
    ):
        """Preprocesses an image and generates line art (standard) using the opencv.

        Args:
            image (`Union[PIL.Image.Image, np.ndarray, torch.Tensor, List[PIL.Image.Image]]`): Input image as PIL Image,
                NumPy array, PyTorch tensor format or a list of PIL Images
            resolution_scale (`float`, optional, defaults to 1.0): Scale factor for image resolution during
                preprocessing and post-processing. Defaults to 1.0 for no scaling.
            invert (`bool`, *optional*, defaults to True): Inverts the generated image if True (white or black background).
            gaussian_sigma (float, optional): Sigma value for Gaussian blur. Defaults to 6.0.
            intensity_threshold (int, optional): Threshold for intensity clipping. Defaults to 8.
            return_type (`str`, *optional*, defaults to "pil"): The desired return type, either "pt" for PyTorch tensor, "np" for NumPy array,
                or "pil" for PIL image.

        Returns:
            `Union[PIL.Image.Image, np.ndarray, torch.Tensor]`: The generated line art (standard) in the
                specified output format.
        """
        if not isinstance(image, np.ndarray):
            image = np.array(image).astype(np.float32)

        # check if image has batch, if not, add it
        if len(image.shape) == 3:
            image = image[None, ...]

        image, resolution_scale = self.resize_numpy_image(image, resolution_scale)

        batch_size, height, width, _channels = image.shape
        processed_images = np.empty((batch_size, height, width), dtype=np.uint8)

        # since we're using just cv2, we can't do batch processing
        for i in range(batch_size):
            gaussian = cv2.GaussianBlur(image[i], (0, 0), gaussian_sigma)
            intensity = np.min(gaussian - image[i], axis=2).clip(0, 255)
            intensity /= max(16, np.median(intensity[intensity > intensity_threshold]))
            intensity *= 127
            edges = intensity.clip(0, 255).astype(np.uint8)

            processed_images[i] = edges

        if invert:
            processed_images = 255 - processed_images

        processed_images = processed_images[..., None]

        if resolution_scale != 1.0:
            processed_images, _ = self.resize_numpy_image(processed_images, 1 / resolution_scale)
            processed_images = processed_images[..., None]  # cv2 resize removes the channel dimension if grayscale

        if return_type == "np":
            return processed_images

        processed_images = np.transpose(processed_images, (0, 3, 1, 2))
        processed_images = [PIL.Image.fromarray(image.squeeze(), mode="L") for image in processed_images]

        return processed_images
