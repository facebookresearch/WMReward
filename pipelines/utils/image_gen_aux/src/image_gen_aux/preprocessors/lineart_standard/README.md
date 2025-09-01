# Line Art Standard

Line Art Standard was copied from the [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) repository.
This preprocessor applies a Gaussian blur to the image to reduce noise and detail, then calculates the intensity difference between the blurred and original images to highlight edges.

## Usage

```python
from image_gen_aux import LineArtStandardPreprocessor
from image_gen_aux.utils import load_image

input_image = load_image(
    "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/simple_upscale/hippowaffle.png"
)

lineart_standard_preprocessor = LineArtStandardPreprocessor()
image = lineart_standard_preprocessor(input_image)[0]
image.save("lineart_standard.png")
```

## Additional resources

* [Original implementation](https://github.com/Fannovel16/comfyui_controlnet_aux/blob/main/src/custom_controlnet_aux/lineart_standard/__init__.py)
