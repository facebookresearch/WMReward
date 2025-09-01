# Preprocessors

Preprocessors in this context refer to the application of machine learning models to images or groups of images as a preliminary step for other tasks. A common use case is to feed preprocessed images into a controlnet to guide the generation of a diffusion model. Examples of preprocessors include depth maps, normals, and edge detection.

## Supported preprocessors

This is a list of the currently supported preprocessors.

* [DepthPreprocessor](https://github.com/asomoza/image_gen_aux/blob/main/src/image_gen_aux/preprocessors/depth/README.md)
* [LineArtPreprocessor](https://github.com/asomoza/image_gen_aux/blob/main/src/image_gen_aux/preprocessors/lineart/README.md)
* [LineArtStandardPreprocessor](https://github.com/asomoza/image_gen_aux/blob/main/src/image_gen_aux/preprocessors/lineart_standard/README.md)
* [TeedPreprocessor](https://github.com/asomoza/image_gen_aux/blob/main/src/image_gen_aux/preprocessors/teed/README.md)

## General preprocessor usage

```python
from image_gen_aux import LineArtPreprocessor
from image_gen_aux.utils import load_image

input_image = load_image(
    "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/simple_upscale/hippowaffle.png"
)

lineart_preprocessor = LineArtPreprocessor.from_pretrained("OzzyGT/lineart").to("cuda")
image = lineart_preprocessor(input_image)
image.save("lineart.png")
```

Use the Hugging Face Hub model id, a URL, or a local path to load a model. You can also load pickle checkpoints, but it's not recommended due to the [vulnerabilities](https://docs.python.org/3/library/pickle.html) of pickled files. Loading [safetensor](https://hf.co/docs/safetensors/index) files is a more secure option.

If the checkpoint has custom imports (high risk!), you'll need to set the `weights_only` argument to `False`.

You can also specify a `filename` and a `subfolder` if needed.

```python
lineart_preprocessor = LineArtPreprocessor.from_pretrained("lllyasviel/Annotators", filename="sk_model.pth", weights_only=False).to("cuda")
```

### List of safetensors checkpoints

This is the current list of safetensor checkpoints available on the Hub.

|Preprocessor|Repository|Author|
|---|---|---|
|Depth Anything V2 Small|depth-anything/Depth-Anything-V2-Small-hf|<https://depth-anything-v2.github.io/>|
|Depth Anything V2 Base|depth-anything/Depth-Anything-V2-Base-hf|<https://depth-anything-v2.github.io/>|
|Depth Anything V2 Large|depth-anything/Depth-Anything-V2-Large-hf|<https://depth-anything-v2.github.io/>|
|LineArt|OzzyGT/lineart|[Caroline Chan](https://github.com/carolineec)|
|Teed|OzzyGT/teed|<https://github.com/xavysp/TEED>|
|ZoeDepth NYU|Intel/zoedepth-nyu|<https://github.com/isl-org/ZoeDepth>|
|ZoeDepth KITTI|Intel/zoedepth-kitti|<https://github.com/isl-org/ZoeDepth>|
|ZoeDepth NYU and KITTI|Intel/zoedepth-nyu-kitti|<https://github.com/isl-org/ZoeDepth>|

If you own the model and want us to change the repository to your name/organization please open an issue.
