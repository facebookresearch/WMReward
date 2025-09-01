# DEPTH

Monocular depth estimation is a computer vision task that involves predicting the depth information of a scene from a single image. In other words, it is the process of estimating the distance of objects in a scene from a single camera viewpoint.

Monocular depth estimation has various applications, including 3D reconstruction, augmented reality, autonomous driving, and robotics. It is a challenging task as it requires the model to understand the complex relationships between objects in the scene and the corresponding depth information, which can be affected by factors such as lighting conditions, occlusion, and texture.

## Usage

```python
from image_gen_aux import DepthPreprocessor
from image_gen_aux.utils import load_image

input_image = load_image("https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/depth/coffee_ship.png")

depth_preprocessor = DepthPreprocessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").to("cuda")
image = depth_preprocessor(input_image)[0]
image.save("depth.png")
```

## Models

The Depth Preprocessor supports any depth estimation model that the `transformers` library supports that doesn't have a fixed image size restriction, but we mainly recommend and ensure the correct functionality for these models:

|Model|License|Project Page|
|---|---|---|
|Depth Anything V2|CC-BY-NC-4.0|<https://depth-anything-v2.github.io/>|
|ZoeDepth|MIT|<https://github.com/isl-org/ZoeDepth>|

Each model has different variations:

### Depth Anything V2

|Variation|Repo ID|
|---|---|
|Small|depth-anything/Depth-Anything-V2-Small-hf|
|Base|depth-anything/Depth-Anything-V2-Base-hf|
|Large|depth-anything/Depth-Anything-V2-Large-hf|

### ZoeDepth

|Variation|Repo ID|
|---|---|
|NYU|Intel/zoedepth-nyu|
|KITTI|Intel/zoedepth-kitti|
|NYU and KITTI|Intel/zoedepth-nyu-kitti|
