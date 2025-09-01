<!---
 Copyright 2024 The HuggingFace Team. All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# Image Generation Auxiliary Tools

Set of auxiliary tools to use with image and video generation libraries. Mainly created to be used with [diffusers](https://github.com/huggingface/diffusers).

## Tools

Each set of tools has its own README where you can find more information on how to use them.

* [Upscalers](https://github.com/asomoza/image_gen_aux/blob/main/src/image_gen_aux/upscalers/README.md)
* [Preprocessors](https://github.com/asomoza/image_gen_aux/blob/main/src/image_gen_aux/preprocessors/README.md)

## Installation

This is a very early version, so only the installation from source is available.

### Install from source

`pip install -e .`

### Install with PyPi

`pip install git+https://github.com/asomoza/image_gen_aux.git`

## Contribution

Soon

## Credits

This library uses inference and models from various authors. If you think you must be credited or if we are missing someone, please let us know.

### Libraries

* [Spandrel](https://github.com/chaiNNer-org/spandrel) for loading super resolution models.

### Super resolution models

* [UltraSharp](https://openmodeldb.info/models/4x-UltraSharp)
* [DAT](https://github.com/zhengchen1999/dat)
* [RealPLKSR|DAT](https://github.com/Phhofm/models)
* [Remacri](https://openmodeldb.info/models/4x-Remacri)

### Preprocessors

* [Depth Anything V2](https://depth-anything-v2.github.io)
* [LineArt](https://github.com/carolineec/informative-drawings)
* [LineArtStandard](https://github.com/Fannovel16/comfyui_controlnet_aux/blob/main/src/custom_controlnet_aux/lineart_standard/__init__.py)
* [TEED](https://github.com/xavysp/TEED)
* [ZoeDepth](https://github.com/isl-org/ZoeDepth)
