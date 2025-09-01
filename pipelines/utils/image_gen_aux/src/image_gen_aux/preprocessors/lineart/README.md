# Line Art

Line Art is based on the original repository, [Informative Drawings: Learning to generate line drawings that convey geometry and semantics](https://github.com/carolineec/informative-drawings).

From the project page [summary](https://carolineec.github.io/informative_drawings/):

*Our method approaches line drawing generation as an unsupervised image translation problem which uses various losses to assess the information communicated in a line drawing. Our key idea is to view the problem as an encoding through a line drawing and to maximize the quality of this encoding through explicit geometry, semantic, and appearance decoding objectives. This evaluation is performed by deep learning methods which decode depth, semantics, and appearance from line drawings. The aim is for the extracted depth and semantic information to match the scene geometry and semantics of the input photographs.*

## Usage

```python
from image_gen_aux import LineArtPreprocessor
from image_gen_aux.utils import load_image

input_image = load_image(
    "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/simple_upscale/hippowaffle.png"
)

lineart_preprocessor = LineArtPreprocessor.from_pretrained("OzzyGT/lineart").to("cuda")
image = lineart_preprocessor(input_image)[0]
image.save("lineart.png")
```

## Additional resources

* [Project page](https://carolineec.github.io/informative_drawings/)
* [Paper](https://arxiv.org/abs/2203.12691)
