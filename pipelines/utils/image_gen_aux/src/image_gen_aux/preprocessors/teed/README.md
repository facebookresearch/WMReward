# TEED: Tiny and Efficient Model for the Edge Detection Generalization

Tiny and Efficient Edge Detector (TEED) is a light convolutional neural network with only 58K parameters, less than 0.2% of the state-of-the-art models. Training on the [BIPED](https://www.kaggle.com/datasets/xavysp/biped) dataset takes less than 30 minutes,
with each epoch requiring less than 5 minutes. Our proposed model is easy to train and it quickly converges within very first few
epochs, while the predicted edge-maps are crisp and of high quality, see image above.
[This paper has been accepted by ICCV 2023-Workshop RCV](https://arxiv.org/abs/2308.06468).

## Usage

```python
from image_gen_aux import TeedPreprocessor
from image_gen_aux.utils import load_image

input_image = load_image(
    "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/teed/20240922043215.png"
)

teed_preprocessor = TeedPreprocessor.from_pretrained("OzzyGT/teed").to("cuda")
image = teed_preprocessor(input_image)[0]
image.save("teed.png")
```

## Additional resources

* [Project page](https://github.com/xavysp/TEED)
* [Paper](https://arxiv.org/abs/2308.06468)
