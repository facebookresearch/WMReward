# UPSCALERS

## Upscale with model

Class to upscale images with a safetensor checkpoint. We use [spandrel](https://github.com/chaiNNer-org/spandrel) for loading, and you can see the list of supported models [here](https://github.com/chaiNNer-org/spandrel?tab=readme-ov-file#model-architecture-support).

Most of the super-resolution models are provided as `pickle` checkpoints, which are considered unsafe. We promote the use of safetensor checkpoints, and for convenient use, we recommend using the Hugging Face Hub. You can still use a locally downloaded model.

### Space

You can test the current super resolution models you can use with this [Hugging Face Space](https://huggingface.co/spaces/OzzyGT/basic_upscaler).

### How to use

```python
from image_gen_aux import UpscaleWithModel
from image_gen_aux.utils import load_image

original = load_image(
    "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/simple_upscale/hippowaffle_small.png"
)

upscaler = UpscaleWithModel.from_pretrained("Kim2091/UltraSharp").to("cuda")
image = upscaler(original)
image.save("upscaled.png")
```

### Tiling

Tiling can be enabled to use less resources.

```python
image = upscaler(original, tiling=True, tile_width=768, tile_height=768, overlap=8)
```

### Scale

The scale is automatically obtained from the model but can be overridden with the `scale` argument:

```python
image = upscaler(original, scale=2)
```

### List of safetensors checkpoints

This is the current list of safetensor checkpoints you can use from the hub.

|Model|Scale|Repository|Owner|
|---|---|---|---|
|UltraSharp|4X|Kim2091/UltraSharp|[Kim2091](https://huggingface.co/Kim2091)|
|DAT|2X|OzzyGT/DAT_X2|[zhengchen1999](https://github.com/zhengchen1999)|
|DAT|3X|OzzyGT/DAT_X3|[zhengchen1999](https://github.com/zhengchen1999)|
|DAT|4X|OzzyGT/DAT_X4|[zhengchen1999](https://github.com/zhengchen1999)|
|RealPLKSR|4X|Phips/4xNomosWebPhoto_RealPLKSR|[Philip Hofmann](https://huggingface.co/Phips)|
|DAT-2|4X|Phips/4xRealWebPhoto_v4_dat2|[Philip Hofmann](https://huggingface.co/Phips)|
|BSRGAN|4X|OzzyGT/4xRemacri|[FoolhardyVEVO](https://openmodeldb.info/users/foolhardy)|

If you own the model and want us to change the repository to your name/organization please open an issue.
