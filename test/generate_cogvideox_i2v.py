import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A close-up shot captures a robot arm picking up a yellow ball"
# image = load_image(
#     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
# )
image = load_image("/home/yjianhao/project/video_guidance/temp/sample.png")
video = pipe(image, prompt, use_dynamic_cfg=True)
export_to_video(video.frames[0], "output.mp4", fps=8)