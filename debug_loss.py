
from transformers import AutoVideoProcessor, AutoModel
from diffusers.utils import export_to_video, load_video
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from compute_vjepa_score import get_score, get_sliding_window_score,  get_sliding_window_score_based
from PIL import Image
import os
import shutil

# init vjepa
processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vith-fpc64-256")
model = AutoModel.from_pretrained(
    "facebook/vjepa2-vith-fpc64-256",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

# raw_path = "/home/yjianhao/project/EvalVideoPhy/data/ball_collision_videos/subgroup_009/valid_00.mp4"
# raw_paths = ["/home/yjianhao/project/EvalVideoPhy/data/ball_collision_videos/subgroup_009/valid_00.mp4", 
#         "/home/yjianhao/project/EvalVideoPhy/data/ball_collision_videos/subgroup_009/temporal_disorder_00.mp4",
#         "/home/yjianhao/project/EvalVideoPhy/data/ball_collision_videos/subgroup_009/invalid_penetration_00.mp4",
#         "/home/yjianhao/project/EvalVideoPhy/data/ball_collision_videos/subgroup_009/invalid_phantom_force_00.mp4",
#         "/home/yjianhao/project/EvalVideoPhy/data/ball_collision_videos/subgroup_009/invalid_teleportation_00.mp4"]

raw_paths = ["/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/valid_00.mp4", 
        "/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/temporal_disorder_00.mp4",
        "/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/invalid_phase_shifting_00.mp4",
        "/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/invalid_sphere_fusion_00.mp4",
        "/home/yjianhao/project/EvalVideoPhy/data/pyramid_videos/subgroup_005/invalid_teleporting_spheres_00.mp4"]

for raw_path in raw_paths:
    video = load_video(raw_path)

    # video is a list of PIL.Image.Image objects
    video_np = np.stack([np.array(frame.resize((256, 256))) for frame in video], axis=0)
    print(video_np.shape)

    score, loss_arr = get_sliding_window_score_based(video, model, processor, kernel_size=16, context_window_size=10, stride=2, return_form='arr')
    print(f"Score: {score}")

    # dir_name = "./debug/ball1"
    dir_name = "./debug/pyramid1"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    save_path = f"{dir_name}/{raw_path.split('/')[-1].split('.')[0]}.mp4"
    shutil.copy2(raw_path, save_path)
    plt.figure(figsize=(6,6))
    plt.plot(loss_arr)

    # Set title and labels
    # plt.title('Loss Over Time')
    plt.xlabel('timestep')
    plt.ylabel('Loss')
    # Save the plot locally
    plt.savefig(f"{dir_name}/loss_{raw_path.split('/')[-1].split('.')[0]}.png")
