from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel
from diffusers.utils import export_to_video, load_video
import torch
from torchcodec.decoders import VideoDecoder
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from compute_vjepa_score import get_score, get_sliding_window_score, get_sliding_window_score_max # Import the process_video function
from PIL import Image


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
#         "/home/yjianhao/project/EvalVideoPhy/data/ball_collision_videos/subgroup_009/invalid_penetration_00.mp4"]
raw_paths = ["/home/yjianhao/project/video_guidance/base_robotarm.mp4"]
for raw_path in raw_paths:
    video = load_video(raw_path)

    # video is a list of PIL.Image.Image objects
    video_np = np.stack([np.array(frame.resize((256, 256))) for frame in video], axis=0)
    print(video_np.shape)



    score, loss_arr, video_processed = get_sliding_window_score_max(video, model, processor, kernel_size=4, context_window_size=2, stride=2, loss_form="mean", return_loss_arr=True)
    print(f"Score: {score}")
    
    save_path = f"./debug/{raw_path.split('/')[-1].split('.')[0]}.mp4"
    B,F,C,H,W = video_processed.shape
    
    video_export = video_processed.squeeze(0).reshape(F,H,W,C).cpu().numpy()
    export_to_video(video_export, save_path, fps=16)
    
    # # Save first frame as PNG
    # first_frame = video_export[0]  # shape: (H, W, C) or possibly (C, H, W)
    # print("Original first_frame shape:", first_frame.shape)

    # # If shape is (C, H, W), transpose to (H, W, C)
    # if first_frame.shape[0] in [1, 3] and first_frame.shape[-1] != 3:
    #     first_frame = np.transpose(first_frame, (1, 2, 0))
    #     print("Transposed first_frame shape:", first_frame.shape)

    # # Convert to uint8 if necessary
    # if first_frame.dtype != np.uint8:
    #     # If values are in [0,1], scale to [0,255]
    #     if first_frame.max() <= 1.0:
    #         first_frame = (first_frame * 255).astype(np.uint8)
    #     else:
    #         first_frame = first_frame.astype(np.uint8)

    # first_frame_pil = Image.fromarray(first_frame)
    # png_save_path = f"./debug/{raw_path.split('/')[-1].split('.')[0]}.png"
    # first_frame_pil.save(png_save_path)

    plt.figure(figsize=(10,6))
    plt.plot(loss_arr)
    # Set title and labels
    plt.title('Loss Over Time')
    plt.xlabel('timestep')
    plt.ylabel('Loss')
    # Save the plot locally
    plt.savefig(f"./debug/loss_{raw_path.split('/')[-1].split('.')[0]}.png")
