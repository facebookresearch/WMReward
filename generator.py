import argparse
from pipelines.wan_pipeline import WanPipeline
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel
from compute_vjepa_score import get_score
from diffusers.utils import export_to_video
import torch
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import math

def get_prompts(prompt_file):
    """Read prompts and negative prompts from a text file."""
    with open(f"../VBench/prompts/prompts_per_dimension/{prompt_file}.txt", 'r') as file:
        prompts = [line.strip() for line in file if line.strip()]
    
    # Define a negative prompt
    negative_prompt = "worst quality, distortion, overexposed, static, blurred details"
    
    return prompts, negative_prompt

def init_pipeline(model_id):
    """Initialize the WanPipeline with the specified model ID."""
    if model_id == "wan":
        pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
    return pipe

def init_vjepa2():
    processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
    model = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    return model, processor


def rejection_sample(pipe, args, prompt, negative_prompt, num_frames, model, processor, num_attempts=10):
    best_score = float('inf')
    best_frames = None

    for i in range(num_attempts):
        frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames).frames[0]
        # export_to_video(frames, f"./temp/wan-t2v{i}.mp4", fps=16)  # Export to video
        # score = get_score(f"./temp/wan-t2v{i}.mp4", model, processor)  # Calculate the score
        score = get_score(frames, model, processor, n_timesteps=args.context_length)  # Calculate the score
        # print(f"{i}, Score: {score}")
        if score < best_score:  # Update the best score and frames
            best_score = score
            best_frames = frames
    return best_frames  # Or return whatever is appropriate for your use case


def generate_videos(pipe, args, prompts, negative_prompt, output_folder, model_id, prompt_name, num_frames=33, fps=16, vjepa=None, vjepa_processor=None):
    """Generate videos for each prompt and save them to the output folder."""
    # Extract the base names for model and prompt file
    model_name = f"{model_id}_rej_cw{args.context_length}" if args.sampling_method == 'rejection' else model_id

    # Create the output directory structure
    model_output_folder = os.path.join(output_folder, model_name, prompt_name)
    os.makedirs(model_output_folder, exist_ok=True)

    # Generate videos for each prompt
    for i, prompt in enumerate(prompts):
        video_path = os.path.join(model_output_folder, f"{prompt}.mp4")
        if os.path.exists(video_path):
            print(f"Video already exists, skipping: {video_path}")
            continue

        # Generate frames
        if args.sampling_method == 'vanilla':
            frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames).frames[0]
        elif args.sampling_method == 'rejection':
            frames = rejection_sample(pipe=pipe, args=args, prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames, model=vjepa, processor=vjepa_processor)

        # Export to video
        
        export_to_video(frames, video_path, fps=fps)
        print(f"Generated video: {video_path}")

def chunk_prompts(prompts, num_chunks, chunk_idx):
    """Divide the prompts into chunks and return the chunk corresponding to the given index."""
    chunk_size = math.ceil(len(prompts) / num_chunks)
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, len(prompts))
    return prompts[start_idx:end_idx]

def main():
    parser = argparse.ArgumentParser(description="Generate videos from text prompts using a specified model.")
    parser.add_argument('prompt_file', type=str, help='Path to the text file containing prompts.')
    parser.add_argument('model_id', type=str, help='Model ID to use for video generation.')
    parser.add_argument('output_folder', type=str, help='Folder to save the generated videos.')
    parser.add_argument('num_gpus', type=int, help='Total number of GPUs available.')
    parser.add_argument('gpu_idx', type=int, help='Index of the GPU to use for this process.')
    parser.add_argument('sampling_method',type=str, default='vanilla', help='Model ID to use for video generation.')
    parser.add_argument('context_length', type=int, default=4, help='context_length')

    args = parser.parse_args()

    # Get prompts and negative prompt
    prompts, negative_prompt = get_prompts(args.prompt_file)

    # Chunk the prompts for distributed processing
    chunked_prompts = chunk_prompts(prompts, args.num_gpus, args.gpu_idx)

    # Initialize pipeline
    pipe = init_pipeline(args.model_id)

    if args.sampling_method == 'rejection':
        model, processor = init_vjepa2()
    else:
        model, processor = None, None

    # Generate videos
    generate_videos(pipe, args, chunked_prompts, negative_prompt, args.output_folder, args.model_id, args.prompt_file, vjepa=model, vjepa_processor=processor)

if __name__ == "__main__":
    main()