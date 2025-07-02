import argparse
from schedulers.unipc_multistep_scheduler import UniPCMultistepScheduler
# from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel
from compute_vjepa_score import get_score, get_sliding_window_score, get_sliding_window_score_based
from diffusers.utils import export_to_video
from simple_benchmarking import get_simple_experiment_name, log_experiment_simple, get_simple_output_folder
import torch
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import math

def get_prompts(prompt_file):
    """Read prompts and negative prompts from a text file."""
    with open(f"../VBench/prompts/prompts_per_dimension/{prompt_file}.txt", 'r') as file:
        prompts = [line.strip() for line in file if line.strip()]
    
    # Define a negative prompt
    negative_prompt = "overexposed, static, blurred details, worst quality, low quality, JPEG compression residue, deformation"
    
    return prompts, negative_prompt

def init_pipeline(args):
    """Initialize the WanPipeline with the specified model ID."""
    if "wan" in args.model_id:
        if args.sampling_method == 'guidance':
            from pipelines.wan_pipeline_guidance_v2 import WanPipeline
        elif args.sampling_method in ['rejection', 'vanilla']:
            from pipelines.wan_pipeline import WanPipeline
        else:
            raise ValueError(f"Unsupported sampling method: {args.sampling_method}")
        pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.bfloat16)
        scheduler = UniPCMultistepScheduler.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="scheduler")
        pipe.scheduler = scheduler
        pipe.enable_model_cpu_offload()
        
        # Set guidance parameters for the guidance pipeline
        if args.sampling_method == 'guidance':
            pipe.guidance_start = args.guidance_start
            pipe.guidance_end = args.guidance_end
            pipe.guidance_rho_scale = args.guidance_rho_scale
            pipe.vjepa_kernel_size = args.kernel_size
            pipe.vjepa_context_length = args.context_length
            pipe.vjepa_stride = args.stride
            pipe.vjepa_mode = args.vjepa_mode
            
    return pipe

def init_vjepa2():
    # jepa_model_id = "facebook/vjepa2-vitl-fpc64-256"  # or "facebook/vjepa2-vith-fpc64-256"
    jepa_model_id = "facebook/vjepa2-vith-fpc64-256"  # Use the ViT-L model for better performance
    processor = AutoVideoProcessor.from_pretrained(jepa_model_id)
    model = AutoModel.from_pretrained(
        jepa_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    return model, processor


def rejection_sample(pipe, args, prompt, negative_prompt, model, processor, generator=None):
    best_score = float('inf')
    best_frames = None

    for i in range(args.num_rejection_attempts):
        frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=args.num_frames, height=args.height, width=args.width, generator=generator, num_inference_steps=args.num_inference_steps, guidance_scale=args.cfg_scale).frames[0]
        
        score = get_sliding_window_score_based(frames, model, processor, 
                                              kernel_size=args.kernel_size, 
                                              context_window_size=args.context_length, 
                                              stride=args.stride, 
                                              return_form='loss', 
                                              mode=args.vjepa_mode, 
                                              require_grad=False)
        # print(f"Attempt {i+1}/{args.num_rejection_attempts}, Score: {score}")
        if score < best_score:  # Update the best score and frames
            best_score = score
            best_frames = frames
    return best_frames  # Or return whatever is appropriate for your use case


def generate_videos(pipe, args, prompts, negative_prompt, experiment_name, fps=16, vjepa=None, vjepa_processor=None):
    """Generate videos for each prompt and save them to the output folder."""
    
    # Get simple output folder
    output_folder = get_simple_output_folder(args, experiment_name)
    
    # Log experiment start
    log_experiment_simple(args, experiment_name, 'started')

    generator = torch.Generator(device="cuda").manual_seed(0)

    # Generate videos for each prompt
    for i, prompt in enumerate(prompts):
        # Clean prompt for filename (remove special characters)
        # safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
        # safe_prompt = safe_prompt.replace(' ', '_')[:50]  # Limit length
        safe_prompt = prompt
        
        video_path = os.path.join(output_folder, f"{safe_prompt}.mp4")
        if os.path.exists(video_path):
            print(f"Video already exists, skipping: {video_path}")
            continue

        print(f"[{experiment_name}] Generating video {i+1}/{len(prompts)} ({args.sampling_method}): {prompt}")

        # Generate frames
        if args.sampling_method == 'vanilla':
            frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=args.num_frames, height=args.height, width=args.width, generator=generator, num_inference_steps=args.num_inference_steps, guidance_scale=args.cfg_scale).frames[0]
        elif args.sampling_method == 'guidance':
            # Guidance uses configurable V-JEPA parameters set on the pipeline
            frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=args.num_frames, height=args.height, width=args.width, generator=generator, num_inference_steps=args.num_inference_steps, guidance_scale=args.cfg_scale).frames[0]
        elif args.sampling_method == 'rejection':
            frames = rejection_sample(pipe=pipe, args=args, prompt=prompt, negative_prompt=negative_prompt, model=vjepa, processor=vjepa_processor, generator=generator)

        # Export to video
        export_to_video(frames, video_path, fps=fps)
        print(f"[{experiment_name}] Generated: {video_path}")
    
    # Log experiment completion
    log_experiment_simple(args, experiment_name, 'completed')
    print(f"[{experiment_name}] Experiment completed! Results saved to: {output_folder}")

def chunk_prompts(prompts, num_chunks, chunk_idx):
    """Divide the prompts into chunks and return the chunk corresponding to the given index."""
    chunk_size = math.ceil(len(prompts) / num_chunks)
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, len(prompts))
    return prompts[start_idx:end_idx]

def main():
    parser = argparse.ArgumentParser(description="Generate videos from text prompts using a specified model.")
    parser.add_argument('--prompt_file', type=str, help='Path to the text file containing prompts.')
    parser.add_argument('--model_id', type=str, help='Model ID to use for video generation.')
    parser.add_argument('--output_folder', type=str, help='Folder to save the generated videos.')
    parser.add_argument('--num_gpus', type=int, help='Total number of GPUs available.')
    parser.add_argument('--gpu_idx', type=int, help='Index of the GPU to use for this process.')
    parser.add_argument('--sampling_method',type=str, default='vanilla', help='Model ID to use for video generation.')
    parser.add_argument('--kernel_size', type=int, default=8, help='Kernel size for V-JEPA sliding window.')
    parser.add_argument('--context_length', type=int, default=6, help='Context length for V-JEPA.')
    parser.add_argument('--stride', type=int, default=2, help='Stride for V-JEPA sliding window.')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps to generate for each video.')
    parser.add_argument('--num_frames', type=int, default=17, help='Number of frames to generate for each video.')
    parser.add_argument('--height', type=int, default=480, help='Height of the generated videos.')
    parser.add_argument('--width', type=int, default=832, help='Width of the generated videos.')
    
    # Ablation parameters for rejection sampling
    parser.add_argument('--num_rejection_attempts', type=int, default=10, help='Number of attempts for rejection sampling.')
    parser.add_argument('--vjepa_mode', type=str, default='mean', choices=['mean', 'max'], help='V-JEPA aggregation mode (mean or max).')
    
    # Ablation parameters for guidance sampling
    parser.add_argument('--cfg_scale', type=float, default=5.0, help='Classifier-free guidance scale.')
    parser.add_argument('--guidance_start', type=int, default=0, help='Timestep to start applying guidance.')
    parser.add_argument('--guidance_end', type=int, default=1001, help='Timestep to end applying guidance.')
    parser.add_argument('--guidance_rho_scale', type=float, default=3.0, help='Gradient scaling factor (rho_scale) for guidance.')

    args = parser.parse_args()

    # Validate V-JEPA parameters
    if args.sampling_method in ['rejection', 'guidance']:
        if args.context_length >= args.kernel_size:
            raise ValueError(f"context_length ({args.context_length}) must be less than kernel_size ({args.kernel_size})")
        if args.kernel_size > args.num_frames:
            raise ValueError(f"kernel_size ({args.kernel_size}) cannot be larger than num_frames ({args.num_frames})")
        if args.stride <= 0:
            raise ValueError(f"stride ({args.stride}) must be positive")

    # Get prompts and negative prompt
    prompts, negative_prompt = get_prompts(args.prompt_file)

    # Generate simple experiment name
    experiment_name = get_simple_experiment_name(args)
    
    # Chunk the prompts for distributed processing
    chunked_prompts = chunk_prompts(prompts, args.num_gpus, args.gpu_idx)

    # Print configuration for this run
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*60}")
    print(f"Sampling method: {args.sampling_method}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Frames per video: {args.num_frames}")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Prompts assigned to this GPU: {len(chunked_prompts)}")
    
    if args.sampling_method in ['rejection', 'guidance']:
        print(f"V-JEPA parameters:")
        print(f"  - Kernel size: {args.kernel_size}")
        print(f"  - Context length: {args.context_length}")
        print(f"  - Stride: {args.stride}")
        print(f"  - Mode: {args.vjepa_mode}")
        
    if args.sampling_method == 'rejection':
        print(f"Rejection sampling:")
        print(f"  - Attempts: {args.num_rejection_attempts}")
        
    if args.sampling_method == 'guidance':
        print(f"Guidance parameters:")
        print(f"  - Timestep range: {args.guidance_start}-{args.guidance_end}")
        print(f"  - Rho scale: {args.guidance_rho_scale}")
    print(f"{'='*60}\n")

    # Initialize pipeline
    pipe = init_pipeline(args)

    # Initialize V-JEPA for methods that need it
    if args.sampling_method in ['rejection']:
        model, processor = init_vjepa2()
    else:
        # Vanilla doesn't need V-JEPA, guidance has it built into pipeline
        model, processor = None, None

    # Generate videos
    generate_videos(pipe, args, chunked_prompts, negative_prompt, experiment_name, vjepa=model, vjepa_processor=processor)

if __name__ == "__main__":
    main()