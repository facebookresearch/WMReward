"""
Simple text-to-video generation using OpenAI's Sora API.

Usage:
    python generate_sora.py --prompt "A cat walking on a beach at sunset"
    python generate_sora.py --prompt_file prompts/my_prompts.txt --output_folder results/
    python generate_sora.py --batch_json prompts/physics_iq.json
"""

import os
import json
import argparse
from datetime import datetime
from pipelines.sora_client import SoraClient


def main():
    parser = argparse.ArgumentParser(description="Generate videos using Sora API")
    
    # Input options
    parser.add_argument('--prompt', type=str, default=None,
                       help='Single text prompt for video generation')
    parser.add_argument('--prompt_file', type=str, default=None,
                       help='Text file with one prompt per line')
    parser.add_argument('--batch_json', type=str, default=None,
                       help='JSON file with list of {prompt, output_video} entries')
    
    # Output options
    parser.add_argument('--output_folder', type=str, default='generated_videos/sora',
                       help='Folder to save generated videos')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Explicit output path for single prompt mode')
    
    # Sora API parameters
    parser.add_argument('--model', type=str, default='sora-2',
                       help='Model to use (default: sora-2)')
    parser.add_argument('--seconds', type=int, default=4, choices=[4, 8, 12],
                       help='Video duration in seconds (4, 8, or 12)')
    parser.add_argument('--size', type=str, default='1280x720',
                       help='Video size as WxH (e.g., 1280x720, 1920x1080)')
    
    # Azure API settings (defaults are pre-configured)
    parser.add_argument('--host', type=str, default=None,
                       help='Azure API host (or set SORA_HOST env var)')
    parser.add_argument('--api_key', type=str, default=None,
                       help='API key (or set SORA_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Validate input
    input_count = sum([
        args.prompt is not None,
        args.prompt_file is not None,
        args.batch_json is not None,
    ])
    if input_count == 0:
        parser.error("Provide --prompt, --prompt_file, or --batch_json")
    if input_count > 1:
        parser.error("Use only one of --prompt, --prompt_file, or --batch_json")
    
    # Initialize client
    client = SoraClient(host=args.host, api_key=args.api_key)
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Collect prompts and output paths
    tasks = []
    
    if args.prompt:
        # Single prompt mode
        if args.output_path:
            output_path = args.output_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = args.prompt[:50].replace(' ', '_').replace('/', '-')
            output_path = os.path.join(args.output_folder, f"{timestamp}_{safe_prompt}.mp4")
        tasks.append({'prompt': args.prompt, 'output_path': output_path})
    
    elif args.prompt_file:
        # Prompt file mode
        with open(args.prompt_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        for i, prompt in enumerate(prompts):
            safe_prompt = prompt[:50].replace(' ', '_').replace('/', '-')
            output_path = os.path.join(args.output_folder, f"{i:04d}_{safe_prompt}.mp4")
            tasks.append({'prompt': prompt, 'output_path': output_path})
    
    elif args.batch_json:
        # Batch JSON mode
        with open(args.batch_json, 'r') as f:
            entries = json.load(f)
        
        for entry in entries:
            prompt = entry.get('prompt')
            output_video = entry.get('output_video')
            if not prompt or not output_video:
                print(f"[skip] Missing prompt or output_video: {entry}")
                continue
            
            output_path = os.path.join(args.output_folder, os.path.basename(output_video))
            tasks.append({'prompt': prompt, 'output_path': output_path})
    
    # Generate videos
    print(f"\n{'='*60}")
    print(f"SORA VIDEO GENERATION")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Duration: {args.seconds}s")
    print(f"Size: {args.size}")
    print(f"Tasks: {len(tasks)}")
    print(f"{'='*60}\n")
    
    for i, task in enumerate(tasks):
        prompt = task['prompt']
        output_path = task['output_path']
        
        # Skip if already exists
        if os.path.exists(output_path):
            print(f"[{i+1}/{len(tasks)}] Already exists, skipping: {output_path}")
            continue
        
        print(f"[{i+1}/{len(tasks)}] Generating: {prompt[:80]}...")
        
        try:
            client.generate_and_download(
                prompt=prompt,
                output_path=output_path,
                model=args.model,
                seconds=args.seconds,
                size=args.size,
                verbose=True,
            )
        except Exception as e:
            print(f"[error] Failed to generate video: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Generation complete! Videos saved to: {args.output_folder}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
