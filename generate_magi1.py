# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.s


"""
MAGI-1 single-prompt I2V generation script with VJEPA guidance support.
This is the main quick-start script for video generation using the MAGI-1 submodule.
"""

import os
import sys
import argparse

# Add MAGI-1 submodule to path
MAGI1_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MAGI-1")

if not os.path.exists(MAGI1_PATH) or len(os.listdir(MAGI1_PATH)) == 0:
    raise RuntimeError(
        "MAGI-1 submodule not found. Please initialize it with:\n"
        "  git submodule update --init --recursive"
    )

# Set SPECIAL_TOKEN_PATH for MAGI-1 if not already set
os.environ.setdefault("SPECIAL_TOKEN_PATH", os.path.join(MAGI1_PATH, "example/assets/special_tokens.npz"))

if MAGI1_PATH not in sys.path:
    sys.path.insert(0, MAGI1_PATH)


def main():
    parser = argparse.ArgumentParser(description="MAGI-1 I2V generation with VJEPA guidance")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing the video")
    parser.add_argument("--config_file", type=str, required=True, help="Path to MAGI-1 configuration JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output video")
    parser.add_argument("--guidance_scale", type=float, default=0.001, help="VJEPA guidance scale.")
    parser.add_argument("--guidance_frequency", type=int, default=5, help="VJEPA guidance frequency.")
    parser.add_argument("--vjepa_type", type=str, default="vitg", help="VJEPA model variant.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["t2v", "i2v", "v2v"],
        default="i2v",
        help="Generation mode: t2v (text-to-video), i2v (image-to-video), v2v (video-to-video)",
    )
    parser.add_argument("--init_image", type=str, default=None, help="Path to initial image for I2V mode")
    parser.add_argument("--init_video", type=str, default=None, help="Path to prefix video for V2V mode")
    args = parser.parse_args()

    # Import MAGI-1 pipeline (after sys.path modification)
    from inference.pipeline.pipeline_w_guidance import MagiPipeline

    # Initialize MAGI-1 pipeline with guidance support
    pipeline = MagiPipeline(args.config_file)
    pipeline.guidance_scale = args.guidance_scale
    pipeline.guidance_frequency = args.guidance_frequency
    pipeline.vjepa_type = args.vjepa_type

    # Run the appropriate mode
    if args.mode == "t2v":
        pipeline.run_text_to_video(prompt=args.prompt, output_path=args.output_path)
    elif args.mode == "i2v":
        if not args.init_image:
            print("Error: --init_image is required for i2v mode.")
            sys.exit(1)
        pipeline.run_image_to_video(prompt=args.prompt, image_path=args.init_image, output_path=args.output_path)
    elif args.mode == "v2v":
        if not args.init_video:
            print("Error: --init_video is required for v2v mode.")
            sys.exit(1)
        pipeline.run_video_to_video(prompt=args.prompt, prefix_video_path=args.init_video, output_path=args.output_path)

    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
