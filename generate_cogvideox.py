import os, cv2, argparse, torch
from PIL import Image
from datetime import datetime
from diffusers.utils import export_to_video
from pipelines.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline

def build_seq(pattern: str, steps: int, is_float: bool):
    tokens = [p.strip() for p in pattern.split(",") if p.strip()]
    seq = []
    for tok in tokens:
        if "x" in tok:
            v, c = tok.split("x")
            seq.extend(([float(v) if is_float else int(v)]) * int(c))
        else:
            v = float(tok) if is_float else int(tok)
            seq = [v] * steps
            break
    if len(seq) != steps:
        raise ValueError(f"bad pattern len {len(seq)} vs {steps}")
    return seq

def parse_range_pair(text: str):
    a, b = (text.split(",", 1) if "," in text else text.split("-", 1))
    return int(a.strip()), int(b.strip())

def load_first_frame(image_path: str | None, video_path: str | None) -> Image.Image:
    if image_path:
        return Image.open(image_path).convert("RGB")
    if not video_path:
        raise ValueError("Provide either --init_image or --init_video")
    cap = cv2.VideoCapture(video_path)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read from video: {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

def main():
    ap = argparse.ArgumentParser("I2V with VJEPA slice_pred guidance")
    ap.add_argument("--prompt", type=str, default=(
        "Two pillows on a table and two grabber tools hanging above them from which a brown tennis ball and an orange block are suspended. The grabber tools let go of the ball and block. Static shot with no camera movement."
    ))
    ap.add_argument("--init_image", type=str, default="./example/0001_switch-frames_anyFPS_perspective-left_trimmed-ball-and-block-fall.jpg")
    ap.add_argument("--init_video", type=str, default=None)
    ap.add_argument("--model_id", type=str, default="THUDM/CogVideoX-5b-I2V")
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guidance_scale", type=float, default=6.0)
    ap.add_argument("--guidance_step_pattern", type=str, default="0x3,1x47")
    ap.add_argument("--guidance_lr_pattern", type=str, default="0.003x50")
    ap.add_argument("--guidance_frequency", type=int, default=1)
    ap.add_argument("--travel_time", type=str, default="0,0")
    # VJEPA slice_pred
    ap.add_argument("--vjepa_variant", type=str, default="vit_giant", choices=["vit_large","vit_huge","vit_giant","vit_giant_384"])
    ap.add_argument("--vjepa_img_size", type=int, default=256)
    ap.add_argument("--vjepa_masking_mode", type=str, default="causal", choices=["causal","random"])
    ap.add_argument("--vjepa_context_frames", type=int, default=8)
    ap.add_argument("--slice_window_size", type=int, default=16)
    ap.add_argument("--slice_stride", type=int, default=8)
    ap.add_argument("--vae_decode_scale", type=float, default=0.8)
    ap.add_argument("--loss_mode", type=str, default="max", choices=["mean","max"])
    # IO
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"{run_name}.mp4")

    init_frame = load_first_frame(args.init_image, args.init_video)
    # Minimal list with the image as the I2V condition
    init_video_list = [init_frame]

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16).to("cuda")
    pipe.vae.enable_tiling(); pipe.vae.enable_slicing(); pipe.vae.enable_gradient_checkpointing()

    steps = int(args.steps)
    guidance_step = build_seq(args.guidance_step_pattern, steps, is_float=False)
    guidance_lr = build_seq(args.guidance_lr_pattern, steps, is_float=True)
    travel_time = parse_range_pair(args.travel_time)

    negative_prompt = "overexposed, static, blurred details, worst quality, low quality, JPEG compression residue, deformation, motion artifacts"

    result = pipe(
        image=init_video_list,
        prompt=args.prompt,
        negative_prompt=negative_prompt,
        num_frames=args.num_frames,
        num_inference_steps=steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
        use_dynamic_cfg=True,
        guidance_step=guidance_step,
        guidance_lr=guidance_lr,
        guidance_frequency=args.guidance_frequency,
        additional_inputs={
            "vjepa_variant": args.vjepa_variant,
            "vjepa_img_size": args.vjepa_img_size,
            "vjepa_masking_mode": args.vjepa_masking_mode,
            "vjepa_context_frames": args.vjepa_context_frames,
            "slice_window_size": args.slice_window_size,
            "slice_stride": args.slice_stride,
            "vae_decode_scale": args.vae_decode_scale,
            "loss_mode": args.loss_mode,
            "save_intermediate_keyframes": False,
            "intermediate_fps": args.fps,
            "intermediate_save_dir": os.path.join(args.out_dir, f"{run_name}_intermediate"),
        },
        travel_time=travel_time,
    ).frames[0]

    export_to_video(result, out_path, fps=args.fps)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
