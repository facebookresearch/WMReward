import os
import pandas as pd
import numpy as np
import argparse
import glob
import cv2
from PIL import Image, ImageDraw, ImageFont
import imageio
from pathlib import Path

def load_videophy2_scores(base_dir, prompt, model_cat, vanilla_model, guidance_model):
    """Load scores from VideoPhY2 CSV files for vanilla and guidance models."""
    vanilla_dir = f"{base_dir}/results/videophy2/{prompt}/{model_cat}/{vanilla_model}"
    guidance_dir = f"{base_dir}/results/videophy2/{prompt}/{model_cat}/{guidance_model}"
    
    vanilla_scores = {}
    guidance_scores = {}
    
    # Load vanilla scores
    for metric in ['pc', 'sa']:
        csv_path = os.path.join(vanilla_dir, f"{metric}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                caption = row['caption']
                score = row['score']
                if caption not in vanilla_scores:
                    vanilla_scores[caption] = {}
                vanilla_scores[caption][metric] = score
                vanilla_scores[caption]['videopath'] = row['videopath']
    
    # Load guidance scores  
    for metric in ['pc', 'sa']:
        csv_path = os.path.join(guidance_dir, f"{metric}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                caption = row['caption']
                score = row['score']
                if caption not in guidance_scores:
                    guidance_scores[caption] = {}
                guidance_scores[caption][metric] = score
                guidance_scores[caption]['videopath'] = row['videopath']
    
    return vanilla_scores, guidance_scores

def calculate_score_deltas(vanilla_scores, guidance_scores, metric='pc'):
    """Calculate score deltas between guidance and vanilla models."""
    deltas = {}
    
    for caption in vanilla_scores:
        if caption in guidance_scores:
            vanilla_score = vanilla_scores[caption].get(metric)
            guidance_score = guidance_scores[caption].get(metric)
            
            if vanilla_score is not None and guidance_score is not None:
                delta = guidance_score - vanilla_score
                deltas[caption] = {
                    'delta': delta,
                    'vanilla_score': vanilla_score,
                    'guidance_score': guidance_score,
                    'vanilla_path': vanilla_scores[caption]['videopath'],
                    'guidance_path': guidance_scores[caption]['videopath']
                }
    
    return deltas

def create_side_by_side_gif(vanilla_path, guidance_path, output_path, caption, 
                           vanilla_score, guidance_score, delta, max_frames=30):
    """Create a side-by-side GIF from two video files with labels."""
    
    # Read videos
    vanilla_cap = cv2.VideoCapture(vanilla_path)
    guidance_cap = cv2.VideoCapture(guidance_path)
    
    frames = []
    frame_count = 0
    
    while frame_count < max_frames:
        ret1, frame1 = vanilla_cap.read()
        ret2, frame2 = guidance_cap.read()
        
        if not ret1 or not ret2:
            break
            
        # Convert BGR to RGB
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        # Resize frames to same height
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        target_height = min(h1, h2, 256)  # Limit height for reasonable file size
        
        new_w1 = int(w1 * target_height / h1)
        new_w2 = int(w2 * target_height / h2)
        
        frame1 = cv2.resize(frame1, (new_w1, target_height))
        frame2 = cv2.resize(frame2, (new_w2, target_height))
        
        # Create combined frame
        combined_width = new_w1 + new_w2 + 10  # 10px separator
        combined_height = target_height + 100  # Space for text labels
        
        combined_frame = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
        
        # Place video frames
        combined_frame[80:80+target_height, :new_w1] = frame1
        combined_frame[80:80+target_height, new_w1+10:] = frame2
        
        # Convert to PIL for text
        pil_img = Image.fromarray(combined_frame)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Add labels
        draw.text((10, 10), f"Caption: {caption[:80]}{'...' if len(caption) > 80 else ''}", 
                 fill=(0, 0, 0), font=small_font)
        
        # Vanilla label
        draw.text((new_w1//2 - 20, 50), "Vanilla", fill=(0, 0, 0), font=font)
        draw.text((new_w1//2 - 30, 65), f"Score: {vanilla_score:.1f}", fill=(0, 0, 0), font=small_font)
        
        # Guidance label  
        draw.text((new_w1 + 10 + new_w2//2 - 20, 50), "Guidance", fill=(0, 0, 0), font=font)
        draw.text((new_w1 + 10 + new_w2//2 - 30, 65), f"Score: {guidance_score:.1f}", fill=(0, 0, 0), font=small_font)
        
        # Delta label
        delta_color = (0, 150, 0) if delta > 0 else (150, 0, 0) if delta < 0 else (100, 100, 100)
        delta_text = f"Δ: {delta:+.1f}"
        draw.text((combined_width//2 - 20, 25), delta_text, fill=delta_color, font=font)
        
        frames.append(np.array(pil_img))
        frame_count += 1
    
    vanilla_cap.release()
    guidance_cap.release()
    
    if frames:
        # Save as GIF
        imageio.mimsave(output_path, frames, duration=0.2, loop=0)
        return True
    
    return False

def generate_comparison_gifs(deltas, output_dir, top_n=10, bottom_n=10):
    """Generate side-by-side GIFs for top and bottom score deltas."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by delta
    sorted_deltas = sorted(deltas.items(), key=lambda x: x[1]['delta'], reverse=True)
    
    generated_gifs = []
    
    # Top improvements
    print(f"\nGenerating GIFs for top {top_n} improvements:")
    for i, (caption, data) in enumerate(sorted_deltas[:top_n]):
        if not os.path.exists(data['vanilla_path']) or not os.path.exists(data['guidance_path']):
            print(f"Skipping {caption} - video files not found")
            continue
            
        safe_caption = "".join(c for c in caption if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
        output_path = os.path.join(output_dir, f"top_{i+1:02d}_{safe_caption}_delta{data['delta']:+.1f}.gif")
        
        print(f"  {i+1}. {caption[:60]}... (Δ: {data['delta']:+.1f})")
        
        success = create_side_by_side_gif(
            data['vanilla_path'], data['guidance_path'], output_path,
            caption, data['vanilla_score'], data['guidance_score'], data['delta']
        )
        
        if success:
            generated_gifs.append(output_path)
    
    # Bottom (worst) changes
    print(f"\nGenerating GIFs for bottom {bottom_n} changes:")
    for i, (caption, data) in enumerate(sorted_deltas[-bottom_n:]):
        if not os.path.exists(data['vanilla_path']) or not os.path.exists(data['guidance_path']):
            print(f"Skipping {caption} - video files not found")
            continue
            
        safe_caption = "".join(c for c in caption if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
        output_path = os.path.join(output_dir, f"bottom_{i+1:02d}_{safe_caption}_delta{data['delta']:+.1f}.gif")
        
        print(f"  {i+1}. {caption[:60]}... (Δ: {data['delta']:+.1f})")
        
        success = create_side_by_side_gif(
            data['vanilla_path'], data['guidance_path'], output_path,
            caption, data['vanilla_score'], data['guidance_score'], data['delta']
        )
        
        if success:
            generated_gifs.append(output_path)
    
    return generated_gifs

def print_delta_summary(deltas, metric='pc'):
    """Print summary statistics of score deltas."""
    delta_values = [data['delta'] for data in deltas.values()]
    
    print(f"\nScore Delta Summary ({metric.upper()}):")
    print(f"  Total comparisons: {len(delta_values)}")
    print(f"  Mean delta: {np.mean(delta_values):.3f}")
    print(f"  Std delta: {np.std(delta_values):.3f}")
    print(f"  Min delta: {np.min(delta_values):.3f}")
    print(f"  Max delta: {np.max(delta_values):.3f}")
    print(f"  Positive deltas: {sum(1 for d in delta_values if d > 0)} ({sum(1 for d in delta_values if d > 0)/len(delta_values)*100:.1f}%)")
    print(f"  Zero deltas: {sum(1 for d in delta_values if d == 0)} ({sum(1 for d in delta_values if d == 0)/len(delta_values)*100:.1f}%)")
    print(f"  Negative deltas: {sum(1 for d in delta_values if d < 0)} ({sum(1 for d in delta_values if d < 0)/len(delta_values)*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample comparison GIFs ranked by score delta")
    parser.add_argument('--prompt', type=str, default='videophy2',
                      help='Evaluation prompt dataset')
    parser.add_argument('--model_cat', type=str, default='cogvideox2b',
                      help='Model category')
    parser.add_argument('--vanilla_model', type=str, default='vanilla_f49_s50_cfg6.0',
                      help='Vanilla model name')
    parser.add_argument('--guidance_model', type=str, default='guidance_f49_s50_c8_cfg6.0_torch',
                      help='Guidance model name')
    parser.add_argument('--metric', type=str, default='pc', choices=['pc', 'sa'],
                      help='Metric to use for ranking (pc=physical correctness, sa=semantic adherence)')
    parser.add_argument('--top_n', type=int, default=5,
                      help='Number of top improvement samples to generate')
    parser.add_argument('--bottom_n', type=int, default=5,
                      help='Number of bottom/worst samples to generate')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Output directory for GIFs (default: based on models and metric)')
    parser.add_argument('--base_dir', type=str, 
                      default='/home/yjianhao/project/frame-guidance',
                      help='Base project directory')
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = f"/home/yjianhao/project/frame-guidance/results/sample_comparisons_{args.prompt}_{args.metric}"
    
    print(f"Loading VideoPhY2 scores for {args.prompt}...")
    print(f"Vanilla model: {args.vanilla_model}")
    print(f"Guidance model: {args.guidance_model}")
    print(f"Metric: {args.metric.upper()}")
    
    # Load scores
    vanilla_scores, guidance_scores = load_videophy2_scores(
        args.base_dir, args.prompt, args.model_cat, args.vanilla_model, args.guidance_model
    )
    
    print(f"Loaded {len(vanilla_scores)} vanilla scores and {len(guidance_scores)} guidance scores")
    
    # Calculate deltas
    deltas = calculate_score_deltas(vanilla_scores, guidance_scores, args.metric)
    print(f"Found {len(deltas)} matching video pairs")
    
    if not deltas:
        print("No matching video pairs found!")
        exit(1)
    
    # Print summary
    print_delta_summary(deltas, args.metric)
    
    # Generate comparison GIFs
    print(f"\nGenerating comparison GIFs...")
    print(f"Output directory: {args.output_dir}")
    
    generated_gifs = generate_comparison_gifs(deltas, args.output_dir, args.top_n, args.bottom_n)
    
    print(f"\nGenerated {len(generated_gifs)} comparison GIFs:")
    for gif_path in generated_gifs:
        print(f"  {gif_path}")
    
    print(f"\nDone! Check {args.output_dir} for the generated comparison GIFs.")
