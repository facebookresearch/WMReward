#!/usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
import subprocess
from PIL import Image, ImageDraw, ImageFont
import tempfile
import glob
import re

def load_dreamgen_pa_scores(base_dir, prompt='gr1_object', model_cat='cogvideox5b_i2v'):
    """Load DreamGen PA scores for experiment models by parsing video paths."""
    experiment_scores = {}
    dreamgen_dir = f"{base_dir}/results/dreamgen/{prompt}/{model_cat}"
    
    if not os.path.exists(dreamgen_dir):
        print(f"DreamGen directory not found: {dreamgen_dir}")
        return experiment_scores
    
    # Process each method directory (guidance, vanilla, rejection, etc.)
    for method_dir in os.listdir(dreamgen_dir):
        method_path = os.path.join(dreamgen_dir, method_dir)
        if not os.path.isdir(method_path):
            continue
            
        # Look for PA CSV files
        pa_csv_file = os.path.join(method_path, f"{prompt}_{method_dir}_pa.csv")
        if os.path.exists(pa_csv_file):
            try:
                # Read PA CSV using the same method as dreamgen_res2tab.py
                print(f"Processing PA scores from: {pa_csv_file}")
                with open(pa_csv_file, "r") as fh:
                    next(fh, None)  # skip header
                    
                    # Parse video paths and predictions
                    video_scores = {}
                    for line in fh:
                        if line.strip():
                            # Get the prediction (last column) using rsplit like dreamgen_res2tab.py
                            prediction = int(line.rsplit(",", 1)[-1].strip())
                            
                            # Get video path (first column) - split on comma and take first part
                            video_path = line.split(',')[0]
                            
                            # Extract experiment name from video path
                            # Path format: .../generated_videos/gr1_object/cogvideox5b_i2v/EXPERIMENT_NAME/video.mp4
                            path_parts = video_path.split('/')
                            if len(path_parts) >= 2:
                                experiment_name = path_parts[-2]  # Parent directory name
                                
                                if experiment_name not in video_scores:
                                    video_scores[experiment_name] = []
                                video_scores[experiment_name].append(prediction)
                    
                    # Calculate PA percentage for each experiment
                    for exp_name, scores in video_scores.items():
                        pa_percentage = (sum(scores) / len(scores) * 100) if scores else 0.0
                        experiment_scores[exp_name] = pa_percentage
                        print(f"  {exp_name}: {pa_percentage:.1f}% PA ({sum(scores)}/{len(scores)} videos)")
                        
            except Exception as e:
                print(f"Error reading PA scores from {pa_csv_file}: {e}")
    
    return experiment_scores

def load_video_level_pa_scores(base_dir, prompt='gr1_object', model_cat='cogvideox5b_i2v'):
    """Load individual video PA scores from DreamGen results with prompt matching."""
    video_pa_scores = {}
    video_prompt_map = {}  # Map video path to prompt text
    dreamgen_dir = f"{base_dir}/results/dreamgen/{prompt}/{model_cat}"
    
    if not os.path.exists(dreamgen_dir):
        return video_pa_scores, video_prompt_map
    
    # Process each method directory (guidance, vanilla, rejection, etc.)
    for method_dir in os.listdir(dreamgen_dir):
        method_path = os.path.join(dreamgen_dir, method_dir)
        if not os.path.isdir(method_path):
            continue
            
        # Look for PA CSV files
        pa_csv_file = os.path.join(method_path, f"{prompt}_{method_dir}_pa.csv")
        if os.path.exists(pa_csv_file):
            try:
                with open(pa_csv_file, "r") as fh:
                    next(fh, None)  # skip header
                    
                    # Parse video paths, prompts and predictions
                    for line in fh:
                        if line.strip():
                            # Get the prediction (last column) using rsplit
                            prediction = int(line.rsplit(",", 1)[-1].strip())
                            
                            # Split the line to get video path and prompt
                            parts = line.split(',')
                            video_path = parts[0]
                            # Prompt is everything between first and last comma
                            prompt_text = ','.join(parts[1:-1]).strip()
                            
                            # Normalize prompt by removing leading number index
                            # E.g., "000 Use the left hand..." -> "Use the left hand..."
                            normalized_prompt = re.sub(r'^\d+\s+', '', prompt_text)
                            
                            # Convert PA score to percentage (0 or 100)
                            video_pa_scores[video_path] = prediction * 100.0
                            video_prompt_map[video_path] = normalized_prompt
                            
                            # Debug: Show first few prompt extractions
                            if len(video_prompt_map) <= 3:
                                print(f"    Debug original: '{prompt_text}'")
                                print(f"    Debug normalized: '{normalized_prompt}'")
                                print(f"    for {os.path.basename(video_path)}")
                                
            except Exception as e:
                print(f"Error reading video PA scores from {pa_csv_file}: {e}")
    
    return video_pa_scores, video_prompt_map

def load_experiment_data(base_dir, dreamgen_prompt='gr1_object', model_cats=['cogvideox5b_i2v']):
    """Load experiment configurations and results from directory structure across multiple model categories."""
    experiments = {}
    
    # Load DreamGen PA scores from all model categories
    print(f"Loading DreamGen PA scores for prompt: {dreamgen_prompt}")
    print(f"Processing model categories: {model_cats}")
    
    for model_cat in model_cats:
        print(f"\nProcessing model category: {model_cat}")
        experiment_pa_scores = load_dreamgen_pa_scores(base_dir, dreamgen_prompt, model_cat)    
        video_pa_scores, video_prompt_map = load_video_level_pa_scores(base_dir, dreamgen_prompt, model_cat)
        print(f"Found PA scores for {len(experiment_pa_scores)} experiments")
        print(f"Found individual video PA scores for {len(video_pa_scores)} videos")
        
        # Only process experiments that have PA scores
        video_base = Path(base_dir) / "generated_videos" / dreamgen_prompt / model_cat
        
        if not video_base.exists():
            print(f"Video base directory not found: {video_base}")
            continue
        
        # Process only experiments that have PA scores
        for exp_name, exp_pa_score in experiment_pa_scores.items():
            # Create unique key that includes model category
            unique_exp_name = f"{model_cat}_{exp_name}"
            exp_dir = video_base / exp_name
            
            if not exp_dir.exists() or not exp_dir.is_dir():
                print(f"Experiment directory not found: {exp_dir}")
                continue
                
            config_file = exp_dir / "experiment_config.json"
            
            # Load config if it exists, otherwise create minimal config from name
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
            else:
                # Create minimal config using experiment name
                config = {
                    'parameters': {},
                    '_source_dir': exp_name
                }
            
            # Add model category to config for labeling
            config['model_category'] = model_cat
            
            # Create results dataframe with individual video PA scores
            video_files = []
            for video_file in exp_dir.glob("*.mp4"):
                video_path = str(video_file)
                # Try to get individual video PA score, fall back to experiment average
                video_score = video_pa_scores.get(video_path, exp_pa_score)
                # Get prompt text for this video
                prompt_text = video_prompt_map.get(video_path, "")
                video_files.append({
                    'videopath': video_path,
                    'score': video_score,
                    'prompt': prompt_text
                })
            
            if video_files:
                results_df = pd.DataFrame(video_files)
                avg_score = results_df['score'].mean()
                
                experiments[unique_exp_name] = {
                    'config': config,
                    'results': results_df,
                    'avg_score': avg_score,
                    'video_dir': exp_dir,
                    'pa_score': exp_pa_score,
                    'model_category': model_cat
                }
                print(f"  {unique_exp_name}: {exp_pa_score:.1f}% PA (avg: {avg_score:.1f}%, {len(video_files)} videos)")
            else:
                print(f"No video files found for experiment: {unique_exp_name}")
    
    print(f"\nLoaded {len(experiments)} total experiments across all model categories")
    return experiments

def _short_key(keyword: str) -> str:
    mapping = {
        'sampling_method': 'meth',
        'cfg_scale': 'cfg',
        'loss_mode': 'loss',
        'vjepa_variant': 'vj',
        'vjepa_context_frames': 'ctx',
        'slice_stride': 'st',
        'slice_window_size': 'w',
        'guidance_frequency': 'freq',
        'guidance_lr_pattern': 'lrp',
        'guidance_step_pattern': 'stp',
        'config_version': 'v',
        'num_frames': 'frames',
        'num_inference_steps': 'steps',
    }
    return mapping.get(keyword, keyword)


def _short_value(keyword: str, value):
    if value is None:
        return None
    # Normalize common values
    if keyword == 'vjepa_variant':
        return 'vitg' if value == 'vit_giant' else ('vith' if value == 'vit_huge' else str(value))
    if keyword == 'sampling_method':
        return 'van' if value == 'vanilla' else ('guid' if value == 'guidance' else str(value))
    if keyword == 'loss_mode':
        return str(value).lower()
    # Compress very long patterns
    if keyword == 'guidance_lr_pattern':
        # Use only first three characters of the pattern string per user request
        s = str(value)
        return s[:3]
    if keyword == 'config_version':
        s = str(value)
        return s.split('_')[0] if s.startswith('v') else s
    # Trim floats like 6.0 -> 6
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _format_label_from_params(params: dict, param_keywords) -> str:
    parts = []
    for key in param_keywords:
        val = _short_value(key, params.get(key))
        if val in (None, ''):
            continue
        sk = _short_key(key)
        # Use compact form key+val for numeric-like, otherwise key:val
        if val.replace('.', '', 1).isdigit():
            parts.append(f"{sk}{val}")
        else:
            parts.append(f"{sk}:{val}")
    return ' '.join(parts) if parts else 'unlabeled'


def extract_key_params(config, param_keywords=None):
    """Extract key parameters and build a concise label from param_keywords order."""
    params = config.get('parameters', {})
    source_dir = config.get('_source_dir', 'unknown')
    model_category = config.get('model_category', 'unknown')

    if param_keywords is None:
        param_keywords = ['sampling_method', 'cfg_scale', 'guidance_lr_pattern']

    # Collect key info
    key_info = {'source_dir': source_dir, 'model_category': model_category}
    for keyword in param_keywords:
        key_info[keyword] = params.get(keyword)

    # Build compact, readable label with model category prefix
    if params:  # If we have parameters, use them
        param_label = _format_label_from_params(params, param_keywords)
        # Add model category as prefix
        label = f"{model_category}: {param_label}"
    else:  # If no parameters (minimal config), use source_dir as label
        label = f"{model_category}: {source_dir[:30]}{'...' if len(source_dir) > 30 else ''}"
    
    return key_info, label



def find_high_delta_videos(experiments, param_keywords, top_k=10):
    """Find videos with highest PA score variance across experiments."""
    # Collect all video files by basename (same prompt across experiments)
    video_scores = defaultdict(dict)
    
    # Debug: Show video counts per experiment
    print("Debug - Videos per experiment:")
    for exp_name, exp_data in experiments.items():
        print(f"  {exp_name}: {len(exp_data['results'])} videos")
    
    for exp_name, exp_data in experiments.items():
        key_params, label = extract_key_params(exp_data['config'], param_keywords)
        
        for _, row in exp_data['results'].iterrows():
            video_path = row['videopath']
            score = row['score']
            prompt_text = row.get('prompt', '')
            
            # Use prompt text as the key for matching across experiments
            # Fall back to basename if no prompt available
            grouping_key = prompt_text if prompt_text else os.path.basename(video_path)
            
            # Use a unique key per experiment to avoid collisions when labels are identical
            video_scores[grouping_key][exp_name] = {
                'score': score,
                'path': video_path,
                'exp_name': exp_name,
                'display_label': label,
                'prompt': prompt_text
            }
    
    # Debug: Show how many experiments each video appears in
    print(f"Debug - Found {len(video_scores)} unique prompt groupings")
    multi_exp_videos = [v for v, exps in video_scores.items() if len(exps) >= 2]
    print(f"Debug - {len(multi_exp_videos)} prompts appear in 2+ experiments")
    if len(multi_exp_videos) < 20:
        print("Debug - Sample prompt matches:")
        for prompt, exps in list(video_scores.items())[:10]:
            if len(exps) >= 2:
                print(f"  {prompt[:50]}...: appears in {list(exps.keys())}")
    
    # Debug: Show experiments that have 3-way matches
    three_way_matches = [v for v, exps in video_scores.items() if len(exps) >= 3]
    print(f"Debug - {len(three_way_matches)} prompts appear in 3+ experiments")
    if three_way_matches:
        for prompt, exps in list(video_scores.items())[:3]:
            if len(exps) >= 3:
                print(f"  THREE-WAY: {prompt[:50]}... in {list(exps.keys())}")
    
    # Calculate PA score variance for each prompt/video group
    video_deltas = []
    for grouping_key, exp_scores in video_scores.items():
        if len(exp_scores) >= 2:  # Need at least 2 experiments to compare
            scores = [data['score'] for data in exp_scores.values()]
            score_variance = np.var(scores)
            score_range = max(scores) - min(scores)
            
            # Use prompt as basename, or first video filename if no prompt
            first_video_data = list(exp_scores.values())[0]
            if first_video_data['prompt']:
                display_name = first_video_data['prompt']
            else:
                display_name = os.path.basename(first_video_data['path'])
            
            video_deltas.append({
                'video_basename': display_name,
                'score_variance': score_variance,
                'score_range': score_range,
                'scores': exp_scores,
                'mean_score': np.mean(scores)
            })
    
    # Sort by score range (highest delta first)
    video_deltas.sort(key=lambda x: x['score_range'], reverse=True)
    
    return video_deltas[:top_k]

def create_video_grid_gif(video_data, output_path, fps=10, duration_sec=5):
    """Create side-by-side GIF from multiple videos."""
    try:
        # Get all video paths and labels
        video_paths = []
        labels = []
        
        # Sort experiments by display label only
        temp_list = []
        for exp_key, data in video_data['scores'].items():
            if not os.path.exists(data['path']):
                continue
            display_label = data.get('display_label', exp_key)
            temp_list.append((display_label, data))
        sorted_experiments = sorted(temp_list, key=lambda x: x[0])

        # Build lists
        for display_label, data in sorted_experiments:
            video_paths.append(data['path'])
            labels.append(f"{display_label}\nPA Score: {data['score']:.1f}")
        
        if len(video_paths) < 2:
            print(f"Not enough videos found for {video_data['video_basename']}")
            return False
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_dirs = []
            
            # Extract frames from each video
            for i, video_path in enumerate(video_paths):
                frame_dir = os.path.join(temp_dir, f"video_{i}")
                os.makedirs(frame_dir)
                
                # Extract frames using ffmpeg with smaller size for smaller GIFs
                cmd = [
                    'ffmpeg', '-i', video_path, '-vf', 
                    f'fps={fps},scale=320:240', '-t', str(duration_sec),
                    '-y', os.path.join(frame_dir, 'frame_%04d.png')
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error extracting frames from {video_path}: {result.stderr}")
                    continue
                
                frame_dirs.append(frame_dir)
            
            if not frame_dirs:
                return False
            
            # Get frame count (use minimum across all videos)
            frame_counts = []
            for frame_dir in frame_dirs:
                frames = [f for f in os.listdir(frame_dir) if f.endswith('.png')]
                frame_counts.append(len(frames))
            
            if not frame_counts:
                return False
            
            max_frames = min(frame_counts)
            if max_frames == 0:
                return False
            
            # Create grid frames
            grid_frames = []
            
            for frame_idx in range(1, max_frames + 1):
                frame_images = []
                
                # Load corresponding frame from each video
                for frame_dir in frame_dirs:
                    frame_path = os.path.join(frame_dir, f'frame_{frame_idx:04d}.png')
                    if os.path.exists(frame_path):
                        img = Image.open(frame_path)
                        frame_images.append(img)
                
                if len(frame_images) != len(frame_dirs):
                    continue
                
                # Use 3-column layout to group experiments by CFG
                cols = 3  # Fixed 3 columns
                rows = (len(frame_images) + cols - 1) // cols
                
                img_width = frame_images[0].width
                img_height = frame_images[0].height
                
                # Create initial grid image to get fonts and measure text
                grid_img = Image.new('RGB', (100, 100), 'white')  # Temporary image for font loading
                draw = ImageDraw.Draw(grid_img)
                
                # Load fonts
                try:
                    title_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 16)
                    label_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 14)
                    score_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 12)
                except:
                    title_font = label_font = score_font = ImageFont.load_default()
                
                # Calculate required label height based on actual content
                max_label_width = 0
                max_label_height = 0
                
                # Helper to wrap text to a given width
                def _wrap_text(text, font, max_width):
                    if not text:
                        return []
                    words = text.split(' ')
                    lines_wrapped = []
                    current = ''
                    for word in words:
                        tentative = word if current == '' else current + ' ' + word
                        bbox = draw.textbbox((0, 0), tentative, font=font)
                        w = bbox[2] - bbox[0]
                        if w <= max_width:
                            current = tentative
                        else:
                            if current:
                                lines_wrapped.append(current)
                                current = word
                            else:
                                # single very long token; hard-wrap by characters
                                token = word
                                while token:
                                    # binary search best cut
                                    lo, hi, best = 1, len(token), 1
                                    while lo <= hi:
                                        mid = (lo + hi) // 2
                                        part = token[:mid]
                                        bw = draw.textbbox((0, 0), part, font=font)
                                        if (bw[2] - bw[0]) <= max_width:
                                            best = mid
                                            lo = mid + 1
                                        else:
                                            hi = mid - 1
                                    lines_wrapped.append(token[:best])
                                    token = token[best:]
                                    current = ''
                    if current:
                        lines_wrapped.append(current)
                    return lines_wrapped

                # Measure maximum label size across all labels with wrapping
                line_height_label = (draw.textbbox((0, 0), "Ag", font=label_font)[3] - draw.textbbox((0, 0), "Ag", font=label_font)[1])
                line_height_score = (draw.textbbox((0, 0), "Ag", font=score_font)[3] - draw.textbbox((0, 0), "Ag", font=score_font)[1])
                for label in labels:
                    lines = label.split('\n')
                    exp_name = lines[0]
                    score_text = lines[1] if len(lines) > 1 else ""
                    wrapped = _wrap_text(exp_name, label_font, img_width - 10)
                    wrapped_widths = [draw.textbbox((0, 0), l, font=label_font)[2] - draw.textbbox((0, 0), l, font=label_font)[0] for l in wrapped] or [0]
                    exp_width = max(wrapped_widths)
                    exp_height = len(wrapped) * line_height_label + (len(wrapped) - 1) * 3
                    if score_text:
                        score_bbox = draw.textbbox((0, 0), score_text, font=score_font)
                        score_width = score_bbox[2] - score_bbox[0]
                        score_height = line_height_score
                        total_width = max(exp_width, score_width)
                        total_height = exp_height + 5 + score_height
                    else:
                        total_width = exp_width
                        total_height = exp_height
                    max_label_width = max(max_label_width, total_width)
                    max_label_height = max(max_label_height, total_height)
                
                # Add padding to label dimensions
                label_width = max_label_width + 20  # 10px padding on each side
                label_height = max_label_height + 20  # 10px padding on top/bottom
                
                # Ensure minimum label dimensions
                label_width = max(label_width, img_width + 10)
                label_height = max(label_height, 40)
                
                # Calculate grid dimensions with dynamic label sizing and safe top margin
                grid_width = cols * img_width + (cols + 1) * 15 + 30
                spacing = 20
                top_margin = label_height + 15  # ensure top labels fit within canvas
                bottom_extra = 80
                grid_height = top_margin + rows * (img_height + label_height + spacing) + bottom_extra
                    
                # Create the actual grid image with proper dimensions
                grid_img = Image.new('RGB', (grid_width, grid_height), 'white')
                draw = ImageDraw.Draw(grid_img)
                
                # Videos start at the top (no title above)
                # Title and info will be added below the videos
                
                # Create clean 4-column grid layout
                for i, (img, label) in enumerate(zip(frame_images, labels)):
                    row = i // cols
                    col = i % cols
                    
                    # Calculate position with dynamic label sizing and safe top margin
                    x = col * (img_width + 15) + 20
                    y = top_margin + row * (img_height + label_height + spacing)
                    
                    # Use consistent neutral colors
                    label_bg_color = '#f5f5f5'  # Light gray
                    text_color = '#333333'      # Dark gray
                    border_color = '#666666'    # Medium gray
                    
                    # Split label into experiment name and score
                    lines = label.split('\n')
                    exp_name = lines[0]
                    score_text = lines[1] if len(lines) > 1 else ""
                    
                    # Draw label background with dynamic sizing
                    label_y_start = y - label_height - 5
                    label_y_end = y - 5
                    draw.rectangle([(x-5, label_y_start), (x + img_width + 5, label_y_end)], 
                                 fill=label_bg_color, outline=border_color, width=2)
                    
                    # Wrap and draw experiment name across multiple lines
                    wrapped = _wrap_text(exp_name, label_font, img_width - 10)
                    current_y = label_y_start + 5
                    for line in wrapped:
                        line_bbox = draw.textbbox((0, 0), line, font=label_font)
                        line_w = line_bbox[2] - line_bbox[0]
                        line_x = x + (img_width - line_w) // 2
                        draw.text((line_x, current_y), line, fill=text_color, font=label_font)
                        current_y += line_height_label + 3
                    
                    # Draw score
                    if score_text:
                        score_bbox = draw.textbbox((0, 0), score_text, font=score_font)
                        score_x = x + (img_width - (score_bbox[2] - score_bbox[0])) // 2
                        # Position score below experiment name with some spacing
                        score_y = current_y
                        draw.text((score_x, score_y), score_text, fill=text_color, font=score_font)
                    
                    # Draw clean border around video frame
                    draw.rectangle([(x-2, y-2), (x + img_width + 2, y + img_height + 2)], 
                                 outline=border_color, width=2)
                    
                    # Paste video frame
                    grid_img.paste(img, (x, y))
                
                # Add title and info below the videos
                video_title = f"Comparison: {video_data['video_basename'][:70]}..."
                score_info = f"PA Score Range: {video_data['score_range']:.1f} | Mean: {video_data['mean_score']:.1f}"
                
                # Calculate position below the last row of videos
                last_video_bottom = top_margin + (rows - 1) * (img_height + label_height + spacing) + img_height
                title_y = last_video_bottom + 15
                score_y = title_y + 20
                
                # Center the title
                title_bbox = draw.textbbox((0, 0), video_title, font=title_font)
                title_x = (grid_width - (title_bbox[2] - title_bbox[0])) // 2
                draw.text((title_x, title_y), video_title, fill='black', font=title_font)
                
                # Center the score info
                score_bbox = draw.textbbox((0, 0), score_info, font=score_font)
                score_x = (grid_width - (score_bbox[2] - score_bbox[0])) // 2
                draw.text((score_x, score_y), score_info, fill='gray', font=score_font)
                
                grid_frames.append(grid_img)
            
            if grid_frames:
                # Save as GIF
                grid_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=grid_frames[1:],
                    duration=int(1000/fps),  # milliseconds per frame
                    loop=0
                )
                return True
    
    except Exception as e:
        print(f"Error creating GIF for {video_data['video_basename']}: {e}")
        return False
    
    return False

def generate_delta_gifs(experiments, output_dir, top_k=10, fps=10, param_keywords=None):
    """Generate GIFs for videos with highest PA score deltas."""
    print(f"\nFinding top {top_k} videos with highest PA score variance...")
    
    high_delta_videos = find_high_delta_videos(experiments, param_keywords or ['sampling_method','cfg_scale','guidance_lr_pattern'], top_k)
    
    if not high_delta_videos:
        print("No videos with sufficient variance found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating {len(high_delta_videos)} comparison GIFs...")
    print("-" * 80)
    
    successful_gifs = 0
    
    for i, video_data in enumerate(high_delta_videos, 1):
        basename = video_data['video_basename']
        score_range = video_data['score_range']
        mean_score = video_data['mean_score']
        
        # Simple, robust filename independent of label structure
        safe_name = "".join(c for c in basename if c.isalnum() or c in "._- ").strip()[:50]
        filename = f"{i:02d}_range{score_range:.1f}_{safe_name}.gif"
        output_path = os.path.join(output_dir, filename)
        
        print(f"{i:2d}. {basename[:60]}...")
        print(f"     PA Score range: {score_range:.2f} (mean: {mean_score:.2f})")
        
        # Show PA scores for each experiment sorted by label
        sorted_experiments = sorted(video_data['scores'].items(), key=lambda x: x[0])
        for label, data in sorted_experiments:
            score = data['score']
            print(f"     • {label}: {score:.1f}")
        
        success = create_video_grid_gif(video_data, output_path, fps)
        if success:
            print(f"     ✓ GIF saved: {output_path}")
            successful_gifs += 1
        else:
            print(f"     ✗ Failed to create GIF")
        print()
    
    print(f"Successfully created {successful_gifs}/{len(high_delta_videos)} GIFs in {output_dir}")
    
    # Create summary markdown file
    create_visualization_summary(high_delta_videos, output_dir, successful_gifs)

def create_visualization_summary(high_delta_videos, output_dir, successful_count):
    """Create a markdown summary of the generated visualizations."""
    summary_path = os.path.join(output_dir, "README.md")
    
    with open(summary_path, 'w') as f:
        f.write("# Grid Search Comparison Visualizations\n\n")
        f.write(f"Generated {successful_count} side-by-side GIF comparisons showing videos with highest PA score variance across experiments.\n\n")
        f.write("## Videos Ranked by PA Score Variance\n\n")
        
        for i, video_data in enumerate(high_delta_videos, 1):
            basename = video_data['video_basename']
            score_range = video_data['score_range']
            mean_score = video_data['mean_score']
            
            # Create safe filename
            safe_name = "".join(c for c in basename if c.isalnum() or c in "._- ").strip()
            if len(safe_name) > 100:
                safe_name = safe_name[:97] + "..."
            
            score_range_str = f"range{score_range:.1f}"
            mean_score_str = f"avg{mean_score:.1f}"
            gif_filename = f"delta_{i:02d}_{score_range_str}_{mean_score_str}_{safe_name}.gif"
            
            f.write(f"### {i}. {basename}\n\n")
            f.write(f"- **PA Score Range**: {score_range:.2f}\n")
            f.write(f"- **Mean PA Score**: {mean_score:.2f}\n")
            f.write(f"- **GIF**: `{gif_filename}`\n\n")
            
            f.write("**PA Scores by Experiment:**\n")
            for label, data in sorted(video_data['scores'].items(), key=lambda x: x[1]['score'], reverse=True):
                f.write(f"- {label}: {data['score']:.1f}\n")
            f.write("\n")
            
            if os.path.exists(os.path.join(output_dir, gif_filename)):
                f.write(f"![{basename}]({gif_filename})\n\n")
            else:
                f.write("*GIF generation failed*\n\n")
            f.write("---\n\n")
    
    print(f"Summary saved to: {summary_path}")

def print_summary_table(experiments, param_keywords):
    """Print a summary table of all experiments."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    # Create summary data
    summary_data = []
    for exp_name, exp_data in experiments.items():
        key_params, label = extract_key_params(exp_data['config'], param_keywords)
        
        # Extract parameter info for display based on parameter keywords
        param_parts = []
        for keyword in param_keywords:
            if keyword in ['sampling_method', 'cfg_scale']:
                continue  # Skip these as they're shown separately
            
            value = key_params.get(keyword)
            if value is not None and value != '':
                if keyword == 'vjepa_variant':
                    if value == 'vit_giant':
                        param_parts.append('vitg')
                    elif value == 'vit_huge':
                        param_parts.append('vith')
                    else:
                        param_parts.append(value)
                elif keyword == 'loss_mode':
                    param_parts.append(value)
                elif keyword == 'config_version':
                    # Extract just the version number for display
                    try:
                        version_match = value.split('_')[0]
                        if version_match.startswith('v'):
                            param_parts.append(version_match)
                    except:
                        param_parts.append(value[:10] + '...' if len(value) > 10 else value)
                else:
                    param_parts.append(f"{keyword}_{value}")
        
        param_display = "_".join(param_parts) if param_parts else "N/A"
        
        summary_data.append({
            'Experiment': label,
            'Method': key_params.get('sampling_method', 'unknown'),
            'CFG': str(key_params.get('cfg_scale', 'unknown')),
            'Parameters': param_display,
            'PA Score': exp_data['avg_score'],  # Now using PA score
            'Video Count': len(exp_data['results'])
        })
    
    # Sort by PA score
    summary_data.sort(key=lambda x: x['PA Score'], reverse=True)
    
    # Print table
    print(f"{'Rank':<4} {'Experiment':<25} {'Method':<8} {'CFG':<4} {'Parameters':<18} {'PA Score':<9} {'Videos':<6}")
    print("-" * 80)
    
    for i, exp in enumerate(summary_data, 1):
        print(f"{i:<4} {exp['Experiment']:<25} {exp['Method']:<8} {exp['CFG']:<4} {exp['Parameters']:<18} {exp['PA Score']:<9.2f} {exp['Video Count']:<6}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate side-by-side GIF comparisons from grid search results using DreamGen PA scores',
        epilog='''
Examples:
  # Run with default parameters (gr1_object prompt, single model)
  python visualize_dreamgen.py
  
  # Use different DreamGen evaluation prompt
  python visualize_dreamgen.py --dreamgen_prompt gr1_behavior
  
  # Compare across multiple model categories
  python visualize_dreamgen.py --model_cats Cosmos-Predict2-14B-Video2World cogvideox5b_i2v
  
  # Custom parameter keywords and prompt
  python visualize_dreamgen.py --param_keywords sampling_method cfg_scale guidance_lr_pattern --dreamgen_prompt gr1_env
  
  # Custom output and top-k
  python visualize_dreamgen.py --output_dir my_vis --top_k_deltas 10
        '''
    )
    parser.add_argument('--base_dir', default='/home/yjianhao/project/frame-guidance',
                       help='Base directory containing generated_videos and results')
    parser.add_argument('--output_dir', default='visualization/dreamgen_visualization_multi_model',
                       help='Output directory for GIF comparisons')
    parser.add_argument('--model_cats', nargs='+', default=['cogvideox5b_i2v','Cosmos-Predict2-2B-Video2World', 'Cosmos-Predict2-14B-Video2World'],
                       help='Model categories to use for PA scores (can specify multiple)')
    parser.add_argument('--top_k_deltas', type=int, default=10,
                       help='Number of highest delta videos to create GIFs for')
    parser.add_argument('--fps', type=int, default=16,
                       help='FPS for generated GIFs')
    parser.add_argument('--duration', type=int, default=5,
                       help='Duration in seconds for each GIF')
    parser.add_argument('--param_keywords', nargs='+', 
                       default=['sampling_method', 'cfg_scale', 'guidance_lr_pattern', 'guidance_step_pattern', 'loss_mode', 'guidance_frequency'],
                       help='Parameter keywords to include in experiment labels')
    parser.add_argument('--dreamgen_prompt', default='gr1_object',
                       help='DreamGen evaluation prompt to use for PA scores (default: gr1_object)')
    
    args = parser.parse_args()
    
    print("Loading experiment data...")
    experiments = load_experiment_data(args.base_dir, args.dreamgen_prompt, args.model_cats)
    
    if not experiments:
        print("No experiments found! Check the directory structure.")
        return
    
    print(f"Found {len(experiments)} experiments")
    
    # Print summary (commented out for now)
    # print_summary_table(experiments, args.param_keywords)
    
    # Generate GIFs for highest PA score delta videos
    print(f"\nGenerating side-by-side GIFs for top {args.top_k_deltas} PA score delta videos...")
    generate_delta_gifs(experiments, args.output_dir, args.top_k_deltas, args.fps, args.param_keywords)

if __name__ == "__main__":
    main()
