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

def load_experiment_data(base_dir, search_dir):
    """Load experiment configurations and results from a single directory."""
    experiments = {}
    
    # Scan for experiment directories in the specified search directory
    video_base = Path(base_dir) / "generated_videos" / search_dir / "cogvideox2b"
    results_base = Path(base_dir) / "results" / "videophy2" / search_dir / "cogvideox2b"
    
    if not video_base.exists():
        print(f"Video directory not found: {video_base}")
        return experiments
    
    if not results_base.exists():
        print(f"Results directory not found: {results_base}")
        return experiments
    
    print(f"Scanning {search_dir} directory...")
    print(f"  Video base: {video_base}")
    print(f"  Results base: {results_base}")
    
    for exp_dir in video_base.iterdir():
        if not exp_dir.is_dir():
            continue
            
        config_file = exp_dir / "experiment_config.json"
        results_file = results_base / exp_dir.name / "pc.csv"
        
        if config_file.exists() and results_file.exists():
            # Load config
            with open(config_file) as f:
                config = json.load(f)
            
            # Add directory info to config for label generation
            config['_source_dir'] = search_dir
            
            # Load results
            try:
                results_df = pd.read_csv(results_file)
                avg_score = results_df['score'].mean()
                
                experiments[exp_dir.name] = {
                    'config': config,
                    'results': results_df,
                    'avg_score': avg_score,
                    'video_dir': exp_dir,
                    'results_file': results_file
                }
            except Exception as e:
                print(f"Error loading results for {exp_dir.name}: {e}")
    
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

    if param_keywords is None:
        param_keywords = ['sampling_method', 'cfg_scale', 'loss_mode']

    # Collect key info
    key_info = {'source_dir': source_dir}
    for keyword in param_keywords:
        key_info[keyword] = params.get(keyword)

    # Build compact, readable label
    label = _format_label_from_params(params, param_keywords)
    return key_info, label



def find_high_delta_videos(experiments, param_keywords, top_k=10):
    """Find videos with highest score variance across experiments."""
    # Collect all video files by basename (same prompt across experiments)
    video_scores = defaultdict(dict)
    
    for exp_name, exp_data in experiments.items():
        key_params, label = extract_key_params(exp_data['config'], param_keywords)
        
        for _, row in exp_data['results'].iterrows():
            video_path = row['videopath']
            video_basename = os.path.basename(video_path)
            score = row['score']
            
            # Use a unique key per experiment to avoid collisions when labels are identical
            video_scores[video_basename][exp_name] = {
                'score': score,
                'path': video_path,
                'exp_name': exp_name,
                'display_label': label
            }
    
    # Calculate score variance for each video
    video_deltas = []
    for video_basename, exp_scores in video_scores.items():
        if len(exp_scores) >= 2:  # Need at least 2 experiments to compare
            scores = [data['score'] for data in exp_scores.values()]
            score_variance = np.var(scores)
            score_range = max(scores) - min(scores)
            
            video_deltas.append({
                'video_basename': video_basename,
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
            labels.append(f"{display_label}\nScore: {data['score']:.1f}")
        
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
                score_info = f"Score Range: {video_data['score_range']:.1f} | Mean: {video_data['mean_score']:.1f}"
                
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
    """Generate GIFs for videos with highest score deltas."""
    print(f"\nFinding top {top_k} videos with highest score variance...")
    
    high_delta_videos = find_high_delta_videos(experiments, param_keywords or ['sampling_method','cfg_scale','loss_mode'], top_k)
    
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
        
        # Create safe filename
        safe_name = "".join(c for c in basename if c.isalnum() or c in "._- ").strip()
        if len(safe_name) > 100:
            safe_name = safe_name[:97] + "..."
        
        # Simple, robust filename independent of label structure
        safe_name = "".join(c for c in basename if c.isalnum() or c in "._- ").strip()[:50]
        filename = f"{i:02d}_range{score_range:.1f}_{safe_name}.gif"
        output_path = os.path.join(output_dir, filename)
        
        print(f"{i:2d}. {basename[:60]}...")
        print(f"     Score range: {score_range:.2f} (mean: {mean_score:.2f})")
        
        # Show scores for each experiment sorted by label
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
        f.write(f"Generated {successful_count} side-by-side GIF comparisons showing videos with highest score variance across experiments.\n\n")
        f.write("## Videos Ranked by Score Variance\n\n")
        
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
            f.write(f"- **Score Range**: {score_range:.2f}\n")
            f.write(f"- **Mean Score**: {mean_score:.2f}\n")
            f.write(f"- **GIF**: `{gif_filename}`\n\n")
            
            f.write("**Scores by Experiment:**\n")
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
            'CFG': key_params.get('cfg_scale', 'unknown'),
            'Parameters': param_display,
            'Avg Score': exp_data['avg_score'],
            'Video Count': len(exp_data['results'])
        })
    
    # Sort by average score
    summary_data.sort(key=lambda x: x['Avg Score'], reverse=True)
    
    # Print table
    print(f"{'Rank':<4} {'Experiment':<25} {'Method':<8} {'CFG':<4} {'Parameters':<18} {'Avg Score':<9} {'Videos':<6}")
    print("-" * 80)
    
    for i, exp in enumerate(summary_data, 1):
        print(f"{i:<4} {exp['Experiment']:<25} {exp['Method']:<8} {exp['CFG']:<4} {exp['Parameters']:<18} {exp['Avg Score']:<9.2f} {exp['Video Count']:<6}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate side-by-side GIF comparisons from grid search results',
        epilog='''
Examples:
  # Scan search_t2v3 directory with default parameters
  python visualize_t2v_gridsearch.py --search_dir search_t2v3
  
  # Scan search_t2v2 with custom parameter keywords
  python visualize_t2v_gridsearch.py --search_dir search_t2v2 --param_keywords cfg_scale loss_mode vjepa_variant
  
  # Scan search_t2v3 with custom output and top-k
  python visualize_t2v_gridsearch.py --search_dir search_t2v3 --output_dir my_vis --top_k_deltas 10
        '''
    )
    parser.add_argument('--base_dir', default='/home/yjianhao/project/frame-guidance',
                       help='Base directory containing generated_videos and results')
    parser.add_argument('--search_dir', default="search_t2v3",
                       help='Search directory to scan (e.g., search_t2v2, search_t2v3)')
    parser.add_argument('--output_dir', default="visualization/t2v3_cfgrho",
                       help='Output directory for GIF comparisons (defaults to visualization/{search_dir})')
    parser.add_argument('--top_k_deltas', type=int, default=10,
                       help='Number of highest delta videos to create GIFs for')
    parser.add_argument('--fps', type=int, default=10,
                       help='FPS for generated GIFs')
    parser.add_argument('--duration', type=int, default=5,
                       help='Duration in seconds for each GIF')
    parser.add_argument('--param_keywords', nargs='+', 
                       default=['sampling_method', 'cfg_scale', 'loss_mode', 'guidance_lr_pattern', 'guidance_step_pattern', 'guidance_frequency'],
                       help='Parameter keywords to include in experiment labels')
    # 'vjepa_variant', 'guidance_lr_pattern', 'vjepa_context_frames', 'slice_stride', 'slice_window_size', 'guidance_frequency'],
    args = parser.parse_args()
    
    print("Loading experiment data...")
    experiments = load_experiment_data(args.base_dir, args.search_dir)
    
    if not experiments:
        print("No experiments found! Check the directory structure.")
        return
    
    print(f"Found {len(experiments)} experiments")
    
    # Print summary
    print_summary_table(experiments, args.param_keywords)
    
    # Generate GIFs for highest delta videos
    print(f"\nGenerating side-by-side GIFs for top {args.top_k_deltas} delta videos...")
    generate_delta_gifs(experiments, args.output_dir, args.top_k_deltas, args.fps, args.param_keywords)

if __name__ == "__main__":
    main()
