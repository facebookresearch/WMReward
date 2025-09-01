#!/usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
import cv2
import tempfile
import re
from PIL import Image, ImageDraw, ImageFont

def parse_list_of_floats(value):
    """Parse a string or list representing a list of floats."""
    if isinstance(value, str):
        if not (value.startswith("[") and value.endswith("]")):
            raise ValueError("Invalid string format for list of floats.")
        try:
            parsed_floats = [round(float(x), 4) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", value)]
            if not parsed_floats:
                raise ValueError("No valid floats found in the input string.")
            return parsed_floats
        except ValueError:
            raise ValueError("Failed to parse floats from the input string.")
    elif isinstance(value, list):
        if all(isinstance(x, (int, float)) for x in value):
            return [round(float(x), 4) for x in value]
        else:
            raise ValueError("List contains non-numeric values.")
    raise TypeError("Input must be a string representing a list of floats or a list of numbers.")

def load_scenario_descriptions():
    """Load scenario descriptions from the Physics-IQ descriptions file."""
    descriptions_path = "/home/yjianhao/project/PhysicsIQ/physics-IQ-benchmark/descriptions/descriptions.csv"
    descriptions = {}
    
    try:
        desc_df = pd.read_csv(descriptions_path)
        for _, row in desc_df.iterrows():
            scenario_name = row['scenario']
            # Extract the base scenario name (remove perspective and take info)
            if 'trimmed-' in scenario_name:
                base_scenario = scenario_name.split('trimmed-')[1]
                descriptions[base_scenario] = row['description']
    except Exception as e:
        print(f"Warning: Could not load scenario descriptions: {e}")
    
    return descriptions

def calculate_physics_iq_metrics(csv_file_path):
    """Calculate Physics-IQ metrics from a CSV file."""
    VIEWS = ["perspective-left", "perspective-center", "perspective-right"]
    
    df = pd.read_csv(csv_file_path)
    
    # Parse list columns
    list_columns = [
        f"v1_mse_{view}" for view in VIEWS
    ] + [
        f"spatiotemporal_iou_v1_{view}" for view in VIEWS
    ]
    
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_of_floats)
    
    # Calculate individual metrics
    total_sum_v1_mse = df.apply(
        lambda row: np.mean(np.concatenate([row[f"v1_mse_{view}"] for view in VIEWS])),
        axis=1
    ).mean()
    
    total_sum_spatiotemporal_iou_v1 = df.apply(
        lambda row: np.mean(np.concatenate([row[f"spatiotemporal_iou_v1_{view}"] for view in VIEWS])),
        axis=1
    ).mean()
    
    total_sum_spatial_iou = df[[f"spatial_iou_v1_{view}" for view in VIEWS]].mean().mean()
    total_sum_weighted_spatial_iou = df[[f"weighted_spatial_iou_v1_{view}" for view in VIEWS]].mean().mean()
    
    # Calculate variance metrics
    physical_variance_mse = round(np.mean([
        df[f"variance_mse_{view}"].apply(parse_list_of_floats).explode().mean()
        for view in VIEWS
        if f"variance_mse_{view}" in df.columns
    ]), 4) if any(f"variance_mse_{view}" in df.columns for view in VIEWS) else 0.001
    
    physical_variance_spatiotemporal_iou = round(np.mean([
        df[f"variance_spatiotemporal_iou_{view}"].apply(parse_list_of_floats).explode().mean()
        for view in VIEWS
        if f"variance_spatiotemporal_iou_{view}" in df.columns
    ]), 4) if any(f"variance_spatiotemporal_iou_{view}" in df.columns for view in VIEWS) else 0.001
    
    physical_variance_spatial = round(np.mean([
        df[f"variance_spatial_{view}"].mean()
        for view in VIEWS
        if f"variance_spatial_{view}" in df.columns
    ]), 5) if any(f"variance_spatial_{view}" in df.columns for view in VIEWS) else 0.001
    
    physical_variance_weighted_spatial = round(np.mean([
        df[f"variance_weighted_spatial_{view}"].mean()
        for view in VIEWS
        if f"variance_weighted_spatial_{view}" in df.columns
    ]), 4) if any(f"variance_weighted_spatial_{view}" in df.columns for view in VIEWS) else 0.001
    
    # Add safeguards against division by zero
    physical_variance_mse = max(physical_variance_mse, 0.001)
    physical_variance_spatiotemporal_iou = max(physical_variance_spatiotemporal_iou, 0.001)
    physical_variance_spatial = max(physical_variance_spatial, 0.001)
    physical_variance_weighted_spatial = max(physical_variance_weighted_spatial, 0.001)
    
    # Calculate final Physics-IQ score
    final_score = (
        (
            (total_sum_spatiotemporal_iou_v1 / physical_variance_spatiotemporal_iou) +
            (total_sum_spatial_iou / physical_variance_spatial) +
            (total_sum_weighted_spatial_iou / physical_variance_weighted_spatial)
        ) / 3
    ) - (total_sum_v1_mse - physical_variance_mse)
    
    final_score *= 100
    final_score = round(max(min(final_score, 100.0), 0.0), 2)
    
    return {
        'mse': round(total_sum_v1_mse, 4),
        'spatiotemporal_iou': round(total_sum_spatiotemporal_iou_v1, 4),
        'spatial_iou': round(total_sum_spatial_iou, 4),
        'weighted_spatial_iou': round(total_sum_weighted_spatial_iou, 4),
        'final_score': final_score
    }

def load_physics_iq_data(base_dir, model_cats=['physics_iq'], include_ground_truth=True):
    """Load Physics-IQ experiment data from CSV files and video directories for multiple model categories."""
    experiments = {}
    
    # Physics-IQ results are stored as CSV files
    video_base = Path(base_dir) / "generated_videos" / "physics_iq"
    
    for model_cat in model_cats:
        results_base = Path(base_dir) / "results" / "physics_iq" / model_cat
        
        if not results_base.exists():
            print(f"Results directory not found: {results_base}")
            continue
        
        # Find all CSV files for generated models in this category
        for csv_file in results_base.glob("*.csv"):
            model_name = csv_file.stem
            
            # Try to find corresponding video directory
            video_dir = None
            for potential_base in video_base.iterdir():
                if potential_base.is_dir():
                    for exp_dir in potential_base.iterdir():
                        if exp_dir.is_dir() and model_name in exp_dir.name:
                            video_dir = exp_dir
                            break
                    if video_dir:
                        break
            
            if not video_dir:
                print(f"Video directory not found for {model_name}")
                continue
            
            try:
                # Load CSV results 
                df = pd.read_csv(csv_file)
                
                # Calculate final score using embedded metrics calculation
                metrics = calculate_physics_iq_metrics(str(csv_file))
                
                # Create unique key that includes model category
                unique_key = f"{model_cat}_{model_name}"
                
                experiments[unique_key] = {
                    'results': df,
                    'video_dir': video_dir,
                    'avg_final_score': metrics['final_score'],
                    'metrics': metrics,
                    'is_ground_truth': False,
                    'model_cat': model_cat,
                    'model_name': model_name
                }
                
            except Exception as e:
                print(f"Error loading data for {model_name}: {e}")
    
    # Add ground truth videos if requested
    if include_ground_truth:
        gt_video_dir = Path("/home/yjianhao/project/PhysicsIQ/physics-IQ-benchmark/split-videos/testing/8FPS")
        if gt_video_dir.exists():
            # For ground truth, we don't have CSV results, so create dummy data based on one of the existing CSVs
            if experiments:
                # Use the first experiment's results as template for scenario names
                first_exp = next(iter(experiments.values()))
                gt_results = first_exp['results'].copy()
                
                experiments['ground_truth'] = {
                    'results': gt_results,  # Same scenarios as other models
                    'video_dir': gt_video_dir,
                    'avg_final_score': 100.0,  # Perfect score for ground truth
                    'metrics': {
                        'mse': 0.0,
                        'spatiotemporal_iou': 1.0,
                        'spatial_iou': 1.0,
                        'weighted_spatial_iou': 1.0,
                        'final_score': 100.0
                    },
                    'is_ground_truth': True,
                    'model_cat': 'ground_truth',
                    'model_name': 'ground_truth'
                }
                print("Added ground truth videos")
        else:
            print(f"Ground truth directory not found: {gt_video_dir}")
    
    return experiments

def find_high_variance_scenarios(experiments, top_k=10):
    """Find scenarios with highest score variance across different models."""
    scenario_scores = defaultdict(dict)
    
    # Collect per-scenario final scores for each experiment (excluding ground truth for variance calculation)
    for exp_name, exp_data in experiments.items():
        is_ground_truth = exp_data.get('is_ground_truth', False)
        df = exp_data['results']
        
        for _, row in df.iterrows():
            scenario = row['scenario']
            
            if is_ground_truth:
                # For ground truth, use perfect score (but don't include in variance calculation)
                scenario_scores[scenario][exp_name] = {
                    'score': 1.0,  # Perfect score
                    'exp_name': exp_name,
                    'is_ground_truth': True,
                    'model_cat': exp_data.get('model_cat', 'unknown'),
                    'model_name': exp_data.get('model_name', 'unknown')
                }
            else:
                # Calculate per-scenario final score using same logic as overall calculation
                # For simplicity, use spatiotemporal IoU as the primary metric
                VIEWS = ["perspective-left", "perspective-center", "perspective-right"]
                
                # Get spatiotemporal IoU values for all views
                spatiotemporal_scores = []
                for view in VIEWS:
                    col_name = f"spatiotemporal_iou_v1_{view}"
                    if col_name in row:
                        value = row[col_name]
                        # Parse the list value if it's a string
                        if isinstance(value, str):
                            try:
                                parsed_values = parse_list_of_floats(value)
                                spatiotemporal_scores.extend(parsed_values)
                            except:
                                continue
                        elif isinstance(value, (list, np.ndarray)):
                            spatiotemporal_scores.extend(value)
                        elif isinstance(value, (int, float)):
                            spatiotemporal_scores.append(value)
                
                if spatiotemporal_scores:
                    scenario_score = np.mean(spatiotemporal_scores)
                    scenario_scores[scenario][exp_name] = {
                        'score': scenario_score,
                        'exp_name': exp_name,
                        'is_ground_truth': False,
                        'model_cat': exp_data.get('model_cat', 'unknown'),
                        'model_name': exp_data.get('model_name', 'unknown')
                    }
    
    # Calculate variance for each scenario (only for non-ground-truth models)
    scenario_variances = []
    for scenario, exp_scores in scenario_scores.items():
        # Filter out ground truth for variance calculation
        non_gt_scores = {k: v for k, v in exp_scores.items() if not v.get('is_ground_truth', False)}
        
        if len(non_gt_scores) >= 2:  # Need at least 2 non-GT experiments to compare
            scores = [data['score'] for data in non_gt_scores.values()]
            score_variance = np.var(scores)
            score_range = max(scores) - min(scores)
            
            scenario_variances.append({
                'scenario': scenario,
                'score_variance': score_variance,
                'score_range': score_range,
                'scores': exp_scores,  # Include all scores (including GT) for display
                'mean_score': np.mean(scores)  # Mean of non-GT scores
            })
    
    # Sort by score range (highest delta first)
    scenario_variances.sort(key=lambda x: x['score_range'], reverse=True)
    
    return scenario_variances[:top_k]

def create_scenario_comparison_gif(scenario_data, output_path, experiments, fps=10, descriptions=None, model_cats=None):
    """Create side-by-side GIF from videos of the same scenario across different models."""
    try:
        scenario = scenario_data['scenario']
        scenario_basename = scenario.replace('.mp4', '')
        
        # Collect video paths for each perspective view
        VIEWS = ["perspective-left", "perspective-center", "perspective-right"]
        model_videos = {}
        
        for exp_name, score_data in scenario_data['scores'].items():
            exp_data = experiments[exp_name]
            video_dir = exp_data['video_dir']
            
            # Find videos for this scenario
            videos_for_model = {}
            exp_data = experiments[exp_name]
            is_ground_truth = exp_data.get('is_ground_truth', False)
            
            for view in VIEWS:
                matching_files = []
                
                if is_ground_truth:
                    # Ground truth pattern: NNNN_testing-videos_8FPS_perspective-{view}_take-1_trimmed-{scenario}.mp4
                    pattern = f"*_testing-videos_8FPS_{view}_take-1_{scenario_basename}.mp4"
                    matching_files = list(video_dir.glob(pattern))
                else:
                    # Generated model patterns: NNNN_{view}_{scenario_basename}.mp4 or NNNN_{scenario_basename}.mp4
                    pattern = f"*{view}_{scenario_basename}.mp4"
                    matching_files = list(video_dir.glob(pattern))
                
                if matching_files:
                    videos_for_model[view] = matching_files[0]
            
            # If no perspective-specific videos found for generated models, try to find a single video for this scenario
            if len(videos_for_model) == 0 and not is_ground_truth:
                single_video_pattern = f"*{scenario_basename}.mp4"
                single_video_files = list(video_dir.glob(single_video_pattern))
                if single_video_files:
                    # Use the same video for all perspectives
                    single_video = single_video_files[0]
                    for view in VIEWS:
                        videos_for_model[view] = single_video
                    print(f"Using single video for all perspectives: {exp_name} {scenario_basename}")
            
            if len(videos_for_model) == len(VIEWS):  # All views found (or single video used for all)
                model_videos[exp_name] = videos_for_model
            else:
                print(f"Warning: {exp_name} missing videos for {scenario_basename} (found {len(videos_for_model)}/{len(VIEWS)} views)")
        
        if len(model_videos) < 2:
            print(f"Not enough video sets found for scenario {scenario}")
            return False
        
        # For each model-view combination, load video frames
        frame_data = {}
        
        for exp_name, videos in model_videos.items():
            frame_data[exp_name] = {}
            
            for view, video_path in videos.items():
                # Load video frames using OpenCV
                cap = cv2.VideoCapture(str(video_path))
                
                if not cap.isOpened():
                    print(f"Error opening video: {video_path}")
                    continue
                
                # Get video properties
                original_fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Calculate how many frames to extract
                max_frames = min(int(fps * 5), total_frames)  # 5 seconds duration
                frame_step = max(1, total_frames // max_frames) if max_frames > 0 else 1
                
                frames = []
                frame_idx = 0
                
                while len(frames) < max_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    # Resize frame to standard size
                    frame = cv2.resize(frame, (240, 180))
                    frames.append(frame)
                    frame_idx += frame_step
                
                cap.release()
                frame_data[exp_name][view] = frames
            
        # Get frame count (use minimum across all videos)
        frame_counts = []
        for exp_name, view_frames in frame_data.items():
            for view, frames in view_frames.items():
                frame_counts.append(len(frames))
        
        if not frame_counts:
            return False
        
        max_frames = min(frame_counts)
        if max_frames == 0:
            return False
            
        # Create grid frames: models as rows, views as columns
        grid_frames = []
        
        # Order models: ground truth first, then by model category order
        gt_models = []
        other_models = []
        
        for exp_name in frame_data.keys():
            if exp_name == 'ground_truth':
                gt_models.append(exp_name)
            else:
                other_models.append(exp_name)
        
        # Sort other models by model category (use the order from model_cats argument)
        def get_model_category_order(exp_name):
            exp_data = experiments[exp_name]
            model_cat = exp_data.get('model_cat', 'unknown')
            # Return a high number for unknown categories to put them at the end
            if model_cat == 'unknown':
                return 999
            # Find the index in the model_cats list, or put at end if not found
            if model_cats is None:
                return 999
            try:
                return model_cats.index(model_cat)
            except ValueError:
                return 999
        
        other_models.sort(key=get_model_category_order)
        
        # Combine: ground truth first, then other models in category order
        models = gt_models + other_models
        
        frame_width, frame_height = 240, 180
        label_height = 60
        
        # Grid dimensions: 3 columns (views) x N rows (models)
        grid_width = len(VIEWS) * (frame_width + 10) + 20
        # Allocate extra space for potentially multi-line titles
        grid_height = len(models) * (frame_height + label_height + 10) + 150
        
        # Store frames for GIF creation
        grid_frames = []
        
        for frame_idx in range(max_frames):
            # Create grid image using OpenCV
            grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
            
            # Add title - use description as prompt
            # title = ""
            # if descriptions and scenario_basename in descriptions:
            title = descriptions[f"{scenario_basename.replace('trimmed-', '')}.mp4"]

            
            # Handle multi-line title if it's too long
            max_title_width = grid_width - 20
            words = title.split()
            lines = []
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                # Estimate text width (rough approximation)
                estimated_width = len(test_line) * 10  # Rough estimate
                if estimated_width <= max_title_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        # Single word is too long, break it
                        lines.append(word)
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Limit to 3 lines maximum
            if len(lines) > 3:
                lines = lines[:3]
                lines[2] = lines[2][:50] + "..." if len(lines[2]) > 50 else lines[2]
            
            # Draw multi-line title using OpenCV
            title_height = max(len(lines) * 18, 18)
            for i, line in enumerate(lines):
                line_y = 10 + i * 18
                # Center the text
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                line_x = (grid_width - text_size[0]) // 2
                cv2.putText(grid_img, line, (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Add view headers
            header_y = 10 + title_height + 10
            
            for col, view in enumerate(VIEWS):
                view_label = view.replace('perspective-', '').title()
                x = col * (frame_width + 10) + 10 + frame_width // 2
                y = header_y
                text_size = cv2.getTextSize(view_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                view_x = x - text_size[0] // 2
                cv2.putText(grid_img, view_label, (view_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
            # Add frames and model labels
            content_start_y = header_y + 25
            
            for row, exp_name in enumerate(models):
                score = scenario_data['scores'][exp_name]['score']
                
                # Calculate label position for this row
                label_y = content_start_y + row * (frame_height + label_height + 10)
                
                # Model label with category
                model_cat = scenario_data['scores'][exp_name].get('model_cat', 'unknown')
                model_name = scenario_data['scores'][exp_name].get('model_name', exp_name)
                
                # Model category label (above the video)
                cat_label = f"{model_cat}"
                cat_y = label_y - 15
                cv2.putText(grid_img, cat_label, (10, cat_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                
                # Model name and score label
                model_label = f"{model_name}: {score:.3f}"
                cv2.putText(grid_img, model_label, (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Frames for each view
                for col, view in enumerate(VIEWS):
                    if view in frame_data[exp_name] and frame_idx < len(frame_data[exp_name][view]):
                        frame = frame_data[exp_name][view][frame_idx]
                        x = col * (frame_width + 10) + 10
                        y = label_y + 20
                        
                        # Place frame in grid
                        grid_img[y:y+frame_height, x:x+frame_width] = frame
                        
                        # Draw border
                        cv2.rectangle(grid_img, (x-1, y-1), (x + frame_width + 1, y + frame_height + 1), (0, 0, 0), 1)
            
            # Add score info at bottom
            score_info = f"Score Range: {scenario_data['score_range']:.3f} | Mean: {scenario_data['mean_score']:.3f}"
            score_y = grid_height - 25
            text_size = cv2.getTextSize(score_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            score_x = (grid_width - text_size[0]) // 2
            cv2.putText(grid_img, score_info, (score_x, score_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
            # Convert OpenCV BGR to RGB for PIL
            grid_img_rgb = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(grid_img_rgb)
            grid_frames.append(pil_img)
        
        if grid_frames:
            # Save as GIF
            grid_frames[0].save(
                output_path,
                save_all=True,
                append_images=grid_frames[1:],
                duration=int(1000/fps),
                loop=0
            )
            return True
        
        return False
        
    except Exception as e:
        print(f"Error creating GIF for scenario {scenario_data['scenario']}: {e}")
        return False

def generate_physics_iq_visualizations(experiments, output_dir, top_k=10, fps=10, model_cats=None):
    """Generate comparison GIFs for Physics-IQ scenarios with highest variance."""
    print(f"\nFinding top {top_k} scenarios with highest score variance...")
    
    # Load scenario descriptions
    descriptions = load_scenario_descriptions()
    print(f"Loaded {len(descriptions)} scenario descriptions")
    
    high_variance_scenarios = find_high_variance_scenarios(experiments, top_k)
    
    if not high_variance_scenarios:
        print("No scenarios with sufficient variance found!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating {len(high_variance_scenarios)} comparison GIFs...")
    print("-" * 80)
    
    successful_gifs = 0
    
    for i, scenario_data in enumerate(high_variance_scenarios, 1):
        scenario = scenario_data['scenario']
        score_range = scenario_data['score_range']
        mean_score = scenario_data['mean_score']
        
        # Create safe filename
        safe_name = "".join(c for c in scenario if c.isalnum() or c in "._-").strip()[:50]
        filename = f"{i:02d}_range{score_range:.3f}_{safe_name}.gif"
        output_path = os.path.join(output_dir, filename)
        
        print(f"{i:2d}. {scenario}")
        print(f"     Score range: {score_range:.3f} (mean: {mean_score:.3f})")
        
        # Show scores for each model in the order they'll appear in the GIF
        # First show ground truth, then other models in category order
        gt_scores = []
        other_scores = []
        
        for exp_name, score_data in scenario_data['scores'].items():
            if exp_name == 'ground_truth':
                gt_scores.append((exp_name, score_data))
            else:
                other_scores.append((exp_name, score_data))
        
        # Sort other scores by model category order
        if model_cats:
            def get_score_category_order(item):
                exp_name, score_data = item
                model_cat = score_data.get('model_cat', 'unknown')
                try:
                    return model_cats.index(model_cat)
                except ValueError:
                    return 999
            
            other_scores.sort(key=get_score_category_order)
        
        # Display in order: ground truth first, then others
        for exp_name, score_data in gt_scores + other_scores:
            score = score_data['score']
            model_cat = score_data.get('model_cat', 'unknown')
            model_name = score_data.get('model_name', exp_name)
            print(f"     • [{model_cat}] {model_name}: {score:.3f}")
        
        success = create_scenario_comparison_gif(scenario_data, output_path, experiments, fps, descriptions=descriptions, model_cats=model_cats)
        if success:
            print(f"     ✓ GIF saved: {output_path}")
            successful_gifs += 1
        else:
            print(f"     ✗ Failed to create GIF")
        print()
    
    print(f"Successfully created {successful_gifs}/{len(high_variance_scenarios)} GIFs in {output_dir}")
    
    # Create summary
    create_summary_report(high_variance_scenarios, experiments, output_dir, successful_gifs)

def create_summary_report(high_variance_scenarios, experiments, output_dir, successful_count):
    """Create a summary report of the Physics-IQ visualizations."""
    summary_path = os.path.join(output_dir, "README.md")
    
    with open(summary_path, 'w') as f:
        f.write("# Physics-IQ Comparison Visualizations\n\n")
        f.write(f"Generated {successful_count} side-by-side GIF comparisons showing scenarios with highest score variance across models.\n\n")
        
        # Overall model performance summary
        f.write("## Model Performance Summary\n\n")
        f.write("| Model Category | Model Name | Avg Final Score | Spatiotemporal IoU | Spatial IoU | MSE |\n")
        f.write("|----------------|------------|-----------------|-------------------|-------------|-----|\n")
        
        for exp_name, exp_data in sorted(experiments.items(), 
                                       key=lambda x: x[1]['avg_final_score'], reverse=True):
            metrics = exp_data['metrics']
            model_cat = exp_data.get('model_cat', 'unknown')
            model_name = exp_data.get('model_name', exp_name)
            f.write(f"| {model_cat} | {model_name} | {metrics['final_score']:.2f} | "
                   f"{metrics['spatiotemporal_iou']:.4f} | {metrics['spatial_iou']:.4f} | "
                   f"{metrics['mse']:.4f} |\n")
        
        f.write("\n## Scenarios Ranked by Score Variance\n\n")
        
        for i, scenario_data in enumerate(high_variance_scenarios, 1):
            scenario = scenario_data['scenario']
            score_range = scenario_data['score_range']
            mean_score = scenario_data['mean_score']
            
            safe_name = "".join(c for c in scenario if c.isalnum() or c in "._-").strip()[:50]
            gif_filename = f"{i:02d}_range{score_range:.3f}_{safe_name}.gif"
            
            f.write(f"### {i}. {scenario}\n\n")
            f.write(f"- **Score Range**: {score_range:.3f}\n")
            f.write(f"- **Mean Score**: {mean_score:.3f}\n")
            f.write(f"- **GIF**: `{gif_filename}`\n\n")
            
            f.write("**Scores by Model:**\n")
            for exp_name, score_data in sorted(scenario_data['scores'].items(), 
                                             key=lambda x: x[1]['score'], reverse=True):
                model_cat = score_data.get('model_cat', 'unknown')
                model_name = score_data.get('model_name', exp_name)
                f.write(f"- **{model_cat}** - {model_name}: {score_data['score']:.3f}\n")
            f.write("\n")
            
            if os.path.exists(os.path.join(output_dir, gif_filename)):
                f.write(f"![{scenario}]({gif_filename})\n\n")
            else:
                f.write("*GIF generation failed*\n\n")
            f.write("---\n\n")
    
    print(f"Summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate side-by-side GIF comparisons from Physics-IQ results'
    )
    parser.add_argument('--base_dir', default='/home/yjianhao/project/frame-guidance',
                       help='Base directory containing generated_videos and results')
    parser.add_argument('--model_cats', nargs='+', default=['Cosmos-Predict2-14B-Video2World', 'Cosmos-Predict2-2B-Video2World', 'cogvideox5b_i2v'],
                       help='Model category subdirectories in results/physics_iq/ (space-separated)')
    parser.add_argument('--output_dir', default='./visualization/physics_iq_14b',
                       help='Output directory for GIF comparisons')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of highest variance scenarios to create GIFs for')
    parser.add_argument('--fps', type=int, default=16,
                       help='FPS for generated GIFs')
    parser.add_argument('--include_gt', action='store_true', default=True,
                       help='Include ground truth videos in comparison')
    
    args = parser.parse_args()
    
    print("Loading Physics-IQ experiment data...")
    experiments = load_physics_iq_data(args.base_dir, args.model_cats, args.include_gt)
    
    if not experiments:
        print("No experiments found! Check the directory structure.")
        return
    
    print(f"Found {len(experiments)} experiments:")
    for exp_name, exp_data in experiments.items():
        model_cat = exp_data.get('model_cat', 'unknown')
        model_name = exp_data.get('model_name', exp_name)
        print(f"  - [{model_cat}] {model_name}: Final Score = {exp_data['avg_final_score']:.2f}")
    
    # Generate visualizations
    generate_physics_iq_visualizations(experiments, args.output_dir, args.top_k, args.fps, args.model_cats)

if __name__ == "__main__":
    main()
