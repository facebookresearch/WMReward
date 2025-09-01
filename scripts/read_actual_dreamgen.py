from pathlib import Path
import argparse
import os
import glob
import pandas as pd

"""
DreamGen and VideoPhy1 Results to Table Converter

This script combines DreamGen evaluation results with VideoPhy1 physics adherence scores
to provide a comprehensive evaluation of video generation models.

Key Features:
1. DreamGen Metrics: SR (whole) and PA (Qwen-based) percentages
2. VideoPhy1 Metrics: PC (Physics Consistency) and SA (Spatial Alignment) scores (0-1 range)
3. Combined PA: Average of binary VideoPhy1 PC scores (≥threshold = 1, <threshold = 0) 
   and Qwen PA scores for each sample

The combined PA metric represents the actual physics adherence by averaging:
- Qwen-VL-2.5 PA scores (binary 0/1)
- VideoPhy1 PC scores converted to binary using a threshold (default: 0.5)

Usage:
    python read_actual_dreamgen.py --prompt gr1_object --model_cat cogvideox5b_i2v --threshold 0.6
"""

def auto_discover_models(base_dir, prompt, model_cat):
    """Automatically discover evaluated models from DreamGen results directories."""
    models = set()
    dreamgen_dir = f"{base_dir}/results/dreamgen/{prompt}/{model_cat}"
    if os.path.exists(dreamgen_dir):
        for model_dir in os.listdir(dreamgen_dir):
            if os.path.isdir(os.path.join(dreamgen_dir, model_dir)):
                models.add(model_dir)
    return sorted(list(models))

def auto_discover_videophy1_models(base_dir, prompt, model_cat):
    """Automatically discover evaluated models from VideoPhy1 results directories."""
    models = set()
    videophy_dir = f"{base_dir}/results/videophy1/{prompt}/{model_cat}"
    if os.path.exists(videophy_dir):
        for model_dir in os.listdir(videophy_dir):
            if os.path.isdir(os.path.join(videophy_dir, model_dir)):
                models.add(model_dir)
    return sorted(list(models))

def filter_models(models, include_patterns=None, exclude_patterns=None):
    """Filter models based on include/exclude patterns."""
    if include_patterns:
        filtered = []
        for model in models:
            if any(pattern in model for pattern in include_patterns):
                filtered.append(model)
        models = filtered
    
    if exclude_patterns:
        filtered = []
        for model in models:
            if not any(pattern in model for pattern in exclude_patterns):
                filtered.append(model)
        models = filtered
    
    return models

def get_base_model_name(model_name):
    """Extract base model name (vanilla, guidance, rejection) from full model name."""
    if 'vanilla' in model_name:
        return 'vanilla'
    elif 'guidance' in model_name:
        return 'guidance'
    elif 'rejection' in model_name:
        return 'rejection'
    return model_name

def map_model_name_for_dreamgen(model_name):
    """Map VideoPhy1 model names to DreamGen model names."""
    base_name = get_base_model_name(model_name)
    return base_name

def parse_videophy1_results(models, base_dir, model_cat, prompt, threshold=0.5):
    """Parse VideoPhy1 results to get actual PA scores (0-1 range) and convert to binary."""
    results = {}
    
    for model in models:
        directory_path = f"{base_dir}/results/videophy1/{prompt}/{model_cat}/{model}"
        if not os.path.exists(directory_path):
            continue
            
        csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
        model_results = {}
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            if 'pc' in filename:
                metric = 'pc'
            elif 'sa' in filename:
                metric = 'sa'
            else:
                continue
            
            # Read VideoPhy1 CSV - scores are probabilities in 2nd column
            try:
                df = pd.read_csv(csv_file, header=None, names=['video_path', 'score'])
                # Calculate mean score (already a probability 0-1)
                avg_score = df['score'].mean()
                # Calculate proportion of videos with score >= threshold
                proportion_ge_threshold = (df['score'] >= threshold).mean() * 100
                # Convert to binary: 1 if score >= threshold, 0 otherwise
                binary_scores = (df['score'] >= threshold).astype(int)
                binary_avg = binary_scores.mean()
                
                model_results[f'{metric}_avg'] = avg_score
                model_results[f'{metric}_pct'] = proportion_ge_threshold
                model_results[f'{metric}_binary'] = binary_avg
                model_results[f'{metric}_raw_scores'] = df['score'].tolist()  # Store raw scores for sample-level averaging
            except Exception as e:
                print(f"Warning: Could not parse {csv_file}: {e}")
                continue
        
        if model_results:
            results[model] = model_results
    
    return results

def parse_results(models, base_dir, model_cat, prompt):
    """Parse DreamGen results from structured directories."""
    results = {}
    
    for model in models:
        directory_path = f"{base_dir}/results/dreamgen/{prompt}/{model_cat}/{model}"
        if not os.path.exists(directory_path):
            continue
            
        csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
        model_results = {}
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            if 'whole' in filename:
                metric = 'whole'
            elif 'pa' in filename:
                metric = 'pa'
            else:
                continue
            
            # Read CSV and calculate percentage
            with open(csv_file, "r") as fh:
                next(fh, None)  # skip header
                vals = [int(line.rsplit(",", 1)[-1].strip()) for line in fh if line.strip()]
            n = len(vals)
            pct = (sum(vals) / n * 100) if n else 0.0
            model_results[metric] = pct
            # Store raw binary values for sample-level averaging
            model_results[f'{metric}_raw_scores'] = vals
        
        if model_results:
            results[model] = model_results
    
    return results

def calculate_combined_pa(qwen_pa_scores, videophy1_pc_scores, threshold=0.5):
    """Calculate combined PA by averaging PA scores with VideoPhy1 PC scores per sample."""
    if not qwen_pa_scores or not videophy1_pc_scores:
        return None
    
    # Ensure both lists have the same length
    min_length = min(len(qwen_pa_scores), len(videophy1_pc_scores))
    if min_length == 0:
        return None
    
    # Get corresponding scores (PA scores are binary 0/1, PC scores are 0-1 range)
    qwen_binary = qwen_pa_scores[:min_length]
    videophy1_scores = videophy1_pc_scores[:min_length]
    
    # Calculate sample-level averages between PA (0/1) and PC (0-1) scores
    sample_averages = []
    for qwen_score, videophy1_score in zip(qwen_binary, videophy1_scores):
        print(qwen_score, videophy1_score)
        avg = (qwen_score + videophy1_score) / 2.0
        sample_averages.append(avg)
    
    # Return overall average
    return sum(sample_averages) / len(sample_averages) * 100  # Convert to percentage

def generate_aggregated_table(models, prompts, all_results, all_videophy1_results):
    """Generate aggregated markdown table with both DreamGen and VideoPhy1 results."""
    # Get all unique models
    all_models = sorted(list(models))
    
    # Create headers - Model + DreamGen metrics + VideoPhy1 metrics + Combined PA
    headers = ["Model/Prompt", "SR (whole, %)", "Qwen PA (%)", "VideoPhy1-PC", "VideoPhy1-SA", "Combined PA (%)"]
    
    # Generate table
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    # Add data rows for each model, with prompts stacked as rows
    for model in all_models:
        # Add a separator row for each model
        lines.append(f"| **{model}** |" + " |".join([" "] * (len(headers) - 1)) + " |")
        
        for prompt in prompts:
            prompt_results = all_results.get(prompt, {})
            prompt_videophy1 = all_videophy1_results.get(prompt, {})
            
            # Try to get DreamGen results using the model name directly first
            model_data = prompt_results.get(model, {})
            
            # If not found, try mapping the model name to base name for DreamGen lookup
            if not model_data:
                base_model = map_model_name_for_dreamgen(model)
                model_data = prompt_results.get(base_model, {})
            
            model_videophy1 = prompt_videophy1.get(model, {})
            
            row = [f"└─ {prompt}"]
            
            # DreamGen metrics
            sr = model_data.get("whole")
            pa = model_data.get("pa")
            row.append(f"{sr:.1f}" if sr is not None else "N/A")
            row.append(f"{pa:.1f}" if pa is not None else "N/A")
            
            # VideoPhy1 metrics
            pc_avg = model_videophy1.get("pc_avg")
            sa_avg = model_videophy1.get("sa_avg")
            
            row.append(f"{pc_avg:.3f}" if pc_avg is not None else "N/A")
            row.append(f"{sa_avg:.3f}" if sa_avg is not None else "N/A")
            
            # Combined PA (average of binary VideoPhy1 PC and Qwen PA)
            combined_pa = None
            if model_data.get("pa_raw_scores") and model_videophy1.get("pc_raw_scores"):
                combined_pa = calculate_combined_pa(
                    model_data["pa_raw_scores"], 
                    model_videophy1["pc_raw_scores"],
                    args.threshold
                )
            row.append(f"{combined_pa:.1f}" if combined_pa is not None else "N/A")
            
            lines.append("| " + " | ".join(row) + " |")
        
        # Add empty row between models for clarity
        lines.append("| |" + " |".join([" "] * (len(headers) - 1)) + " |")
    
    return "\n".join(lines)

def generate_individual_tables(model_list, results, videophy1_results):
    """Generate individual tables for each prompt with both metrics."""
    lines = []
    lines.append("| Model | SR (whole, %) | Qwen PA (%) | VideoPhy1-PC | VideoPhy1-SA | Combined PA (%) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    
    for model in model_list:
        # Try to get DreamGen results using the model name directly first
        model_data = results.get(model, {})
        
        # If not found, try mapping the model name to base name for DreamGen lookup
        if not model_data:
            base_model = map_model_name_for_dreamgen(model)
            model_data = results.get(base_model, {})
        
        # Get VideoPhy1 results directly
        model_videophy1 = videophy1_results.get(model, {})
        
        sr = model_data.get("whole")
        pa = model_data.get("pa")
        pc_avg = model_videophy1.get("pc_avg")
        sa_avg = model_videophy1.get("sa_avg")
        
        # Calculate combined PA
        combined_pa = None
        if model_data.get("pa_raw_scores") and model_videophy1.get("pc_raw_scores"):
            combined_pa = calculate_combined_pa(
                model_data["pa_raw_scores"], 
                model_videophy1["pc_raw_scores"],
                args.threshold
            )
        
        sr_str = f"{sr:.1f}" if sr is not None else "N/A"
        pa_str = f"{pa:.1f}" if pa is not None else "N/A"
        pc_avg_str = f"{pc_avg:.3f}" if pc_avg is not None else "N/A"
        sa_avg_str = f"{sa_avg:.3f}" if sa_avg is not None else "N/A"
        combined_pa_str = f"{combined_pa:.1f}" if combined_pa is not None else "N/A"
        
        lines.append(f"| {model} | {sr_str} | {pa_str} | {pc_avg_str} | {sa_avg_str} | {combined_pa_str} |")
    
    return "\n".join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DreamGen and VideoPhy1 Results to Table Converter")
    parser.add_argument('--prompt', nargs='+', default=['gr1_object'],
                      help='Evaluation prompt(s)')
    parser.add_argument('--model_cat', type=str, default='Cosmos-Predict2-14B-Video2World',
                      help='Model category')
    parser.add_argument('--models', nargs='+', default=None,
                      help='List of model names (if not provided, will auto-discover)')
    parser.add_argument('--auto_discover', action='store_true', default=True,
                      help='Automatically discover models from results directories')
    parser.add_argument('--include_patterns', nargs='+', default=None,
                      help='Only include models containing these patterns')
    parser.add_argument('--exclude_patterns', nargs='+', default=None,
                      help='Exclude models containing these patterns')
    parser.add_argument('--base_dir', type=str, 
                      default='/home/yjianhao/project/frame-guidance',
                      help='Base project directory')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Threshold for converting VideoPhy1 PC scores to binary (default: 0.5)')
    args = parser.parse_args()

    # Collect results from all prompts
    all_results = {}
    all_videophy1_results = {}
    all_model_lists = {}

    # Process each prompt
    for prompt in args.prompt:
        print(f"\n{'='*60}")
        print(f"Processing prompt: {prompt}")
        print(f"{'='*60}")
        
        # Determine which models to process for this prompt
        if args.models is None or args.auto_discover:
            print("Auto-discovering models from results directories...")
            discovered_models = auto_discover_models(args.base_dir, prompt, args.model_cat)
            discovered_videophy1_models = auto_discover_videophy1_models(args.base_dir, prompt, args.model_cat)
            
            # Combine models from both sources
            all_discovered = set(discovered_models + discovered_videophy1_models)
            
            if not all_discovered:
                print("No models found in results directories!")
                print(f"Checked DreamGen: {args.base_dir}/results/dreamgen/{prompt}/{args.model_cat}")
                print(f"Checked VideoPhy1: {args.base_dir}/results/videophy1/{prompt}/{args.model_cat}")
                continue
                
            print(f"Discovered models from DreamGen: {discovered_models}")
            print(f"Discovered models from VideoPhy1: {discovered_videophy1_models}")
            print(f"Combined models: {sorted(list(all_discovered))}")
            
            # Apply filtering if specified
            if args.include_patterns or args.exclude_patterns:
                all_discovered = filter_models(
                    list(all_discovered), 
                    args.include_patterns, 
                    args.exclude_patterns
                )
                print(f"After filtering: {all_discovered}")
            
            # Use discovered models, or combine with manually specified ones
            if args.models:
                model_list = list(set(args.models + list(all_discovered)))
            else:
                model_list = list(all_discovered)
        else:
            model_list = args.models
        
        if not model_list:
            print(f"No models to process for prompt {prompt}!")
            continue
            
        print(f"\nProcessing models: {model_list}")

        # Parse DreamGen results
        print("\nParsing DreamGen results...")
        results = parse_results(model_list, args.base_dir, args.model_cat, prompt)
        
        # Parse VideoPhy1 results for actual PA scores
        print("\nParsing VideoPhy1 results for actual PA scores...")
        videophy1_results = parse_videophy1_results(model_list, args.base_dir, args.model_cat, prompt, args.threshold)
        
        # Generate individual table for this prompt
        print(f"\nCombined Results for {prompt}:")
        individual_table = generate_individual_tables(model_list, results, videophy1_results)
        print(individual_table)

        print(f"\nResults Summary for {prompt}:")
        for model in model_list:
            print(f"{model}:")
            
            # Try to get DreamGen results using the model name directly first
            model_data = results.get(model, {})
            
            # If not found, try mapping the model name to base name for DreamGen lookup
            if not model_data:
                base_model = map_model_name_for_dreamgen(model)
                model_data = results.get(base_model, {})
                
            sr = model_data.get('whole')
            pa = model_data.get('pa')
            print(f"  DreamGen SR (whole) = {sr:.2f}%" if sr is not None else f"  DreamGen SR (whole) = N/A")
            print(f"  DreamGen PA = {pa:.2f}%" if pa is not None else f"  DreamGen PA = N/A")
            
            # VideoPhy1 metrics
            model_videophy1 = videophy1_results.get(model, {})
            pc_avg = model_videophy1.get('pc_avg')
            sa_avg = model_videophy1.get('sa_avg')
            
            print(f"  VideoPhy1 PC (avg) = {pc_avg:.4f}" if pc_avg is not None else f"  VideoPhy1 PC (avg) = N/A")
            print(f"  VideoPhy1 SA (avg) = {sa_avg:.4f}" if sa_avg is not None else f"  VideoPhy1 SA (avg) = N/A")
            
            # Combined PA metric
            combined_pa = None
            if model_data.get("pa_raw_scores") and model_videophy1.get("pc_raw_scores"):
                combined_pa = calculate_combined_pa(
                    model_data["pa_raw_scores"], 
                    model_videophy1["pc_raw_scores"],
                    args.threshold
                )
            print(f"  Combined PA (Qwen + VideoPhy1) = {combined_pa:.2f}%" if combined_pa is not None else f"  Combined PA (Qwen + VideoPhy1) = N/A")
        
        # Store results for aggregation
        all_results[prompt] = results
        all_videophy1_results[prompt] = videophy1_results
        all_model_lists[prompt] = model_list

    # Generate aggregated table
    print(f"\n{'='*80}")
    print("AGGREGATED RESULTS ACROSS ALL PROMPTS (DreamGen + VideoPhy1)")
    print(f"{'='*80}")
    
    # Get all unique models across all prompts
    all_models = set()
    for model_list in all_model_lists.values():
        all_models.update(model_list)
    all_models = sorted(list(all_models))
    
    if all_models and all_results:
        print("\nAggregated Markdown Table (Prompts stacked as rows):")
        aggregated_md = generate_aggregated_table(all_models, args.prompt, all_results, all_videophy1_results)
        print(aggregated_md)

