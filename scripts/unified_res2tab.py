import json
import os
import pandas as pd
import argparse
import pathlib
import glob
from pathlib import Path

def auto_discover_models(base_dir, prompt, model_cat):
    """Automatically discover evaluated models from all results directories."""
    models = set()
    
    # Check all evaluation directories
    eval_dirs = [
        f"{base_dir}/results/videophy1/{prompt}/{model_cat}",
        f"{base_dir}/results/videophy2/{prompt}/{model_cat}", 
        f"{base_dir}/results/vbench/{prompt}/{model_cat}",
        f"{base_dir}/results/dreamgen/{prompt}/{model_cat}"
    ]
    
    for eval_dir in eval_dirs:
        if os.path.exists(eval_dir):
            for item in os.listdir(eval_dir):
                if os.path.isdir(os.path.join(eval_dir, item)):
                    models.add(item)
    
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

def load_videophy1_results(models, base_dir, model_cat, prompt):
    """Load VideoPhY1 results (probability scores with 0.5 threshold)."""
    results = {}
    for model in models:
        directory_path = f"{base_dir}/results/videophy1/{prompt}/{model_cat}/{model}"
        if not os.path.exists(directory_path):
            continue
            
        csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
        
        pc_averages = []
        sa_averages = []
        pc_proportions = []
        sa_proportions = []
        pc_dfs = []
        sa_dfs = []
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, header=None, names=['video_path', 'score'])
            avg = df['score'].mean()
            proportion_ge_05 = (df['score'] >= 0.5).mean() * 100
            
            if 'pc' in os.path.basename(csv_file):
                pc_averages.append(avg)
                pc_proportions.append(proportion_ge_05)
                pc_dfs.append(df)
            elif 'sa' in os.path.basename(csv_file):
                sa_averages.append(avg)
                sa_proportions.append(proportion_ge_05)
                sa_dfs.append(df)
        
        # Calculate joint proportion
        joint_proportions = []
        if pc_dfs and sa_dfs:
            for pc_df, sa_df in zip(pc_dfs, sa_dfs):
                if len(pc_df) == len(sa_df):
                    joint_ge_05 = ((pc_df['score'] >= 0.5) & (sa_df['score'] >= 0.5)).mean() * 100
                    joint_proportions.append(joint_ge_05)
        
        results[model] = {
            'phy1_pc': sum(pc_averages)/len(pc_averages) if pc_averages else None,
            'phy1_sa': sum(sa_averages)/len(sa_averages) if sa_averages else None,
            'phy1_pcp': sum(pc_proportions)/len(pc_proportions) if pc_proportions else None,
            'phy1_sap': sum(sa_proportions)/len(sa_proportions) if sa_proportions else None,
            'phy1_joint': sum(joint_proportions)/len(joint_proportions) if joint_proportions else None
        }
    
    return results

def load_videophy2_results(models, base_dir, model_cat, prompt):
    """Load VideoPhY2 results (1-5 scores with ≥4 threshold)."""
    results = {}
    for model in models:
        directory_path = f"{base_dir}/results/videophy2/{prompt}/{model_cat}/{model}"
        if not os.path.exists(directory_path):
            continue
            
        csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
        
        pc_averages = []
        sa_averages = []
        pc_proportions = []
        sa_proportions = []
        pc_dfs = []
        sa_dfs = []
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            avg = df['score'].mean()
            proportion_ge_4 = (df['score'] >= 4).mean() * 100
            
            if 'pc' in os.path.basename(csv_file):
                pc_averages.append(avg)
                pc_proportions.append(proportion_ge_4)
                pc_dfs.append(df)
            elif 'sa' in os.path.basename(csv_file):
                sa_averages.append(avg)
                sa_proportions.append(proportion_ge_4)
                sa_dfs.append(df)
        
        # Calculate joint proportion
        joint_proportions = []
        if pc_dfs and sa_dfs:
            for pc_df, sa_df in zip(pc_dfs, sa_dfs):
                if len(pc_df) == len(sa_df):
                    joint_ge_4 = ((pc_df['score'] >= 4) & (sa_df['score'] >= 4)).mean() * 100
                    joint_proportions.append(joint_ge_4)
        
        results[model] = {
            'phy2_pc': sum(pc_averages)/len(pc_averages) if pc_averages else None,
            'phy2_sa': sum(sa_averages)/len(sa_averages) if sa_averages else None,
            'phy2_pcp': sum(pc_proportions)/len(pc_proportions) if pc_proportions else None,
            'phy2_sap': sum(sa_proportions)/len(sa_proportions) if sa_proportions else None,
            'phy2_joint': sum(joint_proportions)/len(joint_proportions) if joint_proportions else None
        }
    
    return results

def load_vbench_results(models, base_dir, model_cat, prompt):
    """Load VBench results."""
    evaluation_dir = f"{base_dir}/results/vbench/{prompt}/{model_cat}"
    metric_list = [
        'subject_consistency',
        'temporal_flickering',
        'aesthetic_quality',
        'dynamic_degree',
        'imaging_quality',
        'motion_smoothness'
    ]
    
    results = {}
    for model in models:
        results[model] = {}
        for metric in metric_list:
            path = pathlib.Path(f"{evaluation_dir}/{model}/{metric}/")
            if not path.exists():
                continue
            filename = next(path.glob("results_*_eval_results.json"), None)
            if filename:
                with open(filename, 'r') as f:
                    data = json.load(f)
                results[model][f'vb_{metric}'] = data[metric][0] * 100  # Convert to percentage
    
    return results

def load_dreamgen_results(models, base_dir, model_cat, prompt):
    """Load DreamGen results."""
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
            model_results[f'dg_{metric}'] = pct
        
        if model_results:
            results[model] = model_results
    
    return results

def generate_unified_table(models, prompts, all_results):
    """Generate unified markdown table with all evaluation metrics."""
    
    # Define all possible metrics in order
    metric_columns = [
        # VideoPhY1 metrics
        ('phy1_pc', 'Phy1 PC'),
        ('phy1_sa', 'Phy1 SA'), 
        ('phy1_pcp', 'Phy1 PCp'),
        ('phy1_sap', 'Phy1 SAp'),
        ('phy1_joint', 'Phy1 Joint'),
        # VideoPhY2 metrics
        ('phy2_pc', 'Phy2 PC'),
        ('phy2_sa', 'Phy2 SA'),
        ('phy2_pcp', 'Phy2 PCp'), 
        ('phy2_sap', 'Phy2 SAp'),
        ('phy2_joint', 'Phy2 Joint'),
        # VBench metrics
        ('vb_subject_consistency', 'VB SubjCons'),
        ('vb_temporal_flickering', 'VB TempFlick'),
        ('vb_aesthetic_quality', 'VB AesthQual'),
        ('vb_dynamic_degree', 'VB DynDeg'),
        ('vb_imaging_quality', 'VB ImagQual'),
        ('vb_motion_smoothness', 'VB MotSmooth'),
        # DreamGen metrics
        ('dg_whole', 'DG SR'),
        ('dg_pa', 'DG PA')
    ]
    
    headers = ["Model/Prompt"] + [col[1] for col in metric_columns]
    
    md_table = "| " + " | ".join(headers) + " |\n"
    md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    # Add data rows for each model, with prompts stacked as rows
    for model in models:
        # Add a separator row for each model
        md_table += f"| **{model}** |" + " |".join([" "] * (len(headers) - 1)) + " |\n"
        
        for prompt in prompts:
            prompt_results = all_results.get(prompt, {})
            
            # Combine all metrics for this model and prompt
            combined_metrics = {}
            for eval_type in ['videophy1', 'videophy2', 'vbench', 'dreamgen']:
                eval_results = prompt_results.get(eval_type, {})
                model_data = eval_results.get(model, {})
                combined_metrics.update(model_data)
            
            row = [f"└─ {prompt}"]
            
            # Add values for each metric
            for metric_key, _ in metric_columns:
                value = combined_metrics.get(metric_key)
                if value is not None:
                    if metric_key.startswith('phy1_') and metric_key in ['phy1_pc', 'phy1_sa']:
                        row.append(f"{value:.3f}")
                    elif metric_key.startswith('phy2_') and metric_key in ['phy2_pc', 'phy2_sa']:
                        row.append(f"{value:.2f}")
                    else:
                        row.append(f"{value:.1f}")
                else:
                    row.append("N/A")
            
            md_table += "| " + " | ".join(row) + " |\n"
        
        # Add empty row between models for clarity
        md_table += "| |" + " |".join([" "] * (len(headers) - 1)) + " |\n"
    
    return md_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Results to Table Converter")
    parser.add_argument('--prompt', nargs='+', default=['gr1_object', 'gr1_env', 'gr1_behavior'],
                      help='Evaluation prompt(s)')
    parser.add_argument('--model_cat', type=str, default='cogvideox5b_i2v',
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
    args = parser.parse_args()

    # Collect results from all prompts and all evaluation types
    all_results = {}

    # Process each prompt
    for prompt in args.prompt:
        print(f"\n{'='*60}")
        print(f"Processing prompt: {prompt}")
        print(f"{'='*60}")
        
        # Determine which models to process for this prompt
        if args.models is None or args.auto_discover:
            print("Auto-discovering models from results directories...")
            discovered_models = auto_discover_models(args.base_dir, prompt, args.model_cat)
            
            if not discovered_models:
                print("No models found in results directories!")
                continue
                
            print(f"Discovered models: {discovered_models}")
            
            # Apply filtering if specified
            if args.include_patterns or args.exclude_patterns:
                discovered_models = filter_models(
                    discovered_models, 
                    args.include_patterns, 
                    args.exclude_patterns
                )
                print(f"After filtering: {discovered_models}")
            
            # Use discovered models, or combine with manually specified ones
            if args.models:
                model_list = list(set(args.models + discovered_models))
            else:
                model_list = discovered_models
        else:
            model_list = args.models
        
        if not model_list:
            print(f"No models to process for prompt {prompt}!")
            continue
            
        print(f"\nProcessing models: {model_list}")

        # Load results from all evaluation types
        prompt_results = {}
        
        print("Loading VideoPhY1 results...")
        prompt_results['videophy1'] = load_videophy1_results(model_list, args.base_dir, args.model_cat, prompt)
        
        print("Loading VideoPhY2 results...")
        prompt_results['videophy2'] = load_videophy2_results(model_list, args.base_dir, args.model_cat, prompt)
        
        print("Loading VBench results...")
        prompt_results['vbench'] = load_vbench_results(model_list, args.base_dir, args.model_cat, prompt)
        
        print("Loading DreamGen results...")
        prompt_results['dreamgen'] = load_dreamgen_results(model_list, args.base_dir, args.model_cat, prompt)
        
        all_results[prompt] = prompt_results

    # Generate unified table
    print(f"\n{'='*80}")
    print("UNIFIED RESULTS ACROSS ALL EVALUATION METRICS AND PROMPTS")
    print(f"{'='*80}")
    
    # Get all unique models across all prompts and evaluations
    all_models = set()
    for prompt_results in all_results.values():
        for eval_results in prompt_results.values():
            all_models.update(eval_results.keys())
    all_models = sorted(list(all_models))
    
    if all_models and all_results:
        print("\nUnified Markdown Table:")
        unified_table = generate_unified_table(all_models, args.prompt, all_results)
        print(unified_table)
    else:
        print("No results found to generate unified table.")
