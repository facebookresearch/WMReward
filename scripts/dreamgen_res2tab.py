from pathlib import Path
import argparse
import os
import glob

EXP = "dreamgen4"

def auto_discover_models(base_dir, prompt, model_cat):
    """Automatically discover evaluated models from DreamGen results directories."""
    models = set()
    dreamgen_dir = f"{base_dir}/results/{EXP}/{prompt}/{model_cat}"
    if os.path.exists(dreamgen_dir):
        for model_dir in os.listdir(dreamgen_dir):
            if os.path.isdir(os.path.join(dreamgen_dir, model_dir)):
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

def parse_results(models, base_dir, model_cat, prompt):
    """Parse DreamGen results from structured directories."""
    results = {}
    
    for model in models:
        directory_path = f"{base_dir}/results/{EXP}/{prompt}/{model_cat}/{model}"
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
        
        if model_results:
            results[model] = model_results
    
    return results

def generate_aggregated_table(models, prompts, all_results):
    """Generate aggregated markdown table with DreamGen results from all prompts stacked as rows."""
    # Get all unique models
    all_models = sorted(list(models))
    
    # Create headers - Model + DreamGen metrics
    headers = ["Model/Prompt", "SR (whole, %)", "PA (%)"]
    
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
            model_data = prompt_results.get(model, {})
            
            row = [f"└─ {prompt}"]
            
            sr = model_data.get("whole")
            pa = model_data.get("pa")
            
            row.append(f"{sr:.1f}" if sr is not None else "N/A")
            row.append(f"{pa:.1f}" if pa is not None else "N/A")
            
            lines.append("| " + " | ".join(row) + " |")
        
        # Add empty row between models for clarity
        lines.append("| |" + " |".join([" "] * (len(headers) - 1)) + " |")
    
    return "\n".join(lines)

def generate_individual_tables(model_list, results):
    """Generate individual tables for each prompt."""
    lines = []
    lines.append("| Model | SR (whole, %) | PA (%) |")
    lines.append("| --- | ---: | ---: |")
    
    for model in model_list:
        model_data = results.get(model, {})
        sr = model_data.get("whole")
        pa = model_data.get("pa")
        
        sr_str = f"{sr:.1f}" if sr is not None else "N/A"
        pa_str = f"{pa:.1f}" if pa is not None else "N/A"
        
        lines.append(f"| {model} | {sr_str} | {pa_str} |")
    
    return "\n".join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DreamGen Results to Table Converter")
    # ,'gr1_env','gr1_behavior'
    parser.add_argument('--prompt', nargs='+', default=['gr1_object','gr1_env','gr1_behavior'],
                      help='Evaluation prompt(s)')
    # parser.add_argument('--model_cat', type=str, default='cogvideox5b_i2v',
    #                   help='Model category')
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
    args = parser.parse_args()

    # Collect results from all prompts
    all_results = {}
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
            
            if not discovered_models:
                print("No models found in results directories!")
                print(f"Checked DreamGen: {args.base_dir}/results/{EXP}/{prompt}/{args.model_cat}")
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

        # Parse DreamGen results
        print("\nParsing DreamGen results...")
        results = parse_results(model_list, args.base_dir, args.model_cat, prompt)
        
        # Generate individual table for this prompt
        print(f"\nDreamGen Results for {prompt}:")
        individual_table = generate_individual_tables(model_list, results)
        print(individual_table)

        print(f"\nDreamGen Results Summary for {prompt}:")
        for model, metrics in results.items():
            sr = metrics.get('whole')
            pa = metrics.get('pa')
            print(f"{model}:")
            print(f"  SR (whole) = {sr:.2f}%" if sr is not None else f"  SR (whole) = N/A")
            print(f"  PA = {pa:.2f}%" if pa is not None else f"  PA = N/A")
        
        # Store results for aggregation
        all_results[prompt] = results
        all_model_lists[prompt] = model_list

    # Generate aggregated table
    print(f"\n{'='*80}")
    print("AGGREGATED DREAMGEN RESULTS ACROSS ALL PROMPTS")
    print(f"{'='*80}")
    
    # Get all unique models across all prompts
    all_models = set()
    for model_list in all_model_lists.values():
        all_models.update(model_list)
    all_models = sorted(list(all_models))
    
    if all_models and all_results:
        print("\nAggregated Markdown Table (Prompts stacked as rows):")
        aggregated_md = generate_aggregated_table(all_models, args.prompt, all_results)
        print(aggregated_md)

