import json
import os
import pandas as pd
import argparse
import pathlib

def auto_discover_models(base_dir, prompt, model_cat):
    """Automatically discover evaluated models from VBench results directories."""
    models = set()
    
    # Discover from VBench results only
    vbench_dir = f"{base_dir}/results/vbench/{prompt}/{model_cat}"
    if os.path.exists(vbench_dir):
        for model_dir in os.listdir(vbench_dir):
            if os.path.isdir(os.path.join(vbench_dir, model_dir)):
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

def generate_latex_table(model_list, prompt, model_cat, base_dir):
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
    for model in model_list:
        results[model] = {}
        for metric in metric_list:
            path = pathlib.Path(f"{evaluation_dir}/{model}/{metric}/")
            filename = next(path.glob("results_*_eval_results.json"), None)
            if filename:
                with open(filename, 'r') as f:
                    data = json.load(f)
                results[model][metric] = data[metric][0]

    latex_table = r"""
    \begin{table*}[t]
        \vspace{-4mm}
        \centering
        \tablestyle{3.6pt}{1.0}
        \caption{
            \textbf{Model Evaluation Metrics.} A comparison of VBench metrics across different models.
        }
        \resizebox{1.0\linewidth}{!}{
        \begin{tabular}{l""" + "c" * len(metric_list) + r"""}
            \toprule
            \textbf{Model}
        """
    
    for metric in metric_list:
        latex_table += f" & \\textbf{{{metric.replace('_', ' ')}}}"
    latex_table += r" \\"
    latex_table += r"        \midrule" + "\n"

    for model in model_list:
        row = [f"        {model}"]
        for metric in metric_list:
            score = results[model].get(metric, None)
            if score is not None:
                row.append(f"{score * 100:.2f}")
            else:
                row.append("N/A")
        
        latex_table += " & ".join(row) + r" \\\\ " + "\n"

    latex_table += r"        \bottomrule" + "\n"
    latex_table += r"    \end{tabular}}" + "\n"
    latex_table += r"\end{table*}" + "\n"
    
    return latex_table

def generate_markdown_table(model_list, metric_list, results):
    # Create headers - Model + VBench metrics only
    headers = ["Model"] + [metric.replace('_', ' ').title() for metric in metric_list]
    
    md_table = "| " + " | ".join(headers) + " |\n"
    md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    for model in model_list:
        row = [model]
        
        # Add VBench metric values only
        for metric in metric_list:
            score = results[model].get(metric, None)
            if score is not None:
                row.append(f"{score * 100:.2f}")
            else:
                row.append("N/A")
        
        md_table += "| " + " | ".join(row) + " |\n"
    
    return md_table

def load_vbench_results(model_list, prompt, base_dir, model_cat):
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
    for model in model_list:
        results[model] = {}
        for metric in metric_list:
            path = pathlib.Path(f"{evaluation_dir}/{model}/{metric}/")
            filename = next(path.glob("results_*_eval_results.json"), None)
            if filename:
                with open(filename, 'r') as f:
                    data = json.load(f)
                results[model][metric] = data[metric][0]
    return results

def generate_aggregated_markdown_table(models, prompts, all_results):
    """Generate an aggregated markdown table with VBench results from all prompts stacked as rows."""
    
    metric_list = [
        'subject_consistency',
        'temporal_flickering', 
        'aesthetic_quality',
        'dynamic_degree',
        'imaging_quality',
        'motion_smoothness'
    ]
    
    # Create headers - Model + VBench metrics
    headers = ["Model/Prompt"] + [metric.replace('_', ' ').title() for metric in metric_list]
    
    md_table = "| " + " | ".join(headers) + " |\n"
    md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    # Add data rows for each model, with prompts stacked as rows
    for model in models:
        # Add a separator row for each model
        md_table += f"| **{model}** |" + " |".join([" "] * len(metric_list)) + " |\n"
        
        for prompt in prompts:
            vbench_results = all_results.get(prompt, {})
            model_metrics = vbench_results.get(model, {})
            
            row = [f"└─ {prompt}"]
            
            # Add VBench metric values
            for metric in metric_list:
                score = model_metrics.get(metric)
                if score is not None:
                    row.append(f"{score * 100:.2f}")
                else:
                    row.append("N/A")
            
            md_table += "| " + " | ".join(row) + " |\n"
        
        # Add empty row between models for clarity
        md_table += "| |" + " |".join([" "] * len(metric_list)) + " |\n"
    
    return md_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VBench Results to LaTeX and Markdown Converter")
    parser.add_argument('--prompt', nargs='+', default=['gr1_object', 'gr1_env', 'gr1_behavior'],
                      help='Evaluation prompt(s)')
    parser.add_argument('--model_cat', type=str, default='cogvideox5b_i2v',
                      help='Model category')
    parser.add_argument('--models', nargs='+', default=None,
                      help='List of model names (if not provided, will auto-discover)')
    parser.add_argument('--auto_discover', action='store_true', default=True,
                      help='Automatically discover models from VBench results directories')
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
            print("Auto-discovering models from VBench results directories...")
            discovered_models = auto_discover_models(args.base_dir, prompt, args.model_cat)
            
            if not discovered_models:
                print("No models found in VBench results directories!")
                print(f"Checked: {args.base_dir}/results/vbench/{prompt}/{args.model_cat}")
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

        # Generate LaTeX table from VBench results
        print("\nGenerating LaTeX table from VBench results...")
        latex_output = generate_latex_table(
            model_list=model_list,
            prompt=prompt,
            model_cat=args.model_cat,
            base_dir=args.base_dir
        )
        print(latex_output)

        # Load VBench results for Markdown table
        print("\nLoading VBench results for Markdown table...")
        vbench_results = load_vbench_results(
            model_list=model_list,
            prompt=prompt,
            base_dir=args.base_dir,
            model_cat=args.model_cat
        )

        # Generate Markdown table
        print("\nGenerating Markdown table from VBench results...")
        metric_list = [
            'subject_consistency',
            'temporal_flickering',
            'aesthetic_quality',
            'dynamic_degree',
            'imaging_quality',
            'motion_smoothness'
        ]
        md_output = generate_markdown_table(
            model_list=model_list,
            metric_list=metric_list,
            results=vbench_results
        )
        print(md_output)

        # Store results for aggregation
        all_results[prompt] = vbench_results
        all_model_lists[prompt] = model_list

    # Generate aggregated markdown table
    print(f"\n{'='*80}")
    print("AGGREGATED VBENCH RESULTS ACROSS ALL PROMPTS")
    print(f"{'='*80}")
    
    # Get all unique models across all prompts
    all_models = set()
    for model_list in all_model_lists.values():
        all_models.update(model_list)
    all_models = sorted(list(all_models))
    
    if all_models and all_results:
        print("\nAggregated Markdown Table (Prompts stacked as rows):")
        aggregated_md = generate_aggregated_markdown_table(all_models, args.prompt, all_results)
        print(aggregated_md)
    
    print("Script completed successfully - VBench metrics processed for all prompts.")

