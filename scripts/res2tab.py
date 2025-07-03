import json
import os
import pandas as pd
import argparse
import pathlib
import glob

def auto_discover_models(base_dir, prompt):
    """Automatically discover evaluated models from results directories."""
    models = set()
    
    # Discover from VBench results
    vbench_dir = f"{base_dir}/results/vbench/{prompt}"
    if os.path.exists(vbench_dir):
        for model_dir in os.listdir(vbench_dir):
            if os.path.isdir(os.path.join(vbench_dir, model_dir)):
                models.add(model_dir)
    
    # Discover from VideoPhy results  
    videophy_dir = f"{base_dir}/results/videophy"
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

def generate_latex_table(model_list, prompt, metrics_to_average, base_dir, csv_results):
    evaluation_dir = f"{base_dir}/results/vbench"
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
            path = pathlib.Path(f"{evaluation_dir}/{prompt}/{model}/{metric}/")
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
        \textbf{Model Evaluation Metrics.} A comparison of metrics across different models.
    }
    \resizebox{1.0\linewidth}{!}{
    \begin{tabular}{l""" + "c" * (len(metric_list) + 4) + r"""}
        \toprule
        \textbf{Model}"""
    
    for metric in metric_list:
        latex_table += f" & \\textbf{{{metric.replace('_', ' ')}}}"
    latex_table += r" & \textbf{PC} & \textbf{SA} & \textbf{PCp} & \textbf{SAp} \\"
    latex_table += r"        \midrule" + "\n"

    for model in model_list:
        row = [f"        {model}"]
        for metric in metric_list:
            score = results[model].get(metric, None)
            if score is not None:
                row.append(f"{score * 100:.2f}")
            else:
                row.append("N/A")
        
        pc_avg = csv_results.get(model, {}).get('pc', "N/A")
        sa_avg = csv_results.get(model, {}).get('sa', "N/A")
        pcp_score = csv_results.get(model, {}).get('pcp', "N/A")
        sap_score = csv_results.get(model, {}).get('sap', "N/A")
        
        row.append(f"{pc_avg:.2f}" if isinstance(pc_avg, float) else "N/A")
        row.append(f"{sa_avg:.2f}" if isinstance(sa_avg, float) else "N/A")
        row.append(f"{pcp_score:.2f}" if isinstance(pcp_score, float) else "N/A")
        row.append(f"{sap_score:.2f}" if isinstance(sap_score, float) else "N/A")
        
        latex_table += " & ".join(row) + r" \\\\ " + "\n"

    latex_table += r"        \bottomrule" + "\n"
    latex_table += r"    \end{tabular}}" + "\n"
    latex_table += r"\end{table*}" + "\n"
    
    return latex_table

def generate_markdown_table(model_list, metric_list, results, csv_results):
    header = ["Model"] + [metric.replace('_', ' ').title() for metric in metric_list] + ["PC", "SA", "PCp", "SAp"]
    md_table = "| " + " | ".join(header) + " |\n"
    md_table += "| " + " | ".join(["---"] * len(header)) + " |\n"
    
    for model in model_list:
        row = [model]
        for metric in metric_list:
            score = results[model].get(metric, None)
            if score is not None:
                row.append(f"{score * 100:.2f}")
            else:
                row.append("N/A")
        
        pc_avg = csv_results.get(model, {}).get('pc', "N/A")
        sa_avg = csv_results.get(model, {}).get('sa', "N/A")
        pcp_score = csv_results.get(model, {}).get('pcp', "N/A")
        sap_score = csv_results.get(model, {}).get('sap', "N/A")
        
        row.append(f"{pc_avg:.2f}" if isinstance(pc_avg, float) else "N/A")
        row.append(f"{sa_avg:.2f}" if isinstance(sa_avg, float) else "N/A")
        row.append(f"{pcp_score:.2f}" if isinstance(pcp_score, float) else "N/A")
        row.append(f"{sap_score:.2f}" if isinstance(sap_score, float) else "N/A")
        
        md_table += "| " + " | ".join(row) + " |\n"
    return md_table

def calculate_csv_averages(models, base_dir):
    results = {}
    for model in models:
        directory_path = f"{base_dir}/results/videophy/{model}"
        csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
        
        pc_averages = []
        sa_averages = []
        pc_proportions = []
        sa_proportions = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # Calculate mean score
            avg = df['score'].mean()
            # Calculate proportion of videos rated >= 4
            proportion_ge_4 = (df['score'] >= 4).mean() * 100  # Convert to percentage
            if 'pc' in os.path.basename(csv_file):
                pc_averages.append(avg)
                pc_proportions.append(proportion_ge_4)
            elif 'sa' in os.path.basename(csv_file):
                sa_averages.append(avg)
                sa_proportions.append(proportion_ge_4)
        
        results[model] = {
            'pc': sum(pc_averages)/len(pc_averages) if pc_averages else None,
            'sa': sum(sa_averages)/len(sa_averages) if sa_averages else None,
            'pcp': sum(pc_proportions)/len(pc_proportions) if pc_proportions else None,
            'sap': sum(sa_proportions)/len(sa_proportions) if sa_proportions else None
        }
    
    return results

def load_vbench_results(model_list, prompt, base_dir):
    evaluation_dir = f"{base_dir}/results/vbench"
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
            path = pathlib.Path(f"{evaluation_dir}/{prompt}/{model}/{metric}/")
            filename = next(path.glob("results_*_eval_results.json"), None)
            if filename:
                with open(filename, 'r') as f:
                    data = json.load(f)
                results[model][metric] = data[metric][0]
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Results to LaTeX and Markdown Converter")
    parser.add_argument('--prompt', type=str, default='subject_consistency',
                      help='Evaluation prompt')
    parser.add_argument('--models', nargs='+', default=None,
                      help='List of model names (if not provided, will auto-discover)')
    parser.add_argument('--auto_discover', action='store_true', default=True,
                      help='Automatically discover models from results directories')
    parser.add_argument('--include_patterns', nargs='+', default=None,
                      help='Only include models containing these patterns')
    parser.add_argument('--exclude_patterns', nargs='+', default=None,
                      help='Exclude models containing these patterns')
    parser.add_argument('--base_dir', type=str, 
                      default='/home/yjianhao/project/video_guidance',
                      help='Base project directory')
    args = parser.parse_args()

    # Determine which models to process
    if args.models is None or args.auto_discover:
        print("Auto-discovering models from results directories...")
        discovered_models = auto_discover_models(args.base_dir, args.prompt)
        
        if not discovered_models:
            print("No models found in results directories!")
            print(f"Checked VBench: {args.base_dir}/results/vbench/{args.prompt}")
            print(f"Checked VideoPhy: {args.base_dir}/results/videophy")
            exit(1)
            
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
        print("No models to process!")
        exit(1)
        
    print(f"\nProcessing models: {model_list}")

    # Calculate CSV averages from VIDEOPHY results
    print("\nCalculating VIDEOPHY averages...")
    csv_results = calculate_csv_averages(
        models=model_list,
        base_dir=args.base_dir
    )
    
    # Generate LaTeX table from JSON results
    print("\nGenerating LaTeX table from VBench results...")
    latex_output = generate_latex_table(
        model_list=model_list,
        prompt=args.prompt,
        metrics_to_average=[],
        base_dir=args.base_dir,
        csv_results=csv_results
    )
    print(latex_output)

    # Load VBench results for Markdown table
    print("\nLoading VBench results for Markdown table...")
    vbench_results = load_vbench_results(
        model_list=model_list,
        prompt=args.prompt,
        base_dir=args.base_dir
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
        results=vbench_results,
        csv_results=csv_results
    )
    print(md_output)

    print("\nVIDEOPHY Results Summary:")
    for model, metrics in csv_results.items():
        pc_avg = metrics['pc']
        sa_avg = metrics['sa']
        pcp_score = metrics['pcp']
        sap_score = metrics['sap']
        print(f"{model}:")
        print(f"  PC (average) = {pc_avg:.4f}" if pc_avg is not None else f"  PC (average) = N/A")
        print(f"  SA (average) = {sa_avg:.4f}" if sa_avg is not None else f"  SA (average) = N/A")
        print(f"  PCp (% videos ≥ 4) = {pcp_score:.2f}%" if pcp_score is not None else f"  PCp (% videos ≥ 4) = N/A")
        print(f"  SAp (% videos ≥ 4) = {sap_score:.2f}%" if sap_score is not None else f"  SAp (% videos ≥ 4) = N/A")