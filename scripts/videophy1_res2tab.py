import json
import os
import pandas as pd
import argparse
import pathlib
import glob

def auto_discover_models(base_dir, prompt, model_cat):
    """Automatically discover evaluated models from results directories."""
    models = set()
    
    # Discover from VideoPhy1 results  
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

def generate_latex_table(model_list, prompt, model_cat, base_dir, csv_results):
    """Generate LaTeX table for VideoPhY1 results (no VBench metrics)."""
    
    latex_table = r"""
    \begin{table*}[t]
        \vspace{-4mm}
        \centering
        \tablestyle{3.6pt}{1.0}
        \caption{
            \textbf{VideoPhY1 Model Evaluation Metrics.} A comparison of physics understanding across different models.
        }
        \resizebox{0.6\linewidth}{!}{
        \begin{tabular}{lccccc}
            \toprule
            \textbf{Model} & \textbf{PC} & \textbf{SA} & \textbf{PCp} & \textbf{SAp} & \textbf{Joint} \\
            \midrule
        """

    for model in model_list:
        pc_avg = csv_results.get(model, {}).get('pc', "N/A")
        sa_avg = csv_results.get(model, {}).get('sa', "N/A")
        pcp_score = csv_results.get(model, {}).get('pcp', "N/A")
        sap_score = csv_results.get(model, {}).get('sap', "N/A")
        joint_score = csv_results.get(model, {}).get('joint', "N/A")
        
        row = [f"        {model}"]
        row.append(f"{pc_avg:.3f}" if isinstance(pc_avg, float) else "N/A")
        row.append(f"{sa_avg:.3f}" if isinstance(sa_avg, float) else "N/A")
        row.append(f"{pcp_score:.1f}" if isinstance(pcp_score, float) else "N/A")
        row.append(f"{sap_score:.1f}" if isinstance(sap_score, float) else "N/A")
        row.append(f"{joint_score:.1f}" if isinstance(joint_score, float) else "N/A")
        
        latex_table += " & ".join(row) + r" \\\\ " + "\n"

    latex_table += r"        \bottomrule" + "\n"
    latex_table += r"    \end{tabular}}" + "\n"
    latex_table += r"\end{table*}" + "\n"
    
    return latex_table

def generate_markdown_table(model_list, csv_results):
    """Generate markdown table for VideoPhY1 results only."""
    headers = ["Model", "PC", "SA", "PCp", "SAp", "Joint"]
    
    md_table = "| " + " | ".join(headers) + " |\n"
    md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    for model in model_list:
        pc_avg = csv_results.get(model, {}).get('pc', "N/A")
        sa_avg = csv_results.get(model, {}).get('sa', "N/A")
        pcp_score = csv_results.get(model, {}).get('pcp', "N/A")
        sap_score = csv_results.get(model, {}).get('sap', "N/A")
        joint_score = csv_results.get(model, {}).get('joint', "N/A")
        
        row = [model]
        row.append(f"{pc_avg:.3f}" if isinstance(pc_avg, float) else "N/A")
        row.append(f"{sa_avg:.3f}" if isinstance(sa_avg, float) else "N/A")
        row.append(f"{pcp_score:.1f}" if isinstance(pcp_score, float) else "N/A")
        row.append(f"{sap_score:.1f}" if isinstance(sap_score, float) else "N/A")
        row.append(f"{joint_score:.1f}" if isinstance(joint_score, float) else "N/A")
        
        md_table += "| " + " | ".join(row) + " |\n"
    return md_table

def calculate_csv_averages(models, base_dir, model_cat, prompt):
    results = {}
    for model in models:
        directory_path = f"{base_dir}/results/videophy1/{prompt}/{model_cat}/{model}"
        csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
        
        pc_averages = []
        sa_averages = []
        pc_proportions = []
        sa_proportions = []
        pc_dfs = []
        sa_dfs = []
        
        for csv_file in csv_files:
            # VideoPhY1 CSV has different format - scores are probabilities in 2nd column
            df = pd.read_csv(csv_file, header=None, names=['video_path', 'score'])
            # Calculate mean score (already a probability 0-1)
            avg = df['score'].mean()
            # Calculate proportion of videos with score >= 0.5 (equivalent to high confidence)
            proportion_ge_05 = (df['score'] >= 0.5).mean() * 100  # Convert to percentage
            if 'pc' in os.path.basename(csv_file):
                pc_averages.append(avg)
                pc_proportions.append(proportion_ge_05)
                pc_dfs.append(df)
            elif 'sa' in os.path.basename(csv_file):
                sa_averages.append(avg)
                sa_proportions.append(proportion_ge_05)
                sa_dfs.append(df)
        
        # Calculate joint proportion (both PC >= 0.5 and SA >= 0.5)
        joint_proportions = []
        if pc_dfs and sa_dfs:
            for pc_df, sa_df in zip(pc_dfs, sa_dfs):
                # Assume both dataframes have same order/index for videos
                if len(pc_df) == len(sa_df):
                    joint_ge_05 = ((pc_df['score'] >= 0.5) & (sa_df['score'] >= 0.5)).mean() * 100
                    joint_proportions.append(joint_ge_05)
        
        results[model] = {
            'pc': sum(pc_averages)/len(pc_averages) if pc_averages else None,
            'sa': sum(sa_averages)/len(sa_averages) if sa_averages else None,
            'pcp': sum(pc_proportions)/len(pc_proportions) if pc_proportions else None,
            'sap': sum(sa_proportions)/len(sa_proportions) if sa_proportions else None,
            'joint': sum(joint_proportions)/len(joint_proportions) if joint_proportions else None
        }
    
    return results

def generate_aggregated_markdown_table(models, prompts, all_csv_results):
    """Generate an aggregated markdown table with VideoPhY1 results from all prompts stacked as rows."""
    
    # Create headers - Model + VideoPhY1 metrics
    headers = ["Model/Prompt", "PC", "SA", "PCp", "SAp", "Joint"]
    
    md_table = "| " + " | ".join(headers) + " |\n"
    md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    # Add data rows for each model, with prompts stacked as rows
    for model in models:
        # Add a separator row for each model
        md_table += f"| **{model}** |" + " |".join([" "] * (len(headers) - 1)) + " |\n"
        
        for prompt in prompts:
            csv_results = all_csv_results.get(prompt, {})
            model_metrics = csv_results.get(model, {})
            
            row = [f"└─ {prompt}"]
            
            # Add VideoPhY1 metric values
            pc_avg = model_metrics.get('pc')
            sa_avg = model_metrics.get('sa')
            pcp_score = model_metrics.get('pcp')
            sap_score = model_metrics.get('sap')
            joint_score = model_metrics.get('joint')
            
            row.append(f"{pc_avg:.3f}" if pc_avg is not None else "N/A")
            row.append(f"{sa_avg:.3f}" if sa_avg is not None else "N/A")
            row.append(f"{pcp_score:.1f}" if pcp_score is not None else "N/A")
            row.append(f"{sap_score:.1f}" if sap_score is not None else "N/A")
            row.append(f"{joint_score:.1f}" if joint_score is not None else "N/A")
            
            md_table += "| " + " | ".join(row) + " |\n"
        
        # Add empty row between models for clarity
        md_table += "| |" + " |".join([" "] * (len(headers) - 1)) + " |\n"
    
    return md_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Results to LaTeX and Markdown Converter")
    parser.add_argument('--prompt', nargs='+', default=['search_t2v'],
                      help='Evaluation prompt(s)')
    parser.add_argument('--model_cat', type=str, default='cogvideox2b',
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
    all_csv_results = {}
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
            print(f"Discovered models: {discovered_models}")
            
            if not discovered_models:
                print("No models found in results directories!")
                print(f"Checked VideoPhy1: {args.base_dir}/results/videophy1/{prompt}/{args.model_cat}")
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

        # Calculate CSV averages from VIDEOPHY1 results
        print("\nCalculating VIDEOPHY1 averages...")
        csv_results = calculate_csv_averages(
            models=model_list,
            base_dir=args.base_dir,
            model_cat=args.model_cat,
            prompt=prompt
        )
        
        # Generate LaTeX table
        print("\nGenerating LaTeX table from VideoPhY1 results...")
        latex_output = generate_latex_table(
            model_list=model_list,
            prompt=prompt,
            model_cat=args.model_cat,
            base_dir=args.base_dir,
            csv_results=csv_results
        )
        print(latex_output)

        # Generate Markdown table
        print("\nGenerating Markdown table from VideoPhY1 results...")
        md_output = generate_markdown_table(
            model_list=model_list,
            csv_results=csv_results
        )
        print(md_output)

        print(f"\nVIDEOPHY1 Results Summary for {prompt}:")
        for model, metrics in csv_results.items():
            pc_avg = metrics['pc']
            sa_avg = metrics['sa']
            pcp_score = metrics['pcp']
            sap_score = metrics['sap']
            joint_score = metrics['joint']
            print(f"{model}:")
            print(f"  PC (average) = {pc_avg:.4f}" if pc_avg is not None else f"  PC (average) = N/A")
            print(f"  SA (average) = {sa_avg:.4f}" if sa_avg is not None else f"  SA (average) = N/A")
            print(f"  PCp (% videos ≥ 0.5) = {pcp_score:.2f}%" if pcp_score is not None else f"  PCp (% videos ≥ 0.5) = N/A")
            print(f"  SAp (% videos ≥ 0.5) = {sap_score:.2f}%" if sap_score is not None else f"  SAp (% videos ≥ 0.5) = N/A")
            print(f"  Joint (% videos PC≥0.5 & SA≥0.5) = {joint_score:.2f}%" if joint_score is not None else f"  Joint (% videos PC≥0.5 & SA≥0.5) = N/A")
        
        # Store results for aggregation
        all_csv_results[prompt] = csv_results
        all_model_lists[prompt] = model_list

    # Generate aggregated markdown table
    print(f"\n{'='*80}")
    print("AGGREGATED VIDEOPHY1 RESULTS ACROSS ALL PROMPTS")
    print(f"{'='*80}")
    
    # Get all unique models across all prompts
    all_models = set()
    for model_list in all_model_lists.values():
        all_models.update(model_list)
    all_models = sorted(list(all_models))
    
    if all_models and all_csv_results:
        print("\nAggregated Markdown Table (Prompts stacked as rows):")
        aggregated_md = generate_aggregated_markdown_table(all_models, args.prompt, all_csv_results)
        print(aggregated_md)
