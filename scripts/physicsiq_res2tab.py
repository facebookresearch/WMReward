import json
import os
import numpy as np
import pandas as pd
import argparse
import pathlib
import glob
import re

def auto_discover_models(base_dir, model_cat):
    """Automatically discover evaluated models from results directories."""
    models = set()
    
    # Discover from Physics-IQ results
    physicsiq_dir = f"{base_dir}/results/physics_iq/{model_cat}"
    if os.path.exists(physicsiq_dir):
        for model_dir in os.listdir(physicsiq_dir):
            if model_dir.endswith('.csv'):
                # Extract model name from CSV filename
                model_name = model_dir.replace('.csv', '')
                models.add(model_name)
    
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

def parse_list_of_floats(value):
    """Parse a string or list representing a list of floats and round each number to 4 decimal places."""
    if isinstance(value, str):
        if not (value.startswith("[") and value.endswith("]")):
            raise ValueError("Invalid string format for list of floats.")
        try:
            # Extract numbers from the string and convert to floats
            parsed_floats = [round(float(x), 4) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", value)]
            if not parsed_floats:
                raise ValueError("No valid floats found in the input string.")
            return parsed_floats
        except ValueError:
            raise ValueError("Failed to parse floats from the input string.")
    
    elif isinstance(value, list):
        if all(isinstance(x, (int, float)) for x in value):
            parsed_floats = [round(float(x), 4) for x in value]
            if not parsed_floats:
                raise ValueError("Input list is empty or contains no valid numeric values.")
            return parsed_floats
        else:
            raise ValueError("List contains non-numeric values.")
    
    raise TypeError("Input must be a string representing a list of floats or a list of numbers.")

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
        # if col in df.columns:
        df[col] = df[col].apply(parse_list_of_floats)
    
    
    # 1. MSE (lower is better) - using concatenate and mean like original
    total_sum_v1_mse = df.apply(
        lambda row: np.mean(np.concatenate([row[f"v1_mse_{view}"] for view in VIEWS])),
        axis=1
    ).mean()
    
    # 2. Spatiotemporal IoU (higher is better) - using concatenate and mean like original
    total_sum_spatiotemporal_iou_v1 = df.apply(
        lambda row: np.mean(np.concatenate([row[f"spatiotemporal_iou_v1_{view}"] for view in VIEWS])),
        axis=1
    ).mean()
    
    # 3. Spatial IoU (higher is better)
    total_sum_spatial_iou = df[[f"spatial_iou_v1_{view}" for view in VIEWS]].mean().mean()
    
    # 4. Weighted Spatial IoU (higher is better)
    total_sum_weighted_spatial_iou = df[[f"weighted_spatial_iou_v1_{view}" for view in VIEWS]].mean().mean()
    
    # Calculate variance metrics exactly like the original
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
    
    # Calculate final Physics-IQ score exactly like the original
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

# def calculate_physics_iq_metrics(csv_file_path):
#     """
#     Calculate Physics-IQ metrics from a CSV file and aggregate to the final score.
#     Uses a single ratio-of-sums normalization:
#         score = 100 * ((ST-IoU + S-IoU + WS-IoU - MSE) /
#                        (PV_ST-IoU + PV_S-IoU + PV_WS-IoU - PV_MSE))
#     """
#     # import numpy as np
#     # import pandas as pd

#     VIEWS = ["perspective-left", "perspective-center", "perspective-right"]
#     df = pd.read_csv(csv_file_path)

#     # Parse list-like columns (strings like "[...]" → list[float])
#     list_columns = (
#         [f"v1_mse_{view}" for view in VIEWS] +
#         [f"spatiotemporal_iou_v1_{view}" for view in VIEWS]
#     )
#     for col in list_columns:
#         if col in df.columns:
#             df[col] = df[col].apply(parse_list_of_floats)

#     # ---- Aggregate per-metric (model vs GT) ----
#     # 1) MSE (lower is better): mean over all frames (concat views) then mean over scenarios
#     total_sum_v1_mse = df.apply(
#         lambda row: np.mean(np.concatenate([row[f"v1_mse_{view}"] for view in VIEWS])),
#         axis=1
#     ).mean()

#     # 2) Spatiotemporal IoU (higher is better): mean over all frames (concat views) then mean over scenarios
#     total_sum_spatiotemporal_iou_v1 = df.apply(
#         lambda row: np.mean(np.concatenate([row[f"spatiotemporal_iou_v1_{view}"] for view in VIEWS])),
#         axis=1
#     ).mean()

#     # 3) Spatial IoU (higher is better): mean across views then across scenarios
#     total_sum_spatial_iou = df[[f"spatial_iou_v1_{view}" for view in VIEWS]].mean().mean()

#     # 4) Weighted Spatial IoU (higher is better): mean across views then across scenarios
#     total_sum_weighted_spatial_iou = df[[f"weighted_spatial_iou_v1_{view}" for view in VIEWS]].mean().mean()

#     # ---- Physical variance (real vs real) aggregates ----
#     # Variance MSE & ST-IoU are list-valued; Spatial / Weighted Spatial are usually scalars in CSVs.
#     def _pv_list_mean(colname):
#         return df[colname].apply(parse_list_of_floats).explode().astype(float).mean()

#     def _pv_scalar_or_list_mean(colname):
#         s = df[colname]
#         if s.dtype == object:
#             # try list parse; if it fails, coerce numerics
#             try:
#                 return s.apply(parse_list_of_floats).explode().astype(float).mean()
#             except Exception:
#                 s = pd.to_numeric(s, errors='coerce')
#         return float(s.mean())

#     # PV: MSE
#     if any(f"variance_mse_{v}" in df.columns for v in VIEWS):
#         physical_variance_mse = float(np.mean([_pv_list_mean(f"variance_mse_{v}") for v in VIEWS]))
#     else:
#         physical_variance_mse = 0.001

#     # PV: Spatiotemporal IoU
#     if any(f"variance_spatiotemporal_iou_{v}" in df.columns for v in VIEWS):
#         physical_variance_spatiotemporal_iou = float(np.mean([_pv_list_mean(f"variance_spatiotemporal_iou_{v}") for v in VIEWS]))
#     else:
#         physical_variance_spatiotemporal_iou = 0.001

#     # PV: Spatial IoU (column name without "_iou" in your CSVs)
#     if any(f"variance_spatial_{v}" in df.columns for v in VIEWS):
#         physical_variance_spatial = float(np.mean([_pv_scalar_or_list_mean(f"variance_spatial_{v}") for v in VIEWS]))
#     else:
#         physical_variance_spatial = 0.001

#     # PV: Weighted Spatial IoU (column name without "_iou" in your CSVs)
#     if any(f"variance_weighted_spatial_{v}" in df.columns for v in VIEWS):
#         physical_variance_weighted_spatial = float(np.mean([_pv_scalar_or_list_mean(f"variance_weighted_spatial_{v}") for v in VIEWS]))
#     else:
#         physical_variance_weighted_spatial = 0.001

#     # Floor PV terms to avoid zero/neg issues in denominator
#     physical_variance_mse = max(physical_variance_mse, 1e-6)
#     physical_variance_spatiotemporal_iou = max(physical_variance_spatiotemporal_iou, 1e-6)
#     physical_variance_spatial = max(physical_variance_spatial, 1e-6)
#     physical_variance_weighted_spatial = max(physical_variance_weighted_spatial, 1e-6)

#     # ---- Final Physics-IQ score: ratio-of-sums normalization ----
#     num = (
#         total_sum_spatiotemporal_iou_v1
#         + total_sum_spatial_iou
#         + total_sum_weighted_spatial_iou
#         - total_sum_v1_mse
#     )
#     den = (
#         physical_variance_spatiotemporal_iou
#         + physical_variance_spatial
#         + physical_variance_weighted_spatial
#         - physical_variance_mse
#     )
#     den = max(den, 1e-6)  # guard
#     final_score = 100.0 * (num / den)
#     final_score = round(max(min(final_score, 100.0), 0.0), 2)

#     return {
#         'mse': round(float(total_sum_v1_mse), 4),
#         'spatiotemporal_iou': round(float(total_sum_spatiotemporal_iou_v1), 4),
#         'spatial_iou': round(float(total_sum_spatial_iou), 4),
#         'weighted_spatial_iou': round(float(total_sum_weighted_spatial_iou), 4),
#         'final_score': final_score
#     }


def generate_latex_table(model_list, model_cat, base_dir, csv_results):
    """Generate LaTeX table for Physics-IQ results."""
    
    latex_table = r"""
    \begin{table*}[t]
        \vspace{-4mm}
        \centering
        \tablestyle{3.6pt}{1.0}
        \caption{
            \textbf{Physics-IQ Model Evaluation Metrics.} A comparison of physics understanding across different models.
        }
        \resizebox{0.8\linewidth}{!}{
        \begin{tabular}{lccccc}
            \toprule
            \textbf{Model} & \textbf{MSE↓} & \textbf{Spatio-temporal IoU↑} & \textbf{Spatial IoU↑} & \textbf{Weighted Spatial IoU↑} & \textbf{Final Score↑} \\
            \midrule
        """

    for model in model_list:
        mse = csv_results.get(model, {}).get('mse', "N/A")
        spatiotemporal_iou = csv_results.get(model, {}).get('spatiotemporal_iou', "N/A")
        spatial_iou = csv_results.get(model, {}).get('spatial_iou', "N/A")
        weighted_spatial_iou = csv_results.get(model, {}).get('weighted_spatial_iou', "N/A")
        final_score = csv_results.get(model, {}).get('final_score', "N/A")
        
        row = [f"        {model}"]
        row.append(f"{mse:.4f}" if isinstance(mse, float) else "N/A")
        row.append(f"{spatiotemporal_iou:.4f}" if isinstance(spatiotemporal_iou, float) else "N/A")
        row.append(f"{spatial_iou:.4f}" if isinstance(spatial_iou, float) else "N/A")
        row.append(f"{weighted_spatial_iou:.4f}" if isinstance(weighted_spatial_iou, float) else "N/A")
        row.append(f"{final_score:.2f}" if isinstance(final_score, float) else "N/A")
        
        latex_table += " & ".join(row) + r" \\\\ " + "\n"

    latex_table += r"        \bottomrule" + "\n"
    latex_table += r"    \end{tabular}}" + "\n"
    latex_table += r"\end{table*}" + "\n"
    
    return latex_table

def generate_markdown_table(model_list, csv_results):
    """Generate markdown table for Physics-IQ results."""
    headers = ["Model", "MSE↓", "Spatio-temporal IoU↑", "Spatial IoU↑", "Weighted Spatial IoU↑", "Final Score↑"]
    
    md_table = "| " + " | ".join(headers) + " |\n"
    md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    for model in model_list:
        mse = csv_results.get(model, {}).get('mse', "N/A")
        spatiotemporal_iou = csv_results.get(model, {}).get('spatiotemporal_iou', "N/A")
        spatial_iou = csv_results.get(model, {}).get('spatial_iou', "N/A")
        weighted_spatial_iou = csv_results.get(model, {}).get('weighted_spatial_iou', "N/A")
        final_score = csv_results.get(model, {}).get('final_score', "N/A")
        
        row = [model]
        row.append(f"{mse:.4f}" if isinstance(mse, float) else "N/A")
        row.append(f"{spatiotemporal_iou:.4f}" if isinstance(spatiotemporal_iou, float) else "N/A")
        row.append(f"{spatial_iou:.4f}" if isinstance(spatial_iou, float) else "N/A")
        row.append(f"{weighted_spatial_iou:.4f}" if isinstance(weighted_spatial_iou, float) else "N/A")
        row.append(f"{final_score:.2f}" if isinstance(final_score, float) else "N/A")
        
        md_table += "| " + " | ".join(row) + " |\n"
    return md_table

def calculate_csv_metrics(models, base_dir, model_cat):
    """Calculate Physics-IQ metrics from CSV files."""
    results = {}
    for model in models:
        csv_file_path = f"{base_dir}/results/physics_iq/{model_cat}/{model}.csv"
        if os.path.exists(csv_file_path):
            try:
                metrics = calculate_physics_iq_metrics(csv_file_path)
                results[model] = metrics
            except Exception as e:
                print(f"Error processing {csv_file_path}: {e}")
                results[model] = {
                    'mse': None,
                    'spatiotemporal_iou': None,
                    'spatial_iou': None,
                    'weighted_spatial_iou': None,
                    'final_score': None
                }
        else:
            print(f"CSV file not found: {csv_file_path}")
            results[model] = {
                'mse': None,
                'spatiotemporal_iou': None,
                'spatial_iou': None,
                'weighted_spatial_iou': None,
                'final_score': None
            }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics-IQ Results to LaTeX and Markdown Converter")
    parser.add_argument('--model_cat', type=str, default='Cosmos-Predict2-2B-Video2World',
                      help='Model category')
    parser.add_argument('--models', nargs='+', default=['cogvideox5b_i2v'],
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

    print(f"{'='*60}")
    print(f"Processing Physics-IQ results for model category: {args.model_cat}")
    print(f"{'='*60}")
    
    # Determine which models to process
    if args.models is None or args.auto_discover:
        print("Auto-discovering models from results directories...")
        discovered_models = auto_discover_models(args.base_dir, args.model_cat)
        print(f"Discovered models: {discovered_models}")
        
        if not discovered_models:
            print("No models found in results directories!")
            print(f"Checked Physics-IQ: {args.base_dir}/results/physics_iq/{args.model_cat}")
            exit(1)
            
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

    # Calculate metrics from Physics-IQ CSV results
    print("\nCalculating Physics-IQ metrics...")
    csv_results = calculate_csv_metrics(
        models=model_list,
        base_dir=args.base_dir,
        model_cat=args.model_cat
    )
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table from Physics-IQ results...")
    latex_output = generate_latex_table(
        model_list=model_list,
        model_cat=args.model_cat,
        base_dir=args.base_dir,
        csv_results=csv_results
    )
    print(latex_output)

    # Generate Markdown table
    print("\nGenerating Markdown table from Physics-IQ results...")
    md_output = generate_markdown_table(
        model_list=model_list,
        csv_results=csv_results
    )
    print(md_output)

    print(f"\nPhysics-IQ Results Summary:")
    for model, metrics in csv_results.items():
        mse = metrics['mse']
        spatiotemporal_iou = metrics['spatiotemporal_iou']
        spatial_iou = metrics['spatial_iou']
        weighted_spatial_iou = metrics['weighted_spatial_iou']
        final_score = metrics['final_score']
        
        print(f"{model}:")
        print(f"  MSE = {mse:.4f}" if mse is not None else f"  MSE = N/A")
        print(f"  Spatiotemporal IoU = {spatiotemporal_iou:.4f}" if spatiotemporal_iou is not None else f"  Spatiotemporal IoU = N/A")
        print(f"  Spatial IoU = {spatial_iou:.4f}" if spatial_iou is not None else f"  Spatial IoU = N/A")
        print(f"  Weighted Spatial IoU = {weighted_spatial_iou:.4f}" if weighted_spatial_iou is not None else f"  Weighted Spatial IoU = N/A")
        print(f"  Final Score = {final_score:.2f}" if final_score is not None else f"  Final Score = N/A")
