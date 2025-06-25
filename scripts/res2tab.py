# import json
# import os
# import pandas as pd
# import argparse
# import pathlib
# import glob

# def generate_latex_table(model_list, prompt, metrics_to_average, base_dir, csv_results):
#     # Setup paths
#     evaluation_dir = f"{base_dir}/results/vbench"
    
#     # Define metrics
#     metric_list = [
#         'subject_consistency',
#         'temporal_flickering',
#         'aesthetic_quality',
#         'dynamic_degree',
#         'imaging_quality',
#         'motion_smoothness'
#     ]

#     # Load results
#     results = {}
#     for model in model_list:
#         results[model] = {}
#         for metric in metric_list:
#             path = pathlib.Path(f"{evaluation_dir}/{prompt}/{model}/{metric}/")
#             filename = next(path.glob("results_*_eval_results.json"), None)
            
#             if filename:
#                 with open(filename, 'r') as f:
#                     data = json.load(f)
#                 results[model][metric] = data[metric][0]

#     # Generate LaTeX table
#     latex_table = r"""
# \begin{table*}[t]
#     \vspace{-4mm}
#     \centering
#     \tablestyle{3.6pt}{1.0}
#     \caption{
#         \textbf{Model Evaluation Metrics.} A comparison of metrics across different models.
#     }
#     \resizebox{1.0\linewidth}{!}{
#     \begin{tabular}{l""" + "c" * (len(metric_list) + 3) + r"""}
#         \toprule
# """

#     # Table header
#     latex_table += r"        \textbf{Model}"
#     for metric in metric_list:
#         latex_table += f" & \\textbf{{{metric.replace('_', ' ')}}}"
#     latex_table += r" & \textbf{Average} & \textbf{PC} & \textbf{SA} \\\\"
#     latex_table += r"        \midrule" + "\n"

#     # Table rows
#     for model in model_list:
#         row = [f"        {model}"]
#         scores = []
#         for metric in metric_list:
#             score = results[model].get(metric, None)
#             if score is not None:
#                 row.append(f"{score * 100:.2f}")
#                 scores.append(score)
#             else:
#                 row.append("N/A")
        
#         # Calculate average
#         valid_scores = [s for s in scores if s is not None]
#         avg = sum(valid_scores)/len(valid_scores)*100 if valid_scores else "N/A"
#         row.append(f"{avg:.2f}" if isinstance(avg, float) else "N/A")
        
#         # Add PC and SA averages
#         pc_avg = csv_results.get(model, {}).get('pc', "N/A")
#         sa_avg = csv_results.get(model, {}).get('sa', "N/A")
#         row.append(f"{pc_avg:.2f}" if isinstance(pc_avg, float) else "N/A")
#         row.append(f"{sa_avg:.2f}" if isinstance(sa_avg, float) else "N/A")
        
#         latex_table += " & ".join(row) + r" \\\\ " + "\n"

#     latex_table += r"        \bottomrule" + "\n"
#     latex_table += r"    \end{tabular}}" + "\n"
#     latex_table += r"\end{table*}" + "\n"
    
#     return latex_table

# def calculate_csv_averages(models, base_dir):
#     results = {}
#     for model in models:
#         directory_path = f"{base_dir}/results/videophy/{model}"
#         csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
        
#         pc_averages = []
#         sa_averages = []
#         for csv_file in csv_files:
#             df = pd.read_csv(csv_file)
#             avg = df['score'].mean()
#             if 'pc' in os.path.basename(csv_file):
#                 pc_averages.append(avg)
#             elif 'sa' in os.path.basename(csv_file):
#                 sa_averages.append(avg)
#             print(f"Model: {model}, File: {os.path.basename(csv_file)}, Average: {avg:.4f}")
        
#         results[model] = {
#             'pc': sum(pc_averages)/len(pc_averages) if pc_averages else None,
#             'sa': sum(sa_averages)/len(sa_averages) if sa_averages else None
#         }
    
#     return results

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Results to LaTeX Converter")
#     parser.add_argument('--prompt', type=str, default='subject_consistency',
#                       help='Evaluation prompt')
#     parser.add_argument('--models', nargs='+', default=["wan_vanilla", "wan_rej_cw2", "wan_rej_cw4"],
#                       help='List of model names')
#     parser.add_argument('--base_dir', type=str, 
#                       default='/home/yjianhao/project/video_guidance',
#                       help='Base project directory')
#     args = parser.parse_args()

#     # Calculate CSV averages from VIDEOPHY results
#     print("\nCalculating VIDEOPHY averages...")
#     csv_results = calculate_csv_averages(
#         models=args.models,
#         base_dir=args.base_dir
#     )
    
#     # Generate LaTeX table from JSON results
#     print("\nGenerating LaTeX table from VBench results...")
#     latex_output = generate_latex_table(
#         model_list=args.models,
#         prompt=args.prompt,
#         metrics_to_average=[],
#         base_dir=args.base_dir,
#         csv_results=csv_results
#     )
#     print(latex_output)

#     print("\nVIDEOPHY Results Summary:")
#     for model, metrics in csv_results.items():
#         pc_avg = metrics['pc']
#         sa_avg = metrics['sa']
#         print(f"{model}: PC Average = {pc_avg:.4f}" if pc_avg is not None else f"{model}: PC Average = N/A", 
#               f"SA Average = {sa_avg:.4f}" if sa_avg is not None else "SA Average = N/A")


import json
import os
import pandas as pd
import argparse
import pathlib
import glob

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
    \begin{tabular}{l""" + "c" * (len(metric_list) + 3) + r"""}
        \toprule
        \textbf{Model}"""
    
    for metric in metric_list:
        latex_table += f" & \\textbf{{{metric.replace('_', ' ')}}}"
    latex_table += r" & \textbf{Average} & \textbf{PC} & \textbf{SA} \\"
    latex_table += r"        \midrule" + "\n"

    for model in model_list:
        row = [f"        {model}"]
        scores = []
        for metric in metric_list:
            score = results[model].get(metric, None)
            if score is not None:
                row.append(f"{score * 100:.2f}")
                scores.append(score)
            else:
                row.append("N/A")
        
        valid_scores = [s for s in scores if s is not None]
        avg = sum(valid_scores)/len(valid_scores)*100 if valid_scores else "N/A"
        row.append(f"{avg:.2f}" if isinstance(avg, float) else "N/A")
        
        pc_avg = csv_results.get(model, {}).get('pc', "N/A")
        sa_avg = csv_results.get(model, {}).get('sa', "N/A")
        row.append(f"{pc_avg:.2f}" if isinstance(pc_avg, float) else "N/A")
        row.append(f"{sa_avg:.2f}" if isinstance(sa_avg, float) else "N/A")
        
        latex_table += " & ".join(row) + r" \\\\ " + "\n"

    latex_table += r"        \bottomrule" + "\n"
    latex_table += r"    \end{tabular}}" + "\n"
    latex_table += r"\end{table*}" + "\n"
    
    return latex_table

def generate_markdown_table(model_list, metric_list, results, csv_results):
    header = ["Model"] + [metric.replace('_', ' ').title() for metric in metric_list] + ["Average", "PC", "SA"]
    md_table = "| " + " | ".join(header) + " |\n"
    md_table += "| " + " | ".join(["---"] * len(header)) + " |\n"
    
    for model in model_list:
        row = [model]
        scores = []
        for metric in metric_list:
            score = results[model].get(metric, None)
            if score is not None:
                row.append(f"{score * 100:.2f}")
                scores.append(score)
            else:
                row.append("N/A")
        
        valid_scores = [s for s in scores if s is not None]
        avg = sum(valid_scores)/len(valid_scores)*100 if valid_scores else "N/A"
        row.append(f"{avg:.2f}" if isinstance(avg, float) else "N/A")
        
        pc_avg = csv_results.get(model, {}).get('pc', "N/A")
        sa_avg = csv_results.get(model, {}).get('sa', "N/A")
        row.append(f"{pc_avg:.2f}" if isinstance(pc_avg, float) else "N/A")
        row.append(f"{sa_avg:.2f}" if isinstance(sa_avg, float) else "N/A")
        
        md_table += "| " + " | ".join(row) + " |\n"
    return md_table

def calculate_csv_averages(models, base_dir):
    results = {}
    for model in models:
        directory_path = f"{base_dir}/results/videophy/{model}"
        csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
        
        pc_averages = []
        sa_averages = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            avg = df['score'].mean()
            if 'pc' in os.path.basename(csv_file):
                pc_averages.append(avg)
            elif 'sa' in os.path.basename(csv_file):
                sa_averages.append(avg)
        
        results[model] = {
            'pc': sum(pc_averages)/len(pc_averages) if pc_averages else None,
            'sa': sum(sa_averages)/len(sa_averages) if sa_averages else None
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
    # parser.add_argument('--models', nargs='+', default=["wan_vanilla", "wan_rej_cw2", "wan_rej_cw4", "wan_rej_cw8", "wan_rej_cw16", "wan_rej_w4c2s2", "wan_rej_w8c4s4", "wan_rej_w8c6s2", "wan_rej_w16c10s6"],
    #                   help='List of model names')
    parser.add_argument('--models', nargs='+', default=["wan_vanilla", "wan_rej_w8c4s2", "wan_rej_w8c4s2_max"],
                      help='List of model names')
    parser.add_argument('--base_dir', type=str, 
                      default='/home/yjianhao/project/video_guidance',
                      help='Base project directory')
    args = parser.parse_args()

    # Calculate CSV averages from VIDEOPHY results
    print("Calculating VIDEOPHY averages...")
    csv_results = calculate_csv_averages(
        models=args.models,
        base_dir=args.base_dir
    )
    
    # Generate LaTeX table from JSON results
    print("\nGenerating LaTeX table from VBench results...")
    latex_output = generate_latex_table(
        model_list=args.models,
        prompt=args.prompt,
        metrics_to_average=[],
        base_dir=args.base_dir,
        csv_results=csv_results
    )
    print(latex_output)

    # Load VBench results for Markdown table
    print("\nLoading VBench results for Markdown table...")
    vbench_results = load_vbench_results(
        model_list=args.models,
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
        model_list=args.models,
        metric_list=metric_list,
        results=vbench_results,
        csv_results=csv_results
    )
    print(md_output)

    print("\nVIDEOPHY Results Summary:")
    for model, metrics in csv_results.items():
        pc_avg = metrics['pc']
        sa_avg = metrics['sa']
        print(f"{model}: PC Average = {pc_avg:.4f}" if pc_avg is not None else f"{model}: PC Average = N/A", 
              f"SA Average = {sa_avg:.4f}" if sa_avg is not None else "SA Average = N/A")