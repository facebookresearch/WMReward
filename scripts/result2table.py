import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pathlib

prompt = 'subject_consistency'

modelclass = 'wan'

evaluation_dir = f"./results/{modelclass}/{prompt}"
model_list = [d for d in os.listdir(evaluation_dir) if os.path.isdir(os.path.join(evaluation_dir, d))]

# Sort the model_list to ensure consistent ordering
model_list.sort()

# You can keep your specific metric_list or generate it dynamically as well
metric_list = [
    'subject_consistency',
    # 'overall_consistency',
    # 'background_consistency',
    'dynamic_degree',
    'imaging_quality',
    'aesthetic_quality',
    'motion_smoothness',
    'temporal_flickering',
    # 'spatial_relationship'
    # 'temporal_style'
]

# Initialize a dictionary to store the results
results = {}

# Iterate over models and metrics to populate results
for model in model_list:
    results[model] = {}
    for metric in metric_list:
        # # Construct the file path
        # filename = f"./results/{modelclass}/{prompt}/{model}/metric/results_{xx}_eval_results.json"
        path = pathlib.Path(f"./results/{modelclass}/{prompt}/{model}/{metric}/")
        filename = next(path.glob("results_*_eval_results.json"), None)

        # Load the JSON data from a file
        with open(filename, 'r') as f:
            data = json.load(f)

        # Extract the overall score
        overall_score = data[metric][0]
        results[model][metric] = overall_score

# Define metrics to average
metrics_to_average = [
    # 'subject_consistency',
    # 'dynamic_degree',
    # 'temporal_flickering',
    # 'motion_smoothness'
]

# Modify the LaTeX table header to include the average column
latex_table = """
\\begin{table*}[t]
    \\vspace{-4mm}
    \\centering
    \\tablestyle{3.6pt}{1.0}
    \\caption{
        \\textbf{Model Evaluation Metrics.} A comparison of metrics across different models.
    }
    \\resizebox{1.0\\linewidth}{!}{
    \\begin{tabular}{l""" + "c" * (len(metric_list) + 1) + """}
        \\toprule
"""

# Header row with metric names
latex_table += "        \\textbf{Model}"
for metric in metric_list:
    metric_name = metric.replace('_', '\\_')
    latex_table += f" & \\textbf{{{metric_name}}}"
latex_table += " & \\textbf{Average} \\\\\n        \\midrule\n"

# Data rows for each model
for model in model_list:
    latex_table += f"        {model}"
    for metric in metric_list:
        score = results[model].get(metric, 'N/A')
        if score is None or score == 'N/A':
            score_str = 'N/A'
        else:
            score = score * 100
            score_str = f"{score:.2f}"
        latex_table += f" & {score_str}"
    
    # Calculate and add average
    valid_scores = [results[model].get(metric) for metric in metrics_to_average 
                   if results[model].get(metric) is not None]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores) * 100
        latex_table += f" & {avg_score:.2f}"
    else:
        latex_table += " & N/A"
    latex_table += " \\\\\n"

latex_table += """        \\bottomrule
    \\end{tabular}}
\\end{table*}
"""

print(latex_table)