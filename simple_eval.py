#!/usr/bin/env python3
"""Simple evaluation script for video generation experiments."""

import os
import pandas as pd
import argparse

def find_experiments(output_folder):
    """Find all experiments from the simple folder structure."""
    experiments = []
    
    # Look for experiments.csv
    exp_log = os.path.join(output_folder, 'experiments.csv')
    if os.path.exists(exp_log):
        df = pd.read_csv(exp_log)
        for _, row in df.iterrows():
            exp_path = os.path.join(output_folder, row['name'])
            if os.path.exists(exp_path):
                experiments.append({
                    'name': row['name'],
                    'path': exp_path,
                    'method': row['method'],
                    'status': row['status']
                })
    
    return experiments

def run_evaluation(experiment_path, eval_type='vbench'):
    """Run evaluation on a single experiment."""
    if eval_type == 'vbench':
        cmd = f"cd /home/yjianhao/VBench && python evaluate.py --video_path {experiment_path}"
    elif eval_type == 'videophys':
        cmd = f"cd /home/yjianhao/VideoPhys && python evaluate.py --video_path {experiment_path}"
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")
    
    print(f"Running: {cmd}")
    os.system(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', required=True, help='Output folder containing experiments')
    parser.add_argument('--eval_type', default='vbench', choices=['vbench', 'videophys'])
    parser.add_argument('--method', help='Filter by method (vanilla, rejection, guidance)')
    parser.add_argument('--status', default='completed', help='Filter by status')
    
    args = parser.parse_args()
    
    # Find experiments
    experiments = find_experiments(args.output_folder)
    
    # Filter experiments
    if args.method:
        experiments = [e for e in experiments if e['method'] == args.method]
    if args.status:
        experiments = [e for e in experiments if e['status'] == args.status]
    
    print(f"Found {len(experiments)} experiments to evaluate")
    
    # Run evaluations
    for exp in experiments:
        print(f"\nEvaluating: {exp['name']}")
        run_evaluation(exp['path'], args.eval_type)

if __name__ == "__main__":
    main() 