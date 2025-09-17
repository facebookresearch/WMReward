import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to the path to import physicsiq_res2tab
sys.path.append(str(Path(__file__).parent))
from physicsiq_res2tab import auto_discover_models, filter_models, calculate_csv_metrics

def extract_rejection_number(model_name):
    """Extract the rejection sample number from model name."""
    # Vanilla models represent BoN=1 (no rejection sampling)
    if 'vanilla' in model_name:
        return 1
    
    if 'reject' in model_name:
        # Extract number after 'reject'
        parts = model_name.split('reject')
        if len(parts) > 1:
            # Take the last part which should contain the rejection number
            remaining = parts[-1]
            # Extract digits from the beginning
            num_str = ''
            for char in remaining:
                if char.isdigit():
                    num_str += char
                else:
                    break
            try:
                return int(num_str) if num_str else None
            except ValueError:
                return None

    return None

def plot_bon_curves(base_dir, model_cat, include_patterns=None, exclude_patterns=None):
    """Plot Best-of-N curves showing Physics-IQ final score vs rejection samples."""
    
    print(f"Processing Physics-IQ results for model category: {model_cat}...")
    
    # Discover models
    models = auto_discover_models(base_dir, model_cat)
    if include_patterns or exclude_patterns:
        models = filter_models(models, include_patterns, exclude_patterns)
    else:
        # By default, include vanilla and rejection models for BoN curve
        default_patterns = ['vanilla', 'reject']
        models = filter_models(models, default_patterns, None)
    
    print(f"Found models: {models}")
    
    # Calculate Physics-IQ metrics
    results = calculate_csv_metrics(models, base_dir, model_cat)
    
    # Extract final score values with rejection counts
    plot_data = []
    for model, metrics in results.items():
        rejection_count = extract_rejection_number(model)
        if rejection_count is not None and metrics['final_score'] is not None:
            plot_data.append((rejection_count, metrics['final_score']))
            print(f"Added data point: {model} -> reject{rejection_count}, score={metrics['final_score']}")
    
    # Sort by rejection count
    plot_data.sort(key=lambda x: x[0])
    
    print(f"Plot data points: {plot_data}")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    if plot_data:  # Only plot if we have data
        rejection_counts, final_scores = zip(*plot_data)
        plt.plot(rejection_counts, final_scores, 
                marker='o', color='blue',
                linewidth=2, markersize=8,
                label='Physics-IQ Final Score')
        
        # Set x-axis to show integer values
        plt.xticks(sorted(set(rejection_counts)))
    else:
        print("No data points found for plotting!")
    
    plt.xlabel('Number of Rejection Samples', fontsize=12)
    plt.ylabel('Physics-IQ Final Score', fontsize=12)
    plt.title('Best-of-N Curve: Physics-IQ Performance vs Rejection Samples', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    return plt

def main():
    parser = argparse.ArgumentParser(description="Plot Best-of-N curves for Physics-IQ performance")
    parser.add_argument('--model_cat', type=str, default='cogvideox5b_i2v',
                      help='Model category')
    parser.add_argument('--include_patterns', nargs='+', default=None,
                      help='Only include models containing these patterns')
    parser.add_argument('--exclude_patterns', nargs='+', default=None,
                      help='Exclude models containing these patterns')
    parser.add_argument('--base_dir', type=str, 
                      default='/home/yjianhao/project/frame-guidance',
                      help='Base project directory')
    parser.add_argument('--save', type=str, default='physicsiq_bon_curve.png',
                      help='Save plot to file (e.g., physicsiq_bon_curve.png)')
    parser.add_argument('--show', action='store_true', default=True,
                      help='Show the plot')
    
    args = parser.parse_args()
    
    # Create the plot
    plt = plot_bon_curves(
        args.base_dir, 
        args.model_cat, 
        args.include_patterns, 
        args.exclude_patterns
    )
    

    plt.savefig(args.save, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {args.save}")

    # Show if requested
    # if args.show:
    #     plt.show()

if __name__ == "__main__":
    main()
