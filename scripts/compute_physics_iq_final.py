import os
import re
import sys
import json
import glob
import argparse
import pathlib
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader

OFFICIAL_CODE_DIR = "/home/yjianhao/project/PhysicsIQ/code"  # <- your repo code dir (no fallback)

# -------------- util: discover and import the official scoring function --------------
FUNC_NAME_PREFERENCES = [
    "calculate_physics_iq_metrics",
    "calculate_physics_iq_score",
    "calculate_iq_score",
    "compute_overall_score",
    "compute_final_score",
    "compute_score",
]

FILE_KEYWORDS_ORDER = [
    "calculate_iq", "calculate_physics_iq", "physics_iq",
    "run_physics_iq", "score", "evaluate", "eval",
]

def _score_file_rank(path: str) -> tuple:
    """Rank files by keyword priority to check most likely modules first."""
    fname = os.path.basename(path).lower()
    rank = []
    for i, kw in enumerate(FILE_KEYWORDS_ORDER):
        rank.append((i, kw in fname))
    # more True earlier is better; convert to a tuple that sorts correctly
    # (negate booleans so True sorts before False)
    return tuple((-int(b), i) for i, b in rank)

def _load_official_scorer(code_dir: str):
    if not os.path.isdir(code_dir):
        raise RuntimeError(f"Official code dir not found: {code_dir}")

    # collect candidate .py files
    py_files = []
    for root, _, files in os.walk(code_dir):
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))

    # prioritize likely files
    py_files.sort(key=_score_file_rank)

    # try each file; look for a preferred function name
    last_error = None
    for fpath in py_files:
        try:
            mod = SourceFileLoader(f"physics_iq_mod_{abs(hash(fpath))}", fpath).load_module()
        except Exception as e:
            last_error = e
            continue

        for fn in FUNC_NAME_PREFERENCES:
            if hasattr(mod, fn):
                func = getattr(mod, fn)
                if callable(func):
                    return func

    raise RuntimeError(
        "Could not locate an official scoring function in the repo. "
        f"Tried files under {code_dir}. Last import error: {last_error}"
    )

# load the official scorer (no fallback)
OFFICIAL_SCORER = _load_official_scorer(OFFICIAL_CODE_DIR)

def _call_official_scorer(csv_file_path: str):
    """
    Call the official scorer with flexible signature handling.
    NO FALLBACK math here—if we can't call it, we raise.
    """
    # Try: function(csv_path)
    try:
        out = OFFICIAL_SCORER(csv_file_path)
        return out
    except TypeError:
        pass

    # Try: function(DataFrame)
    try:
        df = pd.read_csv(csv_file_path)
        out = OFFICIAL_SCORER(df)
        return out
    except Exception as e:
        raise RuntimeError(
            "Failed to call the official PhysicsIQ scoring function with both "
            "csv_path and DataFrame. Please check the function signature in the repo."
        ) from e

def _normalize_official_output(out):
    """
    Normalize the official result into a dict with keys:
      mse, spatiotemporal_iou, spatial_iou, weighted_spatial_iou, final_score
    If the official function returns only a score, we surface just 'final_score'.
    """
    # If it's a pandas object, convert to dict
    if hasattr(out, "to_dict"):
        out = out.to_dict()

    # If it's JSON string
    if isinstance(out, str):
        try:
            out = json.loads(out)
        except Exception:
            pass

    # If it’s already a dict, try to map common key variants
    if isinstance(out, dict):
        # common aliases
        aliases = {
            "mse": ["mse", "MSE"],
            "spatiotemporal_iou": ["spatiotemporal_iou", "st_iou", "ST_IoU", "STIoU"],
            "spatial_iou": ["spatial_iou", "s_iou", "S_IoU", "SIoU"],
            "weighted_spatial_iou": ["weighted_spatial_iou", "ws_iou", "WS_IoU", "WSIoU"],
            "final_score": ["final_score", "score", "PhysicsIQ", "physics_iq", "iq_score", "IQ_Score"],
        }
        norm = {}
        for k, keys in aliases.items():
            for kk in keys:
                if kk in out:
                    norm[k] = float(out[kk]) if out[kk] is not None else None
                    break
            if k not in norm:
                norm[k] = None
        # require at least final_score
        if norm["final_score"] is None:
            # maybe the dict is just a mapping of metric->value without 'final_score'
            # in that case we cannot invent it—surface what we have and raise
            raise RuntimeError(
                "Official scorer returned a dict without 'final_score'. "
                "Please check the repo function to ensure it outputs the combined score."
            )
        return norm

    # If it’s a single numeric score
    if isinstance(out, (int, float, np.floating)):
        return {
            "mse": None,
            "spatiotemporal_iou": None,
            "spatial_iou": None,
            "weighted_spatial_iou": None,
            "final_score": float(out),
        }

    raise RuntimeError(
        "Official scorer returned an unsupported type. "
        f"Got: {type(out)}. Please adjust the wrapper to the repo’s actual return value."
    )

# ----------------- your existing helpers (unchanged) -----------------
def auto_discover_models(base_dir, model_cat):
    models = set()
    physicsiq_dir = f"{base_dir}/results/physics_iq/{model_cat}"
    if os.path.exists(physicsiq_dir):
        for fn in os.listdir(physicsiq_dir):
            if fn.endswith(".csv"):
                models.add(fn[:-4])
    return sorted(models)

def filter_models(models, include_patterns=None, exclude_patterns=None):
    if include_patterns:
        models = [m for m in models if any(p in m for p in include_patterns)]
    if exclude_patterns:
        models = [m for m in models if not any(p in m for p in exclude_patterns)]
    return models

_num_pat = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
def parse_list_of_floats(value):
    if isinstance(value, str):
        if not (value.startswith("[") and value.endswith("]")):
            raise ValueError("Invalid string format for list of floats.")
        vals = _num_pat.findall(value)
        if not vals:
            raise ValueError("No valid floats found in the input string.")
        return [round(float(x), 4) for x in vals]
    elif isinstance(value, list):
        if not all(isinstance(x, (int, float)) for x in value):
            raise ValueError("List contains non-numeric values.")
        return [round(float(x), 4) for x in value]
    raise TypeError("Input must be a string or a list of numbers.")

def calculate_physics_iq_metrics(csv_file_path):
    """
    STRICT: Use official code only. No fallback.
    """
    out = _call_official_scorer(csv_file_path)
    return _normalize_official_output(out)

def generate_latex_table(model_list, model_cat, base_dir, csv_results):
    latex = r"""
\begin{table*}[t]
    \vspace{-4mm}
    \centering
    \tablestyle{3.6pt}{1.0}
    \caption{\textbf{Physics-IQ Model Evaluation Metrics.} A comparison of physics understanding across different models.}
    \resizebox{0.8\linewidth}{!}{
    \begin{tabular}{lccccc}
        \toprule
        \textbf{Model} & \textbf{MSE↓} & \textbf{Spatio-temporal IoU↑} & \textbf{Spatial IoU↑} & \textbf{Weighted Spatial IoU↑} & \textbf{Final Score↑} \\
        \midrule
"""
    for model in model_list:
        m = csv_results.get(model, {})
        row = [
            f"        {model}",
            f"{m.get('mse', 'N/A'):.4f}" if isinstance(m.get('mse'), float) else "N/A",
            f"{m.get('spatiotemporal_iou', 'N/A'):.4f}" if isinstance(m.get('spatiotemporal_iou'), float) else "N/A",
            f"{m.get('spatial_iou', 'N/A'):.4f}" if isinstance(m.get('spatial_iou'), float) else "N/A",
            f"{m.get('weighted_spatial_iou', 'N/A'):.4f}" if isinstance(m.get('weighted_spatial_iou'), float) else "N/A",
            f"{m.get('final_score', 'N/A'):.2f}" if isinstance(m.get('final_score'), float) else "N/A",
        ]
        latex += " & ".join(row) + r" \\ " + "\n"
    latex += r"""        \bottomrule
    \end{tabular}}
\end{table*}
"""
    return latex

def generate_markdown_table(model_list, csv_results):
    headers = ["Model", "MSE↓", "Spatio-temporal IoU↑", "Spatial IoU↑", "Weighted Spatial IoU↑", "Final Score↑"]
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for model in model_list:
        m = csv_results.get(model, {})
        row = [
            model,
            f"{m.get('mse', 'N/A'):.4f}" if isinstance(m.get('mse'), float) else "N/A",
            f"{m.get('spatiotemporal_iou', 'N/A'):.4f}" if isinstance(m.get('spatiotemporal_iou'), float) else "N/A",
            f"{m.get('spatial_iou', 'N/A'):.4f}" if isinstance(m.get('spatial_iou'), float) else "N/A",
            f"{m.get('weighted_spatial_iou', 'N/A'):.4f}" if isinstance(m.get('weighted_spatial_iou'), float) else "N/A",
            f"{m.get('final_score', 'N/A'):.2f}" if isinstance(m.get('final_score'), float) else "N/A",
        ]
        md += "| " + " | ".join(row) + " |\n"
    return md

def calculate_csv_metrics(models, base_dir, model_cat):
    results = {}
    for model in models:
        csv_file_path = f"{base_dir}/results/physics_iq/{model_cat}/{model}.csv"
        if os.path.exists(csv_file_path):
            try:
                results[model] = calculate_physics_iq_metrics(csv_file_path)
            except Exception as e:
                # strictly no fallback; surface the error
                raise
        else:
            print(f"CSV file not found: {csv_file_path}")
            results[model] = dict(mse=None, spatiotemporal_iou=None, spatial_iou=None,
                                  weighted_spatial_iou=None, final_score=None)
    return results

def main():
    parser = argparse.ArgumentParser(description="Physics-IQ Results to LaTeX and Markdown Converter (official scorer only)")
    parser.add_argument('--model_cat', type=str, default='Cosmos-Predict2-2B-Video2World')
    parser.add_argument('--models', nargs='+', default=['cogvideox5b_i2v'])
    parser.add_argument('--auto_discover', action='store_true', default=True)
    parser.add_argument('--include_patterns', nargs='+', default=None)
    parser.add_argument('--exclude_patterns', nargs='+', default=None)
    parser.add_argument('--base_dir', type=str, default='/home/yjianhao/project/frame-guidance')
    args = parser.parse_args()

    print("="*60)
    print(f"Processing Physics-IQ results for model category: {args.model_cat}")
    print("="*60)

    # STRICT: require the official code dir to exist
    if not os.path.isdir(OFFICIAL_CODE_DIR):
        raise RuntimeError(f"Official code dir not found: {OFFICIAL_CODE_DIR}")

    if args.models is None or args.auto_discover:
        print("Auto-discovering models from results directories...")
        discovered = auto_discover_models(args.base_dir, args.model_cat)
        print(f"Discovered models: {discovered}")
        if not discovered:
            print("No models found in results directories!")
            print(f"Checked Physics-IQ: {args.base_dir}/results/physics_iq/{args.model_cat}")
            raise SystemExit(1)
        if args.include_patterns or args.exclude_patterns:
            discovered = filter_models(discovered, args.include_patterns, args.exclude_patterns)
            print(f"After filtering: {discovered}")
        model_list = list(set((args.models or []) + discovered))
    else:
        model_list = args.models or []

    if not model_list:
        print("No models to process!")
        raise SystemExit(1)

    print(f"\nUsing official scorer from: {OFFICIAL_CODE_DIR}")
    print(f"Processing models: {model_list}")

    print("\nCalculating Physics-IQ metrics...")
    csv_results = calculate_csv_metrics(model_list, args.base_dir, args.model_cat)

    print("\nGenerating LaTeX table from Physics-IQ results...")
    print(generate_latex_table(model_list, args.model_cat, args.base_dir, csv_results))

    print("\nGenerating Markdown table from Physics-IQ results...")
    print(generate_markdown_table(model_list, csv_results))

    print("\nPhysics-IQ Results Summary:")
    for model, m in csv_results.items():
        print(f"{model}:")
        print(f"  MSE = {m['mse']:.4f}" if m['mse'] is not None else "  MSE = N/A")
        print(f"  Spatiotemporal IoU = {m['spatiotemporal_iou']:.4f}" if m['spatiotemporal_iou'] is not None else "  Spatiotemporal IoU = N/A")
        print(f"  Spatial IoU = {m['spatial_iou']:.4f}" if m['spatial_iou'] is not None else "  Spatial IoU = N/A")
        print(f"  Weighted Spatial IoU = {m['weighted_spatial_iou']:.4f}" if m['weighted_spatial_iou'] is not None else "  Weighted Spatial IoU = N/A")
        print(f"  Final Score = {m['final_score']:.2f}" if m['final_score'] is not None else "  Final Score = N/A")

if __name__ == "__main__":
    main()
