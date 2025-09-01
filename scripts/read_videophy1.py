import os
import json
import pandas as pd
import argparse
import glob


def auto_discover_models(base_dir, prompt, model_cat):
    """Automatically discover evaluated models from VideoPhY1 results directories."""
    models = set()
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


def generate_csv_markdown_table(model_list, csv_results):
    headers = ["Model", "PC", "SA", "PCp", "SAp", "Joint"]
    md_table = "| " + " | ".join(headers) + " |\n"
    md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    for model in model_list:
        metrics = csv_results.get(model, {})
        pc_avg = metrics.get('pc')
        sa_avg = metrics.get('sa')
        pcp_score = metrics.get('pcp')
        sap_score = metrics.get('sap')
        joint_score = metrics.get('joint')

        row = [
            model,
            f"{pc_avg:.3f}" if pc_avg is not None else "N/A",
            f"{sa_avg:.3f}" if sa_avg is not None else "N/A",
            f"{pcp_score:.1f}" if pcp_score is not None else "N/A",
            f"{sap_score:.1f}" if sap_score is not None else "N/A",
            f"{joint_score:.1f}" if joint_score is not None else "N/A",
        ]
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
            # VideoPhY1 CSV format: two columns without header -> [video_path, score]
            df = pd.read_csv(csv_file, header=None, names=['video_path', 'score'])
            # Mean score (already in [0,1])
            avg = df['score'].mean()
            # Proportion of videos with score >= 0.5 (percentage)
            proportion_ge_05 = (df['score'] >= 0.5).mean() * 100
            if 'pc' in os.path.basename(csv_file):
                pc_averages.append(avg)
                pc_proportions.append(proportion_ge_05)
                pc_dfs.append(df)
            elif 'sa' in os.path.basename(csv_file):
                sa_averages.append(avg)
                sa_proportions.append(proportion_ge_05)
                sa_dfs.append(df)

        # Joint proportion: both PC >= 0.5 and SA >= 0.5
        joint_proportions = []
        if pc_dfs and sa_dfs:
            for pc_df, sa_df in zip(pc_dfs, sa_dfs):
                if len(pc_df) == len(sa_df):
                    joint_ge_05 = ((pc_df['score'] >= 0.5) & (sa_df['score'] >= 0.5)).mean() * 100
                    joint_proportions.append(joint_ge_05)

        results[model] = {
            'pc': sum(pc_averages) / len(pc_averages) if pc_averages else None,
            'sa': sum(sa_averages) / len(sa_averages) if sa_averages else None,
            'pcp': sum(pc_proportions) / len(pc_proportions) if pc_proportions else None,
            'sap': sum(sa_proportions) / len(sa_proportions) if sa_proportions else None,
            'joint': sum(joint_proportions) / len(joint_proportions) if joint_proportions else None,
        }

    return results


def generate_aggregated_markdown_table(models, prompts, all_csv_results):
    """Generate an aggregated markdown table with VideoPhY1 results from all prompts stacked as rows."""
    headers = ["Model/Prompt", "PC", "SA", "PCp", "SAp", "Joint"]
    md_table = "| " + " | ".join(headers) + " |\n"
    md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    for model in models:
        md_table += f"| **{model}** |" + " |".join([" "] * (len(headers) - 1)) + " |\n"
        for prompt in prompts:
            csv_results = all_csv_results.get(prompt, {})
            csv_metrics = csv_results.get(model, {})
            pc_avg = csv_metrics.get('pc')
            sa_avg = csv_metrics.get('sa')
            pcp_score = csv_metrics.get('pcp')
            sap_score = csv_metrics.get('sap')
            joint_score = csv_metrics.get('joint')

            row = [f"└─ {prompt}"]
            row.append(f"{pc_avg:.3f}" if pc_avg is not None else "N/A")
            row.append(f"{sa_avg:.3f}" if sa_avg is not None else "N/A")
            row.append(f"{pcp_score:.1f}" if pcp_score is not None else "N/A")
            row.append(f"{sap_score:.1f}" if sap_score is not None else "N/A")
            row.append(f"{joint_score:.1f}" if joint_score is not None else "N/A")
            md_table += "| " + " | ".join(row) + " |\n"

        md_table += "| |" + " |".join([" "] * (len(headers) - 1)) + " |\n"

    return md_table


# ----- Label helpers (ported and reused from read_videophy2.py) -----


def _short_key(keyword: str) -> str:
    mapping = {
        'sampling_method': 'meth',
        'cfg_scale': 'cfg',
        'loss_mode': 'loss',
        'vjepa_variant': 'vj',
        'vjepa_context_frames': 'ctx',
        'slice_stride': 'st',
        'slice_window_size': 'w',
        'guidance_frequency': 'freq',
        'config_version': 'v',
    }
    return mapping.get(keyword, keyword)


def _short_value(keyword: str, value):
    if value is None:
        return None
    if keyword == 'vjepa_variant':
        return 'vitg' if value == 'vit_giant' else ('vith' if value == 'vit_huge' else str(value))
    if keyword == 'sampling_method':
        return 'van' if value == 'vanilla' else ('guid' if value == 'guidance' else str(value))
    if keyword == 'loss_mode':
        return str(value).lower()
    if keyword == 'config_version':
        s = str(value)
        return s.split('_')[0] if s.startswith('v') else s
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _format_label_from_params(params: dict, param_keywords) -> str:
    parts = []
    for key in param_keywords:
        val = _short_value(key, params.get(key))
        if val in (None, ''):
            continue
        sk = _short_key(key)
        if str(val).replace('.', '', 1).isdigit():
            parts.append(f"{sk}{val}")
        else:
            parts.append(f"{sk}:{val}")
    return ' '.join(parts) if parts else 'unlabeled'


def build_model_label(base_dir: str, prompt: str, model_cat: str, model_dir: str, param_keywords):
    """Read experiment_config.json and build a compact label; fallback to folder name."""
    config_path = os.path.join(base_dir, 'generated_videos', prompt, model_cat, model_dir, 'experiment_config.json')
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            params = config.get('parameters', {})
            return _format_label_from_params(params, param_keywords)
    except Exception:
        pass
    return model_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoPhY1 Results to Markdown Converter")
    parser.add_argument('--prompt', nargs='+', default=['videophy1'],
                        help='Evaluation prompt(s)')
    parser.add_argument('--model_cat', type=str, default='cogvideox5b',
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
    parser.add_argument('--param_keywords', nargs='+',
                        default=['sampling_method', 'cfg_scale', 'loss_mode', 'guidance_lr_pattern', 'guidance_step_pattern', 'guidance_frequency'],
                        help='Parameter keywords to include in labels')
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
                print(f"Checked VideoPhY1: {args.base_dir}/results/videophy1/{prompt}/{args.model_cat}")
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

        # Build display labels from configs
        model_labels = {m: build_model_label(args.base_dir, prompt, args.model_cat, m, args.param_keywords) for m in model_list}

        # Calculate CSV averages from VideoPhY1 results
        print("\nCalculating VideoPhY1 averages...")
        csv_results = calculate_csv_averages(
            models=model_list,
            base_dir=args.base_dir,
            model_cat=args.model_cat,
            prompt=prompt
        )

        # Generate Markdown table from VideoPhY1 CSV results
        print("\nGenerating Markdown table from VideoPhY1 results...")
        labeled_md_order = [model_labels.get(m, m) for m in model_list]
        relabeled_results = {model_labels.get(m, m): v for m, v in csv_results.items()}
        md_output = generate_csv_markdown_table(
            model_list=labeled_md_order,
            csv_results=relabeled_results
        )
        print(md_output)

        print(f"\nVideoPhY1 Results Summary for {prompt}:")
        for model, metrics in csv_results.items():
            label = model_labels.get(model, model)
            pc_avg = metrics['pc']
            sa_avg = metrics['sa']
            pcp_score = metrics['pcp']
            sap_score = metrics['sap']
            joint_score = metrics['joint']
            print(f"{label}:")
            print(f"  PC (average) = {pc_avg:.4f}" if pc_avg is not None else f"  PC (average) = N/A")
            print(f"  SA (average) = {sa_avg:.4f}" if sa_avg is not None else f"  SA (average) = N/A")
            print(f"  PCp (% videos ≥ 0.5) = {pcp_score:.2f}%" if pcp_score is not None else f"  PCp (% videos ≥ 0.5) = N/A")
            print(f"  SAp (% videos ≥ 0.5) = {sap_score:.2f}%" if sap_score is not None else f"  SAp (% videos ≥ 0.5) = N/A")
            print(f"  Joint (% videos PC≥0.5 & SA≥0.5) = {joint_score:.2f}%" if joint_score is not None else f"  Joint (% videos PC≥0.5 & SA≥0.5) = N/A")

        # Store results for aggregation
        all_csv_results[prompt] = csv_results
        all_model_lists[prompt] = model_list

    # Generate aggregated markdown table (display with labels)
    print(f"\n{'='*80}")
    print("AGGREGATED VIDEOPHY1 RESULTS ACROSS ALL PROMPTS")
    print(f"{'='*80}")

    all_models = set()
    for model_list in all_model_lists.values():
        all_models.update(model_list)
    all_models = sorted(list(all_models))

    if all_models and all_csv_results:
        # Build a unified label map using the first prompt for lookup
        any_prompt = args.prompt[0]
        labels_map = {m: build_model_label(args.base_dir, any_prompt, args.model_cat, m, args.param_keywords) for m in all_models}
        labeled_models = [labels_map[m] for m in all_models]
        # Relabel results per prompt for printing
        relabeled_all = {}
        for prompt in args.prompt:
            labels_for_prompt = {m: build_model_label(args.base_dir, prompt, args.model_cat, m, args.param_keywords) for m in all_models}
            relabeled_all[prompt] = {labels_for_prompt.get(m, m): v for m, v in all_csv_results.get(prompt, {}).items()}
        print("\nAggregated Markdown Table (Prompts stacked as rows):")
        aggregated_md = generate_aggregated_markdown_table(labeled_models, args.prompt, relabeled_all)
        print(aggregated_md)


