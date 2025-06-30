# Simple Video Generation Benchmarking

This is a simplified version of the benchmarking system that's much easier to use.

## Quick Start

### 1. Generate Videos
```bash
# Vanilla sampling
python generator.py --sampling_method vanilla --prompt_file action --model_id wan --output_folder ./results --num_gpus 1 --gpu_idx 0 --cfg_scale 5.0

# Rejection sampling  
python generator.py --sampling_method rejection --prompt_file action --model_id wan --output_folder ./results --num_gpus 1 --gpu_idx 0 --num_rejection_attempts 5 --cfg_scale 5.0

# Guidance sampling
python generator.py --sampling_method guidance --prompt_file subject_consistency --model_id wan --output_folder ./results --num_gpus 1 --gpu_idx 0 --guidance_rho_scale 3.0 --cfg_scale 5.0
```

### 2. Check Results
Your videos will be organized like this:
```
results/
├── experiments.csv                                    # Simple log of all experiments
├── vanilla_f17_s50_cfg5.0_1201_1430/                # Experiment folder with timestamp
│   ├── 001_a_cat_jumping.mp4
│   ├── 002_a_dog_running.mp4
│   └── ...
└── rejection_f17_s50_w8c6_a5_cfg5.0_1201_1445/
    ├── 001_a_cat_jumping.mp4 
    └── ...
```

### 3. Evaluate
```bash
# Evaluate all completed experiments
python simple_eval.py --output_folder ./results

# Evaluate only rejection sampling experiments  
python simple_eval.py --output_folder ./results --method rejection
```

## That's It!

- **Folder names** are much shorter and include timestamps for uniqueness
- **Simple CSV log** tracks all experiments  
- **Easy evaluation** with automatic experiment discovery
- **Still configurable** with all the same ablation parameters

## Key Parameters

### General Parameters
- `--cfg_scale 5.0` - Classifier-free guidance scale (for all methods)

### V-JEPA Parameters (for rejection/guidance)
- `--kernel_size 8` - Window size for V-JEPA
- `--context_length 6` - Context length (must be < kernel_size) 
- `--stride 2` - Stride for sliding window

### Rejection Parameters
- `--num_rejection_attempts 10` - How many samples to try

### Guidance Parameters  
- `--guidance_rho_scale 3.0` - Rho scaling factor for gradient guidance  
- `--guidance_start 0` - When to start guidance (timestep)
- `--guidance_end 1001` - When to stop guidance 