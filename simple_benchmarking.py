import os
import hashlib
import json
from datetime import datetime

def get_simple_experiment_name(args):
    """Generate a clean, short experiment folder name."""
    
    # Base name with method and key params
    if args.sampling_method == 'vanilla':
        name = f"vanilla_f{args.num_frames}_s{args.num_inference_steps}_cfg{args.cfg_scale}"
    elif args.sampling_method == 'rejection':
        name = f"rejection_f{args.num_frames}_s{args.num_inference_steps}_w{args.kernel_size}c{args.context_length}_a{args.num_rejection_attempts}_cfg{args.cfg_scale}"
    elif args.sampling_method == 'guidance':
        name = f"guidance_f{args.num_frames}_s{args.num_inference_steps}_w{args.kernel_size}c{args.context_length}_rho{args.guidance_rho_scale}_cfg{args.cfg_scale}"
    
    # Add timestamp to make unique
    timestamp = datetime.now().strftime("%m%d_%H%M")
    name += f"_{timestamp}"
    
    return name

def log_experiment_simple(args, experiment_name, status='started'):
    """Simple logging to CSV."""
    log_file = os.path.join(args.output_folder, 'experiments.csv')
    
    # Create header if file doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('name,method,frames,steps,kernel_size,context_length,attempts,guidance_rho_scale,cfg_scale,timestamp,status\n')
    
    # Add entry
    with open(log_file, 'a') as f:
        f.write(f"{experiment_name},{args.sampling_method},{args.num_frames},{args.num_inference_steps},")
        f.write(f"{getattr(args, 'kernel_size', '')},{getattr(args, 'context_length', '')},")
        f.write(f"{getattr(args, 'num_rejection_attempts', '')},{getattr(args, 'guidance_rho_scale', '')},")
        f.write(f"{args.cfg_scale},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{status}\n")

def get_simple_output_folder(args, experiment_name):
    """Get simple output folder structure."""
    # Just: output_folder/prompt_file/experiment_name/
    folder = os.path.join(args.output_folder, args.prompt_file, experiment_name)
    os.makedirs(folder, exist_ok=True)
    return folder 