#!/bin/bash
#SBATCH --job-name=n1_b5
#SBATCH --output=jobs/bon_n1_b5_%j.out
#SBATCH --error=jobs/bon_n1_b5_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --qos=h200_dream_high

cd /home/reyhaneaskari/WMReward
source /checkpoint/dream/yjianhao/VideoGuidance/conda/etc/profile.d/conda.sh
conda activate vg

python3 -u << 'EOF'
import os, json, random, subprocess
from pathlib import Path

N = 1
B = 4  # 5th bootstrap (0-indexed)

with open('./generated_videos/physics_iq/sora/sora2_i2v_s8_1280x720_n16_8s_best/bon_results.json') as f:
    bon_data = json.load(f)

trimmed_dir = Path(f'./results/physics_iq/bon_analysis/trimmed_n{N}_b{B}')
eval_dir = Path(f'./results/physics_iq/bon_analysis/eval_n{N}_b{B}')
trimmed_dir.mkdir(parents=True, exist_ok=True)

print(f"=== N={N} Bootstrap {B+1} ===", flush=True)

for p in bon_data:
    random.seed(B * 1000 + hash(p['prompt']) % 1000)
    selected = random.sample(p['all_scores'], min(N, len(p['all_scores'])))
    best = min(selected, key=lambda x: x[1])[0]
    src = f"./generated_videos/physics_iq/sora/sora2_i2v_s8_1280x720_n16/{best}"
    dst = trimmed_dir / (best.rsplit('_sample', 1)[0] + '.mp4')
    if not dst.exists():
        subprocess.run(['/checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin/ffmpeg',
            '-y', '-i', src, '-vf', 'fps=30', '-vframes', '150', '-an',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', str(dst)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"Trimmed: {len(list(trimmed_dir.glob('*.mp4')))}", flush=True)

env = os.environ.copy()
env['PATH'] = '/checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin:' + env.get('PATH', '')
result = subprocess.run(['/checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin/python',
    'code/run_physics_iq.py', '--input_folders', str(trimmed_dir.resolve()),
    '--output_folder', str(eval_dir.resolve()),
    '--descriptions_file', '/checkpoint/dream/yjianhao/PhysicsIQ/code/physics-IQ-benchmark/descriptions/descriptions.csv'],
    cwd='/checkpoint/dream/yjianhao/PhysicsIQ/code/physics-IQ-benchmark',
    capture_output=True, text=True, env=env)

for line in result.stdout.split('\n'):
    if 'Physics-IQ score' in line:
        print(f"Score: {line.split(':')[-1].strip()}", flush=True)
EOF
