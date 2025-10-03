
### Align Video Diffusion Model to Latent World Model

Minimal tooling to run image/video-to-video (I2V/V2V) generation with CogVideoX, Cosmos, and MAGI1.

## Usage
1. pull the repo, which contains CogVideoX and Cosmos pipelines
```bash
git clone https://github.com/fairinternal/video_guidance.git
cd ./vejap2
git clone https://github.com/facebookresearch/vjepa2.git
```

1.1 To run Cosmos, need approval from HF: https://huggingface.co/nvidia/Cosmos-Predict2-14B-Video2World

1.2 To run MAGI-1
Script can be copy from the storage (optional)
```bash
cp -r /checkpoint/dream/yjianhao/storage/project/MAGI-1 ./[WORKING_DIR]
```
or directly run script under
```bash
/checkpoint/dream/yjianhao/MAGI-1
```


2. (Optional) Activate conda env locally to debug
For CogVideoX:
```bash
source /checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin/activate
conda activate vg
```
For MAGI-1:
```bash
source /checkpoint/dream/yjianhao/VideoGuidance/conda/envs/vg/bin/activate
eval "$(/checkpoint/dream/yjianhao/VideoGuidance/conda/bin/conda shell.bash hook)"
conda activate /checkpoint/dream/yjianhao/VideoGuidance/conda/envs/magi_share
```
3. (Optional) Debug with a single prompt run
Quick start (single prompt for debug)
Use these script entrypoints for quick local runs.

- CogVideoX:
```bash
python run_i2v_cogvideox.py
```

- Cosmos:
```bash
python run_i2v_cosmos.py
```

4. Send Batch Generation Jobs
Batch mode - PhysicsIQ generation
- SLURM multi-node (for example):
```bash
sbatch generation/generate_i2v_cosmos_multinode.sh
sbatch generation/generate_batch_1chunk_abl.sh
```
- Key Hyperparameters inside batch generation:
```bash
- SBATCH --array=0-2 (line5) and NUM_NODES=3 (line21) --> these control how many node to use, the script will distribute the generation as 198 // (NUM_NODE * 8)
- SAMPLE_METHODS=("vanilla") --> control what sampling nethod to use
```
- Place to find generated videos
```bash
/checkpoint/dream/yjianhao/generated_videos/physics_iq
```
All videos are stored in structure of
```bash
/checkpoint/dream/yjianhao/generated_videos/[DATASET]/[MODEL]/[SAMPLING_CONFIG]
```

### Setup External
For CogVideoX
```bash
git clone https://github.com/fairinternal/video_guidance.git

conda create -n reproduce python=3.12
pip install torch torchvision
pip install diffusers transformers
pip install -r requirements.txt
```
For MAGI-1
Follow instruction from [MAGI-1](https://github.com/SandAI-org/MAGI-1/tree/main)


### /checkpoint/dream/yjianhao Breakdown
```bash
.
├── evaluation - Store all evaluation scripts
├── example - Store config files for generation
├── generated_videos - Store all generated videos files for generation
├── generation - Store all generation scripts
├── inference - Store all inference pipelines
├── interactive_session.sh - Ask for a interactive session
├── jobs - store slurm job logs
├── MagiAttention - MAGI-1 attention module
├── physics-IQ-benchmark
├── plots - Store all plots
├── prompts - Store all generation prompts
├── scripts - Store all small helper scripts
├── utils.py - Utils for VJEPA score and video processing
└── visualization - Store all visualization
└── results - Store all evaluation results
```

The pipeline generally goes as (1) use generation script under /generation to generate videos (2) use evaluation script under /evaluation to evaluate the generated videos (3) use ./scripts/physicsiq_res2tab.py to view results.

For generation it goes as (1) entry point under /generation (2) go to ./inference/generator.py to handle vanilla/rejection/smc (3) go to corresponding pipeline i.e. /inference/pipeline_w_guidance.py to handle the generation pipeline (4) go to corresponding video generation i.e. ./inference/video_generate_longcontext_guidance.py to handle forward process.

important scripts:
Run PhysicsIQ evaluation
```bash
bash evaluation/eval_physics_iq_parallel.sh
```
Specify the generated video path under generated_videos_dirs_list

View PhysicsIQ scores
```bash
python3 /home/yjianhao/project/MAGI-1/scripts/physicsiq_res2tab.py
```
by default results will be stored to ./results/physics_iq/results, change --model_cat to other subfolders to view specific grouped ablation results
