
### Align Video Diffusion Model to Latent World Model

Minimal tooling to run image-to-video (I2V) generation with three modes:
- vanilla: baseline generation
- guidance: V-JEPA surprise guidance during sampling
- rejection: generate k candidates, score with V-JEPA, keep best

Supports both CogVideoX and NVIDIA Cosmos I2V checkpoints via customized pipelines in `pipelines/`.

### Setup
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
- Install Python deps:
```bash
pip install -r requirements.txt
```

### Quick start (single prompt)
Use these script entrypoints for quick local runs.

- CogVideoX:
```bash
python ./run_i2v_cogvideox.py
```

- Cosmos:
```bash
python ./run_i2v_cosmos.py
```

### Batch mode (JSON) 
- SLURM multi-node (guidance):
```bash
sbatch generate_i2v_cosmos_multinode.sh
```
- SLURM multi-node (rejection, with rejection buffer reuse this will save all rejection samples for later analysis):
```bash
sbatch generate_i2v_cosmos_rejection_multinode.sh
```