#!/bin/bash

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=sample
## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=./jobs/sample-%j.out
## filename for job standard error output (stderr)
#SBATCH --error=./jobs/sample-%j.err
#SBATCH --mail-user=yjianhao@meta.com
#SBATCH --mail-type=end # mail once the job finishes

## partition name
#SBATCH --partition=learn
## number of nodes
#SBATCH --nodes=1

## number of tasks per node
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=80

## number of GPUs per node


# SBATCH --time=8:00:00

### Section 3:
source /home/yjianhao/miniconda3/bin/activate
conda activate vg
nvidia-smi
echo "Running batch sample script env activated"
bash generate.sh
