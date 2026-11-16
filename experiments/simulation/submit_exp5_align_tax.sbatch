#!/bin/bash
#SBATCH --job-name=exp5-align-tax
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=44G
#SBATCH --time=01:30:00
#SBATCH --output=/users/ali.gunder/eduterm-hybrid-sim/slurm-%j.out

mkdir -p /tmp/site-packages
export PYTHONPATH=/tmp/site-packages:/users/ali.gunder/eduterm-hybrid-sim:$PYTHONPATH
pip install --target=/tmp/site-packages --no-deps /users/ali.gunder/wheels_clean/*.whl

export HF_HOME=/users/ali.gunder/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd /users/ali.gunder/eduterm-hybrid-sim
python3 -u run_hpc_alignment_tax_cefr.py
echo "Exp 5 Finished at $(date)"
