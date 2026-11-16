#!/bin/bash
#SBATCH --job-name=opt1-7b-bf16
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=44G
#SBATCH --time=01:00:00
#SBATCH --output=/users/ali.gunder/eduterm-hybrid-sim/slurm-%j.out

mkdir -p /tmp/site-packages
export PYTHONPATH=/tmp/site-packages:/users/ali.gunder/eduterm-hybrid-sim:$PYTHONPATH
pip install --target=/tmp/site-packages --no-deps /users/ali.gunder/wheels_clean/*.whl

export HF_HOME=/users/ali.gunder/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd /users/ali.gunder/eduterm-hybrid-sim
python3 -u run_hpc_7b_bf16_cross_judge.py
echo "7B-bf16 Cross-Judge Finished at $(date)"
