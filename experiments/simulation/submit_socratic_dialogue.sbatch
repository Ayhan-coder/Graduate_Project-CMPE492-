#!/bin/bash
#SBATCH --job-name=socratic-a40
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=44G
#SBATCH --time=01:00:00
#SBATCH --output=/users/ali.gunder/eduterm-socratic/slurm-%j.out

echo "================================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "================================================================="

mkdir -p /tmp/site-packages
export PYTHONPATH=/tmp/site-packages:$PYTHONPATH
pip install --target=/tmp/site-packages --no-deps /users/ali.gunder/wheels_clean/*.whl

export HF_HOME=/users/ali.gunder/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p /users/ali.gunder/eduterm-socratic/outputs
cd /users/ali.gunder/eduterm-socratic

python3 -u run_hpc_socratic_dialogue.py \
    --model /users/ali.gunder/hf_cache/models--unsloth--mistral-7b-instruct-v0.3-bnb-4bit/snapshots/d5f623888f1415cf89b5c208d09cb620694618ee \
    --out /users/ali.gunder/eduterm-socratic/outputs/socratic_tutoring_report.json

echo "Socratic Tutoring Dialogue Benchmark Finished at $(date)"
