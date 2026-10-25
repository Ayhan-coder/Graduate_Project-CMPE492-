#!/bin/bash
#SBATCH --job-name=eduterm-scaling
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=44G
#SBATCH --output=/users/ali.gunder/eduterm-finetune/slurm-%j.out

echo "================================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "================================================================="

cd /users/ali.gunder/eduterm-finetune

echo "Setting up verified wheels in /tmp/site-packages..."
mkdir -p /tmp/site-packages
export PYTHONPATH=/tmp/site-packages:$PYTHONPATH
pip install --target=/tmp/site-packages --no-deps /users/ali.gunder/wheels_clean/*.whl

echo "Executing Empirical Scaling Law Benchmark (0.5B -> 1.5B -> 7B -> 14B)..."
python3 run_hpc_scaling_law.py \
    --models-dir /users/ali.gunder/models \
    --lora-dir /users/ali.gunder/eduterm-finetune/checkpoints/eduterm_7b_tr_sft \
    --output-json /users/ali.gunder/eduterm-finetune/outputs/scaling_law_report.json

echo "Scaling Law Benchmark completed at $(date)"
