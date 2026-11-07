#!/bin/bash
#SBATCH --job-name=eduterm-dpo
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=44G
#SBATCH --output=/users/ali.gunder/eduterm-dpo/slurm-%j.out

echo "================================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "================================================================="

cd /users/ali.gunder/eduterm-dpo

echo "Setting up verified wheels in /tmp/site-packages..."
mkdir -p /tmp/site-packages
export PYTHONPATH=/tmp/site-packages:$PYTHONPATH
pip install --target=/tmp/site-packages --no-deps /users/ali.gunder/wheels_clean/*.whl

echo "Executing Direct Preference Optimization (DPO) for Pedagogical B1 Alignment..."
python3 train_dpo_qwen.py \
    --base-dir /users/ali.gunder/eduterm-dpo \
    --model-path /users/ali.gunder/models/Qwen2.5-7B-Instruct \
    --sft-lora-path /users/ali.gunder/eduterm-finetune/checkpoints/eduterm_7b_tr_sft \
    --output-dir /users/ali.gunder/eduterm-dpo/checkpoints/eduterm_7b_tr_dpo \
    --beta 0.1 \
    --lr 5e-6 \
    --epochs 3

echo "DPO Alignment Job completed at $(date)"
