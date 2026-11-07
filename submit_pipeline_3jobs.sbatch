#!/bin/bash
#SBATCH --job-name=eduterm-3jobs
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=44G
#SBATCH --time=06:00:00
#SBATCH --output=/users/ali.gunder/eduterm-dpo-14b/slurm-3jobs-%j.out

echo "================================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "================================================================="

cd /users/ali.gunder/eduterm-dpo-14b

echo "Setting up verified wheels in /tmp/site-packages..."
mkdir -p /tmp/site-packages
export PYTHONPATH=/tmp/site-packages:$PYTHONPATH
pip install --target=/tmp/site-packages --no-deps /users/ali.gunder/wheels_clean/*.whl

export HF_HOME=/users/ali.gunder/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo ">>> [STAGE 1/3] Executing Job 1: Head-to-Head Win-Rate Arena (Base vs SFT vs DPO)..."
python3 -u run_hpc_job1_arena_winrate.py

echo ">>> [STAGE 2/3] Executing Job 2: Test-Time Compute (TTC) & CoT Reasoning Scaling..."
python3 -u run_hpc_job2_cot_ttc.py

echo ">>> [STAGE 3/3] Executing Job 3: Multi-Turn Socratic Tutoring & Misconception Stress-Test..."
python3 -u run_hpc_job3_socratic_misconception.py

echo "================================================================="
echo "All 3 Jobs Completed Successfully at $(date)"
echo "================================================================="

