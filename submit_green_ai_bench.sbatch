#!/bin/bash
#SBATCH --job-name=eval-green
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=44G
#SBATCH --time=00:45:00
#SBATCH --output=/users/ali.gunder/eduterm-finetune/slurm-green-%j.out

echo "================================================================="
echo "Experiment 3: Green AI, Energy Profiling & Hardware Efficiency"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname) | Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "================================================================="

mkdir -p /tmp/site-packages
export PYTHONPATH=/tmp/site-packages:$PYTHONPATH
pip install --target=/tmp/site-packages --no-deps /users/ali.gunder/wheels_clean/*.whl

export HF_HOME=/users/ali.gunder/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd /users/ali.gunder/eduterm-finetune
python3 -u run_hpc_green_ai_bench.py \
    --output /users/ali.gunder/eduterm-finetune/outputs/green_ai_efficiency_report.json

echo "Green AI Benchmark finished at: $(date)"

