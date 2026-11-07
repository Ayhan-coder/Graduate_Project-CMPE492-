#!/bin/bash
#SBATCH --job-name=eduterm-72b
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=44G
#SBATCH --output=/users/ali.gunder/eduterm-scaling-72b/slurm-%j.out

echo "================================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "================================================================="

mkdir -p /users/ali.gunder/eduterm-scaling-72b/outputs
cd /users/ali.gunder/eduterm-scaling-72b

echo "Setting up verified wheels in /tmp/site-packages..."
mkdir -p /tmp/site-packages
export PYTHONPATH=/tmp/site-packages:$PYTHONPATH
pip install --target=/tmp/site-packages --no-deps /users/ali.gunder/wheels_clean/*.whl

export HF_HOME=/tmp/hf_cache_72b
mkdir -p /tmp/hf_cache_72b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Executing Qwen2.5-72B-Instruct Flagship Scaling Law Benchmark on NVIDIA A40 (48GB)..."
python3 run_hpc_scaling_law_72b.py \
    --model-path unsloth/Qwen2.5-72B-Instruct-bnb-4bit \
    --base-report /users/ali.gunder/eduterm-finetune/outputs/scaling_law_report.json \
    --output-json /users/ali.gunder/eduterm-scaling-72b/outputs/scaling_law_full_72b_report.json

echo "72B Scaling Law Benchmark completed at $(date)"
