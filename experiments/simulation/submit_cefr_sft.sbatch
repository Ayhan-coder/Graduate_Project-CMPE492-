#!/bin/bash
#SBATCH --job-name=eduterm-cefr-14b
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=44G
#SBATCH --time=12:00:00
#SBATCH --output=/users/ali.gunder/eduterm-finetune/slurm-cefr-%j.out

echo "================================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "================================================================="

mkdir -p /users/ali.gunder/eduterm-finetune/outputs
mkdir -p /users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora
cd /users/ali.gunder/eduterm-finetune

echo "Setting up verified wheels in /tmp/site-packages..."
mkdir -p /tmp/site-packages
export PYTHONPATH=/tmp/site-packages:$PYTHONPATH
pip install --target=/tmp/site-packages --no-deps /users/ali.gunder/wheels_clean/*.whl

export HF_HOME=/users/ali.gunder/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Executing Multi-Level CEFR SFT on NVIDIA A40 (48GB)..."
python3 -u run_hpc_cefr_qlora_sft.py \
    --model-path /users/ali.gunder/models/Qwen2.5-14B-Instruct \
    --data-dir /users/ali.gunder/eduterm-finetune/data \
    --checkpoint-dir /users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora \
    --report-json /users/ali.gunder/eduterm-finetune/outputs/eduterm_14b_cefr_sft_report.json \
    --epochs 3 \
    --batch-size 4 \
    --accum-steps 4

echo "Multi-Level CEFR QLoRA SFT completed at $(date)"

