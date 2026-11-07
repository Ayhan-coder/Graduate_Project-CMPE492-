#!/bin/bash
#SBATCH --job-name=eduterm-14b-dpo
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=44G
#SBATCH --time=12:00:00
#SBATCH --output=/users/ali.gunder/eduterm-dpo-14b/slurm-dpo-14b-%j.out

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

echo "Executing Direct Preference Optimization (DPO) for EduTerm-14B..."
python3 -u train_dpo_14b_qwen.py \
    --base-dir /users/ali.gunder/eduterm-dpo-14b \
    --model-path /users/ali.gunder/models/Qwen2.5-14B-Instruct \
    --sft-lora-path /users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_cefr_qlora \
    --output-dir /users/ali.gunder/eduterm-dpo-14b/checkpoints/eduterm_14b_dpo \
    --output-report /users/ali.gunder/eduterm-dpo-14b/outputs/eduterm_14b_dpo_report.json \
    --beta 0.1 \
    --lr 5e-6 \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum-steps 4

echo "EduTerm-14B DPO Alignment Job completed at $(date)"

