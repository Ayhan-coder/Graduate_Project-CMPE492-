#!/bin/bash
#SBATCH --job-name=eduterm-merge-14b
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=44G
#SBATCH --time=03:00:00
#SBATCH --output=/users/ali.gunder/eduterm-finetune/slurm-merge-14b-%j.out

echo "================================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "================================================================="

mkdir -p /users/ali.gunder/eduterm-finetune/outputs
mkdir -p /users/ali.gunder/eduterm-finetune/checkpoints/eduterm_14b_omni_merged
cd /users/ali.gunder/eduterm-finetune

echo "Setting up verified wheels in /tmp/site-packages..."
mkdir -p /tmp/site-packages
export PYTHONPATH=/tmp/site-packages:$PYTHONPATH
pip install --target=/tmp/site-packages --no-deps /users/ali.gunder/wheels_clean/*.whl

export HF_HOME=/users/ali.gunder/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Executing EduTerm-14B Tri-Domain Adapter Fusion (DARE-TIES) on NVIDIA A40 (48GB)..."
python3 -u run_hpc_tri_domain_merge_14b.py

echo "EduTerm-14B Tri-Domain Fusion completed at $(date)"

