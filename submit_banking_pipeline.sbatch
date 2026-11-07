#!/bin/bash
#SBATCH --job-name=eduterm-banking
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=44G
#SBATCH --time=06:00:00
#SBATCH --output=/users/ali.gunder/banking_project/slurm-banking-%j.out

echo "================================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "================================================================="

cd /users/ali.gunder/banking_project

echo "Setting up verified wheels in /tmp/site-packages..."
mkdir -p /tmp/site-packages
export PYTHONPATH=/tmp/site-packages:$PYTHONPATH
pip install --target=/tmp/site-packages --no-deps /users/ali.gunder/wheels_clean/*.whl

export HF_HOME=/users/ali.gunder/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo ">>> [STAGE 1/3] Transcribing Banking Audio with Whisper Large-v3 Turbo..."
python3 -u step1_transcribe_banking.py

echo ">>> [STAGE 2/3] Constructing Banking Domain Dataset from Transcripts..."
python3 -u step2_build_banking_dataset.py

echo ">>> [STAGE 3/3] Fine-Tuning EduTerm-14B on Banking Domain (DAFT QLoRA)..."
python3 -u step3_train_banking_14b_qlora.py

echo "================================================================="
echo "Banking Domain Adaptation Pipeline Completed Successfully at $(date)"
echo "================================================================="

