#!/bin/bash
# ==============================================================================
# SLURM Submission Script for SLM LoRA Fine-Tuning
# Boğaziçi University Computer Engineering & Academic GPU Cluster Template
# Project: Çok Dilli Eğitim Teknolojisi - Dikey Alan SLM Adaptasyonu
# ==============================================================================
#SBATCH --job-name=eduterm_slm_lora
#SBATCH --output=logs/slm_finetune_%j.out
#SBATCH --error=logs/slm_finetune_%j.err
#SBATCH --partition=gpu            # Or: akya, p100, v100, a100, h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1               # Request 1 dedicated GPU (A100 / RTX)
#SBATCH --time=04:00:00            # Estimated max 4 hours for 3 epochs

echo "================================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on Host: $(hostname)"
echo "Assigned GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start Time: $(date)"
echo "================================================================="

# Create local directories
mkdir -p logs checkpoints

# Activate Conda / Python virtual environment
# source ~/miniconda3/bin/activate eduterm-ai

# Verify CUDA availability
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Execute Fine-Tuning on target domain dataset (Turizm, İSG, KVKK, Bankacılık)
python experiments/finetune/train_lora_slm.py \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --epochs 3 \
    --batch-size 4

echo "================================================================="
echo "Fine-Tuning Finished at: $(date)"
echo "Checkpoints saved in experiments/finetune/checkpoints/"
echo "================================================================="
