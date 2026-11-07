#!/bin/bash
#SBATCH --job-name=download-trendyol
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/users/ali.gunder/eduterm-finetune/slurm-download-trendyol-%j.out

echo "================================================================="
echo "Downloading Trendyol-LLM-8B-T1 from Hugging Face"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname) | Date: $(date)"
echo "================================================================="

mkdir -p /tmp/site-packages
export PYTHONPATH=/tmp/site-packages:$PYTHONPATH
pip install --target=/tmp/site-packages --no-deps /users/ali.gunder/wheels_clean/*.whl

export HF_HOME=/users/ali.gunder/hf_cache
export PYTHONUNBUFFERED=1

cd /users/ali.gunder/eduterm-finetune
python3 -u download_trendyol.py
echo "Download finished at $(date)"

