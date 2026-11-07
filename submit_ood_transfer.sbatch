#!/bin/bash
#SBATCH --job-name=eval-ood
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=44G
#SBATCH --time=00:35:00
#SBATCH --output=/users/ali.gunder/eduterm-finetune/slurm-ood-%j.out

echo "================================================================="
echo "Experiment 2: Cross-Disciplinary Out-of-Distribution (OOD) Transfer Benchmark"
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
python3 -u run_hpc_ood_transfer.py \
    --output /users/ali.gunder/eduterm-finetune/outputs/ood_cross_disciplinary_report.json

echo "OOD Transfer Benchmark finished at: $(date)"

