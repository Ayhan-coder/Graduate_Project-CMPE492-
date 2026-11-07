#!/bin/bash
#SBATCH --job-name=eval-cross-family
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=44G
#SBATCH --time=01:00:00
#SBATCH --output=/users/ali.gunder/eduterm-finetune/slurm-cross-family-%j.out

echo "================================================================="
echo "Cross-Family Benchmark: Kumru-2B vs. EduTerm-14B on HPC A40"
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
python3 -u run_hpc_cross_family_eval.py \
    --output /users/ali.gunder/eduterm-finetune/outputs/cross_family_comprehensive_benchmark.json

echo "Cross-Family Evaluation finished at: $(date)"

