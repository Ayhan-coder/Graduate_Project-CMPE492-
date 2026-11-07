#!/bin/bash
#SBATCH --job-name=corpus133-a40
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=44G
#SBATCH --time=02:00:00
#SBATCH --output=/users/ali.gunder/eduterm-corpus-133/slurm-%j.out

echo "================================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "================================================================="

mkdir -p /tmp/site-packages
export PYTHONPATH=/tmp/site-packages:$PYTHONPATH
pip install --target=/tmp/site-packages --no-deps /users/ali.gunder/wheels_clean/*.whl

export HF_HOME=/users/ali.gunder/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd /users/ali.gunder/eduterm-corpus-133

python3 -u run_hpc_corpus_133.py

echo "Corpus-133 Pipeline Finished at $(date)"

