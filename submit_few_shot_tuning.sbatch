#!/bin/bash
#SBATCH --job-name=eval-few-shot
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=44G
#SBATCH --time=00:35:00
#SBATCH --output=/users/ali.gunder/eduterm-finetune/slurm-few-shot-%j.out

echo "================================================================="
echo "Experiment: Few-Shot Dynamic In-Context Exemplar Tuning Benchmark"
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
python3 -u run_hpc_few_shot_tuning.py \
    --output /users/ali.gunder/eduterm-finetune/outputs/few_shot_icl_tuning_report.json

echo "Few-Shot ICL Benchmark finished at: $(date)"

