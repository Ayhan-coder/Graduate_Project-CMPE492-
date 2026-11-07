#!/bin/bash
#SBATCH --job-name=eduterm-judge
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=44G
#SBATCH --output=/users/ali.gunder/eduterm-judge/slurm-%j.out

echo "================================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "================================================================="

cd /users/ali.gunder/eduterm-judge

echo "Setting up verified wheels in /tmp/site-packages..."
mkdir -p /tmp/site-packages
export PYTHONPATH=/tmp/site-packages:$PYTHONPATH
pip install --target=/tmp/site-packages --no-deps /users/ali.gunder/wheels_clean/*.whl

echo "Executing LLM-as-a-Judge Empirical Validation against Human Annotators..."
python3 run_hpc_llm_judge.py \
    --model-path /users/ali.gunder/models/Qwen2.5-7B-Instruct \
    --a1-csv annotation/labels_a1.csv \
    --a2-csv annotation/labels_a2.csv \
    --output-csv annotation/labels_7b_judge.csv \
    --output-json outputs/llm_judge_human_agreement_report.json

echo "LLM Judge Validation completed at $(date)"
