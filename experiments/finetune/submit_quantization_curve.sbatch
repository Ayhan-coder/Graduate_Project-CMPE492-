#!/bin/bash
#SBATCH --job-name=eduterm-quant
#SBATCH --container-image='nvcr.io#nvidia/pytorch:23.10-py3'
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=44G
#SBATCH --output=/users/ali.gunder/eduterm-quantization/slurm-%j.out

echo "================================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "================================================================="

cd /users/ali.gunder/eduterm-quantization

echo "Setting up verified wheels in /tmp/site-packages..."
mkdir -p /tmp/site-packages
export PYTHONPATH=/tmp/site-packages:$PYTHONPATH
pip install --target=/tmp/site-packages --no-deps /users/ali.gunder/wheels_clean/*.whl

echo "Executing Empirical Quantization Degradation Benchmark on NVIDIA A40..."
python3 run_hpc_quantization_curve.py \
    --model-path /users/ali.gunder/models/Qwen2.5-7B-Instruct \
    --sft-adapter /users/ali.gunder/eduterm-finetune/checkpoints/eduterm_7b_tr_sft \
    --output-json /users/ali.gunder/eduterm-quantization/outputs/quantization_degradation_report.json

echo "Quantization Benchmark Job completed at $(date)"
