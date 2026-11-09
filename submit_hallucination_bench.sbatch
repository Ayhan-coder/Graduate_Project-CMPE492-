#!/bin/bash
#SBATCH --job-name=eval-hallucination
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=44G
#SBATCH --time=01:30:00
#SBATCH --output=/users/ali.gunder/eduterm-finetune/slurm-hallucination-%j.out

echo "================================================================="
echo "Hallucination & Faithfulness Adversarial Stress Test on HPC A40"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname) | Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "================================================================="

export ENROOT_DATA_PATH=/dev/shm/enroot-data-22212
export ENROOT_RUNTIME_PATH=/dev/shm/enroot-runtime-22212

mkdir -p /users/ali.gunder/eduterm-finetune/outputs
cd /users/ali.gunder/eduterm-finetune

echo "Starting container execution via enroot on /dev/shm..."
enroot start --rw \
    --mount /users/ali.gunder:/users/ali.gunder \
    pytorch23 bash -c '
        mkdir -p /tmp/site-packages
        export PYTHONPATH=/tmp/site-packages:$PYTHONPATH
        pip install --target=/tmp/site-packages --no-deps /users/ali.gunder/wheels_clean/*.whl
        export HF_HOME=/users/ali.gunder/hf_cache
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export PYTHONUNBUFFERED=1
        cd /users/ali.gunder/eduterm-finetune
        python3 -u run_hpc_hallucination_bench.py --output /users/ali.gunder/eduterm-finetune/outputs/hallucination_faithfulness_report.json
    '

echo "Hallucination Benchmark finished at: $(date)"

