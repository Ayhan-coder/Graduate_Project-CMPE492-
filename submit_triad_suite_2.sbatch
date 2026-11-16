#!/bin/bash
#SBATCH --job-name=eval-triad-2
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=44G
#SBATCH --time=03:30:00
#SBATCH --output=/users/ali.gunder/eduterm-finetune/slurm-triad2-%j.out

echo "================================================================="
echo "Sequential Triad Benchmark Suite 2 on Boğaziçi HPC NVIDIA A40"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname) | Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "================================================================="

export ENROOT_DATA_PATH=/dev/shm/enroot-data-22212
export ENROOT_RUNTIME_PATH=/dev/shm/enroot-runtime-22212
export ENROOT_CACHE_PATH=/var/lib/enroot-cache/users
export ENROOT_TEMP_PATH=/dev/shm

mkdir -p /dev/shm/enroot-data-22212 /dev/shm/enroot-runtime-22212
mkdir -p /users/ali.gunder/eduterm-finetune/outputs
cd /users/ali.gunder/eduterm-finetune

if [ ! -d "/dev/shm/enroot-data-22212/pytorch23" ]; then
    echo "Container pytorch23 not found in /dev/shm. Creating from cached image..."
    enroot import -o /dev/shm/pytorch-23.10.sqsh docker://nvcr.io#nvidia/pytorch:23.10-py3
    enroot create --name pytorch23 /dev/shm/pytorch-23.10.sqsh
    rm -f /dev/shm/pytorch-23.10.sqsh
    echo "Container pytorch23 created successfully."
fi

echo "Starting sequential triad benchmark suite 2 via enroot on /dev/shm..."
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

        echo ">>> [1/3] Launching Pedagogical Safety, Anti-Cheating & Socratic Guardrails Benchmark..."
        python3 -u run_hpc_pedagogical_safety.py --output /users/ali.gunder/eduterm-finetune/outputs/pedagogical_safety_guardrails_report.json
        echo ">>> [1/3] Finished at: $(date)"

        echo ">>> [2/3] Launching BKT & Knowledge Graph Guided Adaptive Curriculum Benchmark..."
        python3 -u run_hpc_adaptive_curriculum.py --output /users/ali.gunder/eduterm-finetune/outputs/adaptive_curriculum_bkt_report.json
        echo ">>> [2/3] Finished at: $(date)"

        echo ">>> [3/3] Launching Zero-Shot Cross-Lingual Pedagogical Transfer Benchmark (TR -> EN/AZ/DE)..."
        python3 -u run_hpc_cross_lingual_transfer.py --output /users/ali.gunder/eduterm-finetune/outputs/cross_lingual_transfer_report.json
        echo ">>> [3/3] Finished at: $(date)"
    '

echo "================================================================="
echo "All 3 Triad Suite 2 Benchmarks completed successfully at: $(date)"
echo "================================================================="

