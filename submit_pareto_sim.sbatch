#!/bin/bash
#SBATCH --job-name=eval-pareto
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/users/ali.gunder/eduterm-finetune/slurm-pareto-%j.out

echo "================================================================="
echo "Experiment 3: Pedagogical Multi-Objective Pareto Optimization"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname) | Date: $(date)"
echo "================================================================="

cd /users/ali.gunder/eduterm-finetune
python3 -u run_hpc_pareto_sim.py \
    --output /users/ali.gunder/eduterm-finetune/outputs/pareto_optimization_report.json

echo "Pareto Optimization finished at: $(date)"

