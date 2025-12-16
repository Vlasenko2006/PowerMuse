#!/bin/bash
#SBATCH --job-name=jingle_compositional
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/compositional_%j.out
#SBATCH --error=logs/compositional_%j.err
#SBATCH --account=gg0302

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
echo ""

# Run training script
bash run_train_compositional.sh

echo ""
echo "Job finished at: $(date)"
