#!/bin/bash

#SBATCH --job-name=multipattern
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --account=gg0302
#SBATCH --partition=gpu
#SBATCH --error=logs/train-%j.err
#SBATCH --output=logs/train-%j.out
#SBATCH --exclusive
#SBATCH --mem=0

# Multi-Pattern Audio Fusion Training on HPC Levante
echo "=================================================="
echo "Multi-Pattern Fusion Training - HPC Levante"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4"
echo "Start time: $(date)"
echo "=================================================="

# Set up distributed training environment
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((10000 + RANDOM % 50000))

module load git
export PATH=/sw/spack-levante/git-2.43.7-2ofazl/bin:$PATH

export MASTER_ADDR MASTER_PORT

# Set NCCL environment for multi-GPU
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0

# Navigate to project directory
cd /work/gg0302/g260141/Jingle

# Create logs directory
mkdir -p logs

# Activate environment (adjust path if needed)
# Option 1: Use dedicated multipattern environment
source multipattern_env/bin/activate
# Option 2: Use existing BART environment (comment out Option 1, uncomment this)
# source /work/gg0302/g260141/BART/bart_env/bin/activate

echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  Master address: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"
echo "  Working directory: $(pwd)"
echo ""

# Run distributed training
echo "Starting distributed training..."
srun python main_multipattern_ddp.py

echo ""
echo "=================================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=================================================="
