#!/bin/bash
#SBATCH --job-name=adaptive_window
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=training/log_adaptive_%j.txt
#SBATCH --error=training/log_adaptive_error_%j.txt
#SBATCH --account=gg0302

# Adaptive Window Selection Training Script for HPC
# Branch: adaptive-window-selection

echo "========================================="
echo "ADAPTIVE WINDOW SELECTION TRAINING"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Configuration
DATA_FOLDER="dataset_pairs_wav_24sec"
CHECKPOINT_DIR="checkpoints_adaptive"
BATCH_SIZE=4          # Small batch size (3x computation per sample)
NUM_WORKERS=8
EPOCHS=100
LR=1e-4
NOVELTY_WEIGHT=0.1
NUM_PAIRS=3

echo "Configuration:"
echo "  Data folder: $DATA_FOLDER"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Batch size: $BATCH_SIZE (per GPU)"
echo "  Num workers: $NUM_WORKERS"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Novelty weight: $NOVELTY_WEIGHT"
echo "  Number of pairs: $NUM_PAIRS"
echo ""

# Create directories
mkdir -p training
mkdir -p $CHECKPOINT_DIR

# Load modules (adjust for your HPC)
module purge
module load python3
module load cuda/11.8
module load nccl

# Activate conda environment
source ~/.bashrc
conda activate your_env_name  # CHANGE THIS!

# Verify GPU availability
echo "GPU Information:"
nvidia-smi
echo ""

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=1  # Single node for now
export RANK=0

# NCCL settings for debugging
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0

# PyTorch settings
export OMP_NUM_THREADS=8

echo "Environment variables set"
echo ""

# Run training
echo "Starting training..."
echo "========================================="
echo ""

python3 train_adaptive_simple.py \
    --data_folder $DATA_FOLDER \
    --checkpoint_dir $CHECKPOINT_DIR \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --epochs $EPOCHS \
    --lr $LR \
    --novelty_weight $NOVELTY_WEIGHT \
    --num_pairs $NUM_PAIRS \
    --device cuda

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Training finished with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "========================================="

exit $EXIT_CODE
