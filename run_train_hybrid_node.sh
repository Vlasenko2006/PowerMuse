#!/bin/bash
#SBATCH --job-name=hybrid_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/hybrid_train_%j.out
#SBATCH --error=logs/hybrid_train_%j.err
#SBATCH --account=gg0302
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andrey.vlasenko@mpimet.mpg.de

################################################################################
# HYBRID TRAINING: Adaptive Windows + GAN + Full Metrics
################################################################################
# Combines:
#   - Adaptive window selection (3 pairs, 24sec → 16sec compression)
#   - GAN discriminator training
#   - Full baseline metrics (RMS, spectral, mel, correlation)
#   - Compositional agent (rhythm/harmony components)
#   - Complete printout formatting
################################################################################

echo "================================================================================"
echo "HYBRID TRAINING: Adaptive Windows + GAN + Full Metrics"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "================================================================================"

# Create logs directory
mkdir -p logs

# Environment setup
echo "Activating environment..."
source /work/gg0302/g260141/Jingle/multipattern_env/bin/activate

# Verify Python and PyTorch
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# NCCL settings (suppress verbose output)
export NCCL_DEBUG=ERROR
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3

# Dataset path
DATASET_DIR="/work/gg0302/g260141/datasets/dataset_pairs_wav_24sec"
CHECKPOINT_DIR="./checkpoints_hybrid"

echo "================================================================================"
echo "Training Configuration:"
echo "================================================================================"
echo "Dataset: $DATASET_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Epochs: 250"
echo "Batch size per GPU: 6"
echo "Total batch size: 24"
echo "Learning rate: 1e-4"
echo ""
echo "Loss Weights:"
echo "  - RMS Input:     0.3"
echo "  - RMS Target:    0.3"
echo "  - Spectral:      0.01"
echo "  - Mel:           0.01"
echo "  - Novelty:       0.1"
echo "  - Correlation:   0.5"
echo "  - Balance:       15.0"
echo "  - GAN:           0.1"
echo ""
echo "Features:"
echo "  ✓ Adaptive window selection (3 pairs)"
echo "  ✓ Temporal compression (1.0-1.5x)"
echo "  ✓ Tonality reduction"
echo "  ✓ Compositional agent (rhythm/harmony)"
echo "  ✓ GAN discriminator training"
echo "  ✓ Full correlation analysis"
echo "  ✓ Complete metrics display"
echo "================================================================================"

# Run training with output buffering disabled
stdbuf -oL -eL python train_hybrid_worker.py \
    --dataset_dir "$DATASET_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --epochs 250 \
    --batch_size 6 \
    --learning_rate 1e-4 \
    --loss_weight_input 0.3 \
    --loss_weight_target 0.3 \
    --loss_weight_spectral 0.01 \
    --loss_weight_mel 0.01 \
    --corr_weight 0.5 \
    --mask_reg_weight 0.1 \
    --balance_loss_weight 15.0 \
    --gan_weight 0.1 \
    --disc_update_freq 1 \
    --world_size 4 \
    2>&1 | tee "logs/hybrid_train_${SLURM_JOB_ID}_console.log"

echo ""
echo "================================================================================"
echo "Training complete!"
echo "Job ID: $SLURM_JOB_ID"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Console log: logs/hybrid_train_${SLURM_JOB_ID}_console.log"
echo "================================================================================"
