#!/bin/bash
#SBATCH --job-name=adaptive_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --output=logs/adaptive_train_%j.out
#SBATCH --error=logs/adaptive_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --account=gg0302

################################################################################
# ADAPTIVE WINDOW SELECTION TRAINING - HPC DEPLOYMENT
################################################################################
#
# This job trains the AdaptiveWindowCreativeAgent with:
# - 24-second input/target segments
# - 3 pairs of 16-second windows selected adaptively per sample
# - Temporal compression (1.0x-1.5x)
# - Tonality transformation
# - CompositionalCreativeAgent integration
#
# Expected training time: 24-36 hours for 50 epochs
# Peak memory usage: ~60GB per GPU (4x A100)
################################################################################

# Environment setup
echo "=================================="
echo "ADAPTIVE WINDOW TRAINING - START"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================="

# Load modules
module purge
module load python3/2023.01
module load cuda/12.1
module load nccl/2.18.3

# Activate conda environment
source /work/gg0302/g260141/miniforge3/bin/activate
conda activate /work/gg0302/g260141/conda/powermusic

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Create logs directory
mkdir -p logs checkpoints_adaptive

# Print configuration
echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Change to project directory
cd /work/gg0302/g260141/Jingle_D

# Verify dataset
echo "Verifying dataset..."
TRAIN_PAIRS=$(ls dataset_pairs_wav_24sec/train/pair_*_input.wav 2>/dev/null | wc -l)
VAL_PAIRS=$(ls dataset_pairs_wav_24sec/val/pair_*_input.wav 2>/dev/null | wc -l)
echo "  Train pairs: $TRAIN_PAIRS"
echo "  Val pairs: $VAL_PAIRS"

if [ "$TRAIN_PAIRS" -lt 100 ]; then
    echo "ERROR: Insufficient training data ($TRAIN_PAIRS pairs)"
    exit 1
fi

# Run training with DDP
echo ""
echo "=================================="
echo "STARTING TRAINING (DDP)"
echo "=================================="
echo ""

python train_adaptive_ddp.py \
    --dataset_folder dataset_pairs_wav_24sec \
    --batch_size 6 \
    --num_workers 8 \
    --epochs 50 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --checkpoint_dir checkpoints_adaptive \
    --world_size 4 \
    --num_pairs 3 \
    --encoding_dim 128 \
    --nhead 8 \
    --num_layers 6 \
    --num_transformer_layers 3 \
    --dropout 0.1 \
    --encodec_bandwidth 6.0 \
    --encodec_sr 24000 \
    --use_compositional_agent true \
    --mask_reg_weight 0.1 \
    --balance_loss_weight 25.0 \
    --corr_weight 0.5 \
    --loss_weight_spectral 0.01 \
    --loss_weight_target 0.00 \
    --loss_weight_input 0.00 \
    --gan_weight 0.01 \
    --disc_lr 5e-5 \
    --disc_update_freq 1 \
    --shuffle_targets true \
    --anti_cheating 0.0 \
    --save_every 1 \
    --patience 50 \
    --seed 42

EXIT_CODE=$?

echo ""
echo "=================================="
echo "TRAINING FINISHED"
echo "=================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=================================="

# Copy checkpoints to backup location if training succeeded
if [ $EXIT_CODE -eq 0 ]; then
    echo "Backing up checkpoints..."
    mkdir -p /work/gg0302/g260141/checkpoints_backup/adaptive_$(date +%Y%m%d_%H%M%S)
    cp -r checkpoints_adaptive/* /work/gg0302/g260141/checkpoints_backup/adaptive_$(date +%Y%m%d_%H%M%S)/
    echo "Backup complete"
fi

exit $EXIT_CODE
