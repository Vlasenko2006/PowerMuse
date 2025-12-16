#!/bin/bash

#SBATCH --job-name=train_spectral
#SBATCH --partition=gpu
#SBATCH --account=gg0302
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:59:00
#SBATCH --mem=64G
#SBATCH --output=logs/train_spectral.out
#SBATCH --error=logs/train_spectral.err

# Train Simple Transformer on WAV Pairs with DDP
echo "=================================================="
echo "Training Simple Transformer (DDP)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4"
echo "Memory: 64G"
echo "Start time: $(date)"
echo "=================================================="

# Load modules
module purge
module load python3/2022.01-gcc-11.2.0

# Navigate to project directory
cd /work/gg0302/g260141/Jingle_D

# Create logs and checkpoint directories
mkdir -p logs
mkdir -p checkpoints_spectral

# Activate environment
source /work/gg0302/g260141/Jingle/multipattern_env/bin/activate

echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  Working directory: $(pwd)"
echo "  CUDA devices: $CUDA_VISIBLE_DEVICES"
echo ""

# Set environment variables for multi-GPU training
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Check GPU availability
nvidia-smi

echo ""
echo "=================================================="
echo "Starting DDP training..."
echo "=================================================="
echo ""
echo "Configuration:"
echo "  - Model: SimpleTransformer"
echo "  - Encoding dim: 128"
echo "  - Attention heads: 8"
echo "  - Transformer layers: 4"
echo "  - Transformer cascade stages: 1 (set to 3 for cascade)"
echo "  - Dropout: 0.1"
echo ""
echo "  - EnCodec: 24kHz, bandwidth=6.0 (FROZEN)"
echo ""
echo "  - Dataset: dataset_pairs_wav/"
echo "  - Batch size: 16 per GPU × 4 GPUs = 64"
echo "  - Learning rate: 1e-3"
echo "  - Optimizer: AdamW (weight_decay=0.01)"
echo "  - Loss: Unity test (output → input)"
echo "  - Loss weights: input=1.0 (all others=0), GAN=0.0 (disabled)"
echo "  - GAN Training: DISABLED (gan_weight=0.0)"
echo "  - Epochs: 200 (with early stopping)"
echo ""
echo "=================================================="
echo ""

# Run training
python train_simple_ddp.py \
    --dataset_folder dataset_pairs_wav \
    --encoding_dim 128 \
    --nhead 8 \
    --num_layers 4 \
    --num_transformer_layers 1 \
    --dropout 0.1 \
    --encodec_bandwidth 6.0 \
    --encodec_sr 24000 \
    --epochs 200 \
    --batch_size 16 \
    --lr 1e-3 \
    --weight_decay 0.01 \
    --num_workers 0 \
    --patience 20 \
    --seed 42 \
    --checkpoint_dir checkpoints_spectral \
    --save_every 10 \
    --world_size 4 \
    --unity_test true \
    --loss_weight_input 1.0 \
    --loss_weight_target 0.0 \
    --loss_weight_spectral 0.0 \
    --loss_weight_mel 0.0 \
    --gan_weight 0.0 \
    --disc_lr 5e-5 \
    --disc_update_freq 1

echo ""
echo "=================================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=================================================="

# Show results
if [ -f checkpoints_spectral/best_model.pt ]; then
    echo ""
    echo "Best model saved: checkpoints_spectral/best_model.pt"
    echo ""
    echo "All checkpoints:"
    ls -lh checkpoints_spectral/
    echo ""
    echo "Training logs:"
    tail -50 logs/train_spectral.out
else
    echo ""
    echo "ERROR: No model checkpoint found!"
    echo "Check logs/train_spectral.err for errors"
fi

echo ""
echo "=================================================="
echo "Job finished"
echo "=================================================="
