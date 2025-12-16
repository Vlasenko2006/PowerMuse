#!/bin/bash

# Direct training script (run on already allocated GPU node)
# Usage: bash run_train_direct.sh

echo "=================================================="
echo "Training Simple Transformer with Spectral Loss"
echo "=================================================="
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "=================================================="

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
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
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
echo "  - Transformer cascade stages: 3"
echo "  - Dropout: 0.1"
echo ""
echo "  - EnCodec: 24kHz, bandwidth=6.0 (FROZEN)"
echo ""
echo "  - Dataset: dataset_pairs_wav/"
echo "  - Batch size: 16 per GPU × 4 GPUs = 64"
echo "  - Learning rate: 1e-3"
echo "  - Optimizer: AdamW (weight_decay=0.01)"
echo "  - Loss: Combined perceptual loss"
echo "  - Loss weights: input=1.0, target=1.0, spectral=0.01, mel=0.01"
echo "  - Shuffle targets: ENABLED (random pairs)"
echo "  - Complementary Masking: temporal (150 frames, ~1s segments)"
echo "  - Epochs: 200 (with early stopping)"
echo ""
echo "=================================================="
echo ""

# Check if we're in a SLURM allocation
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Detected SLURM job: $SLURM_JOB_ID"
    if [ -z "$SLURM_GPUS" ] && [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        echo "⚠️  No GPUs allocated! Using srun to get GPU access..."
        RUN_CMD="srun --gres=gpu:4 --ntasks=4"
    else
        echo "✓ GPUs already allocated"
        RUN_CMD=""
    fi
else
    echo "Not in SLURM allocation, running directly..."
    RUN_CMD=""
fi

echo ""

# Run training (output goes to terminal and log file)
$RUN_CMD python train_simple_ddp.py \
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
    --unity_test false \
    --shuffle_targets true \
    --anti_cheating 0.3 \
    --loss_weight_input 1.0 \
    --loss_weight_target 1.0 \
    --loss_weight_spectral 0.01 \
    --loss_weight_mel 0.01 \
    --mask_type temporal \
    --mask_temporal_segment 150 \
    --mask_freq_split 0.3 \
    --mask_channel_keep 0.5 \
    --mask_energy_threshold 0.7 \
    2>&1 | tee logs/train_spectral_direct.log

echo ""
echo "=================================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=================================================="

# Show results
if [ -f checkpoints_spectral/best_model.pt ]; then
    echo ""
    echo "✅ Best model saved: checkpoints_spectral/best_model.pt"
    echo ""
    echo "All checkpoints:"
    ls -lh checkpoints_spectral/
else
    echo ""
    echo "⚠️  No model checkpoint found!"
    echo "Check the log above for errors"
fi

echo ""
echo "=================================================="
echo "Training finished"
echo "=================================================="
