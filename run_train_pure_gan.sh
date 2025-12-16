#!/bin/bash
# Training script with Pure GAN mode (curriculum learning from music-to-music → noise-to-music)

echo "=================================================="
echo "Pure GAN Mode Training: Creative Agent"
echo "=================================================="
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "=================================================="
echo ""

# Environment setup
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Random port for DDP
export MASTER_PORT=$((29500 + RANDOM % 500))

echo "Environment:"
echo "  Python: $(which python)"
echo "  Working directory: $(pwd)"
echo "  CUDA devices: ${CUDA_VISIBLE_DEVICES:-all}"
echo ""

# Check GPU status
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
    echo ""
fi

echo "=================================================="
echo "Pure GAN Curriculum Learning Configuration"
echo "=================================================="
echo ""
echo "Training Strategy:"
echo "  Phase 1 (Epochs 1-50): Standard music-to-music training"
echo "  Phase 2 (Epochs 51-150): Gradual transition to noise"
echo "    - pure_gan_mode=0.01 (1% per epoch)"
echo "    - Epoch 51: 99% music + 1% noise"
echo "    - Epoch 75: 75% music + 25% noise"  
echo "    - Epoch 100: 50% music + 50% noise"
echo "    - Epoch 125: 25% music + 75% noise"
echo "    - Epoch 150+: 100% noise (pure generation)"
echo ""
echo "Goal: Learn to generate music from pure noise (like traditional GANs)"
echo ""

echo "=================================================="
echo "Model Configuration"
echo "=================================================="
echo "  - Model: SimpleTransformer with Creative Agent"
echo "  - Encoding dim: 128"
echo "  - Attention heads: 8"
echo "  - Transformer layers: 6"
echo "  - Cascade stages: 2"
echo "  - Dropout: 0.1"
echo ""
echo "  🎨 Creative Agent: ENABLED"
echo "    * Learnable attention-based masking"
echo "    * Mask regularization weight: 0.5 (strong)"
echo "    * Balance loss weight: 15.0"
echo ""
echo "  🎲 Pure GAN Mode: ENABLED"
echo "    * Start epoch: 51 (after music-to-music pre-training)"
echo "    * Transition rate: 0.01 (1% noise per epoch)"
echo "    * Full noise at epoch: 150"
echo ""
echo "  - EnCodec: 24kHz, bandwidth=6.0 (FROZEN)"
echo "  - Dataset: dataset_pairs_wav/"
echo "  - Batch size: 8 per GPU × 4 GPUs = 32"
echo "  - Learning rate: 1e-4"
echo "  - Optimizer: AdamW (weight_decay=0.01)"
echo "  - Loss weights: input=0.3, target=0.3, GAN=0.15"
echo "  - Correlation penalty: 0.5"
echo "  - Anti-cheating noise: 0.1"
echo "  - Epochs: 200"
echo ""

# Detect SLURM job
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Detected SLURM job: $SLURM_JOB_ID"
    echo "✓ GPUs already allocated"
else
    echo "Running locally (not in SLURM)"
fi

echo ""
echo "=================================================="
echo ""

# Run training
python train_simple_ddp.py \
    --train_dir dataset_pairs_wav/train \
    --val_dir dataset_pairs_wav/val \
    --checkpoint_dir checkpoints_pure_gan \
    --batch_size 8 \
    --epochs 200 \
    --lr 1e-4 \
    --encoding_dim 128 \
    --nhead 8 \
    --num_layers 6 \
    --num_transformer_layers 2 \
    --dropout 0.1 \
    --loss_weight_input 0.3 \
    --loss_weight_target 0.3 \
    --loss_weight_spectral 0.0 \
    --loss_weight_mel 0.0 \
    --shuffle_targets \
    --use_creative_agent \
    --mask_reg_weight 0.5 \
    --balance_loss_weight 15.0 \
    --anti_cheating 0.1 \
    --gan_weight 0.15 \
    --disc_lr 5e-5 \
    --disc_update_freq 1 \
    --corr_weight 0.5 \
    --pure_gan_mode 0.01 \
    --gan_curriculum_start_epoch 51

echo ""
echo "=================================================="
echo "Training session completed!"
echo "End time: $(date)"
echo "=================================================="
echo ""

echo "⚠️  Check for issues in the log above"
echo ""

echo "=================================================="
echo "Pure GAN Training Progress:"
echo "  Monitor these metrics:"
echo "    - Epochs 1-50: Standard training metrics"
echo "    - Epochs 51-150: Watch alpha increase 0.0 → 1.0"
echo "    - Epoch 150+: Model generates from pure noise"
echo ""
echo "  Expected behavior:"
echo "    - Phase 1: Learn music-to-music transformation"
echo "    - Phase 2: Gradually adapt to noisy inputs"
echo "    - Phase 3: Generate realistic music from noise"
echo ""
echo "  Success criteria:"
echo "    - Val loss remains stable during transition"
echo "    - Discriminator stays balanced (~85% accuracy)"
echo "    - Generated audio sounds musical (not noise)"
echo "=================================================="
