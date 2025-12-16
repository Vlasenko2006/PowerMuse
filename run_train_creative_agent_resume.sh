#!/bin/bash

# Resume training script for Creative Agent from checkpoint
# Usage: bash run_train_creative_agent_resume.sh

echo "=================================================="
echo "RESUMING Training: Creative Agent from Epoch 20"
echo "=================================================="
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "=================================================="

# Navigate to project directory
cd /work/gg0302/g260141/Jingle_D

# Create logs and checkpoint directories
mkdir -p logs
mkdir -p checkpoints_creative_agent

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
echo "RESUMING from checkpoint_epoch_20.pt"
echo "=================================================="
echo ""
echo "Previous Training Results (First 20 Epochs):"
echo "  ✅ Balance loss: WORKING PERFECTLY!"
echo "     - Input mask: 0.49-0.51 (target: 0.50)"
echo "     - Target mask: 0.49-0.51 (target: 0.50)"
echo "     - Balance loss weight=5.0 is optimal"
echo ""
echo "  📈 Complementarity: EXCELLENT PROGRESS!"
echo "     - Epoch 1:  75.4% (baseline)"
echo "     - Epoch 11: 78.0%"
echo "     - Epoch 16: 83.8%"
echo "     - Epoch 17: 85.0%"
echo "     - Epoch 18: 86.3%"
echo "     - Epoch 19: 87.0% (peak)"
echo "     - Epoch 20: 86.0%"
echo ""
echo "  🎯 Goal: Push complementarity to 90%+ by epoch 50"
echo ""
echo "Configuration:"
echo "  - Model: SimpleTransformer with Creative Agent"
echo "  - Encoding dim: 128"
echo "  - Attention heads: 8"
echo "  - Transformer layers: 6"
echo "  - Transformer cascade stages: 3"
echo "  - Dropout: 0.1"
echo ""
echo "  - Creative Agent: ENABLED 🎨"
echo "    * Learnable attention-based masking"
echo "    * Mask generator: ~500K params"
echo "    * Discriminator: ~200K params"
echo "    * Mask regularization weight: 0.1"
echo "    * Balance loss weight: 15.0 ✅ (INCREASED from 5.0)"
echo ""
echo "  - EnCodec: 24kHz, bandwidth=6.0 (FROZEN)"
echo ""
echo "  - Dataset: dataset_pairs_wav/"
echo "  - Batch size: 8 per GPU × 4 GPUs = 32"
echo "  - Learning rate: 1e-4"
echo "  - Optimizer: AdamW (weight_decay=0.01)"
echo "  - Loss: Combined perceptual + mask regularization"
echo "  - Loss weights: input=0.3, target=0.3, spectral=0.0, mel=0.0, GAN=0.1"
echo "  - Correlation weight: 0.5 (anti-modulation penalty)"
echo "  - Shuffle targets: ENABLED (random pairs for creativity)"
echo "  - Anti-cheating noise: 0.1 (stages 2+)"
echo "  - Fixed masking: DISABLED (creative agent replaces it)"
echo "  - GAN Training: ENABLED (gan_weight=0.1)"
echo "    * Discriminator LR: 5e-5"
echo "    * Update frequency: 1 batch"
echo "  - Epochs: 200 (continuing to epoch 200)"
echo ""
echo "Expected Behavior (Epochs 20-50):"
echo "  - Complementarity: 86% → 90%+ (target)"
echo "  - Mask reg loss: 0.75-0.90 → 0.10-0.20 (decreasing)"
echo "  - Masks stay balanced: 50/50 ± 2%"
echo "  - Temporal diversity increases slightly"
echo ""
echo "=================================================="

# Check if checkpoint exists
CHECKPOINT_PATH="checkpoints_creative_agent_fixed/checkpoint_epoch_20.pt"
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo ""
    echo "❌ ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"
    echo ""
    echo "Available checkpoints:"
    ls -lh checkpoints_creative_agent_fixed/*.pt 2>/dev/null || echo "  No checkpoints in checkpoints_creative_agent_fixed/"
    ls -lh checkpoints_creative_agent/*.pt 2>/dev/null || echo "  No checkpoints in checkpoints_creative_agent/"
    echo ""
    echo "Please update CHECKPOINT_PATH in the script"
    exit 1
fi

echo "✓ Found checkpoint: ${CHECKPOINT_PATH}"
echo "  Size: $(du -h ${CHECKPOINT_PATH} | cut -f1)"
echo ""

# Check for SLURM allocation
if [ -n "$SLURM_JOB_ID" ]; then
    echo ""
    echo "Detected SLURM job: $SLURM_JOB_ID"
    if [ -n "$SLURM_STEP_ID" ]; then
        echo "✓ GPUs already allocated"
        RUN_CMD=""
    fi
else
    echo "Not in SLURM allocation, running directly..."
    RUN_CMD=""
fi

echo ""

# Run training with resume
$RUN_CMD python train_simple_ddp.py \
    --dataset_folder dataset_pairs_wav \
    --encoding_dim 128 \
    --nhead 8 \
    --num_layers 6 \
    --num_transformer_layers 3 \
    --dropout 0.1 \
    --encodec_bandwidth 6.0 \
    --encodec_sr 24000 \
    --epochs 200 \
    --batch_size 8 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --num_workers 0 \
    --patience 20 \
    --seed 42 \
    --checkpoint_dir checkpoints_creative_agent_fixed \
    --save_every 10 \
    --world_size 4 \
    --unity_test false \
    --shuffle_targets true \
    --anti_cheating 0.1 \
    --loss_weight_input 0.3 \
    --loss_weight_target 0.3 \
    --loss_weight_spectral 0.0 \
    --loss_weight_mel 0.0 \
    --mask_type none \
    --use_creative_agent true \
    --mask_reg_weight 0.1 \
    --balance_loss_weight 15.0 \
    --corr_weight 0.5 \
    --gan_weight 0.1 \
    --disc_lr 5e-5 \
    --disc_update_freq 1 \
    --resume ${CHECKPOINT_PATH} \
    2>&1 | stdbuf -oL -eL tee logs/train_creative_agent_resume_epoch20.log

echo ""
echo "=================================================="
echo "Training session completed!"
echo "End time: $(date)"
echo "=================================================="

# Show results
if [ -f checkpoints_creative_agent/best_model.pt ]; then
    echo ""
    echo "✅ Training resumed successfully"
    echo ""
    echo "Latest checkpoints:"
    ls -lth checkpoints_creative_agent/*.pt | head -5
    echo ""
    echo "Recent training log (last 30 lines):"
    tail -30 logs/train_creative_agent_resume_epoch20.log
else
    echo ""
    echo "⚠️  Check for issues in the log above"
fi

echo ""
echo "=================================================="
echo "Training Monitoring Tips:"
echo "  Watch for:"
echo "    - Complementarity reaching 90%+ ✨"
echo "    - Mask reg loss decreasing to 0.10-0.20"
echo "    - Balance staying 50/50 (already achieved!)"
echo "    - Val loss improvement"
echo ""
echo "  If training looks good after 10 more epochs:"
echo "    - Run inference to test outputs"
echo "    - Listen to creative mixing results"
echo "    - Check component statistics"
echo "=================================================="
echo ""
