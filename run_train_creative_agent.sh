#!/bin/bash

# Direct training script for Creative Agent (run on already allocated GPU node)
# Usage: bash run_train_creative_agent.sh

echo "=================================================="
echo "Training Simple Transformer with Creative Agent"
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
echo "Starting DDP training with Creative Agent..."
echo "=================================================="
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
echo "    * Balance loss weight: 5.0 (enforces 50/50 input/target mixing)"
echo ""
echo "  - EnCodec: 24kHz, bandwidth=6.0 (FROZEN)"
echo ""
echo "  - Dataset: dataset_pairs_wav/"
echo "  - Batch size: 8 per GPU × 4 GPUs = 32"
echo "  - Learning rate: 1e-4"
echo "  - Optimizer: AdamW (weight_decay=0.01)"
echo "  - Loss: Combined perceptual + mask regularization"
echo "  - Loss weights: input=0.0, target=1.0, spectral=0.0, mel=0.0, GAN=0.0 (disabled)"
echo "  - Shuffle targets: DISABLED (continuation pairs)"
echo "  - Anti-cheating noise: 0.1 (stages 2+)"
echo "  - Fixed masking: DISABLED (creative agent replaces it)"
echo "  - GAN Training: DISABLED (gan_weight=0.0)"
echo "    * Discriminator LR: 5e-5 (if enabled)"
echo "    * Update frequency: 1 batch (if enabled)"
echo "  - Epochs: 200 (with early stopping)"
echo ""
echo "Expected Behavior:"
echo "  - Complementarity: ~75% (untrained) → 85-95% (after 50+ epochs)"
echo "  - Mask reg loss: ~0.25 → 0.05-0.10"
echo "  - Adaptive masking per song pair"
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
# To resume: uncomment --resume_from and specify checkpoint path
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
    --checkpoint_dir checkpoints_creative_agent \
    --save_every 10 \
    --world_size 4 \
    --unity_test false \
    --shuffle_targets false \
    --anti_cheating 0.1 \
    --loss_weight_input 0.0 \
    --loss_weight_target 1.0 \
    --loss_weight_spectral 0.0 \
    --loss_weight_mel 0.0 \
    --mask_type none \
    --use_creative_agent true \
    --mask_reg_weight 0.1 \
    --balance_loss_weight 5.0 \
    --gan_weight 0.0 \
    --disc_lr 5e-5 \
    --disc_update_freq 1 \
    2>&1 | stdbuf -oL -eL tee logs/train_creative_agent_direct.log
    
# RESUME OPTIONS (uncomment one):
# --resume_from checkpoints_creative_agent/best_model.pt \          # Resume from best validation loss
# --resume_from checkpoints_creative_agent/checkpoint_epoch_20.pt \ # Resume from specific epoch

echo ""
echo "=================================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=================================================="

# Show results
if [ -f checkpoints_creative_agent/best_model.pt ]; then
    echo ""
    echo "✅ Best model saved: checkpoints_creative_agent/best_model.pt"
    echo ""
    echo "All checkpoints:"
    ls -lh checkpoints_creative_agent/
    echo ""
    echo "Recent training log:"
    tail -20 logs/train_creative_agent_direct.log
else
    echo ""
    echo "⚠️  No model checkpoint found!"
    echo "Check the log above for errors"
fi

echo ""
echo "=================================================="
echo "Next Steps:"
echo "  1. Check complementarity improvement in logs"
echo "  2. Compare with fixed masking (temporal)"
echo "  3. Test creative agent: python creative_agent.py"
echo "  4. Listen to outputs"
echo "=================================================="
echo ""
echo "Training finished"
echo "=================================================="
