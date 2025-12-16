#!/bin/bash
#SBATCH --job-name=creative_push_comp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/creative_push_comp_%j.out
#SBATCH --error=logs/creative_push_comp_%j.err

echo "=================================================="
echo "PUSH COMPLEMENTARITY: Strong Mask Separation Training"
echo "=================================================="
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "=================================================="
echo ""

# Environment setup
source /work/gg0302/g260141/Jingle/multipattern_env/bin/activate 2>/dev/null || source ~/miniconda3/bin/activate multipattern_env

echo "Environment:"
echo "  Python: $(which python3)"
echo "  Working directory: $(pwd)"
echo "  CUDA devices: ${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo ""

# GPU check
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo ""

echo "=================================================="
echo "RESUMING from checkpoint_epoch_51.pt"
echo "INCREASING complementarity weight to PUSH 90%+"
echo "=================================================="
echo ""

# Checkpoint configuration
CHECKPOINT_PATH="checkpoints_creative_agent_fixed/checkpoint_epoch_51.pt"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "❌ ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"
    echo "Available checkpoints:"
    ls -lh checkpoints_creative_agent_fixed/*.pt 2>/dev/null || echo "No checkpoints found"
    exit 1
fi

echo "✓ Found checkpoint: ${CHECKPOINT_PATH}"
echo "  Size: $(du -h ${CHECKPOINT_PATH} | cut -f1)"
echo ""

echo "Previous Training Results (Epochs 21-51):"
echo "  🎨 Complementarity: STUCK at 73-74%"
echo "     - Epoch 20: 86% (before resume)"
echo "     - Epoch 21-51: 73-74% (regressed and stuck)"
echo "     - Goal: Push back to 90%+"
echo ""
echo "  ⚠️  Problem Diagnosis:"
echo "     - Complementarity weight (10.0 in reg_loss × 0.1 mask_reg_weight = 1.0 effective)"
echo "     - RMS losses (0.13 × 0.3 = 0.04 each) dominate complementarity (0.26 × 1.0 = 0.26)"
echo "     - Model converged to local optimum with overlapping masks"
echo ""
echo "  🔧 Solution:"
echo "     - INCREASE mask_reg_weight: 0.1 → 0.5 (5x stronger)"
echo "     - Effective complementarity: 0.26 × 10.0 × 0.5 = 1.3 (vs 0.26 before)"
echo "     - This makes complementarity penalty >> reconstruction losses"
echo "     - Forces model out of local optimum toward 90%+ separation"
echo ""

echo "Configuration:"
echo "  - Model: SimpleTransformer with Creative Agent"
echo "  - Encoding dim: 128"
echo "  - Attention heads: 8"
echo "  - Transformer layers: 6"
echo "  - Transformer cascade stages: 2"
echo "  - Dropout: 0.1"
echo ""
echo "  - Creative Agent: ENABLED 🎨"
echo "    * Learnable attention-based masking"
echo "    * Mask generator: ~500K params"
echo "    * Discriminator: ~200K params"
echo "    * Mask regularization weight: 0.5 ✅ (INCREASED from 0.1)"
echo "    * Balance loss weight: 15.0 ✅"
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
echo "Expected Behavior (Epochs 52-80):"
echo "  - Complementarity: 74% → 85%+ → 90%+ (target)"
echo "  - Mask overlap: 0.26 → 0.15 → 0.10 (decreasing)"
echo "  - Mask reg loss: 0.76 → 0.40 → 0.20 (decreasing)"
echo "  - RMS losses: May temporarily increase (tradeoff)"
echo "  - Masks stay balanced: 50/50 ± 2%"
echo "  - Temporal diversity maintained: ~0.10"
echo ""
echo "=================================================="
echo ""

# Check if running under SLURM
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Detected SLURM job: $SLURM_JOB_ID"
    
    # GPUs are already allocated by SLURM
    echo "✓ GPUs already allocated"
else
    echo "⚠️  Not running under SLURM - using all visible GPUs"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
fi

# Set random port for DDP communication
export MASTER_PORT=$((29500 + RANDOM % 500))
echo "DDP Communication: MASTER_PORT=${MASTER_PORT}"
echo ""

echo "================================================================================"
echo "SIMPLE TRANSFORMER TRAINING ON WAV PAIRS"
echo "================================================================================"
echo "Configuration:"
echo "  Dataset: dataset_pairs_wav"
echo "  Model:"
echo "    - Encoding dim: 128"
echo "    - Attention heads: 8"
echo "    - Internal transformer layers: 6"
echo "    - Cascade stages: 2"
echo "    - Dropout: 0.1"
echo "  EnCodec:"
echo "    - Sample rate: 24000 Hz"
echo "    - Bandwidth: 6.0"
echo "    - Status: FROZEN (encoder + decoder)"
echo "  Training:"
echo "    - Epochs: 200"
echo "    - Batch size: 8 per GPU × 4 GPUs = 32"
echo "    - Learning rate: 0.0001"
echo "    - Optimizer: AdamW (weight_decay=0.01)"
echo "    - Loss: Combined perceptual loss"
echo "      * Input weight: 0.3 (RMS reconstruction)"
echo "      * Target weight: 0.3 (RMS continuation)"
echo "      * Spectral weight: 0.0 (multi-resolution STFT)"
echo "      * Mel weight: 0.0 (mel-spectrogram)"
echo "      * GAN weight: 0.15 (adversarial loss)"
echo "    - Unity test: DISABLED"
echo "    - Shuffle targets: ENABLED (random pairs)"
echo "  🎨 Attention-Based Creative Agent:"
echo "    - Enabled: True (learnable masking)"
echo "    - Mask regularization weight: 0.5 (INCREASED)"
echo "  GAN Training:"
echo "    - Enabled: True (adversarial training)"
echo "    - Discriminator LR: 5.00e-05"
echo "    - Discriminator update frequency: every 1 batch(es)"
echo "  Checkpoints: checkpoints_creative_agent_fixed/"
echo "================================================================================"

python3 train_simple_ddp.py \
    --dataset_folder dataset_pairs_wav \
    --checkpoint_dir checkpoints_creative_agent_fixed \
    --use_creative_agent \
    --mask_reg_weight 0.5 \
    --balance_loss_weight 15.0 \
    --epochs 200 \
    --batch_size 8 \
    --lr 1e-4 \
    --loss_weight_input 0.3 \
    --loss_weight_target 0.3 \
    --corr_weight 0.5 \
    --shuffle_targets \
    --use_gan \
    --gan_weight 0.1 \
    --discriminator_lr 5e-5 \
    --num_cascade_stages 2 \
    --encoding_dim 128 \
    --attention_heads 8 \
    --transformer_layers 6 \
    --anti_cheating 0.1 \
    --resume "${CHECKPOINT_PATH}" \
    2>&1 | tee logs/push_complementarity_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=================================================="
echo "Training session completed!"
echo "End time: $(date)"
echo "=================================================="
echo ""
echo "⚠️  Check for issues in the log above"
echo ""
echo "=================================================="
echo "Training Monitoring Tips:"
echo "  Watch for:"
echo "    - Complementarity increasing: 74% → 85% → 90%+ ✨"
echo "    - Mask overlap decreasing: 0.26 → 0.15 → 0.10"
echo "    - Mask reg loss decreasing: 0.76 → 0.40 → 0.20"
echo "    - Balance staying 50/50 (already achieved!)"
echo "    - RMS losses may increase temporarily (acceptable tradeoff)"
echo ""
echo "  If complementarity reaches 90%+ by epoch 70:"
echo "    - Reduce mask_reg_weight back to 0.2-0.3 (fine-tuning)"
echo "    - Allow RMS losses to optimize while maintaining 90%+"
echo "    - Run inference to test creative mixing quality"
echo "=================================================="
