#!/bin/bash
# Train CASCADE model with Compositional Creative Agent
# 
# Compositional agent extracts musical components:
# - Rhythm (small kernels 3-5): Temporal patterns, beat, envelope
# - Harmony (medium kernels 7-9): Pitch relationships, chords, melody
# - Timbre (large kernels 15-21): Tone color, texture, instrumentation
#
# Then composes NEW patterns by intelligently recombining components
# from input and target, creating truly creative output.

set -e  # Exit on error

# Activate environment (for SLURM/Levante)
if [ -f /work/gg0302/g260141/Jingle/multipattern_env/bin/activate ]; then
    echo "Activating environment..."
    source /work/gg0302/g260141/Jingle/multipattern_env/bin/activate
fi

echo "================================================================================"
echo "CASCADE TRAINING WITH COMPOSITIONAL CREATIVE AGENT"
echo "================================================================================"

# Configuration
NUM_GPUS=4        # Number of GPUs to use (4 for multi-GPU training)
BATCH_SIZE=16     # Batch size per GPU (total = 16×4 = 64)
NUM_EPOCHS=50
LEARNING_RATE=1e-4

# Cascade configuration
NUM_CASCADE_STAGES=2
ANTI_CHEAT=0.0

# Compositional creative agent (NEW!)
USE_COMPOSITIONAL=true
NOVELTY_WEIGHT=0.3  # Weight for novelty regularization (INCREASED to force creativity!)
CORR_WEIGHT=0.5     # Weight for anti-modulation correlation cost (PREVENTS COPYING!)

# Loss weights
LOSS_WEIGHT_RMS_INPUT=0.0
LOSS_WEIGHT_RMS_TARGET=0.0
LOSS_WEIGHT_SPECTRAL=0.01  # Musical structure guidance - PREVENTS PURE COPYING!
LOSS_WEIGHT_MEL=0.0

# GAN training (optional)
GAN_WEIGHT=0.01  # Combine with compositional agent for best results
DISC_LR=5e-5
DISC_UPDATE_FREQ=1

# Paths
DATASET_FOLDER="dataset_pairs_wav"
CHECKPOINT_DIR="checkpoints_compositional"

# Training parameters
WORLD_SIZE=4      # Number of parallel processes (= NUM_GPUS for DDP)
NUM_WORKERS=0
PATIENCE=20
SEED=42
SAVE_EVERY=10

# Model architecture
ENCODING_DIM=128
NHEAD=8
NUM_LAYERS=4
DROPOUT=0.1

# EnCodec
ENCODEC_BANDWIDTH=6.0
ENCODEC_SR=24000

# Test modes
UNITY_TEST=false
SHUFFLE_TARGETS=false

# Masking (not used with compositional agent, but required by args)
MASK_TYPE=none

echo ""
echo "Training Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Batch size: $BATCH_SIZE (per GPU)"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo ""
echo "Cascade Configuration:"
echo "  Number of stages: $NUM_CASCADE_STAGES"
echo "  Anti-cheating noise: $ANTI_CHEAT"
echo ""
echo "🎼 Compositional Creative Agent:"
echo "  Enabled: $USE_COMPOSITIONAL"
echo "  Novelty weight: $NOVELTY_WEIGHT"
echo "  Components:"
echo "    - Rhythm: Conv1d kernels 3,5 (fast temporal patterns)"
echo "    - Harmony: Conv1d kernels 7,9 (melodic patterns)"
echo "    - Timbre: Conv1d kernels 15,21 (texture, tone color)"
echo "  Composer: 4-layer transformer with 8 attention heads"
echo "  Parameters: ~14.6M (in addition to cascade model)"
echo ""
echo "Loss Configuration:"
echo "  RMS Input: $LOSS_WEIGHT_RMS_INPUT"
echo "  RMS Target: $LOSS_WEIGHT_RMS_TARGET"
echo "  Spectral: $LOSS_WEIGHT_SPECTRAL (musical structure)"
echo "  Mel-spectrogram: $LOSS_WEIGHT_MEL"
echo "  Novelty regularization: $NOVELTY_WEIGHT"
echo "  🚫 Anti-modulation correlation: $CORR_WEIGHT (PREVENTS COPYING!)"
echo ""
echo "Anti-Modulation Correlation Cost:"
echo "  Splits audio into 250 segments, computes max amplitude per segment"
echo "  Calculates correlation between output envelope and input/target envelopes"
echo "  Cost = -ln(1 - |corr_input|) - ln(1 - |corr_target|)"
echo "  Result: Exponential penalty when correlation → 1.0 (copying detected!)"
echo ""
if [ "$GAN_WEIGHT" != "0.0" ]; then
    echo "GAN Training: ENABLED (gan_weight=$GAN_WEIGHT)"
    echo "  * Discriminator LR: $DISC_LR"
    echo "  * Update frequency: every $DISC_UPDATE_FREQ batch"
else
    echo "GAN Training: DISABLED (gan_weight=0.0)"
fi
echo ""
echo "Dataset:"
echo "  Dataset folder: $DATASET_FOLDER"
echo "  Checkpoint: $CHECKPOINT_DIR/"
echo "================================================================================"

# Create checkpoint directory
mkdir -p $CHECKPOINT_DIR

# Run training
python train_simple_ddp.py \
    --dataset_folder $DATASET_FOLDER \
    --checkpoint_dir $CHECKPOINT_DIR \
    --encoding_dim $ENCODING_DIM \
    --nhead $NHEAD \
    --num_layers $NUM_LAYERS \
    --num_transformer_layers $NUM_CASCADE_STAGES \
    --dropout $DROPOUT \
    --encodec_bandwidth $ENCODEC_BANDWIDTH \
    --encodec_sr $ENCODEC_SR \
    --epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --weight_decay 0.01 \
    --num_workers $NUM_WORKERS \
    --patience $PATIENCE \
    --seed $SEED \
    --save_every $SAVE_EVERY \
    --world_size $WORLD_SIZE \
    --unity_test $UNITY_TEST \
    --shuffle_targets $SHUFFLE_TARGETS \
    --anti_cheating $ANTI_CHEAT \
    --loss_weight_input $LOSS_WEIGHT_RMS_INPUT \
    --loss_weight_target $LOSS_WEIGHT_RMS_TARGET \
    --loss_weight_spectral $LOSS_WEIGHT_SPECTRAL \
    --loss_weight_mel $LOSS_WEIGHT_MEL \
    --mask_type $MASK_TYPE \
    --use_compositional_agent $USE_COMPOSITIONAL \
    --mask_reg_weight $NOVELTY_WEIGHT \
    --corr_weight $CORR_WEIGHT \
    --gan_weight $GAN_WEIGHT \
    --disc_lr $DISC_LR \
    --disc_update_freq $DISC_UPDATE_FREQ

echo ""
echo "================================================================================"
echo "✓ Training complete!"
echo "  Checkpoints saved in: $CHECKPOINT_DIR/"
echo "  Best model: $CHECKPOINT_DIR/best_model.pt"
echo ""
echo "Next steps:"
echo "  1. Run inference to test creative output:"
echo "     python inference_cascade.py --checkpoint $CHECKPOINT_DIR/best_model.pt"
echo ""
echo "  2. Listen to generated audio in inference_outputs/"
echo ""
echo "  3. Check component statistics in training logs:"
echo "     - input_rhythm_weight: How much input rhythm is used"
echo "     - target_harmony_weight: How much target harmony is used"
echo "     - novelty: Measures how different output is from inputs"
echo "================================================================================"
