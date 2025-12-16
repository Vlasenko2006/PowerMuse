#!/bin/bash
# Test script to validate arguments before submitting to SLURM

echo "Testing argument parsing for train_simple_ddp.py"
echo "================================================"

# Configuration (same as run_train_creative_agent_fixed.sh)
DATASET_ROOT="dataset_pairs_wav"
ENCODING_DIM=128
ATTENTION_HEADS=8
TRANSFORMER_LAYERS=6
CASCADE_STAGES=3
DROPOUT=0.1
ENCODEC_BW=6.0
ENCODEC_SR=24000
EPOCHS=200
BATCH_SIZE=8
LR=1e-4
WEIGHT_DECAY=0.01
NUM_WORKERS=0
PATIENCE=20
SEED=42
CHECKPOINT_DIR="checkpoints_creative_agent_fixed"
SAVE_EVERY=10
WORLD_SIZE=4
UNITY_TEST=false
SHUFFLE_TARGETS=false
ANTI_CHEATING=0.1
LOSS_WEIGHT_INPUT=0.0
LOSS_WEIGHT_TARGET=0.0
LOSS_WEIGHT_SPECTRAL=0.0
LOSS_WEIGHT_MEL=0.0
MASK_TYPE=none
USE_CREATIVE_AGENT=true
MASK_REG_WEIGHT=0.1
BALANCE_LOSS_WEIGHT=10.0
GAN_WEIGHT=0.1
DISC_LR=5e-5
DISC_UPDATE_FREQ=1
CORR_WEIGHT=0.5

echo "Testing with local-rank=0 (simulating GPU 0)..."
python test_args.py \
    --local-rank 0 \
    --dataset_folder ${DATASET_ROOT} \
    --encoding_dim ${ENCODING_DIM} \
    --nhead ${ATTENTION_HEADS} \
    --num_layers ${TRANSFORMER_LAYERS} \
    --num_transformer_layers ${CASCADE_STAGES} \
    --dropout ${DROPOUT} \
    --encodec_bandwidth ${ENCODEC_BW} \
    --encodec_sr ${ENCODEC_SR} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --num_workers ${NUM_WORKERS} \
    --patience ${PATIENCE} \
    --seed ${SEED} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --save_every ${SAVE_EVERY} \
    --world_size ${WORLD_SIZE} \
    --unity_test ${UNITY_TEST} \
    --shuffle_targets ${SHUFFLE_TARGETS} \
    --anti_cheating ${ANTI_CHEATING} \
    --loss_weight_input ${LOSS_WEIGHT_INPUT} \
    --loss_weight_target ${LOSS_WEIGHT_TARGET} \
    --loss_weight_spectral ${LOSS_WEIGHT_SPECTRAL} \
    --loss_weight_mel ${LOSS_WEIGHT_MEL} \
    --mask_type ${MASK_TYPE} \
    --use_creative_agent ${USE_CREATIVE_AGENT} \
    --mask_reg_weight ${MASK_REG_WEIGHT} \
    --balance_loss_weight ${BALANCE_LOSS_WEIGHT} \
    --gan_weight ${GAN_WEIGHT} \
    --disc_lr ${DISC_LR} \
    --disc_update_freq ${DISC_UPDATE_FREQ} \
    --corr_weight ${CORR_WEIGHT}

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Argument test PASSED - safe to submit job!"
else
    echo ""
    echo "❌ Argument test FAILED - fix errors before submitting!"
    exit 1
fi
