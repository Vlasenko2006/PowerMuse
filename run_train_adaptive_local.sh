#!/bin/bash
# Local testing script for Adaptive Window Selection
# Tests training for 2 epochs on small subset

echo "========================================="
echo "LOCAL TEST: Adaptive Window Selection"
echo "========================================="
echo ""

# Configuration for quick local test
DATA_FOLDER="dataset_pairs_wav_24sec"
CHECKPOINT_DIR="checkpoints_adaptive_test"
BATCH_SIZE=1          # Very small for local CPU/GPU
NUM_WORKERS=0         # No multiprocessing for debugging
EPOCHS=2              # Just 2 epochs for testing
LR=1e-4
NOVELTY_WEIGHT=0.1
NUM_PAIRS=3
DEVICE="cpu"          # Change to "cuda" if you have GPU locally

echo "Configuration (LOCAL TEST):"
echo "  Data folder: $DATA_FOLDER"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Num workers: $NUM_WORKERS"
echo "  Epochs: $EPOCHS (just for testing!)"
echo "  Device: $DEVICE"
echo ""

# Create checkpoint directory
mkdir -p $CHECKPOINT_DIR

# Check if data exists
if [ ! -d "$DATA_FOLDER/train" ]; then
    echo "ERROR: Training data not found at $DATA_FOLDER/train"
    echo "Please create dataset first with create_dataset_pairs_wav.py"
    exit 1
fi

# Count training samples
NUM_TRAIN=$(ls $DATA_FOLDER/train/pair_*_input.wav 2>/dev/null | wc -l)
echo "Found $NUM_TRAIN training pairs"
echo ""

if [ $NUM_TRAIN -lt 10 ]; then
    echo "WARNING: Less than 10 training samples. Consider creating more data."
    echo ""
fi

# Run training
echo "Starting local test..."
echo "========================================="
echo ""

python3 train_adaptive_simple.py \
    --data_folder $DATA_FOLDER \
    --checkpoint_dir $CHECKPOINT_DIR \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --epochs $EPOCHS \
    --lr $LR \
    --novelty_weight $NOVELTY_WEIGHT \
    --num_pairs $NUM_PAIRS \
    --device $DEVICE

EXIT_CODE=$?

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ LOCAL TEST PASSED!"
    echo ""
    echo "Next steps:"
    echo "  1. Review checkpoints in: $CHECKPOINT_DIR/"
    echo "  2. Check if losses are decreasing"
    echo "  3. If all looks good, run on HPC: sbatch run_train_adaptive_hpc.sh"
else
    echo "✗ LOCAL TEST FAILED (exit code: $EXIT_CODE)"
    echo ""
    echo "Please fix errors before running on HPC"
fi
echo "========================================="

exit $EXIT_CODE
