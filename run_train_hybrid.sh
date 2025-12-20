#!/bin/bash

# Hybrid Training - GPU Node Execution (Adaptive Windows + GAN + Full Metrics)
# Usage: bash run_train_hybrid.sh

echo "=================================================="
echo "HYBRID TRAINING: Adaptive Windows + GAN + Full Metrics"
echo "=================================================="
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "=================================================="

# Navigate to project directory
cd /work/gg0302/g260141/Jingle_D

# Create logs and checkpoint directories
mkdir -p logs
mkdir -p checkpoints_hybrid

# Activate environment
source /work/gg0302/g260141/Jingle/multipattern_env/bin/activate

echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "  Working directory: $(pwd)"
echo "  CUDA devices: $CUDA_VISIBLE_DEVICES"
echo ""

# Set environment variables for multi-GPU training
export NCCL_DEBUG=ERROR
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# Check GPU availability
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo ""

echo "=================================================="
echo "HYBRID TRAINING CONFIGURATION"
echo "=================================================="
echo ""
echo "🎯 Combined Features:"
echo "  ✓ Adaptive window selection (3 pairs, 24sec → 16sec)"
echo "  ✓ GAN discriminator training"
echo "  ✓ Full baseline metrics (RMS, spectral, mel)"
echo "  ✓ Compositional agent (rhythm/harmony)"
echo "  ✓ Output correlation analysis"
echo "  ✓ Complete printout formatting"
echo ""
echo "📊 Architecture:"
echo "  - AdaptiveWindowCreativeAgent (3.6M params)"
echo "    • Window selector (3 pairs)"
echo "    • Temporal compressor (1.0-1.5x)"
echo "    • Tonality reducer"
echo "    • Compositional agent (14.6M params)"
echo "  - AudioDiscriminator (GAN)"
echo "  - EnCodec (24.9M params, frozen)"
echo "  - Total trainable: ~21.2M parameters"
echo ""
echo "Configuration:"
echo "  - Input/Target: 24 seconds (576,000 samples, 1200 frames)"
echo "  - Window pairs: 3 x 16 seconds (800 frames each)"
echo "  - Batch size: 6 per GPU × 4 GPUs = 24 total"
echo "  - Learning rate: 1e-4"
echo "  - Optimizer: AdamW"
echo "  - Epochs: 250"
echo ""
echo "Loss Functions:"
echo "  - RMS Input:     0.3"
echo "  - RMS Target:    0.3"
echo "  - Spectral:      0.01"
echo "  - Mel:           0.01"
echo "  - Novelty:       0.1"
echo "  - Correlation:   0.5"
echo "  - Balance:       15.0"
echo "  - GAN:           0.1"
echo ""
echo "Output Metrics (Full Baseline):"
echo "  ✓ Train: loss, RMS (in/tgt), spectral, mel, novelty, correlation"
echo "  ✓ GAN: gen_loss, disc_loss, disc_acc (real/fake)"
echo "  ✓ Correlation: Output→Input, Output→Target (copying detection)"
echo "  ✓ Compositional: rhythm/harmony weights (input/target)"
echo "  ✓ Adaptive: window positions, compression ratios, tonality"
echo "  ✓ Validation: loss, RMS, spectral, mel, novelty"
echo "  ✓ Waveform visualization (first val sample)"
echo ""
echo "Expected Performance:"
echo "  - Epoch 1-5:   Loss 0.015 → 0.008"
echo "  - Epoch 10-25: Loss 0.008 → 0.004"
echo "  - Epoch 30-50: Loss 0.004 → 0.002"
echo "  - Target: Val loss < 0.002"
echo ""
echo "=================================================="

# Verify dataset
echo "Verifying dataset..."
TRAIN_PAIRS=$(ls dataset_pairs_wav_24sec/train/pair_*_input.wav 2>/dev/null | wc -l)
VAL_PAIRS=$(ls dataset_pairs_wav_24sec/val/pair_*_input.wav 2>/dev/null | wc -l)
echo "  Train pairs: $TRAIN_PAIRS"
echo "  Val pairs: $VAL_PAIRS"
echo ""

if [ "$TRAIN_PAIRS" -lt 100 ]; then
    echo "ERROR: Insufficient training data ($TRAIN_PAIRS pairs)"
    echo "Please run: python create_dataset_pairs_wav.py"
    exit 1
fi

echo "=================================================="
echo "STARTING HYBRID TRAINING"
echo "=================================================="
echo ""

# Run hybrid training with DDP
stdbuf -oL -eL python train_hybrid_worker.py \
    --dataset_dir dataset_pairs_wav_24sec \
    --checkpoint_dir checkpoints_hybrid \
    --epochs 250 \
    --batch_size 6 \
    --learning_rate 1e-4 \
    --loss_weight_input 0.3 \
    --loss_weight_target 0.3 \
    --loss_weight_spectral 0.01 \
    --loss_weight_mel 0.01 \
    --corr_weight 0.5 \
    --mask_reg_weight 0.1 \
    --balance_loss_weight 15.0 \
    --gan_weight 0.1 \
    --disc_update_freq 1 \
    --shuffle_targets \
    --world_size 4 \
    2>&1 | tee logs/hybrid_train_$(date +%Y%m%d_%H%M%S).log

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Training session completed!"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================================="

# Show results
if [ -d checkpoints_hybrid ] && [ "$(ls -A checkpoints_hybrid/*.pt 2>/dev/null | wc -l)" -gt 0 ]; then
    echo ""
    echo "✅ Training completed successfully"
    echo ""
    echo "Latest checkpoints:"
    ls -lth checkpoints_hybrid/*.pt | head -5
    echo ""
    echo "Training summary (last 100 lines):"
    tail -100 logs/hybrid_train_*.log | tail -100
    echo ""
    
    # Backup checkpoints
    BACKUP_DIR="/work/gg0302/g260141/checkpoints_backup/hybrid_$(date +%Y%m%d_%H%M%S)"
    echo ""
    echo "Backing up checkpoints to: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    cp -r checkpoints_hybrid/* "$BACKUP_DIR/"
    echo "✅ Backup complete"
else
    echo ""
    echo "⚠️  Training may have failed - check logs above"
    echo "Log file: logs/hybrid_train_*.log"
fi

echo ""
echo "=================================================="
echo "Next Steps:"
echo "  1. Check comprehensive metrics"
echo "     grep 'Train Metrics:' logs/hybrid_train_*.log -A 20"
echo ""
echo "  2. Verify GAN training"
echo "     grep 'GAN Training:' logs/hybrid_train_*.log"
echo ""
echo "  3. Analyze correlation (copying detection)"
echo "     grep 'Correlation Analysis:' logs/hybrid_train_*.log"
echo ""
echo "  4. Check window selection diversity"
echo "     grep 'Adaptive Window Selection:' logs/hybrid_train_*.log -A 5"
echo ""
echo "  5. Monitor compositional weights"
echo "     grep 'Compositional Agent' logs/hybrid_train_*.log"
echo ""
echo "  6. Download checkpoints to local"
echo "     scp -r g260141@levante:/work/gg0302/g260141/Jingle_D/checkpoints_hybrid ."
echo ""
echo "  7. Test inference with trained model"
echo "     python inference_hybrid.py --checkpoint checkpoints_hybrid/hybrid_epoch_100.pt"
echo "=================================================="
echo ""
