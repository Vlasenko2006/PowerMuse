#!/bin/bash

# Adaptive Window Selection Training - GPU Node Execution
# Usage: bash run_train_adaptive_node.sh

echo "=================================================="
echo "TRAINING: Adaptive Window Selection + Creative Agent"
echo "=================================================="
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "=================================================="

# Navigate to project directory
cd /work/gg0302/g260141/Jingle_D

# Create logs and checkpoint directories
mkdir -p logs
mkdir -p checkpoints_adaptive

# Activate environment
source /work/gg0302/g260141/miniforge3/bin/activate
conda activate /work/gg0302/g260141/conda/powermusic

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
export NCCL_DEBUG=WARN
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
echo "ADAPTIVE WINDOW SELECTION TRAINING"
echo "=================================================="
echo ""
echo "🎯 Innovation:"
echo "  - Input segments: 24 seconds (extended from 16)"
echo "  - Window selection: 3 pairs of 16-second windows per sample"
echo "  - Adaptive selection: Agent learns optimal window positions"
echo "  - Temporal compression: 1.0x - 1.5x per window"
echo "  - Tonality transformation: Learnable harmonic adjustment"
echo "  - Gradient flow: Mean loss across 3 pairs"
echo ""
echo "📊 Local Test Results (Dec 20, 2025):"
echo "  ✅ Epoch 1: Train 0.0050 → Val 0.0044"
echo "  ✅ Epoch 2: Train 0.0015 → Val 0.0028"
echo "  ✅ Window diversity: 169-254 frame starts"
echo "  ✅ No memory errors with 3x computation"
echo ""
echo "Configuration:"
echo "  - Base Model: SimpleTransformer (24.9M params, frozen EnCodec)"
echo "  - Creative Agent: CompositionalCreativeAgent (14.6M params)"
echo "  - Adaptive Agent: AdaptiveWindowCreativeAgent (3.6M new params)"
echo "  - Total trainable: ~21.2M parameters"
echo ""
echo "  - Input/Target: 24 seconds per segment (1200 frames)"
echo "  - Window size: 16 seconds (800 frames)"
echo "  - Number of pairs: 3 (multi-window analysis)"
echo "  - Compression range: 1.0x - 1.5x"
echo ""
echo "  - Dataset: dataset_pairs_wav_24sec/"
echo "  - Batch size: 6 per GPU × 4 GPUs = 24"
echo "  - Learning rate: 1e-4"
echo "  - Optimizer: AdamW"
echo "  - Epochs: 50"
echo "  - Novelty weight: 0.1"
echo ""
echo "Loss Functions:"
echo "  - Spectral loss: 0.01 (frequency domain)"
echo "  - Novelty loss: 0.1 (component diversity)"
echo "  - Correlation penalty: 0.5 (anti-modulation)"
echo "  - GAN loss: 0.01 (realistic outputs)"
echo ""
echo "Expected Metrics:"
echo "  - Epoch 1-5:  Loss 0.015 → 0.008"
echo "  - Epoch 10-25: Loss 0.008 → 0.004"
echo "  - Epoch 30-50: Loss 0.004 → 0.002"
echo "  - Target: Val loss < 0.002 (better than baseline)"
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
echo "STARTING TRAINING"
echo "=================================================="
echo ""

# Run training with DDP
python train_adaptive_ddp.py \
    --dataset_folder dataset_pairs_wav_24sec \
    --batch_size 6 \
    --num_workers 8 \
    --epochs 50 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --checkpoint_dir checkpoints_adaptive \
    --world_size 4 \
    --num_pairs 3 \
    --encoding_dim 128 \
    --nhead 8 \
    --num_layers 6 \
    --num_transformer_layers 3 \
    --dropout 0.1 \
    --encodec_bandwidth 6.0 \
    --encodec_sr 24000 \
    --use_compositional_agent true \
    --mask_reg_weight 0.1 \
    --balance_loss_weight 25.0 \
    --corr_weight 0.5 \
    --loss_weight_spectral 0.01 \
    --loss_weight_target 0.00 \
    --loss_weight_input 0.00 \
    --gan_weight 0.01 \
    --disc_lr 5e-5 \
    --disc_update_freq 1 \
    --shuffle_targets true \
    --anti_cheating 0.0 \
    --save_every 1 \
    --patience 50 \
    --seed 42 \
    2>&1 | stdbuf -oL -eL tee logs/train_adaptive_$(date +%Y%m%d_%H%M%S).log

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Training session completed!"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================================="

# Show results
if [ -f checkpoints_adaptive/best_model.pt ]; then
    echo ""
    echo "✅ Training completed successfully"
    echo ""
    echo "Latest checkpoints:"
    ls -lth checkpoints_adaptive/*.pt | head -5
    echo ""
    echo "Training summary (last 50 lines):"
    tail -50 logs/train_adaptive_*.log | tail -50
    echo ""
    echo "Best model saved: checkpoints_adaptive/best_model.pt"
    
    # Backup checkpoints
    BACKUP_DIR="/work/gg0302/g260141/checkpoints_backup/adaptive_$(date +%Y%m%d_%H%M%S)"
    echo ""
    echo "Backing up checkpoints to: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    cp -r checkpoints_adaptive/* "$BACKUP_DIR/"
    echo "✅ Backup complete"
else
    echo ""
    echo "⚠️  Training may have failed - check logs above"
    echo "Log file: logs/train_adaptive_*.log"
fi

echo ""
echo "=================================================="
echo "Next Steps:"
echo "  1. Check validation loss trend"
echo "     grep 'Val Loss:' logs/train_adaptive_*.log"
echo ""
echo "  2. Analyze window selection diversity"
echo "     grep 'pair0_start' logs/train_adaptive_*.log"
echo ""
echo "  3. Download best model to local"
echo "     scp g260141@levante:/work/gg0302/g260141/Jingle_D/checkpoints_adaptive/best_model.pt ."
echo ""
echo "  4. Test inference with trained model"
echo "     python inference_adaptive.py --checkpoint checkpoints_adaptive/best_model.pt"
echo ""
echo "  5. Compare with baseline performance"
echo "     - Baseline (16-sec fixed): Val loss ~0.003"
echo "     - Target (24-sec adaptive): Val loss < 0.002"
echo "=================================================="
echo ""
