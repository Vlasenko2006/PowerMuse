#!/bin/bash
#SBATCH --job-name=jingle_cascade
#SBATCH --account=gg0302
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --output=logs/train_cascade_%j.log
#SBATCH --error=logs/train_cascade_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=g260141@levante.dkrz.de

##################################################
# JINGLE CASCADE TRAINING ON LEVANTE
# Complementary Masking for Style Transfer
##################################################

echo "=================================================="
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

# Load modules
module purge
module load python3/2022.01-gcc-11.2.0
module load cuda/11.8.0

# Environment setup
export PYTHONPATH=/work/gg0302/g260141/Jingle:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Activate conda environment
source /work/gg0302/g260141/miniconda3/bin/activate
conda activate multipattern_env

# Verify environment
echo ""
echo "=================================================="
echo "Environment:"
echo "  Python: $(which python)"
echo "  Working directory: $(pwd)"
echo "  CUDA devices: $CUDA_VISIBLE_DEVICES"
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo "=================================================="

# Create logs directory
mkdir -p logs

# Change to working directory
cd /work/gg0302/g260141/Jingle_D

# Run training with complementary masking
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
echo "  - Transformer cascade stages: 3 (3-stage cascade)"
echo "  - Dropout: 0.1"
echo ""
echo "  - EnCodec: 24kHz, bandwidth=6.0 (FROZEN)"
echo ""
echo "  - Dataset: dataset_pairs_wav/"
echo "  - Batch size: 16 per GPU × 4 GPUs = 64"
echo "  - Learning rate: 1e-3"
echo "  - Optimizer: AdamW (weight_decay=0.01)"
echo "  - Loss: Combined perceptual loss"
echo "  - Loss weights: input=1.0, target=1.0"
echo "  - Epochs: 200 (with early stopping)"
echo ""
echo "  - Complementary Masking: ENABLED"
echo "  - Mask type: temporal (alternating time segments)"
echo "  - Segment length: 150 frames (~1 second)"
echo "  - Shuffle targets: ENABLED (random pairs)"
echo "  - Anti-cheating: 0.3"
echo "  - GAN Training: DISABLED (gan_weight=0.0)"
echo "    * Discriminator LR: 5e-5 (if enabled)"
echo "    * Update frequency: 1 batch (if enabled)"
echo ""
echo "=================================================="
echo ""

python train_simple_ddp.py \
    --dataset_folder dataset_pairs_wav \
    --encoding_dim 128 \
    --nhead 8 \
    --num_layers 4 \
    --num_transformer_layers 3 \
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
    --gan_weight 0.0 \
    --disc_lr 5e-5 \
    --disc_update_freq 1

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=================================================="

# Show results
if [ -f checkpoints_spectral/best_model.pt ]; then
    echo ""
    echo "✅ Best model saved: checkpoints_spectral/best_model.pt"
    echo ""
    echo "Model checkpoints:"
    ls -lh checkpoints_spectral/*.pt | tail -5
else
    echo ""
    echo "⚠️  No model checkpoint found!"
    echo "Check the log for errors"
fi

echo ""
echo "=================================================="
echo "Job finished: $(date)"
echo "=================================================="

exit $EXIT_CODE
