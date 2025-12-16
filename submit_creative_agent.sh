#!/bin/bash
#SBATCH --job-name=creative_agent
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --time=48:00:00
#SBATCH --output=logs/creative_agent_%j.out
#SBATCH --error=logs/creative_agent_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=g260141@levante.dkrz.de

# ============================================================================
# Creative Agent Training on Levante HPC
# ============================================================================
# This script trains the transformer with LEARNABLE creative agent instead
# of fixed complementary masking. The creative agent learns attention-based
# masks that adaptively select which parts of input and target to combine.
#
# Key Difference from Fixed Masking:
#   - Fixed masking: Rule-based (temporal/frequency/etc.), 96% complementary
#   - Creative agent: Learned via attention, ~75% (untrained) → 85-95% (trained)
#
# Parameters:
#   - encoding_dim=128: Encoding dimension
#   - num_transformer_layers=3: Cascade stages
#   - use_creative_agent=true: ENABLE creative agent (replaces fixed masking)
#   - mask_reg_weight=0.1: Weight for complementarity+coverage loss
#   - shuffle_targets=false: Use continuation (not random targets)
#   - anti_cheating=0.1: Noise on target for stages 2+ (prevents copying)
#
# Expected Results:
#   - Complementarity improves from ~75% to 85-95% over 50+ epochs
#   - Output adapts to each song pair (not fixed strategy)
#   - ~700K additional parameters (creative agent)
#
# Author: Andrey Vlasenko
# Date: 2024
# ============================================================================

echo "=============================================================================="
echo "CREATIVE AGENT TRAINING ON LEVANTE"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4x A100"
echo "Start time: $(date)"
echo ""

# Load modules
echo "Loading modules..."
module load python3
module list
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Install dependencies (if needed)
echo "Checking dependencies..."
pip install torch torchaudio transformers encodec tqdm --quiet
echo "Dependencies OK"
echo ""

# Create directories
echo "Creating directories..."
mkdir -p logs
mkdir -p checkpoints_creative_agent
echo "Directories OK"
echo ""

# Print configuration
echo "=============================================================================="
echo "CONFIGURATION"
echo "=============================================================================="
echo "Dataset: dataset_wav_pairs"
echo "Architecture:"
echo "  - Encoding dim: 128"
echo "  - Attention heads: 8"
echo "  - Internal layers: 6"
echo "  - Cascade stages: 3"
echo "  - Dropout: 0.1"
echo "  - Anti-cheating: 0.1 (noise on target, stages 2+)"
echo ""
echo "Creative Agent:"
echo "  - ENABLED: true"
echo "  - Hidden dim: 256 (automatic)"
echo "  - Attention heads: 4 (automatic)"
echo "  - Mask reg weight: 0.1"
echo "  - Parameters: ~700K (in addition to transformer)"
echo ""
echo "Training:"
echo "  - Batch size: 8 per GPU → 32 total (4 GPUs)"
echo "  - Epochs: 200"
echo "  - Learning rate: 1e-4"
echo "  - Weight decay: 0.01"
echo "  - Patience: 20 epochs"
echo ""
echo "Data:"
echo "  - Shuffle targets: false (continuation pairs)"
echo "  - Unity test: false (normal training)"
echo ""
echo "Loss Weights:"
echo "  - Input: 0.0"
echo "  - Target: 1.0"
echo "  - Spectral: 0.0"
echo "  - Mel: 0.0"
echo "  - Mask regularization: 0.1"
echo "  - GAN: 0.0 (disabled)"
echo ""
echo "GAN Training:"
echo "  - Status: DISABLED (gan_weight=0.0)"
echo "  - Discriminator LR: 5e-5 (if enabled)"
echo "  - Update frequency: 1 batch (if enabled)"
echo ""
echo "Masking Strategy:"
echo "  - Fixed masking: DISABLED (creative agent replaces it)"
echo "  - Creative agent: ENABLED (learned attention masks)"
echo "=============================================================================="
echo ""

# Run training
echo "Starting training..."
echo ""

srun python train_simple_ddp.py \
    --dataset_folder dataset_wav_pairs \
    --encoding_dim 128 \
    --nhead 8 \
    --num_layers 6 \
    --dropout 0.1 \
    --num_transformer_layers 3 \
    --anti_cheating 0.1 \
    --use_creative_agent true \
    --mask_reg_weight 0.1 \
    --batch_size 8 \
    --num_epochs 200 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --patience 20 \
    --world_size 4 \
    --checkpoint_dir checkpoints_creative_agent \
    --save_every 10 \
    --shuffle_targets false \
    --unity_test false \
    --loss_weight_input 0.0 \
    --loss_weight_target 1.0 \
    --loss_weight_spectral 0.0 \
    --loss_weight_mel 0.0 \
    --mask_type none \
    --gan_weight 0.0 \
    --disc_lr 5e-5 \
    --disc_update_freq 1

echo ""
echo "=============================================================================="
echo "TRAINING COMPLETE"
echo "=============================================================================="
echo "End time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Checkpoints saved to: checkpoints_creative_agent/"
echo "Logs saved to: logs/creative_agent_${SLURM_JOB_ID}.out"
echo ""
echo "Next steps:"
echo "  1. Check logs: tail -f logs/creative_agent_${SLURM_JOB_ID}.out"
echo "  2. Load checkpoint: torch.load('checkpoints_creative_agent/checkpoint_best.pth')"
echo "  3. Test complementarity: python creative_agent.py"
echo "  4. Compare with fixed masking: Check complementarity % in logs"
echo "=============================================================================="
