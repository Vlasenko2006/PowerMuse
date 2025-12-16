#!/bin/bash
#SBATCH --job-name=jingle_hybrid
#SBATCH --account=gg0302
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --output=logs/train_hybrid_%j.log
#SBATCH --error=logs/train_hybrid_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=g260141@levante.dkrz.de

##################################################
# HYBRID MASKING: Professional Arrangement
# Temporal + Frequency combined
##################################################

echo "================================================="
echo "Hybrid Masking Experiment"
echo "Input: Low freq on odd beats"
echo "Target: High freq on even beats"
echo "Maximum complementarity!"
echo "GAN Training: DISABLED (gan_weight=0.0)"
echo "================================================="

module purge
module load python3/2022.01-gcc-11.2.0
module load cuda/11.8.0

export PYTHONPATH=/work/gg0302/g260141/Jingle:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

source /work/gg0302/g260141/miniconda3/bin/activate
conda activate multipattern_env

mkdir -p logs
cd /work/gg0302/g260141/Jingle_D

python train_simple_ddp.py \
    --dataset_folder dataset_pairs_wav \
    --num_transformer_layers 3 \
    --shuffle_targets true \
    --checkpoint_dir checkpoints_hybrid \
    --mask_type hybrid \
    --mask_temporal_segment 100 \
    --mask_freq_split 0.25 \
    --loss_weight_input 1.0 \
    --loss_weight_target 1.0 \
    --anti_cheating 0.5 \
    --epochs 200 \
    --batch_size 16 \
    --lr 1e-3 \
    --world_size 4 \
    --gan_weight 0.0 \
    --disc_lr 5e-5 \
    --disc_update_freq 1

echo "✅ Hybrid masking training complete!"
