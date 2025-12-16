#!/bin/bash
#SBATCH --job-name=jingle_freq
#SBATCH --account=gg0302
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --output=logs/train_frequency_%j.log
#SBATCH --error=logs/train_frequency_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=g260141@levante.dkrz.de

##################################################
# FREQUENCY MASKING: Bass + Melody Fusion
##################################################

echo "================================================="
echo "Frequency Masking Experiment"
echo "Input: Low frequencies (bass/rhythm)"
echo "Target: High frequencies (melody/vocals)"
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
    --checkpoint_dir checkpoints_frequency \
    --mask_type frequency \
    --mask_freq_split 0.3 \
    --loss_weight_input 1.0 \
    --loss_weight_target 1.0 \
    --anti_cheating 0.3 \
    --epochs 200 \
    --batch_size 16 \
    --lr 1e-3 \
    --world_size 4 \
    --gan_weight 0.0 \
    --disc_lr 5e-5 \
    --disc_update_freq 1

echo "✅ Frequency masking training complete!"
