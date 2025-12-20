#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDP Training for Adaptive Window Selection with Full Creative Agent Features

Combines:
- Adaptive window selection (24-sec → 3 pairs of 16-sec windows)
- All features from train_simple_ddp.py (DDP, GAN, creative agent, etc.)
- Compositional decomposition (rhythm/harmony/timbre)

Multi-GPU training using DistributedDataParallel
"""

import os
import argparse
import torch.multiprocessing as mp

from train_adaptive_worker import train_adaptive_worker


def main():
    parser = argparse.ArgumentParser(description='Train Adaptive Window Selection with Creative Agent')
    
    # Model parameters
    parser.add_argument('--encoding_dim', type=int, default=128,
                       help='EnCodec encoding dimension (128 for 24kHz)')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of internal transformer encoder layers per stage')
    parser.add_argument('--num_transformer_layers', type=int, default=1,
                       help='Number of cascade stages (1=no cascade, 2+=cascade refinement)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # EnCodec parameters
    parser.add_argument('--encodec_bandwidth', type=float, default=6.0,
                       help='EnCodec bandwidth (6.0 for high quality)')
    parser.add_argument('--encodec_sr', type=int, default=24000,
                       help='EnCodec sample rate')
    
    # Dataset parameters
    parser.add_argument('--dataset_folder', type=str, default='dataset_pairs_wav_24sec',
                       help='Dataset folder with 24-second segments (contains train/ and val/ subdirs)')
    
    # Adaptive window parameters
    parser.add_argument('--num_pairs', type=int, default=3,
                       help='Number of window pairs to select per sample')
    parser.add_argument('--window_duration', type=int, default=800,
                       help='Window duration in frames (800 = 16 seconds)')
    parser.add_argument('--compression_min', type=float, default=1.0,
                       help='Minimum temporal compression ratio')
    parser.add_argument('--compression_max', type=float, default=1.5,
                       help='Maximum temporal compression ratio')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=6,
                       help='Batch size per GPU (smaller due to 3x computation)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for AdamW')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='DataLoader workers')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_adaptive',
                       help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # DDP parameters
    parser.add_argument('--world_size', type=int, default=4,
                       help='Number of GPUs')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0,
                       help='Local rank for distributed training (automatically set by torch.distributed.launch)')
    
    # Unity test parameter
    parser.add_argument('--unity_test', type=lambda x: x.lower() == 'true', default=False,
                       help='Unity test: set target=input for sanity check (true/false)')
    
    # Target shuffling parameter
    parser.add_argument('--shuffle_targets', type=lambda x: x.lower() == 'true', default=False,
                       help='Shuffle targets: random target instead of continuation (true/false)')
    
    # Anti-cheating parameter (for cascade stages 2+)
    parser.add_argument('--anti_cheating', type=float, default=0.0,
                       help='Anti-cheating noise level for cascade (0.0=no noise, 1.0=heavy noise on target)')
    
    # Loss weight parameters
    parser.add_argument('--loss_weight_input', type=float, default=0.0,
                       help='Weight for RMS(predicted, input) loss component')
    parser.add_argument('--loss_weight_target', type=float, default=1.0,
                       help='Weight for RMS(predicted, target) loss component')
    parser.add_argument('--loss_weight_spectral', type=float, default=0.01,
                       help='Weight for multi-resolution STFT loss component')
    parser.add_argument('--loss_weight_mel', type=float, default=0.0,
                       help='Weight for mel-spectrogram loss component')
    
    # Complementary masking parameters (for baseline compatibility)
    parser.add_argument('--mask_type', type=str, default='none',
                       choices=['none', 'temporal', 'frequency', 'spectral', 'energy', 'hybrid'],
                       help='Complementary masking strategy (legacy, prefer creative agent)')
    parser.add_argument('--mask_temporal_segment', type=int, default=150,
                       help='Temporal segment length in frames (~150 = 1 second)')
    parser.add_argument('--mask_freq_split', type=float, default=0.3,
                       help='Frequency split ratio (0.3 = low 30%% vs high 70%%)')
    parser.add_argument('--mask_channel_keep', type=float, default=0.5,
                       help='Channel keep ratio for spectral dropout')
    parser.add_argument('--mask_energy_threshold', type=float, default=0.7,
                       help='Energy threshold percentile (0.7 = 70th percentile)')
    
    # Creative agent parameters
    parser.add_argument('--use_creative_agent', type=lambda x: x.lower() == 'true', default=False,
                       help='Use learnable creative agent instead of fixed masking (true/false)')
    parser.add_argument('--use_compositional_agent', type=lambda x: x.lower() == 'true', default=True,
                       help='Use compositional agent (rhythm/harmony/timbre) - default for adaptive windows')
    parser.add_argument('--mask_reg_weight', type=float, default=0.1,
                       help='Weight for novelty loss (compositional) or mask regularization loss')
    parser.add_argument('--balance_loss_weight', type=float, default=5.0,
                       help='Weight for balance loss (enforces 50/50 input/target mixing)')
    parser.add_argument('--corr_weight', type=float, default=0.5,
                       help='Weight for anti-modulation correlation cost')
    
    # GAN parameters
    parser.add_argument('--gan_weight', type=float, default=0.01,
                       help='Weight for GAN adversarial loss')
    parser.add_argument('--disc_lr', type=float, default=None,
                       help='Discriminator learning rate (default: 0.5 * lr)')
    parser.add_argument('--disc_update_freq', type=int, default=1,
                       help='Update discriminator every N batches')
    
    # Pure GAN mode (curriculum learning)
    parser.add_argument('--pure_gan_mode', type=float, default=0.0,
                       help='Curriculum learning rate: transition from music to noise')
    parser.add_argument('--gan_curriculum_start_epoch', type=int, default=0,
                       help='Epoch to start GAN curriculum')
    parser.add_argument('--gan_noise_ceiling', type=float, default=1.0,
                       help='Maximum noise fraction (alpha ceiling)')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("="*80)
    print("ADAPTIVE WINDOW SELECTION TRAINING (DDP)")
    print("="*80)
    print("Configuration:")
    print(f"  Dataset: {args.dataset_folder}")
    print(f"  Model:")
    print(f"    - Base: SimpleTransformer (24.9M params)")
    print(f"    - Creative Agent: CompositionalCreativeAgent (14.6M params)")
    print(f"    - Adaptive Agent: AdaptiveWindowCreativeAgent (3.6M params)")
    print(f"    - Encoding dim: {args.encoding_dim}")
    print(f"    - Attention heads: {args.nhead}")
    print(f"    - Internal transformer layers: {args.num_layers}")
    print(f"    - Cascade stages: {args.num_transformer_layers}")
    print(f"    - Dropout: {args.dropout}")
    print(f"  Adaptive Windows:")
    print(f"    - Input/Target: 24 seconds (1200 frames)")
    print(f"    - Window size: {args.window_duration} frames (16 seconds)")
    print(f"    - Number of pairs: {args.num_pairs}")
    print(f"    - Compression range: {args.compression_min}x - {args.compression_max}x")
    print(f"  EnCodec:")
    print(f"    - Sample rate: {args.encodec_sr} Hz")
    print(f"    - Bandwidth: {args.encodec_bandwidth}")
    print(f"    - Status: FROZEN")
    print(f"  Training:")
    print(f"    - Epochs: {args.epochs}")
    print(f"    - Batch size: {args.batch_size} per GPU × {args.world_size} GPUs = {args.batch_size * args.world_size}")
    print(f"    - Learning rate: {args.lr}")
    print(f"    - Optimizer: AdamW (weight_decay={args.weight_decay})")
    print(f"  Loss weights:")
    print(f"    - Input: {args.loss_weight_input}")
    print(f"    - Target: {args.loss_weight_target}")
    print(f"    - Spectral: {args.loss_weight_spectral}")
    print(f"    - Mel: {args.loss_weight_mel}")
    print(f"    - Novelty: {args.mask_reg_weight}")
    print(f"    - Correlation: {args.corr_weight}")
    print(f"    - GAN: {args.gan_weight}")
    print(f"  Options:")
    print(f"    - Unity test: {'ENABLED' if args.unity_test else 'DISABLED'}")
    print(f"    - Shuffle targets: {'ENABLED' if args.shuffle_targets else 'DISABLED'}")
    print(f"    - Anti-cheating: {args.anti_cheating}")
    if args.use_compositional_agent:
        print(f"  🎼 Compositional Creative Agent:")
        print(f"    - Enabled: True (rhythm/harmony/timbre decomposition)")
        print(f"    - Novelty regularization weight: {args.mask_reg_weight}")
    elif args.use_creative_agent:
        print(f"  🎨 Attention-Based Creative Agent:")
        print(f"    - Enabled: True (learnable masking)")
        print(f"    - Mask regularization weight: {args.mask_reg_weight}")
    if args.gan_weight > 0:
        disc_lr = args.disc_lr if args.disc_lr is not None else args.lr * 0.5
        print(f"  GAN Training:")
        print(f"    - Enabled: True")
        print(f"    - Discriminator LR: {disc_lr:.2e}")
        print(f"    - Update frequency: every {args.disc_update_freq} batch(es)")
        if args.pure_gan_mode > 0:
            print(f"    - Curriculum: {args.pure_gan_mode} (noise ceiling: {args.gan_noise_ceiling})")
    print(f"  Checkpoints: {args.checkpoint_dir}/")
    print("="*80)
    
    # Launch DDP training
    mp.spawn(
        train_adaptive_worker,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )


if __name__ == '__main__':
    main()
