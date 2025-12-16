#!/usr/bin/env python3
"""Test argument parsing for train_simple_ddp.py"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Train Simple Transformer on WAV Pairs')
    
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
    parser.add_argument('--dataset_folder', type=str, required=True,
                       help='Dataset folder (contains train/ and val/ subdirs)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for AdamW')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='DataLoader workers (0 for GPU encoding)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_simple',
                       help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=10,
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
    parser.add_argument('--loss_weight_spectral', type=float, default=0.0,
                       help='Weight for multi-resolution STFT loss component')
    parser.add_argument('--loss_weight_mel', type=float, default=0.0,
                       help='Weight for mel-spectrogram loss component')
    
    # Complementary masking parameters
    parser.add_argument('--mask_type', type=str, default='none',
                       choices=['none', 'temporal', 'frequency', 'spectral', 'energy', 'hybrid'],
                       help='Complementary masking strategy for style transfer')
    parser.add_argument('--mask_temporal_segment', type=int, default=150,
                       help='Temporal segment length in frames (~150 = 1 second)')
    parser.add_argument('--mask_freq_split', type=float, default=0.3,
                       help='Frequency split ratio (0.3 = low 30%% vs high 70%%)')
    parser.add_argument('--mask_channel_keep', type=float, default=0.5,
                       help='Channel keep ratio for spectral dropout')
    parser.add_argument('--mask_energy_threshold', type=float, default=0.7,
                       help='Energy threshold percentile (0.7 = 70th percentile)')
    
    # Creative agent parameters (learnable masking)
    parser.add_argument('--use_creative_agent', type=lambda x: x.lower() == 'true', default=False,
                       help='Use learnable creative agent instead of fixed masking (true/false)')
    parser.add_argument('--use_compositional_agent', type=lambda x: x.lower() == 'true', default=False,
                       help='Use compositional agent (rhythm/harmony/timbre decomposition) - mutually exclusive with use_creative_agent')
    parser.add_argument('--mask_reg_weight', type=float, default=0.1,
                       help='Weight for mask regularization loss (complementarity + coverage) or novelty loss')
    parser.add_argument('--balance_loss_weight', type=float, default=5.0,
                       help='Weight for balance loss (enforces 50/50 input/target mixing). Higher = stronger enforcement. Typical: 5-10')
    parser.add_argument('--corr_weight', type=float, default=0.0,
                       help='Weight for anti-modulation correlation cost (prevents copying amplitude envelopes)')
    
    # GAN parameters
    parser.add_argument('--gan_weight', type=float, default=0.0,
                       help='Weight for GAN adversarial loss (0.0=disabled, 0.01-0.1 typical)')
    parser.add_argument('--disc_lr', type=float, default=None,
                       help='Discriminator learning rate (default: 0.5 * lr for stability)')
    parser.add_argument('--disc_update_freq', type=int, default=1,
                       help='Update discriminator every N batches (1=every batch, higher for stability)')
    
    print("Testing argument parsing...")
    print(f"Command: {' '.join(sys.argv)}")
    print()
    
    try:
        args = parser.parse_args()
        print("✅ All arguments parsed successfully!")
        print()
        print("Parsed values:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
        return 0
    except SystemExit as e:
        if e.code != 0:
            print("❌ Argument parsing failed!")
            return e.code
        return 0

if __name__ == '__main__':
    sys.exit(main())
