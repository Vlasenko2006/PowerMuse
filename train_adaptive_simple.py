#!/usr/bin/env python3
"""
Training script for Adaptive Window Selection with Compositional Creative Agent

Architecture:
    24-sec input/target → AdaptiveWindowAgent (3 pairs) → 
    CompositionalAgent → 2-stage Cascade → Output

Key features:
- Loads 24-second audio pairs
- Adaptive window selection (3 pairs per sample)
- Compositional decomposition (rhythm/harmony/timbre)
- Mean loss across all 3 pairs
- 2-stage cascade transformer
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from encodec import EncodecModel
from pathlib import Path
import argparse
from tqdm import tqdm

# Import custom modules
from dataset_wav_pairs_24sec import AudioPairsDataset24sec, collate_fn
from adaptive_window_agent import AdaptiveWindowCreativeAgent


def parse_args():
    parser = argparse.ArgumentParser(description='Train adaptive window selection')
    
    # Data
    parser.add_argument('--data_folder', type=str, default='dataset_pairs_wav_24sec',
                       help='Folder with train/val subfolders')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (small due to 3x computation)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    
    # Model
    parser.add_argument('--encoding_dim', type=int, default=128,
                       help='EnCodec encoding dimension')
    parser.add_argument('--num_pairs', type=int, default=3,
                       help='Number of window pairs')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--novelty_weight', type=float, default=0.1,
                       help='Weight for novelty loss')
    
    # Checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_adaptive',
                       help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda or cpu)')
    
    return parser.parse_args()


def encode_audio_batch(audio_batch, encodec_model, target_frames=1200):
    """
    Encode batch of audio with EnCodec and resample to target frames.
    
    Args:
        audio_batch: [B, 1, samples] - mono audio
        encodec_model: EnCodec model
        target_frames: Target number of frames (1200 = 24 sec at 50 fps)
    
    Returns:
        encoded: [B, 128, target_frames]
    """
    B = audio_batch.size(0)
    device = audio_batch.device
    
    with torch.no_grad():
        # Encode with EnCodec
        latents = encodec_model.encoder(audio_batch)  # [B, 128, T]
        
        # Resample to target frames if needed
        if latents.shape[-1] != target_frames:
            latents = torch.nn.functional.interpolate(
                latents,
                size=target_frames,
                mode='linear',
                align_corners=False
            )
    
    return latents


def train_epoch(model, dataloader, encodec_model, optimizer, device, novelty_weight, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_novelty = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (audio_inputs, audio_targets) in enumerate(pbar):
        # Move to device
        audio_inputs = audio_inputs.to(device)  # [B, 1, samples]
        audio_targets = audio_targets.to(device)
        
        # Encode audio
        encoded_inputs = encode_audio_batch(audio_inputs, encodec_model)  # [B, 128, 1200]
        encoded_targets = encode_audio_batch(audio_targets, encodec_model)
        
        # Forward pass through adaptive window agent
        outputs, losses, metadata = model(encoded_inputs, encoded_targets)
        # outputs: List of 3 tensors [B, 128, 800]
        # losses: List of 3 scalars
        
        # Compute mean loss
        mean_novelty_loss = torch.mean(torch.stack(losses))
        
        # Total loss
        loss = novelty_weight * mean_novelty_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        total_novelty += mean_novelty_loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'novelty': f'{mean_novelty_loss.item():.4f}',
            'pair0_start': f'{metadata["pairs"][0]["start_input_mean"]:.1f}',
            'pair0_ratio': f'{metadata["pairs"][0]["ratio_input_mean"]:.2f}x'
        })
    
    avg_loss = total_loss / num_batches
    avg_novelty = total_novelty / num_batches
    
    return avg_loss, avg_novelty


def validate(model, dataloader, encodec_model, device, novelty_weight):
    """Validate"""
    model.eval()
    
    total_loss = 0.0
    total_novelty = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for audio_inputs, audio_targets in tqdm(dataloader, desc="Validation"):
            audio_inputs = audio_inputs.to(device)
            audio_targets = audio_targets.to(device)
            
            # Encode
            encoded_inputs = encode_audio_batch(audio_inputs, encodec_model)
            encoded_targets = encode_audio_batch(audio_targets, encodec_model)
            
            # Forward
            outputs, losses, metadata = model(encoded_inputs, encoded_targets)
            mean_novelty_loss = torch.mean(torch.stack(losses))
            
            loss = novelty_weight * mean_novelty_loss
            
            total_loss += loss.item()
            total_novelty += mean_novelty_loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_novelty = total_novelty / num_batches
    
    return avg_loss, avg_novelty


def main():
    args = parse_args()
    
    print("="*80)
    print("TRAINING: ADAPTIVE WINDOW SELECTION + COMPOSITIONAL AGENT")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data folder: {args.data_folder}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Novelty weight: {args.novelty_weight}")
    print(f"  Device: {args.device}")
    print(f"  Number of pairs: {args.num_pairs}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = AudioPairsDataset24sec(args.data_folder, split='train')
    val_dataset = AudioPairsDataset24sec(args.data_folder, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Load EnCodec model
    print("\nLoading EnCodec...")
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(6.0)
    encodec_model = encodec_model.to(args.device)
    encodec_model.eval()
    for param in encodec_model.parameters():
        param.requires_grad = False
    print("  ✓ EnCodec loaded and frozen")
    
    # Initialize adaptive window agent
    print("\nInitializing model...")
    model = AdaptiveWindowCreativeAgent(
        encoding_dim=args.encoding_dim,
        num_pairs=args.num_pairs
    )
    model = model.to(args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params/1e6:.1f}M")
    print(f"  Trainable parameters: {trainable_params/1e6:.1f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_novelty = train_epoch(
            model, train_loader, encodec_model, optimizer,
            args.device, args.novelty_weight, epoch
        )
        
        # Validate
        val_loss, val_novelty = validate(
            model, val_loader, encodec_model, args.device, args.novelty_weight
        )
        
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f} (novelty: {train_novelty:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (novelty: {val_novelty:.4f})")
        
        # Save checkpoint
        if epoch % args.save_every == 0 or val_loss < best_val_loss:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }
            
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                torch.save(checkpoint, best_path)
                print(f"  ✓ New best model! Val loss: {val_loss:.4f}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()
