#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 29 2025

@author: andreyvlasenko

Main training script for multi-pattern fusion model with DistributedDataParallel support.
Implements 3-phase training with 3 input songs → 1 fused output.

Usage:
  Single GPU:
    python main_multipattern_ddp.py
  
  Multi-GPU (4 GPUs):
    torchrun --nproc_per_node=4 main_multipattern_ddp.py
  
  SLURM with multi-node:
    srun python -m torch.distributed.launch --nproc_per_node=4 main_multipattern_ddp.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from model_multipattern import MultiPatternAttentionModel, MultiPatternAudioDataset
from train_and_validate_multipattern import train_and_validate_multipattern
from utilities import load_checkpoint


def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    else:
        # Single GPU/CPU
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        # Initialize process group
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def print_once(*args, **kwargs):
    """Print only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


# ============================================================================
# HYPERPARAMETERS - Multi-Pattern Configuration
# ============================================================================

# Dataset and paths
dataset_folder = "dataset_multipattern"
checkpoint_folder = "checkpoints_multipattern"
music_out_folder = "music_out_multipattern"

# Audio specifications
sample_rate = 22050  # High-quality sample rate
chunk_duration = 16.33  # seconds (compatible with encoder-decoder architecture)
seq_len = 360000  # samples (16.33s @ 22050Hz - architecture requirement)

# Multi-pattern settings
num_patterns = 3  # Number of input patterns per training example

# Model architecture
n_channels = 64  # Encoded channels
n_seq = 9  # Sequence multiplier (required for 360000 sample reconstruction)
num_heads = 8  # Transformer attention heads
num_layers = 4  # Transformer layers
dropout = 0.15  # Dropout for regularization
sound_channels = 2  # Stereo audio

# Training hyperparameters
batch_size = 4  # Per-GPU batch size (reduced for multi-pattern)
accumulation_steps = 16  # Maintain effective batch size
epochs = 50  # Extended for 3-phase training

# Learning rate
learning_rate = 0.0001

# 3-Phase training schedule
phase1_end = 10  # Epochs 1-10: Unmasked encoder-decoder
phase2_end = 20  # Epochs 11-20: Masked encoder-decoder
# Phase 3: Epochs 21+: Full model with transformer fusion

# Resume training
resume_from_checkpoint = None  # Set to path if resuming

# ============================================================================
# SETUP DISTRIBUTED TRAINING
# ============================================================================

rank, world_size, local_rank = setup_distributed()
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

print_once(f"\n{'='*60}")
print_once(f"MULTI-PATTERN FUSION MODEL - DISTRIBUTED TRAINING")
print_once(f"{'='*60}")
print_once(f"World size: {world_size}")
print_once(f"Rank: {rank}")
print_once(f"Local rank: {local_rank}")
print_once(f"Device: {device}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    if rank == 0:
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    print_once(f"⚠️  Multi-pattern training requires ~3x memory per sample")
    print_once(f"Total GPUs: {torch.cuda.device_count()}")
    print_once(f"Per-GPU batch size: {batch_size}")
    print_once(f"Effective batch size: {batch_size * world_size * accumulation_steps}")

# ============================================================================
# DATA LOADING
# ============================================================================

print_once(f"\n{'='*60}")
print_once(f"LOADING MULTI-PATTERN DATASETS")
print_once(f"{'='*60}")

train_data = np.load(os.path.join(dataset_folder, "training_set_multipattern.npy"), allow_pickle=True)
val_data = np.load(os.path.join(dataset_folder, "validation_set_multipattern.npy"), allow_pickle=True)

print_once(f"Training triplets: {len(train_data)}")
print_once(f"Validation triplets: {len(val_data)}")
print_once(f"Patterns per triplet: {num_patterns}")
print_once(f"Sample rate: {sample_rate} Hz")
print_once(f"Chunk duration: {chunk_duration:.2f}s")
print_once(f"Sequence length: {seq_len} samples")

# Verify data format
if rank == 0 and len(train_data) > 0:
    example = train_data[0]
    inputs, targets = example
    print_once(f"\nData format verification:")
    print_once(f"  Inputs: {len(inputs)} patterns")
    print_once(f"  Input shape (per pattern): {inputs[0].shape}")
    print_once(f"  Targets: {len(targets)} patterns")
    print_once(f"  Target shape (per pattern): {targets[0].shape}")

train_dataset = MultiPatternAudioDataset(train_data)
val_dataset = MultiPatternAudioDataset(val_data)

# Create distributed samplers
train_sampler = DistributedSampler(
    train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
) if world_size > 1 else None

val_sampler = DistributedSampler(
    val_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=False
) if world_size > 1 else None

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    sampler=val_sampler,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

print_once(f"\n{'='*60}")
print_once(f"INITIALIZING MULTI-PATTERN MODEL")
print_once(f"{'='*60}")

model = MultiPatternAttentionModel(
    input_dim=sound_channels,
    num_patterns=num_patterns,
    num_heads=num_heads,
    num_layers=num_layers,
    n_channels=n_channels,
    n_seq=n_seq,
    sound_channels=sound_channels,
    batch_size=batch_size,
    seq_len=seq_len,
    dropout=dropout
).to(device)

# Wrap model with DDP
if world_size > 1:
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True  # Set to True for 3-phase training with frozen layers
    )

# Count parameters (only on rank 0)
if rank == 0:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print_once(f"Architecture:")
    print_once(f"  Encoder-Decoder: Shared across {num_patterns} patterns")
    print_once(f"  Transformer: {num_layers} layers, {num_heads} heads")
    print_once(f"  Fusion: Linear projection to single output")
    print_once(f"\nParameters:")
    print_once(f"  Total: {total_params:,}")
    print_once(f"  Trainable: {trainable_params:,}")
    print_once(f"  Model size: ~{total_params * 4 / 1e6:.2f} MB (float32)")

# ============================================================================
# TRAINING SETUP
# ============================================================================

print_once(f"\n{'='*60}")
print_once(f"3-PHASE TRAINING CONFIGURATION")
print_once(f"{'='*60}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print_once(f"Phase 1 (Epochs 1-{phase1_end}): Unmasked Encoder-Decoder")
print_once(f"  - Train: encoder-decoder only")
print_once(f"  - Frozen: transformer, fusion_layer")
print_once(f"  - Loss: reconstruction_only_loss")
print_once(f"\nPhase 2 (Epochs {phase1_end+1}-{phase2_end}): Masked Encoder-Decoder")
print_once(f"  - Train: encoder-decoder with random masking")
print_once(f"  - Frozen: transformer, fusion_layer")
print_once(f"  - Loss: reconstruction_only_loss")
print_once(f"\nPhase 3 (Epochs {phase2_end+1}+): Full Model + Transformer Fusion")
print_once(f"  - Train: all components")
print_once(f"  - Frozen: none")
print_once(f"  - Loss: multi_pattern_loss (chunk-wise MSE)")

print_once(f"\nOptimization:")
print_once(f"  Loss function: MSE")
print_once(f"  Optimizer: Adam")
print_once(f"  Learning rate: {learning_rate}")
print_once(f"  Per-GPU batch size: {batch_size}")
print_once(f"  Accumulation steps: {accumulation_steps}")
print_once(f"  Effective batch size: {batch_size * world_size * accumulation_steps}")
print_once(f"  Learning rate scheduler: ReduceLROnPlateau")

# Load checkpoint if specified (only rank 0 loads, then broadcasts)
start_epoch = 1
if resume_from_checkpoint and rank == 0:
    start_epoch = load_checkpoint(resume_from_checkpoint, model, optimizer)
    print_once(f"\nResuming from checkpoint: {resume_from_checkpoint}")
    print_once(f"Starting at epoch: {start_epoch}")
else:
    print_once(f"\nStarting training from scratch")

# Synchronize start_epoch across all processes
if world_size > 1:
    start_epoch_tensor = torch.tensor(start_epoch, device=device)
    dist.broadcast(start_epoch_tensor, src=0)
    start_epoch = start_epoch_tensor.item()

# Create output directories (only rank 0)
if rank == 0:
    os.makedirs(checkpoint_folder, exist_ok=True)
    os.makedirs(music_out_folder, exist_ok=True)

# Barrier to ensure all processes are synchronized
if world_size > 1:
    dist.barrier()

# ============================================================================
# TRAINING
# ============================================================================

print_once(f"\n{'='*60}")
print_once(f"STARTING MULTI-PATTERN TRAINING")
print_once(f"{'='*60}\n")

try:
    train_and_validate_multipattern(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        start_epoch=start_epoch,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        sample_rate=sample_rate,
        checkpoint_folder=checkpoint_folder,
        music_out_folder=music_out_folder,
        phase1_end=phase1_end,
        phase2_end=phase2_end,
        accumulation_steps=accumulation_steps,
        use_scheduler=True,
        num_patterns=num_patterns,
        rank=rank,  # Pass rank for distributed training
        train_sampler=train_sampler  # Pass sampler to set epoch
    )
    
    print_once(f"\n{'='*60}")
    print_once(f"TRAINING COMPLETE!")
    print_once(f"{'='*60}")
    print_once(f"Checkpoints saved to: {checkpoint_folder}")
    print_once(f"Music outputs saved to: {music_out_folder}")
    print_once(f"Best model: {os.path.join(checkpoint_folder, 'model_best.pt')}")

finally:
    # Clean up distributed
    cleanup_distributed()
