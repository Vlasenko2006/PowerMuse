#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 28 2025

@author: andreyvlasenko

Main training script for multi-pattern fusion model.
Implements 3-phase training with 3 input songs → 1 fused output.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_multipattern import MultiPatternAttentionModel, MultiPatternAudioDataset
from train_and_validate_multipattern import train_and_validate_multipattern
from utilities import load_checkpoint


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
batch_size = 4  # Reduced for multi-pattern (3x memory per sample)
accumulation_steps = 16  # Maintain effective batch size of 64
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
# SETUP
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"MULTI-PATTERN FUSION MODEL - TRAINING")
print(f"{'='*60}")
print(f"Device: {device}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"⚠️  Multi-pattern training requires ~3x memory per sample")

# ============================================================================
# DATA LOADING
# ============================================================================

print(f"\n{'='*60}")
print(f"LOADING MULTI-PATTERN DATASETS")
print(f"{'='*60}")

train_data = np.load(os.path.join(dataset_folder, "training_set_multipattern.npy"), allow_pickle=True)
val_data = np.load(os.path.join(dataset_folder, "validation_set_multipattern.npy"), allow_pickle=True)

print(f"Training triplets: {len(train_data)}")
print(f"Validation triplets: {len(val_data)}")
print(f"Patterns per triplet: {num_patterns}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Chunk duration: {chunk_duration:.2f}s")
print(f"Sequence length: {seq_len} samples")

# Verify data format
if len(train_data) > 0:
    example = train_data[0]
    inputs, targets = example
    print(f"\nData format verification:")
    print(f"  Inputs: {len(inputs)} patterns")
    print(f"  Input shape (per pattern): {inputs[0].shape}")
    print(f"  Targets: {len(targets)} patterns")
    print(f"  Target shape (per pattern): {targets[0].shape}")

train_dataset = MultiPatternAudioDataset(train_data)
val_dataset = MultiPatternAudioDataset(val_data)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

print(f"\n{'='*60}")
print(f"INITIALIZING MULTI-PATTERN MODEL")
print(f"{'='*60}")

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

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Architecture:")
print(f"  Encoder-Decoder: Shared across {num_patterns} patterns")
print(f"  Transformer: {num_layers} layers, {num_heads} heads")
print(f"  Fusion: Linear projection to single output")
print(f"\nParameters:")
print(f"  Total: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")
print(f"  Model size: ~{total_params * 4 / 1e6:.2f} MB (float32)")

# ============================================================================
# TRAINING SETUP
# ============================================================================

print(f"\n{'='*60}")
print(f"3-PHASE TRAINING CONFIGURATION")
print(f"{'='*60}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(f"Phase 1 (Epochs 1-{phase1_end}): Unmasked Encoder-Decoder")
print(f"  - Train: encoder-decoder only")
print(f"  - Frozen: transformer, fusion_layer")
print(f"  - Loss: reconstruction_only_loss")
print(f"\nPhase 2 (Epochs {phase1_end+1}-{phase2_end}): Masked Encoder-Decoder")
print(f"  - Train: encoder-decoder with random masking")
print(f"  - Frozen: transformer, fusion_layer")
print(f"  - Loss: reconstruction_only_loss")
print(f"\nPhase 3 (Epochs {phase2_end+1}+): Full Model + Transformer Fusion")
print(f"  - Train: all components")
print(f"  - Frozen: none")
print(f"  - Loss: multi_pattern_loss (chunk-wise MSE)")

print(f"\nOptimization:")
print(f"  Loss function: MSE")
print(f"  Optimizer: Adam")
print(f"  Learning rate: {learning_rate}")
print(f"  Batch size: {batch_size}")
print(f"  Accumulation steps: {accumulation_steps}")
print(f"  Effective batch size: {batch_size * accumulation_steps}")
print(f"  Learning rate scheduler: ReduceLROnPlateau")

# Load checkpoint if specified
start_epoch = 1
if resume_from_checkpoint:
    start_epoch = load_checkpoint(resume_from_checkpoint, model, optimizer)
    print(f"\nResuming from checkpoint: {resume_from_checkpoint}")
    print(f"Starting at epoch: {start_epoch}")
else:
    print(f"\nStarting training from scratch")

# Create output directories
os.makedirs(checkpoint_folder, exist_ok=True)
os.makedirs(music_out_folder, exist_ok=True)

# ============================================================================
# TRAINING
# ============================================================================

print(f"\n{'='*60}")
print(f"STARTING MULTI-PATTERN TRAINING")
print(f"{'='*60}\n")

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
    num_patterns=num_patterns
)

print(f"\n{'='*60}")
print(f"TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"Checkpoints saved to: {checkpoint_folder}")
print(f"Music outputs saved to: {music_out_folder}")
print(f"Best model: {os.path.join(checkpoint_folder, 'model_best.pt')}")
