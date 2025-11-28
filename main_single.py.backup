#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 12:37:20 2025

@author: andreyvlasenko

Enhanced main training script with improved hyperparameters for 16s audio at 22050 Hz.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import AttentionModel, AudioDataset
from train_and_validate import train_and_validate
from utilities import load_checkpoint


# ============================================================================
# HYPERPARAMETERS - Optimized for 16s audio at 22050 Hz
# ============================================================================

# Dataset and paths
dataset_folder = "../dataset"
checkpoint_folder = "checkpoints_enhanced"
music_out_folder = "music_out_enhanced"

# Audio specifications
sample_rate = 22050  # Upgraded from 12000 Hz for better quality
chunk_duration = 16  # seconds (upgraded from 10s)
seq_len = sample_rate * chunk_duration  # 352,800 samples

# Model architecture
n_channels = 64  # Encoded channels
n_seq = 4  # Sequence multiplier (increased from 3 for 16s audio)
num_heads = 8  # Transformer attention heads (increased from 4)
num_layers = 4  # Transformer layers (increased from 1)
dropout = 0.15  # Dropout for regularization

# Training hyperparameters
batch_size = 8  # Reduced for memory efficiency with 16s audio (was 16)
accumulation_steps = 8  # Increased to maintain effective batch size of 64
epochs = 30000
learning_rate = 0.0001  # Slightly reduced with scheduler
FREEZE_ENCODER_DECODER_AFTER = 14  # Epoch to start transformer training

# Resume training
resume_from_checkpoint = None  # Set to path if resuming, e.g., "checkpoints_enhanced/model_epoch_90.pt"

# ============================================================================
# SETUP
# ============================================================================

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Memory optimization for CUDA
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# DATA LOADING
# ============================================================================

print("\n" + "="*60)
print("LOADING DATASETS")
print("="*60)

# Load datasets
train_data = np.load(os.path.join(dataset_folder, "training_set.npy"), allow_pickle=True)
val_data = np.load(os.path.join(dataset_folder, "validation_set.npy"), allow_pickle=True)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Chunk duration: {chunk_duration}s")
print(f"Sequence length: {seq_len} samples")

train_dataset = AudioDataset(train_data)
val_dataset = AudioDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True if torch.cuda.is_available() else False)

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

print("\n" + "="*60)
print("INITIALIZING MODEL")
print("="*60)

# Infer input dimension from data
input_dim = train_data[0][0].shape[0]  # Should be 2 for stereo
print(f"Input dimension (channels): {input_dim}")

# Create model with enhanced architecture
model = AttentionModel(
    input_dim=input_dim,
    sound_channels=input_dim,
    seq_len=seq_len,
    n_channels=n_channels,
    n_seq=n_seq,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
    batch_size=batch_size
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: ~{total_params * 4 / 1e6:.2f} MB (float32)")

# ============================================================================
# TRAINING SETUP
# ============================================================================

print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(f"Loss function: MSE")
print(f"Optimizer: Adam")
print(f"Learning rate: {learning_rate}")
print(f"Batch size: {batch_size}")
print(f"Accumulation steps: {accumulation_steps}")
print(f"Effective batch size: {batch_size * accumulation_steps}")
print(f"Freeze encoder-decoder after epoch: {FREEZE_ENCODER_DECODER_AFTER}")

# Load checkpoint if specified
start_epoch = 1
if resume_from_checkpoint:
    start_epoch = load_checkpoint(resume_from_checkpoint, model, optimizer)
else:
    print("Starting training from scratch")

# Create output directories
os.makedirs(checkpoint_folder, exist_ok=True)
os.makedirs(music_out_folder, exist_ok=True)

# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

train_and_validate(
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
    FREEZE_ENCODER_DECODER_AFTER=FREEZE_ENCODER_DECODER_AFTER,
    accumulation_steps=accumulation_steps,
    use_scheduler=True
)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
