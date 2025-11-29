#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 20:28:05 2025
Updated for multi-pattern fusion on Nov 29 2025

@author: andrey
"""

import os
import numpy as np
import torch


# Save one validation sample as NumPy files (multi-pattern version)
def save_sample_as_numpy(model, val_loader, device, music_out_folder, epoch, prefix=''):
    """
    Save validation samples for multi-pattern fusion model.
    
    For multi-pattern model:
    - inputs: [batch, num_patterns, channels, samples] -> saves 3 separate input patterns
    - reconstructed: [batch, num_patterns, channels, samples] -> saves 3 reconstructed patterns
    - outputs: [batch, channels, samples] -> saves 1 fused output
    - targets: same as inputs (3 patterns)
    """
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:  # Take one batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Handle both DDP and non-DDP models
            if hasattr(model, 'module'):
                reconstructed, outputs = model.module(inputs)
            else:
                reconstructed, outputs = model(inputs)

            os.makedirs(music_out_folder, exist_ok=True)
            
            # Take the first sample from batch
            # inputs shape: [batch, num_patterns, channels, samples]
            if inputs.dim() == 4:  # Multi-pattern format
                input_np = inputs.cpu().numpy()[0]  # Shape: [num_patterns, channels, samples]
                reconstructed_np = reconstructed.cpu().numpy()[0]  # Shape: [num_patterns, channels, samples]
                
                # Save each input pattern separately
                for i in range(input_np.shape[0]):
                    np.save(
                        os.path.join(music_out_folder, prefix + f"input_pattern{i+1}_epoch_{epoch}.npy"),
                        input_np[i]  # Shape: [channels, samples]
                    )
                
                # Save each reconstructed pattern separately
                for i in range(reconstructed_np.shape[0]):
                    np.save(
                        os.path.join(music_out_folder, prefix + f"reconstructed_pattern{i+1}_epoch_{epoch}.npy"),
                        reconstructed_np[i]  # Shape: [channels, samples]
                    )
                
                print(f"Saved {input_np.shape[0]} input patterns and reconstructions for epoch {epoch}.")
            else:
                # Fallback for single-pattern format
                input_np = inputs.cpu().numpy()[0]
                reconstructed_np = reconstructed.cpu().numpy()[0]
                np.save(os.path.join(music_out_folder, prefix + f"input_epoch_{epoch}.npy"), input_np)
                np.save(os.path.join(music_out_folder, prefix + f"reconstructed_epoch_{epoch}.npy"), reconstructed_np)
            
            # Save fused output (single output for all patterns)
            output_np = outputs.cpu().numpy()[0]  # Shape: [channels, samples]
            np.save(os.path.join(music_out_folder, prefix + f"output_fused_epoch_{epoch}.npy"), output_np)
            
            # Save target (for comparison)
            if targets.dim() == 4:
                # Multi-pattern targets - save first pattern as reference
                target_np = targets.cpu().numpy()[0, 0]  # First sample, first pattern
            else:
                target_np = targets.cpu().numpy()[0]
            np.save(os.path.join(music_out_folder, prefix + f"target_reference_epoch_{epoch}.npy"), target_np)
            
            print(f"Saved fused output and target as NumPy files for epoch {epoch}.")
            print(f"Output shape: {output_np.shape} (stereo audio: [channels, samples])")
            break  # Save only one sample


# Save model checkpoint
def save_checkpoint(model, optimizer, epoch, checkpoint_folder, filename=None):
    """
    Save model checkpoint (handles both DDP and non-DDP models).
    """
    os.makedirs(checkpoint_folder, exist_ok=True)
    if filename is None:
        checkpoint_path = os.path.join(checkpoint_folder, f"model_epoch_{epoch}.pt")
    else:
        checkpoint_path = os.path.join(checkpoint_folder, filename)
    
    # Extract model state dict (handle DDP wrapper)
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


# Load model checkpoint
def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Load model checkpoint (handles both DDP and non-DDP models).
    """
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state (handle DDP wrapper)
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
        return start_epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting from epoch 1.")
        return 1  # Start from the first epoch if no checkpoint is found
