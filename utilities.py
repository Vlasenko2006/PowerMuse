#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 20:28:05 2025

@author: andrey
"""

import os
import numpy as np
import torch


# Save one validation sample as NumPy files
def save_sample_as_numpy(model, val_loader, device, music_out_folder, epoch, prefix=''):
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:  # Take one batch
            inputs, targets = inputs.to(device), targets.to(device)
            reconstructed, outputs = model(inputs)

            # Convert to NumPy and save
            input_np = inputs.cpu().numpy()[0]  # Take the first sample
            reconstructed_np = reconstructed.cpu().numpy()[0]
            output_np = outputs.cpu().numpy()[0]
            target_np = targets.cpu().numpy()[0]

            os.makedirs(music_out_folder, exist_ok=True)
            np.save(os.path.join(music_out_folder, prefix + f"input_epoch_{epoch}.npy"), input_np)
            np.save(os.path.join(music_out_folder, prefix + f"reconstructed_epoch_{epoch}.npy"), reconstructed_np)
            np.save(os.path.join(music_out_folder, prefix + f"output_epoch_{epoch}.npy"), output_np)
            np.save(os.path.join(music_out_folder, prefix + f"target_epoch_{epoch}.npy"), target_np)

            print(f"Saved input, reconstructed, output, and target as NumPy files for epoch {epoch}.")
            break  # Save only one sample


# Save model checkpoint
def save_checkpoint(model, optimizer, epoch, checkpoint_folder, filename=None):
    os.makedirs(checkpoint_folder, exist_ok=True)
    if filename is None:
        checkpoint_path = os.path.join(checkpoint_folder, f"model_epoch_{epoch}.pt")
    else:
        checkpoint_path = os.path.join(checkpoint_folder, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


# Load model checkpoint
def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
        return start_epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting from epoch 1.")
        return 1  # Start from the first epoch if no checkpoint is found
