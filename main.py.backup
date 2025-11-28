#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 12:37:20 2025

@author: andreyvlasenko
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import AttentionModel, AudioDataset
from train_and_validate import train_and_validate
from utilities import load_checkpoint

 # Number of steps after which encoder-decoder weights are frozen







# Constants
dataset_folder = "../dataset"
batch_size = 16 #* 4
epochs = 30000
sample_rate = 16000
learning_rate = 0.0002 *0.25 
FREEZE_ENCODER_DECODER_AFTER = 14
accumulation_steps = 4
checkpoint_folder = "checkpoints_trans2"
music_out_folder = "music_out_trans2"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resume_from_checkpoint = "checkpoints_trans2/model_epoch_90.pt"  # Change this to the checkpoint path if resuming

# Load datasets
train_data = np.load(os.path.join(dataset_folder, "training_set.npy"), allow_pickle=True)
val_data = np.load(os.path.join(dataset_folder, "validation_set.npy"), allow_pickle=True)

train_dataset = AudioDataset(train_data)
val_dataset = AudioDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, criterion, optimizer
input_dim = train_data[0][0].shape[-1]  # Infer input dimension from data
model = AttentionModel(input_dim=input_dim, sound_channels = 2, seq_len = 120000).to(device)
criterion = nn.MSELoss()  # Use MSELoss for reconstruction and task-specific losses
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Load checkpoint if specified
start_epoch = 1
if resume_from_checkpoint:
    start_epoch = load_checkpoint(resume_from_checkpoint, model, optimizer)

# Train and validate
train_and_validate(model,
                   train_loader, 
                   val_loader,
                   start_epoch, 
                   epochs, 
                   criterion,
                   optimizer,
                   device, 
                   sample_rate, 
                   checkpoint_folder, 
                   music_out_folder,
                   FREEZE_ENCODER_DECODER_AFTER = FREEZE_ENCODER_DECODER_AFTER,
                   accumulation_steps = accumulation_steps)
