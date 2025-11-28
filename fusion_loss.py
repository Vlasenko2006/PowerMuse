#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusion loss function for multi-pattern music generation

Implements chunk-wise MSE with min(MSE) selection across overlapping chunks
"""

import torch
import torch.nn.functional as F
from masking_utils import create_overlapping_chunks


def chunk_wise_mse_loss(output, targets, masks, sample_rate=22050, overlap=0.5):
    """
    Compute min(MSE) across overlapping chunks for multiple targets.
    
    For each target:
    1. Determine unmasked length from corresponding input mask
    2. Split both target and output into overlapping chunks of that length
    3. Compute MSE for each chunk pair
    4. Take minimum MSE as cost for this target
    5. Sum costs across all targets
    
    Args:
        output: Single fused prediction [batch, channels, seq_len]
        targets: Multiple target continuations [batch, num_targets, channels, seq_len]
        masks: Masks used for inputs [batch, num_targets, seq_len], True = keep
        sample_rate: Audio sample rate (default 22050)
        overlap: Overlap ratio for chunks (default 0.5 = 50%)
    
    Returns:
        loss: Scalar tensor, average min-MSE across batch and targets
    """
    batch_size = targets.shape[0]
    num_targets = targets.shape[1]
    
    total_loss = 0.0
    
    for b in range(batch_size):
        for t in range(num_targets):
            # Get single target and corresponding mask
            target = targets[b, t]  # [channels, seq_len]
            mask = masks[b, t]      # [seq_len]
            out = output[b]         # [channels, seq_len]
            
            # Calculate unmasked length in samples
            unmasked_samples = mask.sum().item()
            
            # Ensure we have enough samples for chunking
            if unmasked_samples < 1000:  # Safety check
                # Fall back to full MSE if mask is too small
                mse = F.mse_loss(out, target)
                total_loss += mse
                continue
            
            # Split into overlapping chunks based on unmasked length
            target_chunks = create_overlapping_chunks(target, unmasked_samples, overlap)
            output_chunks = create_overlapping_chunks(out, unmasked_samples, overlap)
            
            # Ensure we have chunks
            if len(target_chunks) == 0 or len(output_chunks) == 0:
                mse = F.mse_loss(out, target)
                total_loss += mse
                continue
            
            # Compute MSE for each chunk pair
            chunk_mses = []
            for t_chunk, o_chunk in zip(target_chunks, output_chunks):
                # Ensure chunks have same shape
                if t_chunk.shape != o_chunk.shape:
                    continue
                mse = F.mse_loss(o_chunk, t_chunk)
                chunk_mses.append(mse)
            
            if len(chunk_mses) == 0:
                # Fallback if no valid chunks
                mse = F.mse_loss(out, target)
                total_loss += mse
            else:
                # Take minimum MSE across chunks
                min_mse = min(chunk_mses)
                total_loss += min_mse
    
    # Average across batch and targets
    return total_loss / (batch_size * num_targets)


def multi_pattern_loss(reconstructed, inputs, output, targets, masks, 
                       criterion, use_chunk_wise=True, rec_weight=0.3, pred_weight=0.7):
    """
    Combined loss for multi-pattern training.
    
    Args:
        reconstructed: Reconstructed inputs [batch, num_patterns, channels, seq_len]
        inputs: Original inputs [batch, num_patterns, channels, seq_len]
        output: Single fused prediction [batch, channels, seq_len]
        targets: Target continuations [batch, num_patterns, channels, seq_len]
        masks: Input masks [batch, num_patterns, seq_len]
        criterion: Base loss function (e.g., nn.MSELoss())
        use_chunk_wise: Whether to use chunk-wise MSE for prediction loss
        rec_weight: Weight for reconstruction loss (default 0.3)
        pred_weight: Weight for prediction loss (default 0.7)
    
    Returns:
        total_loss: Combined weighted loss
        rec_loss: Reconstruction loss component
        pred_loss: Prediction loss component
    """
    batch_size, num_patterns = inputs.shape[0], inputs.shape[1]
    
    # Reconstruction loss: average across all patterns
    rec_loss = 0.0
    for i in range(num_patterns):
        rec_loss += criterion(reconstructed[:, i], inputs[:, i])
    rec_loss = rec_loss / num_patterns
    
    # Prediction loss
    if use_chunk_wise:
        pred_loss = chunk_wise_mse_loss(output, targets, masks)
    else:
        # Simple MSE across all targets
        pred_loss = 0.0
        for i in range(num_patterns):
            pred_loss += criterion(output, targets[:, i])
        pred_loss = pred_loss / num_patterns
    
    # Weighted combination
    total_loss = rec_weight * rec_loss + pred_weight * pred_loss
    
    return total_loss, rec_loss, pred_loss


def reconstruction_only_loss(reconstructed, inputs, criterion):
    """
    Reconstruction-only loss for Phase 1 and Phase 2 training.
    
    Args:
        reconstructed: Reconstructed inputs [batch, num_patterns, channels, seq_len]
        inputs: Original inputs [batch, num_patterns, channels, seq_len]
        criterion: Base loss function (e.g., nn.MSELoss())
    
    Returns:
        loss: Average reconstruction loss across all patterns
    """
    num_patterns = inputs.shape[1]
    
    total_loss = 0.0
    for i in range(num_patterns):
        total_loss += criterion(reconstructed[:, i], inputs[:, i])
    
    return total_loss / num_patterns
