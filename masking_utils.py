#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Masking utilities for multi-pattern music fusion

Handles random and fixed masking for 3-pattern input
"""

import torch
import numpy as np


def generate_random_mask(seq_len=352800, sample_rate=22050, min_unmasked_seconds=16, max_mask_ratio=0.3):
    """
    Generate random mask for one pattern, ensuring at least 16s remains unmasked.
    
    Args:
        seq_len: Total sequence length in samples (352800 = 16s @ 22050Hz)
        sample_rate: Audio sample rate
        min_unmasked_seconds: Minimum seconds that must remain unmasked (default 16s)
        max_mask_ratio: Maximum ratio of sequence that can be masked (default 30%)
    
    Returns:
        mask: Boolean tensor [seq_len], True = keep, False = mask out
    """
    total_seconds = seq_len / sample_rate
    min_unmasked_samples = int(min_unmasked_seconds * sample_rate)
    max_masked_samples = int(seq_len * max_mask_ratio)
    
    # Ensure we don't mask too much
    max_masked_samples = min(max_masked_samples, seq_len - min_unmasked_samples)
    
    if max_masked_samples <= 0:
        # Can't mask anything, return all True
        return torch.ones(seq_len, dtype=torch.bool)
    
    # Random mask length
    mask_length = np.random.randint(0, max_masked_samples + 1)
    
    if mask_length == 0:
        return torch.ones(seq_len, dtype=torch.bool)
    
    # Random mask position
    max_start = seq_len - mask_length
    mask_start = np.random.randint(0, max_start + 1)
    mask_end = mask_start + mask_length
    
    # Create mask (True = keep, False = mask out)
    mask = torch.ones(seq_len, dtype=torch.bool)
    mask[mask_start:mask_end] = False
    
    return mask


def generate_batch_masks(batch_size, num_patterns=3, seq_len=352800, sample_rate=22050, 
                         min_unmasked_seconds=16, fixed_mask=None):
    """
    Generate masks for a batch of multi-pattern inputs.
    
    Args:
        batch_size: Number of samples in batch
        num_patterns: Number of patterns per sample (default 3)
        seq_len: Sequence length
        sample_rate: Audio sample rate
        min_unmasked_seconds: Minimum unmasked seconds
        fixed_mask: If provided, use this mask for all patterns (for validation)
                   Shape: [num_patterns, seq_len]
    
    Returns:
        masks: Boolean tensor [batch_size, num_patterns, seq_len]
    """
    if fixed_mask is not None:
        # Use fixed mask, broadcast to batch
        return fixed_mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Generate random masks
    masks = torch.stack([
        torch.stack([
            generate_random_mask(seq_len, sample_rate, min_unmasked_seconds)
            for _ in range(num_patterns)
        ])
        for _ in range(batch_size)
    ])
    
    return masks


def apply_mask(audio, mask):
    """
    Apply mask to audio tensor.
    
    Args:
        audio: Audio tensor [batch, num_patterns, channels, seq_len]
        mask: Boolean mask [batch, num_patterns, seq_len], True = keep
    
    Returns:
        masked_audio: Same shape as audio, masked regions set to 0
    """
    # Expand mask to match audio dimensions
    # mask: [batch, num_patterns, seq_len] -> [batch, num_patterns, 1, seq_len]
    mask_expanded = mask.unsqueeze(2)
    
    return audio * mask_expanded.float()


def create_attention_mask(mask):
    """
    Convert boolean mask to attention mask for transformer.
    
    Args:
        mask: Boolean mask [batch, num_patterns, seq_len], True = keep
    
    Returns:
        attention_mask: Float mask for transformer, 0.0 = attend, -inf = ignore
    """
    # Transformer attention mask: 0.0 for positions to attend, -inf for positions to ignore
    attention_mask = torch.zeros_like(mask, dtype=torch.float)
    attention_mask[~mask] = float('-inf')
    
    return attention_mask


def get_unmasked_length(mask, sample_rate=22050):
    """
    Calculate length of unmasked region in seconds.
    
    Args:
        mask: Boolean mask [seq_len], True = keep
        sample_rate: Audio sample rate
    
    Returns:
        unmasked_seconds: Length of unmasked audio in seconds
    """
    unmasked_samples = mask.sum().item()
    return unmasked_samples / sample_rate


def create_overlapping_chunks(audio, chunk_size_samples, overlap=0.5):
    """
    Split audio into overlapping chunks for chunk-wise MSE computation.
    
    Args:
        audio: Audio tensor [batch, channels, seq_len] or [channels, seq_len]
        chunk_size_samples: Size of each chunk in samples
        overlap: Overlap ratio (0.5 = 50% overlap)
    
    Returns:
        chunks: List of audio chunks
    """
    if audio.dim() == 2:
        # [channels, seq_len]
        channels, seq_len = audio.shape
        audio = audio.unsqueeze(0)  # Add batch dim
        squeeze_output = True
    else:
        # [batch, channels, seq_len]
        squeeze_output = False
        channels, seq_len = audio.shape[1], audio.shape[2]
    
    stride = int(chunk_size_samples * (1 - overlap))
    chunks = []
    
    start = 0
    while start + chunk_size_samples <= seq_len:
        chunk = audio[:, :, start:start + chunk_size_samples]
        chunks.append(chunk)
        start += stride
    
    # Add last chunk if there's remaining audio
    if start < seq_len:
        chunk = audio[:, :, -chunk_size_samples:]
        chunks.append(chunk)
    
    if squeeze_output and len(chunks) > 0:
        chunks = [c.squeeze(0) for c in chunks]
    
    return chunks


# For validation: create fixed masks
def create_fixed_validation_masks(num_patterns=3, seq_len=352800, sample_rate=22050):
    """
    Create fixed masks for validation (reproducibility).
    
    Example masks:
    - Pattern 0: mask first 25%
    - Pattern 1: mask middle 25%
    - Pattern 2: mask last 25%
    
    Returns:
        masks: Boolean tensor [num_patterns, seq_len]
    """
    masks = []
    quarter = seq_len // 4
    
    for i in range(num_patterns):
        mask = torch.ones(seq_len, dtype=torch.bool)
        
        if i == 0:
            # Mask first quarter
            mask[:quarter] = False
        elif i == 1:
            # Mask second quarter
            mask[quarter:2*quarter] = False
        else:
            # Mask third quarter
            mask[2*quarter:3*quarter] = False
        
        masks.append(mask)
    
    return torch.stack(masks)
