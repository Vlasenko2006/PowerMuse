#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test and Visualize Complementary Masking

Quick script to see how different masking strategies affect encodings
"""

import torch
import numpy as np
from complementary_masking import apply_complementary_mask, visualize_mask_effect, get_mask_description


def test_masking(mask_type='temporal', B=4, D=128, T=600):
    """
    Test masking with synthetic data
    
    Args:
        mask_type: Type of masking to test
        B: Batch size
        D: Encoding dimensions
        T: Time frames (~600 = 4 seconds at 24kHz EnCodec)
    """
    print(f"\n{'='*80}")
    print(f"Testing Complementary Masking: {mask_type}")
    print(f"{'='*80}")
    
    # Create synthetic encoded representations
    torch.manual_seed(42)
    encoded_input = torch.randn(B, D, T) * 5.0  # Typical EnCodec scale
    encoded_target = torch.randn(B, D, T) * 5.0
    
    print(f"\nInput shape: {encoded_input.shape}")
    print(f"Target shape: {encoded_target.shape}")
    print(f"Duration: ~{T/150:.1f} seconds")
    
    # Apply masking
    print(f"\nApplying {mask_type} masking...")
    masked_input, masked_target = apply_complementary_mask(
        encoded_input, encoded_target,
        mask_type=mask_type,
        temporal_segment_frames=150,  # ~1 second
        freq_split_ratio=0.3,
        channel_keep_ratio=0.5,
        energy_threshold=0.7
    )
    
    print(f"\n✓ Masking applied!")
    print(f"  Description: {get_mask_description(mask_type)}")
    
    # Compute statistics
    input_coverage = (torch.abs(masked_input) > 0.1).float().mean().item()
    target_coverage = (torch.abs(masked_target) > 0.1).float().mean().item()
    
    # Check overlap
    input_active = (torch.abs(masked_input) > 0.1).float()
    target_active = (torch.abs(masked_target) > 0.1).float()
    overlap = (input_active * target_active).mean().item()
    
    print(f"\nStatistics:")
    print(f"  Input coverage: {input_coverage:.1%}")
    print(f"  Target coverage: {target_coverage:.1%}")
    print(f"  Overlap: {overlap:.1%}")
    print(f"  Complementary: {1.0 - overlap:.1%}")
    
    # Visualize
    print(f"\nGenerating visualization...")
    visualize_mask_effect(
        encoded_input, encoded_target,
        masked_input, masked_target,
        save_path=f'mask_viz_{mask_type}.png',
        mask_type=mask_type
    )
    
    print(f"✓ Saved: mask_viz_{mask_type}.png")
    print(f"\n{'='*80}\n")


def test_all_masks():
    """Test all masking strategies"""
    print("\n" + "="*80)
    print("COMPLEMENTARY MASKING TEST SUITE")
    print("="*80)
    
    mask_types = ['temporal', 'frequency', 'spectral', 'energy', 'hybrid']
    
    for mask_type in mask_types:
        test_masking(mask_type)
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)
    print("\nGenerated visualizations:")
    for mask_type in mask_types:
        print(f"  - mask_viz_{mask_type}.png")
    print("\nOpen these files to see how each masking strategy works!")
    print("="*80 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test complementary masking')
    parser.add_argument('--mask_type', type=str, default='all',
                       choices=['all', 'temporal', 'frequency', 'spectral', 'energy', 'hybrid'],
                       help='Masking type to test (default: all)')
    
    args = parser.parse_args()
    
    if args.mask_type == 'all':
        test_all_masks()
    else:
        test_masking(args.mask_type)
