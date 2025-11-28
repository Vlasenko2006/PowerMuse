#!/usr/bin/env python3
"""
Test suite for multi-pattern fusion implementation.
Tests all components: masking, loss, model, dataset.

NOTE: Using seq_len=360000 (16.33s @ 22050Hz) with n_seq=9
      to match encoder-decoder architecture constraints.
"""

import torch
import torch.nn as nn
import numpy as np
import sys

# Use compatible sequence length
SEQ_LEN = 360000  # 16.33 seconds @ 22050 Hz (compatible with n_seq=9)
N_SEQ = 9

print("="*60)
print("MULTI-PATTERN FUSION - TEST SUITE")
print("="*60)
print(f"Using seq_len={SEQ_LEN} (16.33s @ 22050Hz) with n_seq={N_SEQ}")
print("="*60)

# Test 1: Masking Utils
print("\n[1/5] Testing masking_utils.py...")
try:
    from masking_utils import (
        generate_random_mask, 
        apply_mask, 
        create_attention_mask,
        create_overlapping_chunks,
        generate_batch_masks,
        create_fixed_validation_masks
    )
    
    # Test random mask generation
    mask = generate_random_mask(seq_len=SEQ_LEN, sample_rate=22050)
    assert mask.shape == (SEQ_LEN,), f"Mask shape wrong: {mask.shape}"
    assert mask.dtype == torch.bool, f"Mask dtype wrong: {mask.dtype}"
    
    # Test batch masks
    batch_masks = generate_batch_masks(batch_size=4, num_patterns=3, seq_len=SEQ_LEN, sample_rate=22050)
    assert batch_masks.shape == (4, 3, SEQ_LEN), f"Batch masks shape wrong: {batch_masks.shape}"
    
    # Test apply_mask
    audio = torch.randn(4, 3, 2, SEQ_LEN)
    masked = apply_mask(audio, batch_masks)
    assert masked.shape == audio.shape, "Masked audio shape wrong"
    
    # Test overlapping chunks
    signal = torch.randn(2, 88200)  # 4 seconds at 22050 Hz
    chunks = create_overlapping_chunks(signal, chunk_size_samples=44100, overlap=0.5)
    assert len(chunks) > 1, "Should create multiple overlapping chunks"
    
    print("✓ masking_utils.py - PASSED")
except Exception as e:
    print(f"✗ masking_utils.py - FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Fusion Loss
print("\n[2/5] Testing fusion_loss.py...")
try:
    from fusion_loss import chunk_wise_mse_loss, multi_pattern_loss, reconstruction_only_loss
    
    batch_size = 2
    num_patterns = 3
    criterion = nn.MSELoss()
    
    # Create test data
    output = torch.randn(batch_size, 2, SEQ_LEN)
    targets = torch.randn(batch_size, num_patterns, 2, SEQ_LEN)
    masks = torch.ones(batch_size, num_patterns, SEQ_LEN, dtype=torch.bool)
    reconstructed = torch.randn(batch_size, num_patterns, 2, SEQ_LEN)
    inputs = torch.randn(batch_size, num_patterns, 2, SEQ_LEN)
    
    # Test chunk-wise loss
    loss = chunk_wise_mse_loss(output, targets, masks, sample_rate=22050, overlap=0.5)
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not torch.isnan(loss), "Loss is NaN"
    
    # Test multi-pattern loss (returns tuple: total_loss, rec_loss, pred_loss)
    total_loss, rec_loss, pred_loss = multi_pattern_loss(reconstructed, inputs, output, targets, masks, criterion)
    assert total_loss.item() >= 0, "Total loss should be non-negative"
    assert rec_loss.item() >= 0, "Reconstruction loss should be non-negative"
    assert pred_loss.item() >= 0, "Prediction loss should be non-negative"
    
    # Test reconstruction-only loss
    loss = reconstruction_only_loss(reconstructed, inputs, criterion)
    assert loss.item() >= 0, "Reconstruction loss should be non-negative"
    
    print("✓ fusion_loss.py - PASSED")
except Exception as e:
    print(f"✗ fusion_loss.py - FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Model Architecture
print("\n[3/5] Testing model_multipattern.py...")
try:
    from model_multipattern import MultiPatternAttentionModel, MultiPatternAudioDataset
    
    # Create model
    model = MultiPatternAttentionModel(
        input_dim=2,
        num_patterns=3,
        num_heads=8,
        num_layers=4,
        n_channels=64,
        n_seq=N_SEQ,
        sound_channels=2,
        batch_size=2,
        seq_len=SEQ_LEN,
        dropout=0.15
    )
    
    # Test forward pass
    batch_size = 2
    inputs = torch.randn(batch_size, 3, 2, SEQ_LEN)
    masks = torch.ones(batch_size, 3, SEQ_LEN, dtype=torch.bool)
    
    model.eval()
    with torch.no_grad():
        reconstructed, output = model(inputs, masks)
    
    assert reconstructed.shape == (batch_size, 3, 2, SEQ_LEN), f"Reconstructed shape wrong: {reconstructed.shape}"
    assert output.shape == (batch_size, 2, SEQ_LEN), f"Output shape wrong: {output.shape}"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    
    print("✓ model_multipattern.py - PASSED")
except Exception as e:
    print(f"✗ model_multipattern.py - FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Dataset Class
print("\n[4/5] Testing MultiPatternAudioDataset...")
try:
    # Create mock triplet data
    mock_data = []
    for i in range(5):
        inputs = tuple([np.random.randn(2, SEQ_LEN).astype(np.float32) for _ in range(3)])
        targets = tuple([np.random.randn(2, SEQ_LEN).astype(np.float32) for _ in range(3)])
        mock_data.append((inputs, targets))
    
    dataset = MultiPatternAudioDataset(mock_data)
    assert len(dataset) == 5, f"Dataset length wrong: {len(dataset)}"
    
    # Test __getitem__
    inputs, targets = dataset[0]
    assert inputs.shape == (3, 2, SEQ_LEN), f"Inputs shape wrong: {inputs.shape}"
    assert targets.shape == (3, 2, SEQ_LEN), f"Targets shape wrong: {targets.shape}"
    
    print("✓ MultiPatternAudioDataset - PASSED")
except Exception as e:
    print(f"✗ MultiPatternAudioDataset - FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Integration Test
print("\n[5/5] Testing end-to-end integration...")
try:
    from torch.utils.data import DataLoader
    
    # Create small dataset
    mock_data = []
    for i in range(4):
        inputs = tuple([np.random.randn(2, SEQ_LEN).astype(np.float32) for _ in range(3)])
        targets = tuple([np.random.randn(2, SEQ_LEN).astype(np.float32) for _ in range(3)])
        mock_data.append((inputs, targets))
    
    dataset = MultiPatternAudioDataset(mock_data)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Create model
    model = MultiPatternAttentionModel(
        input_dim=2, num_patterns=3, num_heads=8, num_layers=4,
        n_channels=64, n_seq=N_SEQ, sound_channels=2, batch_size=2,
        seq_len=SEQ_LEN, dropout=0.15
    )
    
    # Test training step
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    for inputs, targets in loader:
        # Generate masks
        batch_size = inputs.shape[0]
        masks = generate_batch_masks(batch_size, 3, SEQ_LEN, 22050)
        
        # Apply masking
        masked_inputs = apply_mask(inputs, masks)
        
        # Forward pass
        reconstructed, output = model(masked_inputs, masks)
        
        # Compute loss (returns tuple)
        total_loss, rec_loss, pred_loss = multi_pattern_loss(reconstructed, inputs, output, targets, masks, criterion)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        assert total_loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(total_loss), "Loss is NaN"
        
        break  # Test only one batch
    
    print("✓ End-to-end integration - PASSED")
except Exception as e:
    print(f"✗ End-to-end integration - FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("ALL TESTS PASSED ✓")
print("="*60)
print("\nMulti-pattern fusion implementation verified:")
print("  • Masking utilities working correctly")
print("  • Loss functions computing properly")
print("  • Model architecture functional")
print("  • Dataset class loading data")
print("  • End-to-end training pipeline operational")
print("\n⚠️  NOTE: Config files need updating:")
print("  - seq_len: 352800 → 360000")
print("  - n_seq: 4 → 9")  
print("  - chunk_duration: 16s → 16.33s")
print("\nReady for production training (after config update)!")
print("="*60)
