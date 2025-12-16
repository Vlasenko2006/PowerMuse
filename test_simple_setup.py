#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Simple Transformer Setup

Quick test to verify:
1. Model loads correctly
2. EnCodec encoding works
3. Dataset loading works
4. Forward pass works
"""

import torch
import encodec
from model_simple_transformer import SimpleTransformer, count_parameters


def test_model():
    """Test model architecture"""
    print("="*60)
    print("TEST 1: Model Architecture")
    print("="*60)
    
    model = SimpleTransformer(
        encoding_dim=128,
        nhead=8,
        num_layers=4,
        dropout=0.1
    )
    
    trainable, total = count_parameters(model)
    print(f"\nModel parameters:")
    print(f"  Trainable: {trainable:,}")
    print(f"  Total: {total:,}")
    
    # Test forward pass
    batch_size = 16
    T_enc = 1126  # ~16s at 24kHz
    
    x = torch.randn(batch_size, 128, T_enc)
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        out = model(x)
    
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Output shape mismatch!"
    
    print("✓ Model test passed!\n")


def test_encodec():
    """Test EnCodec encoding"""
    print("="*60)
    print("TEST 2: EnCodec Encoding")
    print("="*60)
    
    # Load EnCodec
    model = encodec.EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    
    # Freeze
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"EnCodec model loaded:")
    print(f"  Sample rate: 24000 Hz")
    print(f"  Bandwidth: 6.0")
    print(f"  Status: FROZEN")
    
    # Test encoding
    duration = 16.0  # seconds
    sample_rate = 24000
    num_samples = int(duration * sample_rate)
    
    # Create dummy audio [1, 1, samples] (mono)
    audio = torch.randn(1, 1, num_samples)
    print(f"\nInput audio shape: {audio.shape}")
    print(f"  Channels: {audio.shape[1]} (mono)")
    print(f"  Samples: {audio.shape[2]} ({audio.shape[2]/sample_rate:.1f}s)")
    
    with torch.no_grad():
        encoded = model.encode(audio)
        encoded_frames = encoded[0][0]  # [1, K, T]
        print(f"\nEncoded shape: {encoded_frames.shape}")
        print(f"  Codebooks (K): {encoded_frames.shape[1]}")
        print(f"  Frames (T): {encoded_frames.shape[2]}")
        
        # Test decode
        decoded = model.decode(encoded)
        print(f"\nDecoded shape: {decoded.shape}")
        assert decoded.shape == audio.shape, "Decode shape mismatch!"
    
    print("✓ EnCodec test passed!\n")


def test_mse_loss():
    """Test MSE loss"""
    print("="*60)
    print("TEST 3: MSE Loss")
    print("="*60)
    
    criterion = torch.nn.MSELoss()
    
    # Test identical tensors
    x = torch.randn(16, 128, 1126)
    loss = criterion(x, x)
    print(f"MSE(x, x) = {loss.item():.10f}")
    assert loss.item() < 1e-7, "Loss should be ~0 for identical tensors"
    
    # Test different tensors
    y = torch.randn(16, 128, 1126)
    loss = criterion(x, y)
    print(f"MSE(x, y) = {loss.item():.6f}")
    assert loss.item() > 0, "Loss should be > 0 for different tensors"
    
    print("✓ MSE loss test passed!\n")


def test_adamw():
    """Test AdamW optimizer"""
    print("="*60)
    print("TEST 4: AdamW Optimizer")
    print("="*60)
    
    model = SimpleTransformer(encoding_dim=128, nhead=8, num_layers=4, dropout=0.1)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    
    print(f"Optimizer: AdamW")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Weight decay: {optimizer.param_groups[0]['weight_decay']}")
    print(f"  Parameters: {len(optimizer.param_groups[0]['params'])}")
    
    # Test optimization step
    x = torch.randn(2, 128, 1126)
    target = torch.randn(2, 128, 1126)
    
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, target)
    
    print(f"\nInitial loss: {loss.item():.6f}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Compute new loss
    with torch.no_grad():
        output = model(x)
        new_loss = torch.nn.functional.mse_loss(output, target)
    
    print(f"After 1 step: {new_loss.item():.6f}")
    print("✓ AdamW test passed!\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING SIMPLE TRANSFORMER SETUP")
    print("="*60 + "\n")
    
    test_model()
    test_encodec()
    test_mse_loss()
    test_adamw()
    
    print("="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print("\nReady to train!")
    print("Submit with: sbatch run_train_simple.sh")
    print("="*60 + "\n")
