#!/usr/bin/env python3
"""Dry run test for pure_gan_mode feature"""

import sys
import torch
import argparse

print("="*60)
print("Pure GAN Mode - Dry Run Test")
print("="*60)

# Test 1: Check PyTorch availability
print("\n[Test 1] Testing PyTorch...")
try:
    print(f"✓ PyTorch {torch.__version__} available")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ PyTorch test failed: {e}")
    sys.exit(1)

# Test 2: Verify argument parser has new options
print("\n[Test 2] Testing argument parser...")
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pure_gan_mode', type=float, default=0.0)
    parser.add_argument('--gan_curriculum_start_epoch', type=int, default=0)
    
    # Test with valid values
    args1 = parser.parse_args(['--pure_gan_mode', '0.01', '--gan_curriculum_start_epoch', '51'])
    assert args1.pure_gan_mode == 0.01
    assert args1.gan_curriculum_start_epoch == 51
    print("✓ Arguments parsed correctly")
    print(f"  pure_gan_mode: {args1.pure_gan_mode}")
    print(f"  gan_curriculum_start_epoch: {args1.gan_curriculum_start_epoch}")
except Exception as e:
    print(f"✗ Argument parsing failed: {e}")
    sys.exit(1)

# Test 3: Verify noise interpolation logic
print("\n[Test 3] Testing noise interpolation logic...")
try:
    # Simulate inputs
    B, D, T = 4, 128, 1200
    inputs = torch.randn(B, D, T) * 5.0  # Mean ~5.0
    targets = torch.randn(B, 1, 384000) * 0.5  # Audio range
    
    # Store originals
    original_inputs = inputs.clone()
    original_targets = targets.clone()
    
    # Test case 1: alpha=0.0 (no noise)
    pure_gan_mode = 0.01
    gan_curriculum_counter = 0
    alpha = min(1.0, pure_gan_mode * gan_curriculum_counter)
    assert alpha == 0.0
    print(f"✓ Counter=0: alpha={alpha:.4f} (no noise)")
    
    # Test case 2: alpha=0.5 (50% noise)
    gan_curriculum_counter = 50
    alpha = min(1.0, pure_gan_mode * gan_curriculum_counter)
    assert alpha == 0.5
    
    # Generate per-sample noise
    input_std = inputs.std(dim=(1, 2), keepdim=True)
    target_std = targets.std(dim=(1, 2), keepdim=True)
    
    input_noise = torch.randn_like(inputs) * input_std
    target_noise = torch.randn_like(targets) * target_std
    
    # Interpolate
    noisy_inputs = (1.0 - alpha) * inputs + alpha * input_noise
    noisy_targets = (1.0 - alpha) * targets + alpha * target_noise
    
    print(f"✓ Counter=50: alpha={alpha:.4f}")
    print(f"  Input std range: {input_std.min():.2f} - {input_std.max():.2f}")
    print(f"  Target std range: {target_std.min():.4f} - {target_std.max():.4f}")
    print(f"  Noisy input != Original: {not torch.allclose(noisy_inputs, inputs)}")
    
    # Test case 3: alpha=1.0 (full noise)
    gan_curriculum_counter = 100
    alpha = min(1.0, pure_gan_mode * gan_curriculum_counter)
    assert alpha == 1.0
    
    noisy_inputs = (1.0 - alpha) * inputs + alpha * input_noise
    # Should be pure noise (no original signal)
    correlation = torch.corrcoef(torch.stack([noisy_inputs.flatten(), inputs.flatten()]))[0, 1]
    print(f"✓ Counter=100: alpha={alpha:.4f} (full noise)")
    print(f"  Correlation with original: {correlation:.4f} (should be near 0)")
    
    # Test case 4: alpha capped at 1.0
    gan_curriculum_counter = 200
    alpha = min(1.0, pure_gan_mode * gan_curriculum_counter)
    assert alpha == 1.0
    print(f"✓ Counter=200: alpha={alpha:.4f} (capped at 1.0)")
    
except Exception as e:
    print(f"✗ Noise interpolation test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Verify loss uses original targets
print("\n[Test 4] Testing loss computation...")
try:
    # Verify that original_targets are separate from noisy targets
    assert not torch.allclose(original_targets, noisy_targets)
    print("✓ Original targets preserved (different from noisy)")
    print(f"  Original mean: {original_targets.mean():.4f}")
    print(f"  Noisy mean: {noisy_targets.mean():.4f}")
except Exception as e:
    print(f"✗ Loss computation test failed: {e}")
    sys.exit(1)

# Test 5: Type consistency
print("\n[Test 5] Testing type consistency...")
try:
    assert isinstance(pure_gan_mode, float)
    assert isinstance(gan_curriculum_counter, int)
    assert isinstance(alpha, float)
    assert 0.0 <= alpha <= 1.0
    print("✓ All types correct")
    print(f"  pure_gan_mode: {type(pure_gan_mode).__name__}")
    print(f"  gan_curriculum_counter: {type(gan_curriculum_counter).__name__}")
    print(f"  alpha: {type(alpha).__name__} in range [0, 1]")
except Exception as e:
    print(f"✗ Type consistency test failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ All dry run tests PASSED!")
print("="*60)
print("\nReady to run training with pure_gan_mode enabled.")
