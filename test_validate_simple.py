#!/usr/bin/env python3
"""Simple test of validation script components"""

import torch
from pathlib import Path

print("Testing validation script components...\n")

# Test 1: Import model
print("1. Testing model import...")
try:
    from adaptive_window_agent import AdaptiveWindowCreativeAgent
    model = AdaptiveWindowCreativeAgent(encoding_dim=128, num_pairs=3)
    print(f"   ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

# Test 2: Import dataset
print("\n2. Testing dataset import...")
try:
    from dataset_wav_pairs import WavPairsDataset
    print("   ✓ Dataset class imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

# Test 3: Import losses
print("\n3. Testing loss functions...")
try:
    from training.losses import combined_loss
    print("   ✓ Loss function imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

# Test 4: Check if dataset exists
print("\n4. Checking for validation dataset...")
dataset_path = Path("dataset_pairs_wav_24sec/val")
if dataset_path.exists():
    files = list(dataset_path.glob("*.wav"))
    print(f"   ✓ Found {len(files)} files in {dataset_path}")
else:
    print(f"   ⚠ Warning: {dataset_path} not found")

# Test 5: Check for checkpoints
print("\n5. Checking for checkpoints...")
checkpoint_dir = Path("checkpoints_24")
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    print(f"   ✓ Found {len(checkpoints)} checkpoints in {checkpoint_dir}")
    if checkpoints:
        print(f"   Latest: {sorted(checkpoints)[-1].name}")
else:
    print(f"   ⚠ Warning: {checkpoint_dir} not found")

# Test 6: Test forward pass with dummy data
print("\n6. Testing forward pass with dummy data...")
try:
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(2, 128, 1200)  # B=2, D=128, T=1200
        dummy_target = torch.randn(2, 128, 1200)
        outputs, losses, metadata = model(dummy_input, dummy_target)
    
    print(f"   ✓ Forward pass successful")
    print(f"   - Outputs: {len(outputs)} pairs")
    print(f"   - Each pair shape: {outputs[0].shape}")
    print(f"   - Metadata keys: {list(metadata.keys())}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

print("\n" + "="*60)
print("✅ All basic tests passed!")
print("="*60)
print("\nYou can now try running:")
print("  python validate_hybrid.py --checkpoint checkpoints_24/hybrid_epoch_27.pt")
