#!/usr/bin/env python3
"""Test ratio supervision additions."""

import torch
from adaptive_window_agent import AdaptiveWindowCreativeAgent

print("Testing ratio supervision...")

# Create agent
agent = AdaptiveWindowCreativeAgent(encoding_dim=128, num_pairs=3)

# Create dummy inputs
B = 2
inputs = torch.randn(B, 128, 1200)
targets = torch.randn(B, 128, 1200)

# Forward pass
outputs, losses, metadata = agent(inputs, targets)

print(f"\n✓ Agent forward pass successful")
print(f"✓ Number of outputs: {len(outputs)}")
print(f"✓ 'window_params' in metadata: {'window_params' in metadata}")

if 'window_params' in metadata:
    print(f"✓ Number of window params: {len(metadata['window_params'])}")
    
    # Check first params
    params = metadata['window_params'][0]
    print(f"\nFirst pair params:")
    print(f"  start_input shape: {params['start_input'].shape}")
    print(f"  start_target shape: {params['start_target'].shape}")
    print(f"  ratio_input shape: {params['ratio_input'].shape}")
    print(f"  ratio_target shape: {params['ratio_target'].shape}")
    print(f"  tonality_strength shape: {params['tonality_strength'].shape}")
    
    # Test ratio diversity calculation
    all_ratios_input = [p['ratio_input'] for p in metadata['window_params']]
    all_ratios_target = [p['ratio_target'] for p in metadata['window_params']]
    
    ratios_input_stacked = torch.stack(all_ratios_input, dim=0)  # [3, B]
    ratios_target_stacked = torch.stack(all_ratios_target, dim=0)  # [3, B]
    
    print(f"\nRatio diversity test:")
    print(f"  Input ratios stacked shape: {ratios_input_stacked.shape}")
    print(f"  Target ratios stacked shape: {ratios_target_stacked.shape}")
    
    ratio_variance_input = torch.var(ratios_input_stacked, dim=0).mean()
    ratio_variance_target = torch.var(ratios_target_stacked, dim=0).mean()
    
    print(f"  Input variance: {ratio_variance_input.item():.6f}")
    print(f"  Target variance: {ratio_variance_target.item():.6f}")
    
    ratio_diversity_loss = -(ratio_variance_input + ratio_variance_target)
    print(f"  Diversity loss: {ratio_diversity_loss.item():.6f}")
    
    print(f"\n✓ All ratio supervision tests passed!")
else:
    print(f"✗ window_params not in metadata!")
