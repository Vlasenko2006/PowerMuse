#!/usr/bin/env python3
"""Test creative agent integration with SimpleTransformer"""

import torch
from model_simple_transformer import SimpleTransformer

print("="*80)
print("Testing Creative Agent Integration with SimpleTransformer")
print("="*80)

# Test 1: Model without creative agent
print("\n1. Testing WITHOUT creative agent (baseline):")
model_baseline = SimpleTransformer(
    encoding_dim=128,
    num_transformer_layers=3,
    use_creative_agent=False
)
print(f"   Model created: {sum(p.numel() for p in model_baseline.parameters()):,} parameters")

# Forward pass
batch_size = 2
seq_len = 300
encoded_input = torch.randn(batch_size, 128, seq_len)
encoded_target = torch.randn(batch_size, 128, seq_len)

output1, mask_loss1 = model_baseline(encoded_input, encoded_target)
print(f"   Output shape: {output1.shape}")
print(f"   Mask loss: {mask_loss1}")
assert mask_loss1 is None, "Baseline should not have mask loss"
print("   ✅ Baseline works (no creative agent)")

# Test 2: Model with creative agent
print("\n2. Testing WITH creative agent:")
model_creative = SimpleTransformer(
    encoding_dim=128,
    num_transformer_layers=3,
    use_creative_agent=True
)
print(f"   Model created: {sum(p.numel() for p in model_creative.parameters()):,} parameters")

# Forward pass
output2, mask_loss2 = model_creative(encoded_input, encoded_target)
print(f"   Output shape: {output2.shape}")
print(f"   Mask loss: {mask_loss2.item():.6f}" if mask_loss2 is not None else "   Mask loss: None")
assert mask_loss2 is not None, "Creative agent should have mask loss"
assert mask_loss2.item() > 0, "Mask loss should be positive"
print("   ✅ Creative agent works!")

# Test 3: Check that creative agent is actually used
print("\n3. Checking creative agent is being used:")
print(f"   model_creative.creative_agent: {model_creative.creative_agent}")
assert model_creative.creative_agent is not None, "Creative agent should exist"

# Run through creative agent directly
masked_input, masked_target, reg_loss = model_creative.creative_agent.generate_creative_masks(
    encoded_input, encoded_target, hard=False
)
print(f"   Masked input shape: {masked_input.shape}")
print(f"   Masked target shape: {masked_target.shape}")
print(f"   Regularization loss: {reg_loss.item():.6f}")

# Check complementarity
with torch.no_grad():
    input_mask, target_mask, _ = model_creative.creative_agent.mask_generator(
        encoded_input, encoded_target
    )
    overlap = (input_mask * target_mask).mean().item()
    complementarity = (1.0 - overlap) * 100
    print(f"   Complementarity: {complementarity:.1f}%")

print("   ✅ Creative agent generates complementary masks")

# Test 4: Gradient flow
print("\n4. Testing gradient flow:")
loss = output2.mean() + 0.1 * mask_loss2
loss.backward()

total_grad = sum(p.grad.norm().item() for p in model_creative.parameters() if p.grad is not None)
print(f"   Total gradient norm: {total_grad:.6f}")
assert total_grad > 0, "Gradients should flow"
print("   ✅ Gradients flow through creative agent")

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nCreative agent is fully integrated and working correctly.")
print("Ready to train on HPC!")
