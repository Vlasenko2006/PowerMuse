#!/usr/bin/env python3
"""Debug why validation loss is constant"""

import torch
import torch.nn as nn
from model_simple_transformer import SimpleTransformer

# Create model
model = SimpleTransformer(encoding_dim=128, nhead=8, num_layers=4, dropout=0.1, num_transformer_layers=1)
model.eval()

# Simulate validation: same input every time
torch.manual_seed(42)
x1 = torch.randn(16, 128, 1200)

# Run forward pass multiple times with same input
outputs = []
for i in range(5):
    with torch.no_grad():
        out = model(x1)
        outputs.append(out)
    print(f"Pass {i+1}: Output RMS = {torch.sqrt((out**2).mean()).item():.6f}")

# Check if outputs are identical
for i in range(1, 5):
    diff = (outputs[i] - outputs[0]).abs().max().item()
    print(f"Max diff between pass 1 and {i+1}: {diff:.6e}")

if all((outputs[i] - outputs[0]).abs().max().item() < 1e-6 for i in range(1, 5)):
    print("\n✅ Model is deterministic in eval mode (expected)")
else:
    print("\n❌ Model has randomness in eval mode (unexpected)")

# Now test with different inputs
print("\n" + "="*60)
print("Testing with different inputs:")
torch.manual_seed(100)
x2 = torch.randn(16, 128, 1200)

with torch.no_grad():
    out1 = model(x1)
    out2 = model(x2)

print(f"Output1 RMS: {torch.sqrt((out1**2).mean()).item():.6f}")
print(f"Output2 RMS: {torch.sqrt((out2**2).mean()).item():.6f}")
print(f"Difference RMS: {torch.sqrt(((out2-out1)**2).mean()).item():.6f}")

if torch.sqrt(((out2-out1)**2).mean()).item() > 0.1:
    print("✅ Model produces different outputs for different inputs")
else:
    print("❌ Model outputs are too similar!")
