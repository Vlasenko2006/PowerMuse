#!/usr/bin/env python3
"""Quick test of the full pipeline (without actual training)"""

import torch
import encodec
from model_simple_transformer import SimpleTransformer

print("="*60)
print("FULL PIPELINE TEST")
print("="*60)

# 1. Load EnCodec (frozen)
print("\n1. Loading EnCodec...")
encodec_model = encodec.EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(6.0)
encodec_model.eval()
for param in encodec_model.parameters():
    param.requires_grad = False
print("   ✓ EnCodec loaded and frozen")

# 2. Create transformer
print("\n2. Creating SimpleTransformer...")
model = SimpleTransformer(
    encoding_dim=128,
    nhead=8,
    num_layers=4,
    dropout=0.1
)
print(f"   ✓ Model created: {sum(p.numel() for p in model.parameters()):,} params")

# 3. Simulate audio input (16s at 24kHz, mono)
print("\n3. Simulating audio input...")
audio_input = torch.randn(1, 1, 384000)  # [B, C, T]
print(f"   Input audio: {audio_input.shape}")

# 4. Encode
print("\n4. Encoding with EnCodec...")
with torch.no_grad():
    encoded_input = encodec_model.encoder(audio_input)
print(f"   Encoded: {encoded_input.shape}")

# 5. Transform
print("\n5. Applying transformer...")
encoded_input_batch = encoded_input  # Already [B, D, T]
encoded_output = model(encoded_input_batch)  # Keep gradients for training test
print(f"   Transformed: {encoded_output.shape}")

# 6. Decode
print("\n6. Decoding with EnCodec...")
with torch.no_grad():
    audio_output = encodec_model.decoder(encoded_output)
print(f"   Output audio: {audio_output.shape}")

# 7. Verify shapes
print("\n7. Verification...")
assert audio_output.shape == audio_input.shape, "Shape mismatch!"
print(f"   ✓ Input shape: {audio_input.shape}")
print(f"   ✓ Output shape: {audio_output.shape}")

# 8. Test MSE loss
print("\n8. Testing MSE loss...")
target_encoded = torch.randn_like(encoded_output)
loss = torch.nn.functional.mse_loss(encoded_output, target_encoded)
print(f"   MSE loss: {loss.item():.6f}")

# 9. Test backward pass
print("\n9. Testing backward pass...")
loss.backward()
print("   ✓ Gradients computed successfully")

print("\n" + "="*60)
print("✓ FULL PIPELINE TEST PASSED!")
print("="*60)
print("\nReady to train on HPC!")
print("Commands:")
print("  1. sbatch create_dataset_pairs_levante.sh")
print("  2. sbatch run_train_simple.sh")
print("="*60)
