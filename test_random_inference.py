#!/usr/bin/env python3
"""Test inference with randomly initialized model"""

import torch
import soundfile as sf
import numpy as np
import encodec
import os
from model_simple_transformer import SimpleTransformer

# Configuration matching trained model
encoding_dim = 128
nhead = 8
num_layers = 6
num_transformer_layers = 3
dropout = 0.1
anti_cheating = 0.1
use_creative_agent = True
use_compositional = False

print("="*80)
print("TESTING INFERENCE WITH RANDOM MODEL (NO TRAINING)")
print("="*80)

# Create model with random initialization
print("\nCreating model with RANDOM weights...")
model = SimpleTransformer(
    encoding_dim=encoding_dim,
    nhead=nhead,
    num_layers=num_layers,
    num_transformer_layers=num_transformer_layers,
    dropout=dropout,
    anti_cheating=anti_cheating,
    use_creative_agent=use_creative_agent,
    use_compositional=use_compositional
)
model.eval()

print(f"  Creative agent: {use_creative_agent}")
print(f"  Cascade stages: {num_transformer_layers}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Load EnCodec
print("\nLoading EnCodec...")
encodec_model = encodec.EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(6.0)
encodec_model.eval()

# Load sample input and target
val_folder = "dataset_pairs_wav/val"
input_path = os.path.join(val_folder, "pair_0000_input.wav")
target_path = os.path.join(val_folder, "pair_0000_output.wav")

print(f"\nLoading audio:")
print(f"  Input: {input_path}")
print(f"  Target: {target_path}")

input_audio, sr = sf.read(input_path)
target_audio, sr2 = sf.read(target_path)

if input_audio.ndim == 2:
    input_audio = input_audio.mean(axis=1)
if target_audio.ndim == 2:
    target_audio = target_audio.mean(axis=1)

# Convert to tensors
input_tensor = torch.from_numpy(input_audio).float().unsqueeze(0).unsqueeze(0)
target_tensor = torch.from_numpy(target_audio).float().unsqueeze(0).unsqueeze(0)

print(f"  Input shape: {input_tensor.shape}")
print(f"  Target shape: {target_tensor.shape}")

# Run inference
print("\nRunning inference with RANDOM model...")
with torch.no_grad():
    # Encode
    encoded_input = encodec_model.encoder(input_tensor)
    encoded_target = encodec_model.encoder(target_tensor)
    print(f"  Encoded input: {encoded_input.shape}")
    print(f"  Encoded target: {encoded_target.shape}")
    
    # Transform
    model_output = model(encoded_input, encoded_target)
    if isinstance(model_output, tuple):
        encoded_output = model_output[0]
        print(f"  Model returned tuple (output + loss)")
    else:
        encoded_output = model_output
    print(f"  Encoded output: {encoded_output.shape}")
    
    # Decode
    output_audio = encodec_model.decoder(encoded_output)
    output_audio = output_audio.squeeze().numpy()

print(f"\nOutput statistics:")
print(f"  Shape: {output_audio.shape}")
print(f"  Range: [{output_audio.min():.3f}, {output_audio.max():.3f}]")
print(f"  RMS: {np.sqrt(np.mean(output_audio**2)):.6f}")

# Compute correlations
input_corr = np.corrcoef(output_audio, input_audio)[0, 1]
target_corr = np.corrcoef(output_audio, target_audio)[0, 1]

print(f"\nCorrelations:")
print(f"  Output→Input: {input_corr:.3f}")
print(f"  Output→Target: {target_corr:.3f}")

if target_corr > 0.7:
    print("  ⚠️  HIGH correlation with target - model copying target!")
elif input_corr > 0.7:
    print("  ⚠️  HIGH correlation with input - model copying input!")
elif abs(target_corr) < 0.3 and abs(input_corr) < 0.3:
    print("  ✓ LOW correlations - random/noise output (expected for untrained model)")
else:
    print("  ✓ Moderate correlations - some mixing")

# Save outputs
os.makedirs("test_outputs", exist_ok=True)
sf.write("test_outputs/random_output.wav", output_audio, sr)
print(f"\nSaved output: test_outputs/random_output.wav")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
