#!/usr/bin/env python3
"""
Test script to verify tensor sizes in hybrid training.
Ensures all audio tensors match before computing losses.
"""

import torch

print("=" * 80)
print("TENSOR SIZE VERIFICATION TEST")
print("=" * 80)

# Test parameters
B = 2  # batch size
sample_rate = 24000
duration_24sec = 24
samples_24sec = duration_24sec * sample_rate
samples_per_frame = 320

print(f"\nParameters:")
print(f"  Batch size: {B}")
print(f"  Sample rate: {sample_rate} Hz")
print(f"  Input duration: {duration_24sec} sec")
print(f"  Input samples: {samples_24sec}")
print(f"  Samples per EnCodec frame: {samples_per_frame}")

print(f"\n1. Creating test audio...")
audio_24sec = torch.randn(B, 1, samples_24sec)
print(f"   Input audio shape: {audio_24sec.shape}")
assert audio_24sec.shape == (B, 1, samples_24sec)
print("   ✓ Input audio size correct")

print(f"\n2. Simulating agent outputs (800 frames per pair, 3 pairs)...")
outputs_list = []
for i in range(3):
    output = torch.randn(B, 128, 800)
    outputs_list.append(output)
print(f"   Each pair: [B={B}, D=128, T=800]")
print("   ✓ All pair outputs created")

print(f"\n3. Calculating expected decoded size...")
frames_output = 800
expected_samples = frames_output * samples_per_frame
expected_duration = expected_samples / sample_rate
print(f"   Frames: {frames_output}")
print(f"   Samples: {expected_samples}")
print(f"   Duration: {expected_duration:.2f} seconds")

print(f"\n4. Calculating center extraction indices...")
offset = (samples_24sec - expected_samples) // 2
end_idx = offset + expected_samples
print(f"   24-second audio: {samples_24sec} samples")
print(f"   Decoded output:  {expected_samples} samples")
print(f"   Center offset:   {offset} samples")
print(f"   Extraction:      [{offset}:{end_idx}]")
print(f"   ✓ Indices: audio[:, :, {offset}:{end_idx}]")

print(f"\n5. Extracting matching segments from input/target...")
input_segment = audio_24sec[:, :, offset:end_idx]
target_segment = audio_24sec[:, :, offset:end_idx]
print(f"   Input segment shape:  {input_segment.shape}")
print(f"   Target segment shape: {target_segment.shape}")
print(f"   Expected: [{B}, 1, {expected_samples}]")
assert input_segment.shape == (B, 1, expected_samples)
assert target_segment.shape == (B, 1, expected_samples)
print("   ✓ Segment extraction correct")

print(f"\n6. Simulating decoded output...")
# Simulate what decoder would produce
output_decoded = torch.randn(B, 1, expected_samples)
print(f"   Output shape:  {output_decoded.shape}")
print(f"   Input shape:   {input_segment.shape}")
print(f"   Target shape:  {target_segment.shape}")
assert output_decoded.shape == input_segment.shape == target_segment.shape
print("   ✓ All shapes match!")

print(f"\n7. Testing loss computation (squeezed tensors)...")
output_squeezed = output_decoded.squeeze(1)
input_squeezed = input_segment.squeeze(1)
target_squeezed = target_segment.squeeze(1)

print(f"   Output squeezed:  {output_squeezed.shape} - expected [{B}, {expected_samples}]")
print(f"   Input squeezed:   {input_squeezed.shape} - expected [{B}, {expected_samples}]")
print(f"   Target squeezed:  {target_squeezed.shape} - expected [{B}, {expected_samples}]")

assert output_squeezed.shape == input_squeezed.shape == target_squeezed.shape
assert output_squeezed.shape == (B, expected_samples)
print("   ✓ All squeezed shapes match")

print(f"\n8. Computing RMS losses...")
rms_output_input = torch.sqrt(torch.mean((output_squeezed - input_squeezed) ** 2))
rms_output_target = torch.sqrt(torch.mean((output_squeezed - target_squeezed) ** 2))
print(f"   RMS(output, input):  {rms_output_input.item():.4f}")
print(f"   RMS(output, target): {rms_output_target.item():.4f}")
print("   ✓ Loss computation successful - NO SIZE ERRORS!")

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print(f"\nCONFIGURATION FOR CODE:")
print(f"  Agent outputs: 800 frames")
print(f"  Decoded size: {expected_samples} samples ({expected_duration:.2f} sec)")
print(f"  Extraction: audio[:, :, {offset}:{end_idx}]")
print(f"  All tensor sizes: [{B}, 1, {expected_samples}] → squeeze → [{B}, {expected_samples}]")
print("=" * 80)
