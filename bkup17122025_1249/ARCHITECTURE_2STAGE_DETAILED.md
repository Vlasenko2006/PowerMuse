# 2-Stage Cascade Architecture - Actual Implementation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INPUT & TARGET ENCODINGS                        │
│                                                                          │
│  Input Audio (384K samples)  ──► EnCodec ──► [B, 128, 1200]            │
│                                              RMS = 5.66                  │
│                                                                          │
│  Target Audio (384K samples) ──► EnCodec ──► [B, 128, 1200]            │
│                                              RMS = 5.69                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           STAGE 0 (256-dim)                              │
│                                                                          │
│  Input: noisy_input [B, 128, 1200] (input + optional noise)            │
│  Input: encoded_target [B, 128, 1200]                                  │
│                                                                          │
│  Concat [noisy_input + target] ──► [B, 256, 1200]                      │
│         ↓                                                                │
│  Transpose ──► [B, 1200, 256]                                           │
│         ↓                                                                │
│  Positional Encoding                                                     │
│         ↓                                                                │
│  LayerNorm (input_norm)                                                  │
│         ↓                                                                │
│  Transformer (6 layers × 256-dim)                                        │
│    Each layer:                                                           │
│      → Self-Attention (8 heads)                                          │
│      → Add & Norm                                                        │
│      → FFN: Linear(256→1024) → GELU → Dropout → Linear(1024→256)        │
│      → Add & Norm                                                        │
│         ↓                                                                │
│  LayerNorm (post_norm)                                                   │
│         ↓                                                                │
│  Output Proj (256→128) + Spectral Norm                                  │
│         ↓                                                                │
│  Add Residual: out + 1.0 × (0.5 × input + 0.5 × target)                │
│         ↓                                                                │
│  Transpose ──► [B, 128, 1200]                                           │
│         ↓                                                                │
│  RMS Scaling:                                                            │
│    target_rms_combined = sqrt((input_rms² + target_rms²) / 2)          │
│    scale_factor = target_rms_combined / output_rms                      │
│    scale_factor = clamp(scale_factor, 0.7, 1.5)                        │
│    output = output × scale_factor                                       │
│         ↓                                                                │
│  Output: stage0_output [B, 128, 1200]                                   │
│         RMS ≈ 5.73, scale_factor ≈ 1.01                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           STAGE 1 (384-dim)                              │
│                                                                          │
│  Input: noisy_input [B, 128, 1200] (same as stage 0)                   │
│  Input: stage0_output [B, 128, 1200] (prev_output)                     │
│  Input: noisy_target [B, 128, 1200] (target + optional noise)          │
│                                                                          │
│  Concat [noisy_input + stage0_output + noisy_target] ──► [B, 384, 1200]│
│         ↓                                                                │
│  Compute new residual:                                                   │
│    x_residual = (input + stage0_output + target) / 3.0                 │
│         ↓                                                                │
│  Transpose ──► [B, 1200, 384]                                           │
│         ↓                                                                │
│  Positional Encoding                                                     │
│         ↓                                                                │
│  LayerNorm (input_norm)                                                  │
│         ↓                                                                │
│  Transformer (6 layers × 384-dim)                                        │
│    Each layer:                                                           │
│      → Self-Attention (8 heads)                                          │
│      → Add & Norm                                                        │
│      → FFN: Linear(384→1536) → GELU → Dropout → Linear(1536→384)        │
│      → Add & Norm                                                        │
│         ↓                                                                │
│  LayerNorm (post_norm)                                                   │
│         ↓                                                                │
│  Output Proj (384→128) + Spectral Norm                                  │
│         ↓                                                                │
│  Add Residual: out + 1.0 × x_residual                                  │
│    (x_residual = average of input, stage0, target)                     │
│         ↓                                                                │
│  Transpose ──► [B, 128, 1200]                                           │
│         ↓                                                                │
│  RMS Scaling:                                                            │
│    target_rms_combined = sqrt((input_rms² + target_rms²) / 2)          │
│    scale_factor = target_rms_combined / output_rms                      │
│    scale_factor = clamp(scale_factor, 0.7, 1.5)                        │
│    output = output × scale_factor                                       │
│         ↓                                                                │
│  Output: final_output [B, 128, 1200]                                    │
│         RMS ≈ 5.68, scale_factor ≈ 1.02                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                            Final Output [B, 128, 1200]
                                    │
                                    ▼
                         Decode with EnCodec ──► Audio [B, 1, 384000]
```

## Key Implementation Details

### Stage 0 Processing:
```python
# Input concatenation
x_concat = torch.cat([noisy_input, encoded_target_use], dim=1)  # [B, 256, T]

# Residual for blending (computed once before loop)
x_residual = 0.5 * (encoded_input_use + encoded_target_use)  # [B, 128, T]
x_residual_transposed = x_residual.transpose(1, 2)  # [B, T, 128]

# After transformer + projection
output_projected = output_proj(transformed)  # [B, T, 128]
output_projected = output_projected + x_residual_transposed  # Full 1.0× residual
```

### Stage 1 Processing:
```python
# Input concatenation (3 sources)
x_concat = torch.cat([noisy_input, output, noisy_target], dim=1)  # [B, 384, T]

# NEW residual (recomputed for stage 1)
x_residual = (encoded_input_use + output + encoded_target_use) / 3.0  # [B, 128, T]
x_residual_transposed = x_residual.transpose(1, 2)  # [B, T, 128]

# After transformer + projection
output_projected = output_proj(transformed)  # [B, T, 128]
output_projected = output_projected + x_residual_transposed  # Full 1.0× residual
```

### RMS Scaling (both stages):
```python
input_rms = sqrt(mean(input²))
target_rms = sqrt(mean(target²))
output_rms = sqrt(mean(output²))

target_rms_combined = sqrt((input_rms² + target_rms²) / 2)
scale_factor = target_rms_combined / (output_rms + 1e-8)
scale_factor = clamp(scale_factor, min=0.7, max=1.5)

output = output × scale_factor
```

## Spectral Normalization

Applied to all `output_proj` layers (both stages):
```python
output_proj = spectral_norm(nn.Linear(d_model, encoding_dim))
```

This creates:
- `weight_orig`: Original unconstrained weights
- `weight_u`: Singular vector for power iteration
- `weight`: Normalized weights = weight_orig / spectral_norm(weight_orig)

Effect: Constrains Lipschitz constant ≤ 1.0, preventing gradient amplification.

## Anti-Cheating Noise

Optional noise added during training to prevent model from copying:
```python
if self.training and self.anti_cheating > 0:
    noise = randn_like(input) * anti_cheating * input.std()
    noisy_input = input + noise
```

Applied to:
- Stage 0: input only
- Stage 1: input and target

## Parameter Count

- **Stage 0**: 256-dim transformer (6 layers) + output_proj
  - Transformer: ~6.3M parameters
  - Output proj: 256×128 = 33K
  
- **Stage 1**: 384-dim transformer (6 layers) + output_proj  
  - Transformer: ~14.2M parameters
  - Output proj: 384×128 = 49K

- **Creative Agent**: ~4M parameters (mask generator + discriminator)

**Total**: ~24.9M parameters

## Why This Works

1. **Stage 0**: Learns basic transformation from input+target → output
   - Full 1.0× residual maintains signal energy
   - RMS scaling keeps magnitude consistent
   - scale_factor ≈ 1.01 (minimal amplification needed)

2. **Stage 1**: Refines output using all 3 sources
   - Has access to: original input, stage0 output, target
   - Residual averages all 3 sources (prevents any single source dominance)
   - Full 1.0× residual maintains signal energy
   - RMS scaling keeps magnitude consistent
   - scale_factor ≈ 1.02 (minimal amplification needed)

3. **No RMS Collapse**: Both stages use full 1.0× residual
   - output_rms stays healthy (5.7-5.8 range)
   - No need for excessive amplification (5.2×)
   - Gradients remain stable

4. **Spectral Norm**: Constrains weight matrices
   - Prevents gradient explosion during backprop
   - Works synergistically with RMS clamping

5. **Triple Protection**: 
   - Layer 1: Spectral norm on weights
   - Layer 2: [0.7, 1.5] RMS scale clamping
   - Layer 3: Gradient clipping at 10.0
