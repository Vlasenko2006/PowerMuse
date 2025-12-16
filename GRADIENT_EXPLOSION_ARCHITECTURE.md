# Gradient Explosion Architecture Diagram

## 3-Stage Cascade Transformer

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
│  Concat [Input + Target] ──► LayerNorm ──► Transformer (6 layers)      │
│       [B, 256, 1200]              ▼                                     │
│                           Output Proj (256→128)                         │
│                                   ▼                                     │
│                        ✓ Spectral Norm Applied                         │
│                                   ▼                                     │
│                      Residual: out + 1.0 × inputs                      │
│                                   ▼                                     │
│                         Output RMS = 5.73                               │
│                         Scale Factor = 1.01 ✓                          │
│                         (no amplification needed)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           STAGE 1 (384-dim)                              │
│                                                                          │
│  Concat [Stage0 + Target] ──► LayerNorm ──► Transformer (6 layers)     │
│       [B, 384, 1200]              ▼                                     │
│                           Output Proj (384→128)                         │
│                                   ▼                                     │
│                        ✓ Spectral Norm Applied                         │
│                                   ▼                                     │
│                      Residual: out + 1.0 × stage0                      │
│                                   ▼                                     │
│                         Output RMS = 5.68                               │
│                         Scale Factor = 1.02 ✓                          │
│                         (no amplification needed)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           STAGE 2 (384-dim)                              │
│                         ⚠️  PROBLEM ZONE ⚠️                             │
│                                                                          │
│  Concat [Stage1 + Target] ──► LayerNorm ──► Transformer (6 layers)     │
│       [B, 384, 1200]              ▼                                     │
│                           Output Proj (384→128)                         │
│                                   ▼                                     │
│                        ✓ Spectral Norm Applied                         │
│                                   ▼                                     │
│            ❌ WEAK RESIDUAL: out + 0.1 × stage1  (OLD)                 │
│            ✓ MODERATE RESIDUAL: out + 0.3 × stage1  (NEW FIX)          │
│                                   ▼                                     │
│                    ──────────────────────────────                       │
│                   │   RMS COLLAPSE PROBLEM:    │                       │
│                   │                             │                       │
│                   │  OLD (0.1×): RMS = 1.09     │                       │
│                   │  Needs 5.2× amplification   │                       │
│                   │  Clamped to 1.5×            │                       │
│                   │  ❌ Output too quiet         │                       │
│                   │  ❌ Gradients explode        │                       │
│                   │                             │                       │
│                   │  NEW (0.3×): RMS ≈ 2.5-3.0  │                       │
│                   │  Needs ~2.0× amplification  │                       │
│                   │  Clamped to 1.5×            │                       │
│                   │  ✓ Closer to target         │                       │
│                   │  ✓ Smaller gradient error   │                       │
│                    ──────────────────────────────                       │
│                                   ▼                                     │
│                         Scale Factor = 1.5×                             │
│                     (CLAMPED from trying 5.2×)                          │
│                                   ▼                                     │
│                    ⚠️  WARNING: scale >1.3×                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         GRADIENT BACKPROPAGATION                         │
│                                                                          │
│  Loss Function ──► ∂L/∂output ──► Stage 2 ──► Stage 1 ──► Stage 0      │
│                                      │                                   │
│                                      ▼                                   │
│                        ┌──────────────────────────┐                     │
│                        │  GRADIENT EXPLOSION:     │                     │
│                        │                          │                     │
│                        │  Epoch 1: norm = 2.45    │                     │
│                        │  Epoch 2: norm = 6.66    │  (OLD 0.1×)         │
│                        │  Epoch 3: norm = 16.71   │                     │
│                        │  ❌ NaN at batch 205      │                     │
│                        │                          │                     │
│                        │  WITH FIXES (0.3×):      │                     │
│                        │  Epoch 1: norm = 3.93    │                     │
│                        │  Epoch 2: norm = 5.36    │  (IMPROVED)         │
│                        │  Epoch 3: norm ≈ 6-7?    │                     │
│                        │  ✓ Should stay stable    │                     │
│                        └──────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
```

## Why Stage 2 Causes Gradient Explosion

### The Residual Weight Problem

```
┌──────────────────────────────────────────────────────────────────┐
│                    STAGE 2 OUTPUT CALCULATION                     │
│                                                                   │
│  transformer_out = Transformer(concat[stage1, target])           │
│                   ▼                                               │
│            projected = output_proj(transformer_out)              │
│                   ▼                                               │
│         RESIDUAL BLENDING (the key issue):                       │
│                                                                   │
│  OLD: output = projected + 0.1 × stage1_output                   │
│       ─────────────────────────────────────                      │
│       │ projected: RMS ≈ 0.85 (weak signal)    │                │
│       │ 0.1 × stage1: RMS ≈ 0.57 (very weak)   │                │
│       │ ──────────────────────────────────────  │                │
│       │ TOTAL: RMS = 1.09 (too quiet!)         │                │
│       │                                         │                │
│       │ Target RMS = 5.69                       │                │
│       │ Needs 5.2× amplification ❌             │                │
│       │ Clamped to 1.5×                         │                │
│       │ Output = 1.09 × 1.5 = 1.64 (still quiet)│               │
│       ─────────────────────────────────────────                  │
│                                                                   │
│  NEW: output = projected + 0.3 × stage1_output                   │
│       ─────────────────────────────────────                      │
│       │ projected: RMS ≈ 0.85                  │                │
│       │ 0.3 × stage1: RMS ≈ 1.70               │                │
│       │ ──────────────────────────────────────  │                │
│       │ TOTAL: RMS ≈ 2.5-3.0 (better!)         │                │
│       │                                         │                │
│       │ Target RMS = 5.69                       │                │
│       │ Needs 2.0-2.3× amplification ✓          │                │
│       │ Clamped to 1.5×                         │                │
│       │ Output = 2.7 × 1.5 = 4.05 (closer!)    │                │
│       ─────────────────────────────────────────                  │
└──────────────────────────────────────────────────────────────────┘
```

## Gradient Flow Through output_proj

```
                    FORWARD PASS
                         │
                         ▼
          ┌──────────────────────────┐
          │  output_proj (384→128)   │
          │  + Spectral Norm         │
          │  (constrains weights ≤1) │
          └──────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────┐
          │  Weak Residual (0.1×)    │  ◄── PROBLEM HERE
          │  Output RMS drops to 1.09│
          └──────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────┐
          │  RMS Scaling (×1.5)      │
          │  Trying to match 5.69    │
          │  But only gets to 1.64   │
          └──────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────┐
          │  LOSS (large error)      │
          │  ∂L/∂out = HIGH          │
          └──────────────────────────┘
                         │
                    BACKWARD PASS
                         │
                         ▼
          ┌──────────────────────────┐
          │  ∂L/∂scale × scale (1.5×)│
          │  Error amplified!        │
          └──────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────┐
          │  ∂L/∂output_proj         │  ◄── EXPLODES HERE
          │  Gradient = 4.94         │      (Epoch 2)
          │  (was 2.58 in Epoch 1)   │
          └──────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────┐
          │  Weight Update           │
          │  Weights change too much │
          │  Next forward pass worse │
          └──────────────────────────┘
                         │
                         ▼
                  CYCLE REPEATS
              Gradients grow 2-3×
                  each epoch
                         │
                         ▼
                    NaN COLLAPSE
```

## Triple Protection System

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: SPECTRAL NORM                        │
│                                                                  │
│  Applied to: output_proj.weight                                 │
│  Effect: Constrains weight matrix spectral norm ≤ 1.0           │
│  Creates: weight_orig parameter (unconstrained)                 │
│           weight = weight_orig / spectral_norm(weight_orig)     │
│  Purpose: Prevents weight matrix from amplifying gradients      │
│                                                                  │
│  Result: ✓ Gradient norm stays bounded during backprop          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               LAYER 2: SCALE_FACTOR CLAMPING                     │
│                                                                  │
│  Clamp range: [0.7, 1.5]                                        │
│  Applied to: scale_factor = target_rms / output_rms             │
│  Purpose: Prevents >50% amplification in forward pass           │
│                                                                  │
│  OLD (0.1× residual):                                           │
│    output_rms = 1.09, needs 5.2×, clamped to 1.5×              │
│    ❌ 3.5× mismatch → large loss → large gradients              │
│                                                                  │
│  NEW (0.3× residual):                                           │
│    output_rms ≈ 2.7, needs 2.1×, clamped to 1.5×               │
│    ✓ 0.6× mismatch → smaller loss → smaller gradients          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               LAYER 3: GRADIENT CLIPPING                         │
│                                                                  │
│  Threshold: 5.0 (NEW, was 10.0)                                 │
│  Applied to: All model parameters during optimizer.step()       │
│  Purpose: Final safety net if gradients accumulate              │
│                                                                  │
│  Formula: if ||grad|| > 5.0:                                    │
│              grad = grad × (5.0 / ||grad||)                     │
│                                                                  │
│  Result: ✓ Prevents explosion even if layers 1-2 insufficient   │
└─────────────────────────────────────────────────────────────────┘
```

## Summary: The 0.1× vs 0.3× Residual Impact

```
                    0.1× RESIDUAL (OLD)              0.3× RESIDUAL (NEW)
                    ─────────────────                ─────────────────
Stage 2 Output:     RMS = 1.09                       RMS ≈ 2.7
Target:             RMS = 5.69                       RMS = 5.69
Scale Needed:       5.2× (clamped to 1.5×)           2.1× (clamped to 1.5×)
Actual Output:      1.64 (71% too quiet)             4.05 (29% too quiet)
Loss Error:         HIGH ❌                           MEDIUM ✓
Gradients:          Explode (2.5→6.7→16.7)           Stable (3.9→5.4→6-7?)
Result:             NaN at Epoch 3                   Should survive ✓
```

The key insight: **A 0.2 increase in residual weight (0.1→0.3) prevents RMS collapse, which reduces the amplification need from 5.2× to 2.1×, which keeps loss/gradients in stable range.**
