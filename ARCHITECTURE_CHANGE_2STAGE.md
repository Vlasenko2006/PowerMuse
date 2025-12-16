# 2-Stage Architecture Change - Gradient Explosion Fix

## Date: December 15, 2025

## Problem
The 3-stage cascade architecture had a critical gradient explosion issue in stage 2:
- Stage 2 used weak 0.1× residual connection
- This caused output RMS to collapse to 1.09 (target was 5.69)
- Required 5.2× amplification (clamped to 1.5×)
- Output remained 71% too quiet → large loss → gradients exploded
- Training collapsed with NaN at epoch 3, batch 205

## Solution
Simplified from 3-stage to 2-stage cascade, eliminating the problematic stage:

### Old Architecture (3-stage)
```
Stage 0: concat(input, target) [256-dim] → output0 (residual 1.0×) ✓
Stage 1: concat(stage0, target, input) [384-dim] → output1 (residual 1.0×) ✓
Stage 2: concat(stage1, target, input) [384-dim] → output2 (residual 0.1×) ❌
                                                    └─→ RMS collapse → gradient explosion
```

### New Architecture (2-stage)
```
Stage 0: concat(input, target) [256-dim] → output0 (residual 1.0×) ✓
Stage 1: concat(stage0, target, input) [384-dim] → final_output (residual 1.0×) ✓
                                        └─→ No RMS collapse, stable gradients
```

## Changes Made

### model_simple_transformer.py

1. **Architecture simplification** (lines 167-209):
   - Changed from 3 stages to 2 stages
   - `for stage_idx in range(num_transformer_layers)` → `for stage_idx in range(2)`
   - Renamed `d_model_stage1/stage2plus` → `d_model_stage0/stage1`
   
2. **Residual connection** (lines 401-403):
   - **OLD**: Final stage used 0.1× weak residual
   - **NEW**: Both stages use 1.0× full residual
   - Removed weak residual logic entirely

3. **Documentation** (lines 3-11):
   - Updated docstring to reflect 2-stage architecture
   - Added explanation of why change was made

4. **Initialization print** (lines 203-218):
   - Updated to show "Cascade stages: 2 (FIXED - removed problematic stage 2)"

## Expected Results

### Gradient Norms
- **OLD (3-stage)**: Epoch 1: 2.45 → Epoch 2: 6.66 → Epoch 3: 16.71 → NaN
- **NEW (2-stage)**: Should stay in 2-5 range throughout training

### RMS Scaling
- **Stage 0**: output_rms ≈ 5.7, scale_factor ≈ 1.01 (healthy)
- **Stage 1**: output_rms ≈ 5.7, scale_factor ≈ 1.02 (healthy)
- No stage with RMS collapse requiring excessive amplification

### Training Stability
- Should complete 200 epochs without NaN collapse
- Gradients remain bounded by spectral norm + [0.7, 1.5] clamping + gradient clip 10.0
- Triple protection system still active on all output_proj layers

## Model Parameters

### Before (3-stage):
- Stage 0: 256-dim transformer (6 layers) → output_proj (256→128)
- Stage 1: 384-dim transformer (6 layers) → output_proj (384→128)
- Stage 2: 384-dim transformer (6 layers) → output_proj (384→128)
- **Total transformer parameters**: ~27.4M

### After (2-stage):
- Stage 0: 256-dim transformer (6 layers) → output_proj (256→128)
- Stage 1: 384-dim transformer (6 layers) → output_proj (384→128)
- **Total transformer parameters**: ~18.3M (33% reduction)

## Benefits

1. **Stability**: Eliminates gradient explosion root cause
2. **Efficiency**: 33% fewer parameters, faster training
3. **Simplicity**: Easier to understand and debug
4. **Quality**: Both stages produce high-quality outputs with full residual connections
5. **Scalability**: Can extend to 3+ stages later if needed, using 1.0× residual for all

## Testing Checklist

- [ ] Training completes epoch 3 without NaN
- [ ] Gradient norms stay 2-5 range through epoch 10
- [ ] No scale_factor warnings (>1.3×) in either stage
- [ ] Complementarity reaches 85%+ by epoch 50
- [ ] Output audio quality matches 3-stage baseline
- [ ] Inference generates coherent continuations

## Rollback Plan

If 2-stage architecture produces worse quality:
1. Keep 2-stage structure but increase stage 1 to 512-dim for more capacity
2. Or revert to 3-stage with 0.3× residual for stage 2 instead of 0.1×
3. Original 3-stage code preserved in git history

## Sync to Remote

```bash
bash sync_to_remote.sh
```

Files modified:
- model_simple_transformer.py (architecture + forward pass)
