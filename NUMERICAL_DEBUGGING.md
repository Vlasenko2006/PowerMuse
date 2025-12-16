# Detailed Numerical Debugging for Gradient Explosion

## Problem

Gradient explosion occurring in **cascade stage 1** (not correlation penalty):

```
Epoch 8, Batch 0:
Total gradient norm: 776,853,774,304,905,088 (777 QUADRILLION!)

Top exploding parameter:
  module.cascade_stages.1.output_proj.weight
  norm = 757,541,177,707,724,800 (757 quadrillion)
  max value = 44,241,646,367,277,056 (44 quadrillion)
```

This is **not** from the correlation penalty - it's happening inside the transformer cascade.

## Debugging Strategy

Added comprehensive numerical health checks at every layer of the cascade:

### 1. Check Function
```python
def check_tensor_health(tensor, name, stage_info=""):
    """Check for NaN/Inf and print statistics"""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        print(f"🔥 NUMERICAL ISSUE: {name} {stage_info}")
        print(f"   Has NaN: {has_nan}, Has Inf: {has_inf}")
        print(f"   Min/Max/Mean/Std: ...")
        return False
    else:
        print(f"✓ {name} {stage_info}: OK")
        return True
```

### 2. Monitoring Points (Per Stage)

For each cascade stage (0, 1, 2), we now check:

**Inputs:**
- ✓ `noisy_input` - input with anti-cheating noise
- ✓ `encoded_target_use` - target encoding
- ✓ `prev_output` - output from previous stage (stages 1+)
- ✓ `x_concat` - concatenated inputs [B, 2D or 3D, T]

**Layer Operations:**
- ✓ `x (after transpose)` - reshaped for transformer
- ✓ `x (after pos_encoding)` - with positional embeddings
- ✓ `x (after input_norm)` - after layer norm
- ✓ `transformed (after transformer)` - transformer output
- ✓ `transformed (after post_norm)` - after post-norm
- ✓ `output_projected (after output_proj)` - linear projection

**Post-Processing:**
- ✓ `output_projected (after residual)` - with residual connection
- ✓ RMS values: `input_rms`, `target_rms`, `output_rms`
- ✓ `scale_factor` - RMS normalization multiplier
- ✓ `output_projected_transposed (after RMS scaling)` - final stage output

### 3. Activation Conditions

Debugging enabled for **first batch only** of specific epochs:
- Epochs 1, 2, 3, 4, 5, 8 (critical observation points)
- Only on rank 0 (avoid DDP duplication)

```python
if num_batches == 0 and rank == 0:
    if epoch in [1, 2, 3, 4, 5, 8]:
        model_simple_transformer.DEBUG_NUMERICS = True
```

Automatically disabled after first batch to avoid log spam:
```python
if num_batches == 0 and rank == 0:
    model_simple_transformer.DEBUG_NUMERICS = False
```

## Expected Output

When explosion occurs, we'll see exactly which layer/operation introduces NaN/Inf:

### Example: Healthy Stage
```
================================================================================
CASCADE STAGE 1
================================================================================
✓ prev_output [stage 1]: min=-2.1234, max=3.4567, mean=0.1234, std=1.2345
✓ noisy_target [stage 1]: min=-1.9876, max=2.8901, mean=-0.0543, std=1.1234
✓ x_residual [stage 1]: min=-1.5432, max=2.6789, mean=0.0456, std=1.0987
✓ x_concat (after cat) [stage 1]: min=-2.1234, max=3.4567, mean=0.0234, std=1.1567
✓ x (after transpose) [stage 1]: OK
✓ x (after pos_encoding) [stage 1]: OK
✓ x (after input_norm) [stage 1]: OK
✓ transformed (after transformer) [stage 1]: OK
✓ transformed (after post_norm) [stage 1]: OK
✓ output_projected (after output_proj) [stage 1]: OK
   RMS values [stage 1]:
     input_rms: 5.6234
     target_rms: 5.8901
     output_rms (before scaling): 2.3456
     target_rms_combined: 5.7589
     scale_factor: min=2.3456, max=2.5678, mean=2.4567
✓ output_projected_transposed (after RMS scaling) [stage 1]: OK
```

### Example: Explosion Detected
```
================================================================================
CASCADE STAGE 1
================================================================================
✓ prev_output [stage 1]: OK
✓ noisy_target [stage 1]: OK
✓ x_residual [stage 1]: OK
✓ x_concat (after cat) [stage 1]: OK
✓ x (after transpose) [stage 1]: OK
✓ x (after pos_encoding) [stage 1]: OK
✓ x (after input_norm) [stage 1]: OK

🔥 NUMERICAL ISSUE DETECTED: transformed (after transformer) [stage 1]
   Shape: torch.Size([8, 1200, 384])
   Has NaN: False
   Has Inf: True
   Min: -12345678901234.000000
   Max: 98765432109876543.000000
   Mean: 123456789.012345
   Std: 456789012345.678901
```

This tells us the **transformer block** in stage 1 is producing Inf values.

## Likely Root Causes

Based on explosion pattern (stage 1, output_proj.weight):

### 1. LayerNorm Instability
```python
# LayerNorm can explode if variance is near zero
x_norm = (x - mean) / sqrt(variance + eps)

# If variance → 0, division explodes
# eps=1e-5 may be too small
```

**Fix**: Increase LayerNorm epsilon to 1e-6 or 1e-5

### 2. Attention Softmax Overflow
```python
# Softmax can overflow with large logits
attention_weights = softmax(Q @ K.T / sqrt(d_k))

# If Q @ K.T has values > 100, exp() → Inf
```

**Fix**: Ensure proper scaling or clip attention logits

### 3. RMS Scaling Division by Zero
```python
scale_factor = target_rms_combined / (output_rms + 1e-8)

# If output_rms is very small (< 1e-6), scale_factor → huge
```

**Fix**: Increase epsilon or clamp scale_factor

### 4. Residual Connection Amplification
```python
# Stage 1: full residual (weight = 1.0)
output_projected = output_projected + x_residual_transposed

# If both are large, sum explodes
```

**Fix**: Reduce residual weight or add gradient checkpointing

## Next Steps

1. **Run training with debugging enabled**:
   ```bash
   bash run_train_creative_agent_fixed.sh
   ```

2. **Observe epoch 1-5, 8 logs** - look for first NaN/Inf occurrence

3. **Identify exact layer** where explosion starts

4. **Apply targeted fix**:
   - LayerNorm epsilon → 1e-5
   - Attention logit clipping
   - RMS scale_factor clamping
   - Residual weight reduction

5. **Retest** until gradient norms stay < 10

## Files Modified

1. ✅ `model_simple_transformer.py`:
   - Added `DEBUG_NUMERICS` global flag
   - Added `check_tensor_health()` function
   - Inserted 15+ checkpoints in cascade forward pass
   - Added RMS calculation debugging

2. ✅ `train_simple_worker.py`:
   - Import `model_simple_transformer` module (for flag access)
   - Enable `DEBUG_NUMERICS = True` for epochs 1,2,3,4,5,8, batch 0
   - Disable `DEBUG_NUMERICS = False` after batch 0

3. ✅ `correlation_penalty.py`:
   - Already fixed (quadratic penalty)

4. ✅ `run_train_creative_agent_fixed.sh`:
   - Already fixed (CORR_WEIGHT=0.5)

**All files synced to HPC** ✅

## Monitoring Checklist

When training runs, verify:

- [ ] Epoch 1, batch 0: All layers show "✓ OK"
- [ ] Epoch 2, batch 0: Check for first Inf/NaN
- [ ] Epoch 3, batch 0: Confirm explosion point
- [ ] Epoch 4, batch 0: Identify exact operation
- [ ] Epoch 5, batch 0: Verify pattern consistency
- [ ] Epoch 8, batch 0: Check if explosion persists after clipping

## Expected Timeline

- **Epochs 1-2**: Gradients stable (norms 1-10)
- **Epoch 3-4**: Explosion begins (specific layer identified)
- **After fix**: Gradients stay < 10 throughout training

## Summary

Added comprehensive layer-by-layer numerical health monitoring to:
1. Identify exact layer where NaN/Inf first appears
2. Distinguish between LayerNorm, Attention, RMS scaling, or Residual issues
3. Enable targeted fixes instead of broad changes
4. Verify fixes work by comparing before/after debugging output

The debugging will pinpoint whether explosion originates from:
- Transformer blocks (attention overflow)
- Normalization layers (variance collapse)
- RMS scaling (division by near-zero)
- Residual connections (amplification)
