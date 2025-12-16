# Gradient Explosion Fix

## Problem

Training experienced catastrophic gradient explosion on epoch 3:
- **Epoch 1**: gradient norm = 6.9 (normal)
- **Epoch 2**: gradient norm = 45.3 (6.5× increase ⚠️)
- **Epoch 3**: gradient norm = 1,557,911 (34,400× increase 🔥)

Worst parameter: `module.cascade_stages.2.output_proj.weight` with norm = 1,467,765

## Root Causes

1. **Exponential penalty function too strong**
   - Function: `cost = -ln(1 - |corr|)`
   - Goes to infinity as correlation approaches 1.0:
     - corr = 0.9 → cost ≈ 2.3
     - corr = 0.95 → cost ≈ 3.0
     - corr = 0.99 → cost ≈ 4.6
     - corr = 0.999 → cost ≈ 6.9
     - corr = 0.9999 → cost ≈ 9.2
   - Original weight: `CORR_WEIGHT = 0.5` (too high for exponential penalty)

2. **Symmetric losses amplify gradients**
   - Spectral loss now computes: `0.5 * (loss_input + loss_target)`
   - Mel loss now computes: `0.5 * (loss_input + loss_target)`
   - Creates two gradient pathways instead of one
   - Combined with strong correlation penalty → explosion

3. **Insufficient gradient clipping**
   - Original: `max_norm = 1.0`
   - Not enough for exponential penalties which can produce very large gradients

## Solutions Applied

### 1. Reduced Correlation Weight
**File**: `run_train_creative_agent_fixed.sh`
```bash
# Before:
CORR_WEIGHT=0.5

# After:
CORR_WEIGHT=0.1  # Reduced from 0.5 - exponential penalty needs tiny weight
```

**Reasoning**: Exponential penalties need weights in range [0.05, 0.2] to remain stable. Even 0.5 is too aggressive.

### 2. Increased Gradient Clipping
**File**: `train_simple_worker.py`
```python
# Before:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# After:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Increased for exponential penalty stability
```

**Reasoning**: Larger clipping threshold (5.0) allows model to learn but prevents catastrophic explosions.

### 3. Tighter Correlation Clamping
**File**: `correlation_penalty.py`
```python
# Before:
corr_clamped = torch.clamp(corr_abs, 0.0, 0.999)  # Avoid ln(0)

# After:
corr_clamped = torch.clamp(corr_abs, 0.0, 0.95)  # Clamp to prevent numerical explosion (0.95 → cost ≈ 3.0)
```

**Reasoning**: 
- At corr=0.95: cost = -ln(0.05) ≈ 3.0 (reasonable)
- At corr=0.999: cost = -ln(0.001) ≈ 6.9 (too high)
- Clamping to 0.95 limits maximum penalty to ~3.0, preventing numerical instability

## Expected Impact

### Gradient Norms
- **Before**: 6.9 → 45.3 → 1,557,911 (catastrophic explosion)
- **After**: Should stay in range [5, 30] throughout training

### Correlation Penalty
- **Before**: Could reach infinity as corr → 1.0
- **After**: Maximum penalty ≈ 3.0 (at corr=0.95), scaled by weight 0.1 = 0.3

### Total Loss
- **Before**: Dominated by exploding correlation penalty
- **After**: Balanced contribution from all loss terms:
  - Input reconstruction: weight 0.3
  - Target reconstruction: weight 0.3
  - Spectral loss: weight 0.1 (symmetric)
  - Mel loss: weight 0.1 (symmetric)
  - Balance loss: weight 15.0
  - Correlation penalty: weight 0.1 (max contribution ~0.3)
  - GAN loss: weight 0.1

## Verification Steps

After restarting training, monitor these metrics:

1. **Gradient Norm** (should stay < 30):
   ```
   Epoch 1, Batch X: Gradient norm: Y.YY
   ```

2. **Correlation Penalty** (should be < 1.0):
   ```
   Avg Corr Penalty: 0.XXX
   ```

3. **Total Loss** (should decrease smoothly):
   ```
   Train Loss: X.XXX (should decrease from ~5.0 to ~2.0)
   ```

4. **Output Correlations** (should balance):
   ```
   Output → Input: 0.4-0.6 (mixing input features)
   Output → Target: 0.5-0.7 (mixing target features)
   ```

## Alternative Solutions (If Problem Persists)

If gradient explosion continues, consider:

1. **Replace exponential penalty with quadratic**:
   ```python
   cost = correlation ** 2  # Quadratic is numerically stable
   ```
   - No risk of infinity
   - Weight can be higher (0.5-1.0)
   - Linear gradients instead of exponential

2. **Further reduce correlation weight**:
   ```bash
   CORR_WEIGHT=0.05  # Even smaller for extra safety
   ```

3. **Increase clipping threshold**:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
   ```

4. **Use adaptive clipping based on gradient percentiles**:
   ```python
   # Clip to 95th percentile of gradient norms
   clip_value = compute_percentile_clipping(model)
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
   ```

## Files Modified

1. ✅ `run_train_creative_agent_fixed.sh` - Reduced CORR_WEIGHT from 0.5 to 0.1
2. ✅ `train_simple_worker.py` - Increased gradient clipping from 1.0 to 5.0
3. ✅ `correlation_penalty.py` - Clamped correlation to [0.0, 0.95] instead of [0.0, 0.999]

All files synced to HPC: ✅

## Next Steps

1. **Kill current training job** (SLURM job 21449738):
   ```bash
   scancel 21449738
   ```

2. **Restart training with fixed configuration**:
   ```bash
   bash run_train_creative_agent_fixed.sh
   ```

3. **Monitor first 3 epochs carefully**:
   - Watch gradient norms (should stay < 30)
   - Check correlation penalty values (should be < 1.0)
   - Verify loss decreasing smoothly

4. **If stable after 10 epochs**:
   - Continue to 200 epochs
   - Monitor output correlations for balanced mixing

## Technical Details

### Correlation Penalty Function
```python
def correlation_to_exponential_cost(correlation):
    corr_abs = torch.abs(correlation)
    corr_clamped = torch.clamp(corr_abs, 0.0, 0.95)  # Cap at 0.95
    cost = -torch.log(1.0 - corr_clamped)
    return cost
```

**Behavior**:
- corr = 0.0 → cost = 0.0 (no penalty)
- corr = 0.5 → cost ≈ 0.69
- corr = 0.8 → cost ≈ 1.61
- corr = 0.9 → cost ≈ 2.30
- corr = 0.95 → cost ≈ 3.00 (MAXIMUM after clamping)

### Loss Computation
```python
# Symmetric spectral loss (compare to BOTH sources)
spectral_loss_input = stft_loss(output, input)
spectral_loss_target = stft_loss(output, target)
spectral = 0.5 * (spectral_loss_input + spectral_loss_target)

# Symmetric mel loss (compare to BOTH sources)
mel_loss_input = mel_loss_fn(output, input)
mel_loss_target = mel_loss_fn(output, target)
mel_value = 0.5 * (mel_loss_input + mel_loss_target)

# Correlation penalty (decorrelate from BOTH sources)
if weight_correlation > 0:
    corr_penalty = compute_modulation_correlation_penalty(input_audio, target_audio, output_audio)
else:
    corr_penalty = torch.tensor(0.0, device=output.device)

# Total loss
total_loss = (
    weight_input * rms_input +
    weight_target * rms_target +
    weight_spectral * spectral +
    weight_mel * mel_value +
    weight_correlation * corr_penalty
)
```

## Summary

**Root Cause**: Exponential correlation penalty with weight=0.5 caused gradient explosion when combined with symmetric losses.

**Solution**: Three-pronged fix:
1. Reduce correlation weight: 0.5 → 0.1 (5× reduction)
2. Increase gradient clipping: 1.0 → 5.0 (5× increase)
3. Clamp correlation: 0.999 → 0.95 (limits max penalty to ~3.0)

**Expected Result**: Stable training with gradient norms < 30 and smooth loss decrease.
