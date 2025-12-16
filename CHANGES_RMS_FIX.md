# RMS Scaling Stability Fix

## Problem
Training was exploding with NaN at epochs 5, 14, and 24 due to unstable learnable RMS scaling parameters. These scalar parameters received gradients 100-10,000x larger than other parameters, causing training instability.

### Root Cause
- Learnable scalar parameters: `rms_scale_logs`, `rms_input_weights`, `rms_target_weights`
- Scalar parameters accumulate batch gradients without dampening
- RMS computation creates strong gradients through division operations
- Backprop through `exp(log_scale)` amplifies gradients exponentially
- Gradient clipping (1.0) insufficient for scalar parameters

### Evidence from Training Logs
```
Epoch 5: gradient norm 15,438 → NaN
  - Top gradients: rms_target_weights.1 (1576), rms_scale_logs.1 (1560), rms_input_weights.1 (1526)
  
Epoch 14: gradient norm 471 billion → NaN
  
Epoch 24: gradient norm 27 → NaN again
```

## Solution
Replaced **learnable RMS scaling** with **fixed RMS scaling** using equal weighting (50/50) of input and target RMS.

### Changes Made

#### 1. Removed Learnable Parameters (lines 144-150, 183-186)
**Before:**
```python
# Per-stage learnable RMS scaling parameters
self.rms_scale_logs = nn.ParameterList()
self.rms_input_weights = nn.ParameterList()
self.rms_target_weights = nn.ParameterList()

# ... in loop ...
self.rms_scale_logs.append(nn.Parameter(torch.zeros(1)))
self.rms_input_weights.append(nn.Parameter(torch.tensor(0.5)))
self.rms_target_weights.append(nn.Parameter(torch.tensor(0.5)))
```

**After:**
```python
# Removed all learnable RMS parameters
```

#### 2. Simplified RMS Scaling (lines 340-360)
**Before:**
```python
# Weighted combination of input and target RMS (per-stage learnable)
weight_sum = torch.abs(self.rms_input_weights[stage_idx]) + torch.abs(self.rms_target_weights[stage_idx]) + 1e-8
w_in = torch.abs(self.rms_input_weights[stage_idx]) / weight_sum
w_tgt = torch.abs(self.rms_target_weights[stage_idx]) / weight_sum
target_rms_combined = w_in * input_rms + w_tgt * target_rms

# Scale transformer output to match target RMS
scale_factor = torch.exp(self.rms_scale_logs[stage_idx])
output_projected_transposed = output_projected_transposed * (target_rms_combined / output_rms) * scale_factor
```

**After:**
```python
# Fixed equal weighting: target_rms = sqrt((input_rms² + target_rms²) / 2)
target_rms_combined = torch.sqrt((input_rms ** 2 + target_rms ** 2) / 2.0)

# Scale transformer output to match target RMS (no learnable scale factor)
output_projected_transposed = output_projected_transposed * (target_rms_combined / output_rms)
```

## Expected Benefits
1. **Training Stability**: No more NaN explosions from scalar parameter gradients
2. **Consistent RMS Ratios**: Output magnitude will stabilize around 1.0x input/target
3. **Simpler Architecture**: Fewer parameters, easier to debug
4. **Faster Convergence**: No need to learn scaling factors

## Architecture Summary
Current architecture after fix:
- **2 cascade stages** with hybrid residual connections
- **Stage 1**: concat(input, target) → transformer → output1 + 1.0x residual
- **Stage 2 (final)**: concat(input, output1, target) → transformer → output2 + 0.1x residual
- **RMS Scaling**: Fixed equal weighting of input and target RMS (no learnable parameters)
- **Residual Strategy**: 1.0x for early stages, 0.1x for final stage (reduces leakage)

## Recommendations

### 1. Clean Restart (CRITICAL)
```bash
# Remove corrupted checkpoints
rm -rf checkpoints/epoch_5_*.pth
rm -rf checkpoints/epoch_14_*.pth
rm -rf checkpoints/epoch_24_*.pth

# Start fresh training
./run_train_simple.sh
```

### 2. Monitor During Training
Watch for these indicators:
- ✅ Gradient norms stay under 100 (typically 5-50)
- ✅ RMS ratios stabilize around 0.8-1.2
- ✅ Output develops temporal structure (non-constant waveforms)
- ✅ Correlations stay low (<0.3) for creativity

### 3. Address Constant Output (After 10-20 Stable Epochs)
If output remains constant (all ████ bars), consider:
- **Option A**: Add small reconstruction loss (0.01-0.05)
- **Option B**: Increase final residual to 0.2-0.3
- **Option C**: Reduce novelty weight from 0.85 to 0.5-0.7

## Next Steps
1. Start fresh training run
2. Monitor gradient norms for first 30 epochs
3. Verify no NaN explosions
4. Check if temporal structure emerges
5. Adjust reconstruction loss if needed
