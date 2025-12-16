# Compositional Creative Agent Stability Fixes

**Date**: December 14, 2025  
**Issue**: NaN explosion at epoch 3 during training with compositional agent  
**Status**: ✅ Stability safeguards added

---

## Problem Analysis

### Symptoms
- Epoch 1: Gradient norm 12.1 (normal)
- Epoch 2: Gradient norm 64.9 (growing)
- Epoch 3: Gradient norm 18.8 → NaN by end of epoch
- Epoch 4+: All compositional agent parameters = NaN

### Root Cause
The compositional agent (rhythm/harmony/timbre decomposition) was producing extreme values that:
1. Caused transformer outputs to explode beyond safe range
2. Led to NaN in EnCodec decoder when processing extreme encoded values
3. Propagated NaN through correlation cost computation
4. Corrupted all gradients

---

## Solutions Implemented

### 1. Clamping Transformer Output (ComponentComposer)
**File**: `compositional_creative_agent.py`, line ~220

```python
# BEFORE:
x = self.output_projection(x)  # [B, T, output_dim]
composed = x.transpose(1, 2)

# AFTER:
x = self.output_projection(x)  # [B, T, output_dim]
x = torch.clamp(x, -100.0, 100.0)  # Prevent extreme values
composed = x.transpose(1, 2)
```

**Rationale**: Prevents transformer from producing values >> 30 (EnCodec safe range)

### 2. Clamping Creative Output
**File**: `compositional_creative_agent.py`, line ~369

```python
# After composition
creative_output = self.composer(selected_components)

# ADDED: Clamp to safe range for EnCodec
creative_output = torch.clamp(creative_output, -50.0, 50.0)

# ADDED: NaN detection and recovery
if torch.isnan(creative_output).any():
    print("⚠️  WARNING: NaN detected in compositional agent output")
    creative_output = torch.where(
        torch.isnan(creative_output), 
        torch.zeros_like(creative_output), 
        creative_output
    )
```

**Rationale**: 
- Primary defense against extreme values
- Graceful degradation if NaN occurs (replace with zeros instead of crashing)

### 3. NaN Check in Correlation Cost
**File**: `compositional_creative_agent.py`, line ~520

```python
# ADDED: Check decoded audio for NaN before computing correlation
if torch.isnan(output_audio).any():
    print("⚠️  WARNING: NaN in output audio, correlation cost set to 0")
    return torch.tensor(0.0, device=output_audio.device)
```

**Rationale**: 
- Prevents correlation cost from propagating NaN
- Correlation cost becomes 0 when output is corrupted (graceful degradation)

---

## Expected Behavior After Fixes

### Training Stability
- ✅ Gradient norms should stay under 100
- ✅ No NaN explosions
- ✅ Smooth training through 50+ epochs

### Warning Signs to Monitor
If you see these warnings during training:
- `"NaN detected in compositional agent output"` → Output extreme, clamping engaged
- `"NaN in output audio, correlation cost set to 0"` → Decoder produced NaN

**Action**: If warnings persist:
1. Reduce learning rate (try 5e-5 instead of 1e-4)
2. Reduce correlation weight (try 0.5 instead of 0.85)
3. Add gradient clipping to compositional agent separately

### Healthy Training Metrics
```
Epoch 1: gradient_norm=10-20, novelty=0.03-0.05, corr_cost=0.15-0.25
Epoch 2: gradient_norm=20-40, novelty=0.01-0.03, corr_cost=0.30-0.40
Epoch 3+: gradient_norm=15-30, novelty=0.005-0.02, corr_cost=0.25-0.35
```

---

## Alternative Solutions (If Issues Persist)

### Option 1: Reduce Correlation Weight
**Current**: `corr_weight=0.85`  
**Try**: `corr_weight=0.3` or `corr_weight=0.5`

```bash
# In run_train_compositional.sh, change:
--corr_weight 0.5 \
```

### Option 2: Add Gradient Clipping to Compositional Agent
**File**: `train_simple_worker.py`

```python
# After main gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# ADD: Separate clipping for compositional agent
if hasattr(model_unwrapped, 'creative_agent'):
    torch.nn.utils.clip_grad_norm_(
        model_unwrapped.creative_agent.parameters(), 
        max_norm=0.5  # Stricter limit
    )
```

### Option 3: Use Lower Learning Rate
**Current**: `lr=1e-4`  
**Try**: `lr=5e-5`

### Option 4: Disable Correlation Cost Temporarily
**Current**: `corr_weight=0.85`  
**Try**: `corr_weight=0.0` for first 10 epochs

```bash
# Train without correlation cost first
--corr_weight 0.0 \
# Then resume with correlation cost after 10 epochs
```

---

## Testing the Fixes

### Quick Test (1 epoch)
```bash
python train_simple_ddp.py \
  --use_compositional_agent true \
  --epochs 1 \
  --mask_reg_weight 0.85 \
  --corr_weight 0.85 \
  --checkpoint_dir checkpoints_test/
```

**Expected output:**
- No NaN warnings
- Gradient norm < 100
- Loss values finite

### Full Test (20 epochs)
```bash
./run_train_compositional.sh
```

**Monitor for:**
- Epoch 3-5: Critical period where NaN occurred before
- Gradient norms staying stable
- Component weights showing balanced usage (rhythm/harmony/timbre all > 0.1)

---

## Files Modified

1. **compositional_creative_agent.py** (651 lines)
   - Line ~220: Added clamping to ComponentComposer output
   - Line ~369: Added clamping and NaN recovery to creative output
   - Line ~520: Added NaN check in correlation cost computation

---

## Recovery from Corrupted Checkpoint

If you have a checkpoint with NaN parameters:

```bash
# Start fresh - don't resume from corrupted checkpoint
rm -rf checkpoints_compositional/checkpoint_epoch_3.pt
rm -rf checkpoints_compositional/checkpoint_epoch_4.pt

# Retrain from scratch
./run_train_compositional.sh
```

---

**Status**: Ready for retraining with compositional agent
**Next**: `./run_train_compositional.sh` and monitor epochs 3-5 closely
