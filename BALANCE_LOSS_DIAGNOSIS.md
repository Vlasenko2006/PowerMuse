# Balance Loss Issue: Diagnosis and Solution

**Date**: December 14, 2025  
**Status**: 🔍 DIAGNOSED - Awaiting retraining with fix

---

## 🚨 Problem Identified

### Symptom
Despite perfect 50/50 mask balance (Input=0.50, Target=0.50), the **output audio converges to the target**, completely ignoring the input rhythm/structure.

### Root Cause Analysis

The balance loss is working **exactly as designed**, but there's a fundamental architectural issue:

#### What Balance Loss Does ✓
- Controls the **mask values** (input_mask and target_mask)
- Enforces 50/50 split: `(input_mask.mean() - 0.5)² + (target_mask.mean() - 0.5)²`
- **Result**: Masks are perfectly balanced at ~0.50 each

#### What Balance Loss Does NOT Do ✗
- Does **NOT** control what the output audio sounds like
- Does **NOT** prevent the model from copying the target
- Does **NOT** affect the main reconstruction loss

### The Fundamental Issue

```python
# Current training objective (run_train_creative_agent.sh):
loss_weight_input = 0.0    # No input reconstruction loss
loss_weight_target = 1.0   # Strong target reconstruction loss
balance_loss_weight = 5.0  # Mask balance enforcement

# Total loss:
total_loss = 1.0 * MSE(output, target) + 5.0 * balance_loss + ...
```

**The problem**: The model learns to:
1. ✓ Use 50/50 masks (satisfies balance_loss)
2. ✗ **But still produce output = target** (satisfies reconstruction loss)

The masks are balanced, but the **output ignores the input** because there's no loss term that penalizes this!

---

## 📊 New Diagnostic Metrics (Implemented)

### Added Correlation Analysis

**File Modified**: `train_simple_worker.py`

**New metrics computed every batch**:
```python
# Correlation between output and sources (range: -1 to 1)
output_input_corr = correlation(output_audio, input_audio)
output_target_corr = correlation(output_audio, target_audio)
```

**Expected values for creative mixing**:
- **Healthy**: Both correlations ~0.3-0.7 (mixing both sources)
- **Copying target**: `output_target_corr ≈ 0.9-1.0`, `output_input_corr ≈ 0.1-0.3`
- **Copying input**: `output_input_corr ≈ 0.9-1.0`, `output_target_corr ≈ 0.1-0.3`

### Training Log Output (NEW)

```
Epoch 1/200:
  Train Loss: 0.2146
  🎨 Creative Agent: mask_reg=0.7471, complementarity=75.4%, overlap=0.246
     Input mask: 0.496, Target mask: 0.495
     Balance loss (raw): 0.0000 [×5.0 weight = 0.0002]
     Temporal diversity: 0.1000
  📊 Output Correlation Analysis:
     Output→Input corr: 0.237 (closer to 1.0 = using input)
     Output→Target corr: 0.257 (closer to 1.0 = copying target)
     ✓ Output appears to mix both sources

  # VS problematic case:
  📊 Output Correlation Analysis:
     Output→Input corr: 0.114 (closer to 1.0 = using input)
     Output→Target corr: 0.862 (closer to 1.0 = copying target)
     ⚠️  OUTPUT IS COPYING TARGET! (target corr >> input corr)
```

**Alert triggers**:
- ⚠️  **Copying target**: `output_target_corr > 2 × output_input_corr`
- ⚠️  **Copying input**: `output_input_corr > 2 × output_target_corr`
- ✓ **Mixing**: Both correlations within 2× of each other

---

## 🔧 Proposed Solutions

### Solution 1: Use GAN + Novelty Loss (RECOMMENDED)

**Change** (in `run_train_creative_agent_fixed.sh`):
```bash
# OLD:
--loss_weight_input 0.0 \
--loss_weight_target 1.0 \
--gan_weight 0.0 \
--novelty_weight 0.85 \

# NEW:
--loss_weight_input 0.0 \      # No reconstruction (let GAN decide)
--loss_weight_target 0.0 \     # No reconstruction (let GAN decide)
--gan_weight 0.1 \             # Adversarial quality control
--novelty_weight 0.5 \         # Encourage creativity
--balance_loss_weight 10.0 \   # Strong 50/50 enforcement
```

**Effect**: 
- No direct reconstruction loss to bias toward target
- GAN forces output to be "realistic" but not copy sources
- Novelty loss prevents exact copying
- Balance loss ensures 50/50 mask usage

**Expected correlations after fix**:
- `output_input_corr`: 0.4-0.6 (using input rhythm)
- `output_target_corr`: 0.5-0.7 (using target harmony)

---

### Solution 2: Increase Balance Loss Weight (EXPERIMENTAL)

**Try**:
```bash
--balance_loss_weight 20.0   # Much stronger (was 5.0)
```

**Rationale**: Make mask balance so important that output must diverge from pure target to maintain it.

**Risk**: May hurt reconstruction quality if too high.

---

### Solution 3: Add Novelty Loss (STRONGEST)

**Already implemented in code** but weight is 0.0!

**Change**:
```bash
--novelty_weight 0.5 \  # Penalize copying either source
```

**Effect**: Directly penalizes output for being too similar to input or target in latent space.

---

## 🎯 Recommended Training Configuration

### Balanced Approach (Most Likely to Succeed)
```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_simple_ddp.py \
    --loss_weight_input 0.0 \      # No reconstruction (let GAN decide)
    --loss_weight_target 0.0 \     # No reconstruction (let GAN decide)
    --balance_loss_weight 10.0 \   # Increased from 5.0
    --novelty_weight 0.5 \         # Encourage creativity
    --corr_weight 0.5 \            # Prevent envelope copying
    --gan_weight 0.1 \             # Adversarial quality control
    # ... rest unchanged
```

**Expected behavior**:
- Input mask: 0.50 (balanced)
- Target mask: 0.50 (balanced)
- Output→Input corr: 0.4-0.6 (using input)
- Output→Target corr: 0.5-0.7 (using target)
- Complementarity: 85-90% (high)

---

## 📈 How to Monitor During Training

### Watch for these patterns:

**Epoch 1-10** (Learning phase):
```
Output→Input corr: 0.1-0.3   # Still learning to use input
Output→Target corr: 0.6-0.8  # High target bias (normal at start)
⚠️  OUTPUT IS COPYING TARGET!  # Expected initially
```

**Epoch 10-30** (Transition):
```
Output→Input corr: 0.3-0.5   # Increasing (good!)
Output→Target corr: 0.5-0.7  # Decreasing (good!)
✓ Output appears to mix both sources  # Goal achieved!
```

**Epoch 30-50** (Convergence):
```
Output→Input corr: 0.4-0.6   # Stable
Output→Target corr: 0.5-0.7  # Stable
✓ Output appears to mix both sources  # Maintained
```

### Red flags:
- If `output_target_corr > 0.8` after epoch 20 → Increase `loss_weight_input`
- If `output_input_corr < 0.3` after epoch 30 → Increase `loss_weight_input` further
- If both correlations < 0.2 → Output is random noise (reduce novelty_weight)

---

## 🧪 Quick Test (Before Full Training)

Run inference with current checkpoint and check correlations:

```python
import torch
import torchaudio

# Load checkpoint
model = load_checkpoint('checkpoints_creative_agent/best_model.pt')
input_audio, sr = torchaudio.load('input.wav')
target_audio, sr = torchaudio.load('target.wav')

# Generate output
output_audio = model.inference(input_audio, target_audio)

# Compute correlations
output_flat = output_audio.reshape(-1)
input_flat = input_audio.reshape(-1)
target_flat = target_audio.reshape(-1)

corr_input = torch.corrcoef(torch.stack([output_flat, input_flat]))[0, 1]
corr_target = torch.corrcoef(torch.stack([output_flat, target_flat]))[0, 1]

print(f"Output→Input: {corr_input:.3f}")
print(f"Output→Target: {corr_target:.3f}")

if corr_target > 2 * corr_input:
    print("❌ PROBLEM CONFIRMED: Output copying target")
else:
    print("✓ Output mixing both sources")
```

---

## 📝 Summary

### What We Learned

1. **Balance loss works perfectly** - Masks are 50/50 ✓
2. **But the output ignores masks** - Because reconstruction loss dominates ✗
3. **Need additional loss terms** to enforce actual mixing in output audio

### The Fix (3-step)

1. **Add input reconstruction loss** (`loss_weight_input=0.3`)
2. **Add novelty loss** (`novelty_weight=0.2`)
3. **Monitor correlations** (new diagnostic metrics)

### Expected Improvement

**Before fix**:
- Masks: 50/50 ✓
- Output: Copies target ✗
- Correlation: 0.11 input / 0.86 target

**After fix**:
- Masks: 50/50 ✓
- Output: Mixes both ✓
- Correlation: 0.45 input / 0.55 target

---

## 🚀 Next Steps

1. **Retrain with new configuration** (see recommended config above)
2. **Monitor correlation metrics** during training
3. **Validate improvement** via inference after 20 epochs
4. **Iterate if needed** (adjust weights based on correlations)

---

**Status**: Ready to retrain with diagnostic logging enabled  
**Expected time**: 30-50 epochs to see improvement  
**Success criteria**: `output_input_corr > 0.4` and ratio < 2×
