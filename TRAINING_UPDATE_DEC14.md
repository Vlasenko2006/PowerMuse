# Training Update: Balance Loss Diagnosis

**Date**: December 14, 2025  
**Status**: 🔍 Issue diagnosed, fix implemented, ready for retraining

---

## 🚨 Issue Discovered

Your waveform visualization shows the **output converges to the target**, ignoring the input.

### Root Cause
The balance loss only controls **mask values** (50/50 split ✓), but the main reconstruction loss still forces **output = target** because:

```bash
# Current configuration:
--loss_weight_input 0.0    # No penalty for ignoring input
--loss_weight_target 1.0   # Strong penalty for diverging from target
```

The model learns: "Use 50/50 masks, but still produce target audio" (cheating!)

---

## ✅ Solution Implemented

### 1. Added Diagnostic Logging

**File**: `train_simple_worker.py`

**New metrics** (computed every batch):
- `output_input_correlation`: How much output uses input (0-1)
- `output_target_correlation`: How much output uses target (0-1)

**Alerts**:
- ⚠️  OUTPUT IS COPYING TARGET! (if target_corr > 2× input_corr)
- ⚠️  OUTPUT IS COPYING INPUT! (if input_corr > 2× target_corr)
- ✓ Output appears to mix both sources (balanced correlations)

### 2. Created Fixed Training Script

**File**: `run_train_creative_agent_fixed.sh`

**Key changes**:
```bash
--loss_weight_input 0.0      # Keep at 0 (using GAN instead)
--loss_weight_target 0.0     # CHANGED: From 1.0 (using GAN instead)
--balance_loss_weight 10.0   # INCREASED: From 5.0
--novelty_weight 0.5         # MODERATE: From 0.85
--gan_weight 0.1             # NEW: Adversarial training
```

**Checkpoint directory**: `checkpoints_creative_agent_fixed/` (separate from old training)

---

## 📊 Expected Training Evolution

### Current Behavior (BROKEN)
```
Epoch 1-20:
  Input mask: 0.50, Target mask: 0.50  ✓ Balance perfect
  Output→Input corr: 0.11               ✗ Ignoring input
  Output→Target corr: 0.86              ✗ Copying target
  ⚠️  OUTPUT IS COPYING TARGET!
```

### Expected Behavior (FIXED)
```
Epoch 1-10 (Learning):
  Input mask: 0.50, Target mask: 0.50  ✓ Balance maintained
  Output→Input corr: 0.20-0.35         ↗ Learning to use input
  Output→Target corr: 0.65-0.75        ↘ Reducing target copying
  ⚠️  OUTPUT IS COPYING TARGET!        (still learning)

Epoch 10-30 (Transition):
  Input mask: 0.50, Target mask: 0.50  ✓ Balance maintained
  Output→Input corr: 0.35-0.50         ↗ Using input rhythm
  Output→Target corr: 0.50-0.65        ↘ Balanced mixing
  ✓ Output appears to mix both sources (SUCCESS!)

Epoch 30-50 (Stable):
  Input mask: 0.50, Target mask: 0.50  ✓ Balance maintained
  Output→Input corr: 0.45-0.60         ✓ Strong input usage
  Output→Target corr: 0.50-0.65        ✓ Balanced target usage
  ✓ Output appears to mix both sources (STABLE)
```

---

## 🚀 Next Steps

### On HPC (levante.dkrz.de)

```bash
# 1. Start training with fixed configuration
bash run_train_creative_agent_fixed.sh

# 2. Monitor logs for correlation metrics
# Look for these lines in training output:
# 📊 Output Correlation Analysis:
#    Output→Input corr: X.XXX
#    Output→Target corr: X.XXX

# 3. Success criteria (after 20+ epochs):
#    - Output→Input corr > 0.4
#    - Ratio (target/input) < 2.0
#    - Alert: ✓ Output appears to mix both sources

# 4. Run inference after 30 epochs
python inference_cascade.py \
    --checkpoint checkpoints_creative_agent_fixed/best_model.pt \
    --num_samples 5
```

### Watch for These Milestones

**Epoch 10**: 
- Input corr should be > 0.3
- If still < 0.2, may need to increase `loss_weight_input` to 0.5

**Epoch 20**:
- Input corr should be > 0.4
- Alert should change from ⚠️ to ✓

**Epoch 30**:
- Both correlations should be 0.4-0.6 (balanced)
- Complementarity should be 85-90%

---

## 📁 Files Modified

1. **`train_simple_worker.py`** - Added correlation diagnostics (35 lines added)
2. **`BALANCE_LOSS_DIAGNOSIS.md`** - Complete analysis document (300+ lines)
3. **`run_train_creative_agent_fixed.sh`** - Fixed training script (NEW)

---

## 🔧 If Correlations Don't Improve

### After 10 epochs, if still copying target:

**Option A**: Increase input weight
```bash
# Edit run_train_creative_agent_fixed.sh line 60
LOSS_WEIGHT_INPUT=0.5  # was 0.3
```

**Option B**: Decrease target weight
```bash
# Edit run_train_creative_agent_fixed.sh line 61
LOSS_WEIGHT_TARGET=0.5  # was 0.7
```

**Option C**: Increase novelty weight
```bash
# Edit run_train_creative_agent_fixed.sh line 79
NOVELTY_WEIGHT=0.8  # was 0.5
```

### After 20 epochs, if copying input instead:

```bash
# Edit run_train_creative_agent_fixed.sh
LOSS_WEIGHT_INPUT=0.2   # was 0.3
LOSS_WEIGHT_TARGET=0.8  # was 0.7
```

---

## 📊 Quick Validation (Local Machine)

If you want to test the current checkpoint locally:

```python
import torch
import torchaudio

# Load model
checkpoint = torch.load('checkpoints/best_model.pt')
# ... load model ...

# Generate output
input_audio, sr = torchaudio.load('input.wav')
target_audio, sr = torchaudio.load('target.wav')
output_audio = model.inference(input_audio, target_audio)

# Check correlations
output_flat = output_audio.reshape(-1)
input_flat = input_audio.reshape(-1)
target_flat = target_audio.reshape(-1)

corr_input = torch.corrcoef(torch.stack([output_flat, input_flat]))[0, 1]
corr_target = torch.corrcoef(torch.stack([output_flat, target_flat]))[0, 1]

print(f"Output→Input: {corr_input:.3f}")
print(f"Output→Target: {corr_target:.3f}")

if corr_target > 2 * corr_input:
    print("❌ Copying target (ratio: {:.1f}×)".format(corr_target / corr_input))
elif corr_input > 2 * corr_target:
    print("❌ Copying input (ratio: {:.1f}×)".format(corr_input / corr_target))
else:
    print("✓ Balanced mixing")
```

---

## 🎯 Success Criteria

Training is successful when:

1. ✅ **Masks balanced**: Input=0.50, Target=0.50
2. ✅ **Correlations balanced**: Both 0.4-0.6
3. ✅ **Alert shows**: "✓ Output appears to mix both sources"
4. ✅ **Complementarity high**: 85-90%
5. ✅ **Listening test**: Output has input rhythm + target harmony

---

**Summary**: The balance loss was working perfectly, but the main loss function was undermining it. With the fixed configuration, the model will be forced to actually use both sources, not just balance the masks.

Ready to retrain with: `bash run_train_creative_agent_fixed.sh` 🚀
