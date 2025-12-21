# CRITICAL FIX: EnCodec Frame-to-Sample Calculation

**Date:** Dec 18, 2025  
**Status:** ✅ FIXED  
**Branch:** adaptive-window-selection  
**Commit:** 9f6206e

---

## Problem

RuntimeError during training:
```
RuntimeError: The size of tensor a (256000) must match the size of tensor b (384000)
```

---

## Root Cause

**WRONG ASSUMPTION:**
- Believed 800 EnCodec frames = 384,000 samples = 16 seconds
- Used extraction indices [96000:480000] everywhere

**CORRECT CALCULATION:**
- EnCodec at 24kHz: 320 samples per frame
- 800 frames × 320 samples/frame = **256,000 samples = 10.67 seconds**

The error propagated through entire training pipeline!

---

## Impact

**Affected Operations:**
1. Training per-pair loss computation
2. Validation loss computation  
3. Correlation analysis
4. Spectral penalty computation
5. Sample saving for audio outputs

**All these compared:**
- Decoded output: 256,000 samples ✓
- Extracted segments: 384,000 samples ✗ **MISMATCH!**

---

## Solution

**Changed ALL extraction indices:**

### Before (WRONG):
```python
# Extracting 16 seconds (384,000 samples)
input_16sec = audio_inputs[:, :, 96000:480000]   # [B, 1, 384000]
target_16sec = audio_targets[:, :, 96000:480000] # [B, 1, 384000]

# But output is only 10.67 seconds!
output_audio = output_decoded_i.squeeze(1)  # [B, 256000]
```

### After (CORRECT):
```python
# Extracting 10.67 seconds (256,000 samples)
input_10sec = audio_inputs[:, :, 160000:416000]   # [B, 1, 256000]
target_10sec = audio_targets[:, :, 160000:416000] # [B, 1, 256000]

# Now matches output!
output_audio = output_decoded_i.squeeze(1)  # [B, 256000]
```

**Calculation:**
- 24-second input: 576,000 samples
- Agent output: 256,000 samples  
- Center offset: (576,000 - 256,000) / 2 = 160,000
- Extraction: [160000:416000]

---

## Fixes Applied

### train_hybrid_worker.py (6 locations)

1. **Training per-pair loss** (lines ~209-220)
   - Changed: `input_16sec/target_16sec` → `input_10sec/target_10sec`
   - Changed: `[96000:480000]` → `[160000:416000]`

2. **Loss computation call** (line ~213)
   - Changed: `combined_loss(output_audio, input_16sec, target_16sec, ...)`
   - To: `combined_loss(output_audio, input_10sec, target_10sec, ...)`

3. **Correlation/GAN section** (lines ~250-265)
   - Updated spectral penalty extraction
   - Updated correlation analysis extraction

4. **Validation per-pair loss** (lines ~430-440)
   - Same extraction correction as training

5. **Validation sample saving** (lines ~465-470)
   - Changed: `audio_inputs[0:1, :, 96000:480000]`
   - To: `audio_inputs[0:1, :, 160000:416000]`

6. **Comments and variable names**
   - Updated all "16 sec" → "10.67 sec"
   - Updated all "384000" → "256000"

---

## Verification

Created comprehensive test: `test_tensor_sizes.py`

**Test validates:**
```
✓ 800 frames → 256,000 samples  
✓ Duration: 10.67 seconds at 24kHz
✓ Center extraction: [160000:416000]
✓ All tensor shapes match: [B, 1, 256000]
✓ Squeezed shapes match: [B, 256000]
✓ Loss computation succeeds with no size errors
```

**Test output:**
```
================================================================================
✅ ALL TESTS PASSED!
================================================================================

CONFIGURATION FOR CODE:
  Agent outputs: 800 frames
  Decoded size: 256000 samples (10.67 sec)
  Extraction: audio[:, :, 160000:416000]
  All tensor sizes: [2, 1, 256000] → squeeze → [2, 256000]
================================================================================
```

---

## Next Steps

1. ✅ All fixes committed and pushed
2. 🔄 Pull on HPC and restart training
3. 📊 Verify training progresses without size errors
4. 📈 Monitor loss values and metrics

**Expected result:** Training should now progress normally without RuntimeError!

---

## Key Learnings

1. **Always validate EnCodec frame calculations**
   - Don't assume frame counts without verification
   - 800 frames ≠ 16 seconds at 24kHz

2. **Test tensor sizes early**
   - Would have caught this before deployment
   - Comprehensive tests prevent propagation

3. **Document assumptions**
   - "16 seconds" was an assumption, not verified
   - Led to systematic error throughout codebase

---

## Files Modified

- `train_hybrid_worker.py` - 6 extraction index corrections
- `test_tensor_sizes.py` - NEW comprehensive size verification test

**No other files affected** - error was contained to hybrid training script.
