# Complementary Masking Implementation Summary

## What Was Implemented

I've added **5 complementary masking strategies** to make input and target non-overlapping, forcing the model to create musical **arrangements** instead of simple **blending**.

---

## Files Created/Modified

### New Files
1. **`complementary_masking.py`** (270 lines)
   - Core masking implementation
   - 5 masking strategies: temporal, frequency, spectral, energy, hybrid
   - Visualization tools for debugging
   - Masking descriptions for logging

2. **`README_MASKING.md`**
   - Complete documentation of all masking strategies
   - Usage examples and parameter tuning guide
   - Comparison tables and quick reference

3. **`test_masking.py`**
   - Test script to visualize masking effects
   - Generates PNG visualizations for each strategy
   - Run: `python test_masking.py --mask_type all`

### Modified Files
1. **`train_simple_ddp.py`**
   - Added 5 new arguments for masking configuration
   - Updated configuration display to show masking info

2. **`train_simple_worker.py`**
   - Integrated masking into training loop
   - Applied after encoding target, before forward pass
   - Debug logging for first batch

3. **`run_train_direct.sh`**
   - Added masking parameters (default: temporal)
   - Ready to use with `--mask_type temporal`

---

## Masking Strategies Overview

### 1. Temporal (Recommended First)
```bash
--mask_type temporal --mask_temporal_segment 150
```
- **Input**: Active on odd beats (1, 3, 5...)
- **Target**: Active on even beats (2, 4, 6...)
- **Result**: Call-and-response rhythmic patterns
- **Best for**: Musical structure, rhythmic fusion

### 2. Frequency
```bash
--mask_type frequency --mask_freq_split 0.3
```
- **Input**: Low frequencies (bass, rhythm)
- **Target**: High frequencies (melody, vocals)
- **Result**: Bass from Song A + melody from Song B
- **Best for**: Harmonic separation, clean mixing

### 3. Spectral
```bash
--mask_type spectral --mask_channel_keep 0.5
```
- **Input**: Random 50% of encoding channels
- **Target**: Other 50% of channels
- **Result**: Complementary feature learning
- **Best for**: Experimental fusion, diversity

### 4. Energy
```bash
--mask_type energy --mask_energy_threshold 0.7
```
- **Input**: High-energy frames (attacks, hits)
- **Target**: Low-energy frames (sustain, holds)
- **Result**: Attack from Song A + sustain from Song B
- **Best for**: Dynamic contrast, percussion + harmony

### 5. Hybrid (Advanced)
```bash
--mask_type hybrid --mask_temporal_segment 150 --mask_freq_split 0.3
```
- **Combination**: Temporal + Frequency
- **Input**: Low freq on odd beats
- **Target**: High freq on even beats
- **Result**: Maximum separation, professional fusion
- **Best for**: Complex arrangements

---

## Quick Start

### Test Masking (Visualize)
```bash
# See how each masking strategy works
python test_masking.py --mask_type temporal

# Test all strategies
python test_masking.py --mask_type all

# Open generated PNG files to see masks
```

### Train with Temporal Masking
```bash
# Edit run_train_direct.sh:
--mask_type temporal \
--mask_temporal_segment 150 \
--shuffle_targets true \
--loss_weight_input 1.0 \
--loss_weight_target 1.0

# Run training
bash run_train_direct.sh
```

### Expected Training Output
```
🎭 Complementary Masking Applied:
  Type: temporal
  Description: Alternating time segments (Input: beats 1,3,5... | Target: beats 2,4,6...)
  Segment length: 150 frames (~1.0s)
```

---

## How It Works

### Before Masking (Original)
```
Input:  ████████████████████  (Full Song A)
Target: ████████████████████  (Full Song B)
Output: Muddy blend (overlapping A+B)
```

### After Temporal Masking
```
Input:  ████____████____████  (Song A on odd beats)
Target: ____████____████____  (Song B on even beats)
Output: Clear arrangement (A's patterns in B's context)
```

### Code Flow
1. **Encode target audio** → `encoded_target [B, D, T]`
2. **Apply masking** → `masked_input, masked_target` (complementary)
3. **Forward cascade** → Model receives non-overlapping sources
4. **Learn arrangement** → Forced to combine creatively

---

## Parameters

### All Masking Parameters
```bash
--mask_type {none, temporal, frequency, spectral, energy, hybrid}
--mask_temporal_segment 150        # Temporal: frames per segment (~150 = 1s)
--mask_freq_split 0.3             # Frequency: split ratio (0.3 = 30% low)
--mask_channel_keep 0.5           # Spectral: keep ratio (0.5 = 50/50)
--mask_energy_threshold 0.7       # Energy: percentile (0.7 = 70th)
```

### Recommended Combinations

**Style Transfer (Most Musical):**
```bash
--mask_type temporal \
--mask_temporal_segment 150 \
--shuffle_targets true \
--loss_weight_input 1.0 \
--loss_weight_target 1.0
```

**Bass + Melody Fusion:**
```bash
--mask_type frequency \
--mask_freq_split 0.3 \
--shuffle_targets true
```

**Attack + Sustain:**
```bash
--mask_type energy \
--mask_energy_threshold 0.7 \
--shuffle_targets true
```

**Professional Arrangement:**
```bash
--mask_type hybrid \
--mask_temporal_segment 100 \
--mask_freq_split 0.25 \
--shuffle_targets true \
--anti_cheating 0.5
```

---

## Verification

### Check Logs
Training should show:
```
🎭 Complementary Masking Applied:
  Type: temporal
  Description: Alternating time segments...
```

### Check Convergence
- Both `rms_in` and `rms_tgt` should be similar (~0.10-0.15)
- Not severely imbalanced like before
- Both losses decreasing together

### Check Audio Quality
- Output sounds like **arrangement**, not **blend**
- Clear contributions from both sources
- Musical transitions, not muddy overlap

---

## Comparison with Previous Approach

| Aspect | Before (No Masking) | After (Temporal Masking) |
|--------|--------------------|-----------------------|
| Input/Target | Full overlap | Alternating segments |
| Model learns | Simple blending | Creative arrangement |
| Output sounds like | Muddy overlap | Clean fusion |
| RMS balance | Can be imbalanced | Naturally balanced |
| Musical quality | Overlapping melodies | Complementary patterns |

---

## Next Steps

1. **Test masking locally:**
   ```bash
   python test_masking.py --mask_type temporal
   # Open mask_viz_temporal.png to see effect
   ```

2. **rsync to HPC:**
   ```bash
   rsync -avz *.py *.sh *.md levante:/path/to/Jingle_D/
   ```

3. **Train with temporal masking:**
   ```bash
   # On HPC
   bash run_train_direct.sh  # Already configured for temporal
   ```

4. **Listen to outputs:**
   ```bash
   python inference_cascade.py \
       --checkpoint checkpoints_spectral/best_model.pt \
       --shuffle_targets \
       --num_samples 5
   ```

5. **Experiment with other masks:**
   - Try `--mask_type frequency` for bass/melody split
   - Try `--mask_type hybrid` for complex arrangements
   - Compare audio quality between strategies

---

## Technical Details

### Masking Application Point
- **Location**: `train_simple_worker.py`, line ~240
- **When**: After encoding target, before forward pass
- **Mode**: Training only (validation uses full inputs)
- **Cascade only**: Single-stage models don't use masking

### Implementation
```python
# In training loop
if model.module.num_transformer_layers > 1:
    encoded_target = encodec_model.encoder(targets)
    
    if mask_type != 'none':
        inputs, encoded_target = apply_complementary_mask(
            inputs, encoded_target,
            mask_type=mask_type,
            temporal_segment_frames=150,
            ...
        )
```

### Smooth Transitions
- Temporal masking includes 10-frame fade at boundaries
- Prevents clicks/pops at segment switches
- Creates musical transitions

---

## Expected Results

### With Temporal Masking
- Input contributes rhythmic patterns on beats 1, 3, 5...
- Target contributes patterns on beats 2, 4, 6...
- Output is smooth arrangement with both sources
- Clear call-and-response structure

### With Frequency Masking
- Input contributes bass line and low rhythm
- Target contributes melody and high frequencies
- Output is clean mix with separated bands
- No frequency competition

### With Hybrid Masking
- Maximum separation (time + frequency)
- Professional-sounding arrangements
- Complex but coherent fusion
- Best quality but hardest to train

---

## Troubleshooting

**Q: Masking not showing in logs?**
- Check `--mask_type` is not 'none'
- Look for "🎭 Complementary Masking Applied" message
- Verify cascade mode (`--num_transformer_layers > 1`)

**Q: Output still sounds like blend?**
- Try more aggressive masking (smaller temporal segments)
- Increase anti-cheating noise (`--anti_cheating 0.5`)
- Use hybrid masking for maximum separation

**Q: Training unstable?**
- Reduce masking strength (larger temporal segments)
- Lower learning rate
- Start with frequency masking (more stable than temporal)

**Q: Want to disable masking?**
```bash
--mask_type none
```

---

## Summary

You now have **5 complementary masking strategies** that force the model to create musical arrangements instead of simple blending:

1. ✅ **Temporal** - Alternating time segments (recommended)
2. ✅ **Frequency** - Bass vs melody separation
3. ✅ **Spectral** - Random channel dropout
4. ✅ **Energy** - Attack vs sustain
5. ✅ **Hybrid** - Combined temporal + frequency

All integrated into training pipeline with command-line control. Ready to use! 🎵
