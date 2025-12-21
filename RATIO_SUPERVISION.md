# Ratio Supervision Implementation

**Date:** Dec 21, 2025  
**Status:** ✅ IMPLEMENTED  
**Branch:** adaptive-window-selection  
**Commit:** bb915ed

---

## Problem Identified

**Your concern was absolutely correct!**

The `AdaptiveWindowCreativeAgent` had a fundamental flaw:
- `WindowSelector` predicted compression ratios (`ratio_input`, `ratio_target`)
- `TemporalCompressor` **did apply** these ratios (via interpolation)
- BUT: Loss only compared against **already-compressed** segments
- **No gradient signal** to learn meaningful compression patterns

**Result**: Ratios could collapse to meaningless values (all 1.0, or random variations)

---

## Root Cause

```python
# Before supervision:
# 1. Agent compresses windows with predicted ratios
window_compressed = temporal_compressor(window_raw, ratio)  # ✓ Uses ratio

# 2. Loss compares to COMPRESSED segments only
loss = F.mse_loss(output, compressed_input)  # ✗ No original reference!
```

**Missing**: Comparison to the **original uncompressed** ground truth!

---

## Solution: Two Supervision Mechanisms

### 1. Ratio Diversity Loss (Weight: 0.1)

**Purpose**: Prevent collapse to identical ratios

```python
# Stack ratios across 3 pairs: [3, B]
ratios_input_stacked = torch.stack([p['ratio_input'] for p in window_params])
ratios_target_stacked = torch.stack([p['ratio_target'] for p in window_params])

# Compute variance across pairs (higher = more diverse)
variance_input = torch.var(ratios_input_stacked, dim=0).mean()
variance_target = torch.var(ratios_target_stacked, dim=0).mean()

# Negative loss = encourage higher variance
diversity_loss = -(variance_input + variance_target)
loss += 0.1 * diversity_loss
```

**Effect**: Forces the 3 pairs to learn **different** compression strategies

---

### 2. Reconstruction Loss (Weight: 0.05)

**Purpose**: Keep output close to original uncompressed content

```python
for pair_idx, params in enumerate(window_params):
    # Extract ORIGINAL window (before compression)
    # Uses same start positions predicted by WindowSelector
    original_window = extract_window(
        encoded_24sec,
        start_position=params['start_input'],
        duration=800  # No compression!
    )
    
    # Decode both
    output_audio = decode(agent_output[pair_idx])
    original_audio = decode(original_window)
    
    # Compare to ground truth
    recon_loss = F.mse_loss(output_audio, original_audio)
    loss += 0.05 * recon_loss
```

**Effect**: Ensures compressed/transformed output still resembles the original content

---

## What This Achieves

### Before (No Supervision)
- ❌ Ratios could be: `[1.0, 1.0, 1.0]` (no compression)
- ❌ Or random: `[0.98, 1.02, 0.99]` (meaningless)
- ❌ No incentive to learn useful patterns

### After (With Supervision)
- ✅ **Diversity Loss** pushes: `[0.8, 1.0, 1.2]` (varied strategies)
- ✅ **Reconstruction Loss** ensures: output ≈ original content
- ✅ Gradient signal: "use compression when it helps, don't when it doesn't"

---

## Implementation Details

### 1. adaptive_window_agent.py
```python
# Now returns raw window_params in metadata
metadata = {
    'num_pairs': 3,
    'pairs': [...],  # For logging
    'window_params': window_params  # ← NEW: For supervision
}
```

### 2. train_hybrid_worker.py
Added two loss components:

```python
# Ratio diversity: encourage different ratios
ratio_diversity_loss = -(variance_input + variance_target)
loss += 0.1 * ratio_diversity_loss

# Reconstruction: compare to original
mean_reconstruction_loss = stack(recon_losses).mean()
loss += 0.05 * mean_reconstruction_loss
```

### 3. training/debug_utils.py
Updated `MetricsAccumulator` to track:
- `total_ratio_diversity`
- `total_reconstruction`

Updated `print_epoch_summary` to display:
```
Ratio Supervision:
  Ratio Diversity:   -0.006124
  Reconstruction:    0.002345
```

---

## Test Results

Created `test_ratio_supervision.py`:

```
✓ Agent forward pass successful
✓ 'window_params' in metadata: True
✓ Number of window params: 3

First pair params:
  start_input shape: torch.Size([2])
  ratio_input shape: torch.Size([2])
  ratio_target shape: torch.Size([2])

Ratio diversity test:
  Input variance: 0.005391
  Target variance: 0.000733
  Diversity loss: -0.006124

✓ All ratio supervision tests passed!
```

---

## Expected Training Behavior

### Early Training
- Ratios start similar: `[1.01, 0.99, 1.00]`
- Diversity loss high (negative large): `-0.001`
- Reconstruction loss moderate: `0.05`

### Mid Training
- Ratios diversify: `[0.95, 1.00, 1.05]`
- Diversity loss lower: `-0.01`
- Reconstruction loss decreases: `0.02`

### Late Training
- Ratios specialized: `[0.8, 1.0, 1.2]`
  - Pair 0: Heavy compression (fast playback)
  - Pair 1: No compression (original timing)
  - Pair 2: Stretching (slow playback)
- Diversity loss stabilizes: `-0.05`
- Reconstruction loss low: `0.005`

---

## Weight Tuning

**Current weights:**
- Diversity: `0.1` - Moderate encouragement
- Reconstruction: `0.05` - Light constraint

**If ratios don't diversify enough:**
- Increase diversity weight: `0.2` or `0.5`

**If output strays too far from original:**
- Increase reconstruction weight: `0.1` or `0.2`

**If ratios over-diversify (too extreme):**
- Decrease diversity weight: `0.05`

---

## Next Steps

1. ✅ All changes committed and pushed
2. 🔄 Pull on HPC: `git pull origin adaptive-window-selection`
3. 🚀 Train and monitor new metrics:
   - Watch ratio diversity converge
   - Verify reconstruction stays low
   - Check if ratios specialize (3 different strategies)
4. 📊 If needed, tune weights based on training behavior

---

## Key Takeaway

**Your intuition was spot-on!** The model was learning to match compressed patterns but had no incentive to learn meaningful compression. Now it has proper supervision through:

1. **Diversity**: "Learn different strategies"
2. **Reconstruction**: "Stay true to the original"

This creates the gradient signal needed for the WindowSelector to discover when compression/stretching actually helps!
