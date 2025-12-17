# Complementary Masking for Style Transfer

## Overview

Complementary masking makes input and target **non-overlapping** instead of fully blended. This forces the model to create **musical arrangements** where patterns from Song 1 are logically injected into Song 2, rather than simple overlapping/blending.

## Masking Strategies

### 1. **Temporal Masking** (Recommended for Music)
```bash
--mask_type temporal --mask_temporal_segment 150
```

**What it does:**
- Input is active on beats 1, 3, 5... (odd segments)
- Target is active on beats 2, 4, 6... (even segments)
- Creates rhythmic complementarity

**Best for:**
- Creating call-and-response patterns
- Maintaining musical structure
- Rhythmic fusion

**Parameters:**
- `--mask_temporal_segment`: Frames per segment (~150 = 1 second)
  - Smaller values (75): Faster alternation (half-bar)
  - Larger values (300): Slower alternation (2-bar phrases)

**Example:**
```
Input:  ████____████____████____  (beats 1,3,5)
Target: ____████____████____████  (beats 2,4,6)
Result: Smooth transitions between sources
```

---

### 2. **Frequency Masking**
```bash
--mask_type frequency --mask_freq_split 0.3
```

**What it does:**
- Input keeps **low frequencies** (bass, rhythm, kick drum)
- Target keeps **high frequencies** (melody, vocals, hi-hats)
- Creates harmonic complementarity

**Best for:**
- Bass from one song + melody from another
- Preserving rhythmic foundation from input
- Clean frequency separation

**Parameters:**
- `--mask_freq_split`: Split point (0.3 = low 30% vs high 70%)
  - 0.2: Very narrow bass range
  - 0.5: Equal split (mid-point)

**Example:**
```
Frequency
High ▲   Target ████████████████  (melody, vocals)
     │
Low  ▼   Input  ████████████████  (bass, drums)
         Time ──────────────────>
```

---

### 3. **Spectral Dropout**
```bash
--mask_type spectral --mask_channel_keep 0.5
```

**What it does:**
- Input gets random 50% of encoding channels
- Target gets the other 50%
- Forces complementary feature learning

**Best for:**
- Experimental fusion
- Learning diverse features
- Preventing simple copying

**Parameters:**
- `--mask_channel_keep`: Ratio per source (0.5 = 50/50 split)

**Example:**
```
Channels: [0, 1, 2, 3, 4, 5, 6, 7]
Input:    [✓, ✗, ✓, ✗, ✓, ✗, ✓, ✗]  Random selection
Target:   [✗, ✓, ✗, ✓, ✗, ✓, ✗, ✓]  Complementary set
```

---

### 4. **Energy Masking**
```bash
--mask_type energy --mask_energy_threshold 0.7
```

**What it does:**
- Input keeps **high-energy frames** (transients, attacks, hits)
- Target keeps **low-energy frames** (sustain, holds, ambience)
- Creates dynamic complementarity

**Best for:**
- Attack from one song + sustain from another
- Percussive elements vs sustained notes
- Dynamic contrast

**Parameters:**
- `--mask_energy_threshold`: Energy percentile (0.7 = 70th percentile)
  - Lower (0.5): More balanced split
  - Higher (0.9): Only strongest attacks

**Example:**
```
Energy
High ▲   Input  █_█_█__█_█_█__  (drum hits, attacks)
     │
Low  ▼   Target ___████___████  (sustained chords)
         Time ──────────────────>
```

---

### 5. **Hybrid Masking** (Advanced)
```bash
--mask_type hybrid --mask_temporal_segment 150 --mask_freq_split 0.3
```

**What it does:**
- Combines **temporal** and **frequency** masking
- Input: Low frequencies on odd beats
- Target: High frequencies on even beats
- Double complementarity

**Best for:**
- Complex arrangements
- Maximum separation
- Professional-sounding fusion

**Example:**
```
        Time Segment 1  | Time Segment 2  | Time Segment 3
High ▲  Target (silent) | Target (melody) | Target (silent)
     │
Low  ▼  Input  (bass)   | Input (silent)  | Input  (bass)
```

---

## Usage Examples

### Recommended Settings for Different Tasks

#### **Style Transfer (Shuffle Targets)**
Inject rhythm from Song A into Song B:
```bash
--shuffle_targets true \
--mask_type temporal \
--mask_temporal_segment 150 \
--loss_weight_input 1.0 \
--loss_weight_target 1.0 \
--anti_cheating 0.3
```

#### **Bass + Melody Fusion**
Take bass from input, melody from target:
```bash
--shuffle_targets true \
--mask_type frequency \
--mask_freq_split 0.3 \
--loss_weight_input 1.0 \
--loss_weight_target 1.0
```

#### **Attack + Sustain Combination**
Percussive hits from input, sustained notes from target:
```bash
--shuffle_targets true \
--mask_type energy \
--mask_energy_threshold 0.7 \
--loss_weight_input 1.0 \
--loss_weight_target 1.0
```

#### **Complex Arrangement**
Professional-level fusion with all techniques:
```bash
--shuffle_targets true \
--mask_type hybrid \
--mask_temporal_segment 100 \
--mask_freq_split 0.25 \
--loss_weight_input 1.0 \
--loss_weight_target 1.0 \
--anti_cheating 0.5
```

#### **No Masking (Baseline)**
Simple blending (original behavior):
```bash
--mask_type none \
--loss_weight_input 1.0 \
--loss_weight_target 1.0
```

---

## Parameter Tuning Guide

### Temporal Segment Length
- **75 frames** (~0.5s): Very fast alternation, choppy
- **150 frames** (~1s): Musical bar-level, recommended
- **300 frames** (~2s): Phrase-level, smoother
- **600 frames** (~4s): Section-level, very smooth

### Frequency Split Ratio
- **0.2**: Narrow bass (< 200 Hz) vs wide treble
- **0.3**: Standard bass/mid split (recommended)
- **0.5**: Equal low/high split
- **0.7**: Wide bass/mid vs narrow treble

### Energy Threshold
- **0.5**: Balanced split (median energy)
- **0.7**: Only moderate-to-strong transients (recommended)
- **0.9**: Only strongest hits (sparse)

---

## Training Tips

1. **Start with Temporal Masking**
   - Most musical and interpretable
   - Easy to hear what's happening
   - Good baseline for comparison

2. **Use Shuffle Targets**
   - Always use `--shuffle_targets true` with masking
   - Creates random pairings for style transfer
   - Validation stays matched for quality checks

3. **Balanced Loss Weights**
   - Use `--loss_weight_input 1.0 --loss_weight_target 1.0`
   - Model learns to respect both sources equally
   - Don't use unity test with masking

4. **Anti-Cheating Noise**
   - Use `--anti_cheating 0.3-0.5` with masking
   - Prevents model from copying masked regions
   - Forces creative pattern learning

5. **Experiment with Combinations**
   - Try different mask types on same checkpoint
   - Compare temporal vs frequency vs hybrid
   - Listen to outputs to judge quality

---

## Debugging

### Check Masking is Active
Training logs should show:
```
🎭 Complementary Masking Applied:
  Type: temporal
  Description: Alternating time segments (Input: beats 1,3,5... | Target: beats 2,4,6...)
  Segment length: 150 frames (~1.0s)
```

### Verify Complementarity
- Input and target RMS should be **similar** (~0.10-0.15)
- Not severely imbalanced like before
- Both losses decreasing together

### Audio Quality
- Output should sound like **arrangement**, not **blend**
- Clear contributions from both sources
- Musical transitions, not muddy overlap

---

## Implementation Details

**Location:** `complementary_masking.py`

**Function:** `apply_complementary_mask(encoded_input, encoded_target, mask_type, ...)`

**Returns:** `masked_input, masked_target` (complementary, non-overlapping)

**Applied:** During training, after encoding target (cascade mode only)

**Validation:** No masking during validation (uses full inputs for quality check)

---

## Comparison: Before vs After

### Before (No Masking)
```
Input:  ████████████████████  Full song A
Target: ████████████████████  Full song B
Output: ████████████████████  Muddy blend of A+B
```

### After (Temporal Masking)
```
Input:  ████____████____████  Song A on odd beats
Target: ____████____████____  Song B on even beats
Output: ████████████████████  Arrangement: A's patterns in B's context
```

Result: Clean fusion with distinct contributions from both sources!

---

## Quick Reference

| Mask Type  | Input Source         | Target Source        | Best For              |
|------------|---------------------|----------------------|-----------------------|
| `none`     | Full overlap        | Full overlap         | Baseline/blending     |
| `temporal` | Odd time segments   | Even time segments   | Rhythm/structure      |
| `frequency`| Low frequencies     | High frequencies     | Bass + melody         |
| `spectral` | Random 50% channels | Other 50% channels   | Feature diversity     |
| `energy`   | High energy frames  | Low energy frames    | Attack + sustain      |
| `hybrid`   | Low freq, odd beats | High freq, even beats| Professional fusion   |

---

## Example Training Command

```bash
python train_simple_ddp.py \
    --dataset_folder dataset_pairs_wav \
    --num_transformer_layers 3 \
    --shuffle_targets true \
    --mask_type temporal \
    --mask_temporal_segment 150 \
    --loss_weight_input 1.0 \
    --loss_weight_target 1.0 \
    --anti_cheating 0.3 \
    --epochs 200 \
    --lr 1e-3
```

This creates a 3-stage cascade that learns to arrange Song A's rhythmic patterns into Song B's musical context, with alternating 1-second segments!
