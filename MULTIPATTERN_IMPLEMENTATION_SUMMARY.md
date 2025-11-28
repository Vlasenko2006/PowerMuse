# Multi-Pattern Fusion Implementation Summary

**Date:** November 28, 2025  
**Author:** Andrey Vlasenko  
**Project:** NeuroNMusE - PowerMuse Multi-Pattern Fusion

---

## Overview

Successfully implemented complete multi-pattern fusion architecture that processes **3 input songs** and generates **1 fused output** using transformer-based fusion with random masking.

---

## Architecture

### Input/Output Format
- **Input:** 3 stereo audio patterns (16s each @ 22050 Hz) → `[batch, 3, 2, 352800]`
- **Output:** 1 fused stereo pattern (16s @ 22050 Hz) → `[batch, 2, 352800]`
- **Masking:** Random segments per pattern (training), fixed masks (validation)

### Model Components

1. **Encoder-Decoder** (`encoder_decoder.py`)
   - Shared across all 3 patterns
   - Conv1D with BatchNorm1d + ReLU
   - Processes each pattern independently

2. **Transformer Fusion** (`model_multipattern.py`)
   - 4 transformer layers, 8 attention heads
   - Processes encoded patterns together
   - GELU activation, 0.15 dropout
   - Fusion layer: Linear(192→64) to combine patterns

3. **Loss Function** (`fusion_loss.py`)
   - Chunk-wise MSE with min() selection
   - 50% overlapping 4-second chunks
   - Finds best alignment across temporal shifts
   - Reconstruction loss for encoder-decoder training

4. **Masking System** (`masking_utils.py`)
   - Random masking: 3-8 segments per pattern (1-5s each)
   - Fixed validation masks: consistent evaluation
   - Boolean masks: True=keep, False=mask out

---

## 3-Phase Training Strategy

### Phase 1: Unmasked Encoder-Decoder (Epochs 1-10)
- **Goal:** Train encoder-decoder to reconstruct patterns
- **Forward:** 3 patterns → encode → decode → 3 reconstructions
- **Loss:** `reconstruction_only_loss` (no masking)
- **Frozen:** transformer, fusion_layer
- **Active:** encoder-decoder

### Phase 2: Masked Encoder-Decoder (Epochs 11-20)
- **Goal:** Train encoder-decoder to handle incomplete patterns
- **Forward:** 3 masked patterns → encode → decode → 3 reconstructions
- **Loss:** `reconstruction_only_loss` (random masks)
- **Frozen:** transformer, fusion_layer
- **Active:** encoder-decoder

### Phase 3: Full Model with Fusion (Epochs 21+)
- **Goal:** Train transformer to fuse patterns optimally
- **Forward:** 3 masked patterns → encode → transformer fusion → 1 fused output
- **Loss:** `multi_pattern_loss` (chunk-wise MSE)
- **Frozen:** none
- **Active:** encoder-decoder, transformer, fusion_layer

---

## Implementation Files

| File | Size | Lines | Purpose |
|------|------|-------|---------|
| `masking_utils.py` | 6.3K | 182 | Random/fixed masking, chunk splitting |
| `fusion_loss.py` | 5.6K | 162 | Chunk-wise MSE loss with min() |
| `model_multipattern.py` | 6.8K | 160 | Multi-pattern model architecture |
| `create_dataset_multipattern.py` | 7.3K | 198 | Triplet dataset generation |
| `train_and_validate_multipattern.py` | 9.8K | 251 | 3-phase training loop |
| `main_multipattern.py` | 7.8K | 237 | Main training script |
| **Total** | **43.6K** | **846** | Complete multi-pattern system |

---

## Key Features

### Dataset Generation
```python
# Samples 3 different songs for each triplet
# Format: ((input1, input2, input3), (target1, target2, target3))
# Each pattern: [2, 352800] stereo at 22050 Hz
# Consecutive chunks: input_i → target_i from same song
```

### Training Configuration
- **Batch size:** 4 (reduced for 3x memory per sample)
- **Accumulation steps:** 16 (effective batch size = 64)
- **Learning rate:** 0.0001 with ReduceLROnPlateau scheduler
- **Epochs:** 50 (10 per phase 1&2, 30 for phase 3)
- **GPU memory:** ~3x higher than single-pattern

### Loss Function Logic
```python
# For each target pattern:
# 1. Split into 4-second overlapping chunks (50% overlap)
# 2. Compute MSE for each chunk against output
# 3. Take minimum MSE across all chunks
# 4. Average across all 3 patterns

total_loss = mean([min(chunk_mses_pattern1), 
                   min(chunk_mses_pattern2), 
                   min(chunk_mses_pattern3)])
```

---

## Usage Instructions

### 1. Generate Multi-Pattern Dataset
```bash
cd /Users/andreyvlasenko/tst/Jingle
python create_dataset_multipattern.py
```
**Output:**
- `dataset_multipattern/training_set_multipattern.npy`
- `dataset_multipattern/validation_set_multipattern.npy`

### 2. Train Multi-Pattern Model
```bash
python main_multipattern.py
```
**Outputs:**
- Checkpoints: `checkpoints_multipattern/model_epoch_*.pt`
- Best model: `checkpoints_multipattern/model_best.pt`
- Music samples: `music_out_multipattern/`

### 3. Resume Training (if needed)
```python
# In main_multipattern.py, set:
resume_from_checkpoint = "checkpoints_multipattern/model_epoch_20.pt"
```

---

## Memory Optimization

Multi-pattern training requires **~3x memory** per sample:
- Single-pattern: `[batch, 2, 352800]` → ~2.8 MB per sample
- Multi-pattern: `[batch, 3, 2, 352800]` → ~8.4 MB per sample

**Optimizations applied:**
- Reduced batch_size: 8 → 4
- Increased accumulation_steps: 8 → 16
- Maintains effective batch size of 64
- Uses gradient accumulation for memory efficiency

---

## Validation Strategy

### Fixed Validation Masks
- Generated once before training
- Reused across all validation epochs
- Ensures consistent evaluation
- 3-8 segments per pattern (1-5s each)

### Metrics Tracked
- Total loss (weighted combination)
- Per-phase loss type:
  - Phases 1-2: Reconstruction loss only
  - Phase 3: Multi-pattern fusion loss
- Best model saved based on validation loss

---

## Next Steps

1. **Generate Dataset:** Run `create_dataset_multipattern.py` on your music collection
2. **Start Training:** Execute `main_multipattern.py` with GPU
3. **Monitor Progress:** Check validation loss, best model updates
4. **Evaluate:** Listen to generated outputs in `music_out_multipattern/`
5. **Fine-tune:** Adjust phase durations, masking parameters if needed

---

## Technical Specifications

- **Sample Rate:** 22,050 Hz (high quality)
- **Chunk Duration:** 16 seconds
- **Samples per Chunk:** 352,800
- **Channels:** 2 (stereo)
- **Patterns:** 3 input → 1 fused output
- **Model Parameters:** ~1.5M trainable
- **Model Size:** ~6 MB (float32)

---

## Comparison: Single vs Multi-Pattern

| Aspect | Single-Pattern | Multi-Pattern |
|--------|---------------|---------------|
| **Input** | 1 song | 3 songs |
| **Output** | 1 prediction | 1 fused prediction |
| **Architecture** | Encoder-decoder + transformer | Same + fusion layer |
| **Training** | 2-phase | 3-phase |
| **Memory** | 2.8 MB/sample | 8.4 MB/sample |
| **Batch size** | 8 | 4 |
| **Loss** | MSE | Chunk-wise MSE with min() |
| **Masking** | None | Random per pattern |

---

## GitHub Repository

All changes tracked at: https://github.com/Vlasenko2006/PowerMuse

---

**Status:** ✅ All 6 tasks completed. Ready for training.

