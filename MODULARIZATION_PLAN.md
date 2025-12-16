# Code Modularization Plan

## Overview

Splitting large Python files into smaller, focused modules for better maintainability, testability, and code organization.

---

## File Size Analysis

### Current Large Files:
| File | Lines | Status |
|------|-------|--------|
| `train_simple_worker.py` | 1014 | ✅ **Partially split** |
| `compositional_creative_agent.py` | 638 | ✅ **Refactored** (could split further) |
| `inference_cascade.py` | 489 | ⏭️ Keep as-is (single purpose) |
| `creative_agent.py` | 445 | ⏭️ Keep as-is (legacy) |
| `model_simple_transformer.py` | 438 | ⏭️ Keep as-is (single model) |

---

## Implemented: Training Module Split

### ✅ Phase 1: Loss Functions Extracted

**Created**: `training/` package

```
training/
├── __init__.py          # Package exports
└── losses.py            # Loss functions (180 lines)
```

**Extracted from `train_simple_worker.py`**:
- `rms_loss()` - RMS reconstruction loss
- `stft_loss()` - Multi-resolution spectral loss
- `mel_loss()` - Mel-spectrogram perceptual loss
- `combined_loss()` - Weighted combination

**Benefits**:
- ✅ Loss functions now independently testable
- ✅ Clear separation of concerns
- ✅ Reduced `train_simple_worker.py` from 1014 → ~880 lines (-13%)
- ✅ Can reuse losses in other scripts (inference, evaluation)

**Usage**:
```python
from training.losses import combined_loss, stft_loss, mel_loss

# Instead of defining losses inline
loss, rms_in, rms_tgt, spec, mel = combined_loss(
    output_audio, input_audio, target_audio,
    weight_input=0.0, weight_target=0.0,
    weight_spectral=0.01, weight_mel=0.0
)
```

---

## Proposed: Further Modularization

### 🔄 Phase 2: Complete Training Module (Future)

```
training/
├── __init__.py
├── losses.py            # ✅ DONE
├── trainer.py           # Training loop (train_epoch)
├── validator.py         # Validation loop (validate_epoch)
├── metrics.py           # Metric tracking and MLflow logging
└── utils.py             # Helper functions (gradient debugging, etc.)
```

**Would extract**:
- `train_epoch()` → `trainer.py` (~300 lines)
- `validate_epoch()` → `validator.py` (~100 lines)
- Metric tracking → `metrics.py` (~80 lines)
- Gradient debugging → `utils.py` (~50 lines)

**Result**: `train_simple_worker.py` would become ~350 lines (65% reduction)

---

### 🔄 Phase 3: Agent Module Split (Future)

```
agents/
├── __init__.py
├── compositional/
│   ├── __init__.py
│   ├── agent.py         # Main CompositionalCreativeAgent
│   ├── extractors.py    # ComponentExtractor classes
│   ├── composer.py      # ComponentComposer network
│   └── losses.py        # Novelty & correlation losses
└── masking/
    ├── __init__.py
    └── agent.py         # ComplementaryMaskingAgent
```

**Would split `compositional_creative_agent.py`**:
- Main agent class → `agent.py` (~120 lines)
- Extractors → `extractors.py` (~180 lines)
- Composer → `composer.py` (~100 lines)
- Losses (novelty, correlation) → `losses.py` (~180 lines)

**Result**: 638 lines → 4 files of ~120-180 lines each

---

## Benefits of Modularization

### 1. **Maintainability**
- Each file has single, clear responsibility
- Easier to locate and fix bugs
- Changes isolated to relevant modules

### 2. **Testability**
```python
# Can now unit test individual components
from training.losses import stft_loss
import torch

def test_stft_loss():
    pred = torch.randn(2, 16000)
    target = torch.randn(2, 16000)
    loss = stft_loss(pred, target)
    assert loss >= 0
    assert loss.shape == torch.Size([])
```

### 3. **Reusability**
```python
# Use losses in different contexts
from training.losses import mel_loss

# Inference evaluation
def evaluate_quality(model_output, reference):
    return mel_loss(model_output, reference)
```

### 4. **Clarity**
- Import statements show dependencies clearly
- File names describe purpose
- Easier onboarding for new developers

### 5. **Development Speed**
- Faster file navigation
- Reduced merge conflicts
- Parallel development possible

---

## Migration Strategy

### ✅ Completed:
1. Extract loss functions to `training/losses.py`
2. Update imports in `train_simple_worker.py`
3. Verify training still works (backward compatible)

### 📋 Next Steps (Optional):

**Phase 2**: Training module completion
```bash
# 1. Create trainer.py
mv train_epoch() → training/trainer.py

# 2. Create validator.py  
mv validate_epoch() → training/validator.py

# 3. Create metrics.py
mv metric tracking code → training/metrics.py

# 4. Update train_simple_worker.py imports
from training.trainer import train_epoch
from training.validator import validate_epoch
```

**Phase 3**: Agent module split
```bash
# 1. Create agents package
mkdir -p agents/compositional

# 2. Split compositional_creative_agent.py
# Extract classes to separate files

# 3. Update imports in model_simple_transformer.py
from agents.compositional import CompositionalCreativeAgent
```

---

## Testing After Split

### Unit Tests
```python
# tests/test_losses.py
import pytest
from training.losses import rms_loss, stft_loss, mel_loss

def test_rms_loss_zero():
    """RMS loss should be zero for identical signals"""
    x = torch.randn(2, 16000)
    assert rms_loss(x, x) < 1e-6

def test_stft_loss_positive():
    """STFT loss should be positive for different signals"""
    x = torch.randn(2, 16000)
    y = torch.randn(2, 16000)
    assert stft_loss(x, y) > 0
```

### Integration Test
```bash
# Verify training still works
python train_simple_ddp.py \
    --dataset_folder dataset_pairs_wav \
    --epochs 1 \
    --batch_size 4
```

---

## Backward Compatibility

✅ **All changes are backward compatible**:
- Existing scripts work without modification
- Old imports still work (if needed, add compatibility layer)
- Checkpoints load correctly
- No changes to training behavior

---

## File Size Targets

| Module | Current | Target | Reduction |
|--------|---------|--------|-----------|
| `train_simple_worker.py` | 1014 | 350 | -65% |
| `compositional_creative_agent.py` | 638 | 4×150 | Modular |
| Loss functions | (inline) | 180 | Reusable |

**Goal**: No file >400 lines (except legacy/generated code)

---

## Summary

### ✅ Completed (Phase 1):
- Created `training/` package
- Extracted all loss functions to `training/losses.py`
- Updated imports in `train_simple_worker.py`
- Reduced main file by 13% (134 lines)

### 🎯 Impact:
- **Maintainability**: ⬆️ Significantly improved
- **Testability**: ⬆️ Loss functions now unit testable
- **Readability**: ⬆️ Clearer code organization
- **Training**: ✅ Fully backward compatible

### 📈 Next Recommended:
If further modularization is desired:
1. Split `train_epoch()` to `training/trainer.py`
2. Split `validate_epoch()` to `training/validator.py`
3. Split compositional agent into `agents/compositional/` package

This would achieve **65% reduction** in largest file size and create highly maintainable codebase.
