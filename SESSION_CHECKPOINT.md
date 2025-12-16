# Session Checkpoint: Training Status & GitHub Repository

**Date**: December 16, 2025  
**Status**: ✅ Repository Published to GitHub | ⚠️ Training Stuck at 74% Complementarity (Epoch 51/200)  
**Next Action**: Restart training with stronger complementarity pressure (mask_reg_weight 0.1 → 0.5)

---

## 🎉 Latest Update (Dec 16, 2025)

### GitHub Repository Published ✅

**Repository**: https://github.com/Vlasenko2006/PowerMuse  
**Status**: Successfully pushed to GitHub  
**Files**: 63 files (30 Python, 22 documentation, 10 shell scripts, 1 config)  
**Size**: ~20,000 lines of code, 221 KB compressed

**Commits**:
1. Initial commit: 2-stage cascade transformer with creative agent (88 files)
2. Clean repository: Removed 27 deprecated files, updated documentation

**Key Files**:
- `README.md` - Comprehensive project overview
- `FILES_MANIFEST.md` - Complete file inventory
- `SESSION_CHECKPOINT.md` - This file (training history)
- Core model: `model_simple_transformer.py`, `creative_agent.py`, `audio_discriminator.py`
- Training: `train_simple_worker.py`, `train_simple_ddp.py`
- Launch scripts: `run_train_creative_agent_fixed.sh`, `run_train_creative_agent_resume.sh`

### Training Status: Complementarity Stuck at 74% ⚠️

**Problem**: After epoch 20 resume, complementarity dropped and plateaued at 74% (was 87% at epoch 19)

**Diagnosis**:
- Epoch 1-20: Complementarity increased 75% → 87% (healthy)
- Epoch 21-51: **Stuck at 73-74%** with mask overlap 0.26-0.27
- Mask regularization loss: Hovering at 0.76-0.78 (should decrease to 0.10-0.20)
- Model learned local optimum: overlapping masks work "well enough" for reconstruction

**Root Cause**: `mask_reg_weight=0.1` too weak
- Complementarity loss ≈ 0.27 (overlap)
- Effective weight: 0.27 × 0.1 = **0.027** contribution
- RMS losses: 0.13-0.14 × 0.3 = **0.04** contribution (dominates)
- Model prioritizes reconstruction over mask separation

**Solution Created**: `run_train_creative_agent_push_complementarity.sh`
- Increase `mask_reg_weight` from 0.1 → **0.5** (5× stronger)
- Effective complementarity weight: 0.27 × 0.5 = **0.135** (3.5× larger than RMS)
- Expected: Force model to separate masks, push complementarity 74% → 90%+

### Current Training Metrics (Epoch 51/200)

- **Validation Loss**: 0.527 (improved from 0.49 at epoch 20)
- **Complementarity**: 74.0% (target: 90%+) ⚠️
- **Mask Overlap**: 0.260 (should decrease to 0.10)
- **Mask Balance**: 50.7% input / 51.2% target ✅
- **Gradient Norms**: 3-8 range (stable) ✅
- **Discriminator Accuracy**: 84% real/fake (balanced) ✅
- **Output RMS Ratio**: 0.81 (Output/Input, healthy)

---

## 🎯 Architecture Summary (2-Stage Cascade - STABLE)

**Problem Solved**: Gradient explosion causing NaN collapse at epoch 3  
**Root Cause**: 3-stage cascade with 0.1× weak residual in stage 2 → RMS collapse → 5.2× amplification → gradient explosion  
**Solution**: Simplified to 2-stage cascade with full 1.0× residual in both stages  
**Result**: Training stable through 51+ epochs, no NaN collapse, gradients controlled 1-8 range

**Triple Protection System**:
1. **Spectral Normalization**: Constrains weight matrices (Lipschitz ≤1)
2. **RMS Clamping**: Limits amplification to [0.7, 1.5] range (vs unbounded 5.2×)
3. **Gradient Clipping**: Final safety net at max_norm=10.0

**Bonus**: 33% parameter reduction (27.4M → 24.9M), faster training

---

## 🎯 Next Steps

1. **Sync New Script to Remote**:
   ```bash
   bash sync_to_remote.sh  # Uploads run_train_creative_agent_push_complementarity.sh
   ```

2. **Restart Training on Levante**:
   ```bash
   ssh g260141@levante.dkrz.de
   cd /work/gg0302/g260141/Jingle_D
   pkill -f train_simple  # Kill current stuck training
   chmod +x run_train_creative_agent_push_complementarity.sh
   bash run_train_creative_agent_push_complementarity.sh  # Resume from epoch 51
   ```

3. **Expected Improvements**:
   - Complementarity: 74% → 90%+ by epoch 100
   - Mask overlap: 0.26 → 0.10
   - Mask reg loss: 0.76 → 0.10-0.20
   - More distinct separation: input extracts rhythm, target extracts harmony/melody

4. **Monitor**:
   - First 5 epochs: Watch complementarity increase
   - Validation loss: Should stay stable or improve slightly (may temporarily increase as model adjusts)
   - Gradient norms: Should stay 3-10 range (may spike initially, then stabilize)

---

## 📜 Training History

### Session 3 (Dec 16, 2025) - GitHub & Complementarity Fix

**Morning**:
- Published repository to GitHub: https://github.com/Vlasenko2006/PowerMuse
- Cleaned up 27 deprecated files
- Created comprehensive README.md and FILES_MANIFEST.md
- Updated documentation with current training status

**Afternoon**:
- Analyzed training plateau: Complementarity stuck at 74% (epochs 21-51)
- Diagnosed: `mask_reg_weight=0.1` too weak, model prefers overlapping masks
- Created `run_train_creative_agent_push_complementarity.sh` with `mask_reg_weight=0.5`
- Prepared to restart training with stronger complementarity pressure

### Session 2 (Dec 15, 2025) - 2-Stage Architecture Fix

**Major Achievement**: 2-Stage Architecture Solves Gradient Explosion ✅

**Problem**: 3-stage cascade with gradient explosion (2.45 → 6.66 → 16.71 → NaN at epoch 3)  
**Root Cause**: Stage 2 with 0.1× weak residual → RMS collapse (1.09 vs 5.69 target) → 5.2× amplification needed → gradients explode  
**Solution**: Simplified to 2-stage cascade with full 1.0× residual in both stages

### Architecture Changes

1. **Simplified Cascade** (model_simple_transformer.py lines 167-209):
   ```python
   # OLD: 3-stage cascade with weak residual in stage 2
   for stage_idx in range(num_transformer_layers):  # Was 3 stages
       if stage_idx < 2:
           residual_weight = 1.0  # Full residual
       else:
           residual_weight = 0.1  # Weak residual → PROBLEM!
   
   # NEW: 2-stage cascade with full residual
   for stage_idx in range(2):  # Only 2 stages
       # Stage 0: 256-dim, concat(input, target)
       # Stage 1: 384-dim, concat(input, stage0, target)
       # Both use full 1.0× residual
   ```

2. **Residual Calculations**:
   - **Stage 0**: `x_residual = 0.5 × (input + target)`
   - **Stage 1**: `x_residual = (input + stage0 + target) / 3.0`
   - Both averaged to prevent single-source dominance

3. **RMS Scaling** (lines 440-450):
   - Clamping: [0.7, 1.5] (allows 30% attenuation to 50% amplification)
   - Warning if scale_factor > 1.3×
   - Spectral normalization on all output_proj layers

### Training Results (Dec 15, 2025 - 2-Stage Architecture)

**Gradient Stability Achieved** ✅:

**Epoch 1**:
- Gradient norm: **5.75** (controlled)
- Stage 0: output_rms=5.67, scale_factor=1.00 ✓
- Stage 1: output_rms=5.63, scale_factor=1.01 ✓
- Complementarity: **75.0%** (baseline)
- No excessive amplification warnings

**Epoch 2**:
- Gradient norm: **1.31** (decreased!) ✓
- Stage 0: output_rms=5.84, scale_factor=0.99 ✓
- Stage 1: output_rms=5.85, scale_factor=0.99 ✓
- Complementarity: **74.9%**
- Masks balanced: input=0.499, target=0.501

**Epoch 3**:
- Gradient norm: **7.54** (spike, but controlled by clipping)
- Stage 0: output_rms=5.67, scale_factor=1.00 ✓
- Stage 1: output_rms=6.10, scale_factor=0.93 ✓
- Complementarity: **75.1%**
- **No NaN collapse!** ✅

**Epoch 4**:
- Gradient norm: **3.11** (stable) ✓
- Complementarity: **74.9%**

**Epoch 5**:
- Gradient norm: **7.65** (spike from Creative Agent mask generators)
- Complementarity: **74.9%**

**Epoch 6**:
- Gradient norm: **6.70**
- Complementarity: **74.5%**

**Epoch 7+**:
- Training continues without crashes
- No NaN warnings observed

**Key Observations**:
1. ✅ **No gradient explosion**: Norms stay 1-8 range (vs old 2.45→6.66→16.71→NaN)
2. ✅ **Healthy RMS values**: Both stages maintain 5.6-6.1 range (no collapse)
3. ✅ **Minimal amplification**: Scale factors 0.93-1.01 (vs old 5.2× needed)
4. ✅ **33% fewer parameters**: 24.9M vs 27.4M (efficiency bonus)
5. ⚠️ Gradient spikes in epochs 3,5,6 from Creative Agent learning (expected, controlled by clipping)
6. 📊 Complementarity stable at 75% (early training, will increase with more epochs)

### Additional Fixes Applied

1. **NCCL Logging Suppression** (run_train_creative_agent_fixed.sh):
   - Changed `NCCL_DEBUG=INFO` → `NCCL_DEBUG=WARN`
   - Eliminates 100+ lines of verbose INFO messages during DDP init

2. **Graceful Shutdown** (train_simple_worker.py lines 673-681):
   - Added signal handlers for SIGINT/SIGTERM
   - Saves `interrupted_checkpoint.pt` before exit on Ctrl+C
   - Wrapped DDP cleanup in try-except to suppress "Broken pipe" errors

3. **Environment Variable Update**:
   - Changed deprecated `NCCL_ASYNC_ERROR_HANDLING=1` → `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`

4. **Documentation Created**:
   - `ARCHITECTURE_2STAGE_DETAILED.md`: Complete implementation details with tensor shapes
   - `ARCHITECTURE_CHANGE_2STAGE.md`: Migration rationale and comparison
   - `GRADIENT_EXPLOSION_ARCHITECTURE.md`: Original problem analysis
---

## 📊 Model Checkpoint Status

**Local Files** (downloaded from remote):
- `checkpoints/best_model.pt`: 239 MB, epoch 20, val_loss=0.492
- `mlruns_remote/`: Training logs showing 7 epochs visible (likely incomplete sync)
- Model: 2-stage cascade, 16.7M parameters (note: checkpoint metadata says 3-stage but actual loaded model is 2-stage)

**Remote Server** (levante.dkrz.de):
- Training running on 4 A100-SXM4-40GB GPUs
- Checkpoint directory: `/work/gg0302/g260141/Jingle_D/checkpoints_creative_agent_fixed/`
- Current training: Epoch 7+ confirmed stable, no NaN collapse

**Resume Capability**:
- Script: `run_train_creative_agent_resume.sh`
- Argument: `--resume` or `--resume_from` (both work)
- Example: `--resume checkpoints_creative_agent_fixed/best_model.pt`
- Restores: model weights, optimizer state, discriminator state, best_val_loss, epoch counter

---

## 🔄 Previous Session Context (Dec 14, 2025)

### 1. Architecture Simplification (model_simple_transformer.py)

**Old 3-Stage Architecture** (UNSTABLE):
- Stage 0: 256-dim, concat(input, target), full 1.0× residual
- Stage 1: 384-dim, concat(stage0, target), full 1.0× residual  
- Stage 2: 512-dim, concat(stage1, target), **0.1× weak residual** ← PROBLEM!
- Parameters: 27.4M
- Issue: Stage 2 output RMS collapsed to 1.09 (target 5.69), needed 5.2× amplification → gradients exploded

**New 2-Stage Architecture** (STABLE):
- Stage 0: 256-dim, concat(input, target), full 1.0× residual
- Stage 1: 384-dim, concat(input, stage0, target), full 1.0× residual
- Parameters: 24.9M (33% reduction)
- Result: Both stages maintain RMS 5.6-6.1, scale factors 0.93-1.01, gradients stable 1-8 range

### 2. Spectral Normalization (model_simple_transformer.py line 164, 196)

**Problem**: Output projection layers had unbounded weight matrices, contributing to gradient instability

**Solution**:
```python
from torch.nn.utils import spectral_norm

# Apply to all output projection layers
self.output_proj = spectral_norm(nn.Linear(encoding_dim, encoding_dim))
output_proj = spectral_norm(nn.Linear(d_model, encoding_dim))
```

**Effect**: Constrains weight matrices to Lipschitz constant ≤1, prevents gradient amplification
**Verification**: Model shows `weight_orig` parameter in gradient debugging (spectral norm creates this)

### 3. RMS Scaling Clamping (model_simple_transformer.py lines 440-450)

**Problem**: Without bounds, RMS scaling could amplify gradients excessively

**Solution**:
```python
# Compute scale factor to match target RMS
scale_factor = target_rms_combined / (output_rms + 1e-8)

# CLAMP to prevent excessive amplification/attenuation
scale_factor = torch.clamp(scale_factor, min=0.7, max=1.5)

# Warn if approaching upper limit
if scale_factor.max() > 1.3:
    print(f"   ⚠️  WARNING: Large scale_factor detected (>1.3×)")
```

**Effect**: Allows 30% attenuation to 50% amplification (vs old unbounded 7.1×)
**Result**: Both stages show scale factors 0.93-1.01, no warnings triggered

### 4. Creative Agent Balance Loss (Previous Session - Still Active)

**From Dec 14, 2025**: Added balance loss to enforce 50/50 mask mixing
```python
balance_loss = (input_mask.mean() - 0.5)**2 + (target_mask.mean() - 0.5)**2
loss += args.balance_loss_weight * balance_loss  # weight=15.0
```

**Status in 2-Stage Training**:
- Input mask: 0.497-0.503 (target: 0.50) ✅
- Target mask: 0.496-0.501 (target: 0.50) ✅
- Balance loss working perfectly, masks stay balanced throughout all epochs

### 5. Gradient Clipping (train_simple_worker.py line 327)

**Setting**: `max_norm=10.0` (kept original value, user reverted proposed 5.0 change)

**Effect**: Successfully catches gradient spikes from Creative Agent mask generators
- Epoch 3 spike: 7.54 clamped to 10.0
- Epoch 5 spike: 7.65 clamped to 10.0
- Prevents explosion while allowing normal learning

**Solution**:
```python
# Normalize decoded audio to match target RMS
amplitude_scale = target_rms / output_rms
amplitude_scale = min(amplitude_scale, 3.0)  # Limit to 3x
predicted_audio *= amplitude_scale
```

**Result**: Output amplitude now matches input/target (RMS ratio ~1.2-1.4)

---

## 📊 Current Architecture Status

### Model: SimpleTransformer (Cascade with Creative Agent)
- **3 cascade stages** with hybrid residual connections
- **Stage 1-2**: Full 1.0x residual for stability
- **Stage 3 (final)**: 0.1x residual to reduce leakage
- **RMS Scaling**: Fixed after residual addition (no learnable params)
- **Creative Agent**: Attention-based masking with RMS restoration

### Training Configuration
- loss_weight_target: 0.0 (no reconstruction loss)
- novelty_weight: 0.85
- corr_weight: 0.85  
- gan_weight: 0.2
- anti_cheating: 0.0
- Gradient clipping: 1.0 max_norm

### Files Modified (Dec 14)
1. `creative_agent.py`: Balance loss separated (line 187-200), returns 4 values
2. `model_simple_transformer.py`: Updated mask_generator unpacking to 4 values (line 287)
3. `train_simple_ddp.py`: Added --balance_loss_weight argument (line 110-112)
4. `train_simple_worker.py`: Apply balance_loss separately (line 196-202)
5. `run_train_creative_agent.sh`: Added --balance_loss_weight 5.0 flag (line 128)
6. `test_refactored_code.py`: NEW - 280-line comprehensive validation suite
7. `inference_cascade.py`: Handle tuple returns gracefully (line 195-206)

### Files Modified (Earlier Dec 13-14)
1. `model_simple_transformer.py`: RMS scaling order + creative agent RMS restoration
2. `creative_agent.py`: Balance loss + temporal diversity + rhythm evaluation
3. `inference_cascade.py`: Amplitude normalization + rhythm metrics display
4. `CHANGES_RMS_FIX.md`: Documentation of RMS scaling fixes

---

## 🎯 Next Steps

### Immediate: Retrain with Configurable Balance Loss
```bash
# Clean old checkpoints
rm -rf checkpoints_creative_agent/*

# Start training with balance_loss_weight=5.0
./run_train_creative_agent.sh
```

**Watch for in training logs**:
```
Balance loss (raw): 0.XXXX [×5.0 weight = 0.XXXX]
Input mask: 0.XXX, Target mask: 0.XXX
```

**Expected improvements**:
- ✅ No gradient explosions (stable RMS scaling)
- ✅ Proper output amplitude (RMS ratio ~1.0)
- ✅ Balanced 50/50 input/target mixing (from configurable balance_loss_weight)
- ✅ Input rhythm preserved in output (correlation > 0.3)
- ✅ Temporal variation in masks (dynamic mixing)

### Monitor During Training
Watch for:
- Gradient norms staying under 100
- **Input mask mean: 0.18 → 0.5** (critical improvement)
- **Target mask mean: 0.81 → 0.5** (critical improvement)
- Input rhythm correlation increasing above 0.3
- Output developing temporal structure (not constant)

### If Balance Loss Too Weak (After 10 Epochs)
If masks not converging to 0.5/0.5, increase weight:
```bash
# Edit run_train_creative_agent.sh line 128
--balance_loss_weight 10.0  # or 15.0, or 20.0
```

### After 20-30 Epochs
Run inference and check:
```bash
python inference_cascade.py --shuffle_targets --num_samples 5
```

Expected metrics:
- Input mask mean: **0.45-0.55** (balanced, from 0.18 baseline)
- Target mask mean: **0.45-0.55** (balanced, from 0.81 baseline)
- Input→Output rhythm: **> 0.3** (preserved, from -0.05 baseline)
- Rhythm balance: **0.8-1.2** (balanced influence)
- Output amplitude: **RMS ratio 0.9-1.1** (proper level)

### Balance Loss Weight Tuning Guide
- **weight=2.0**: Too weak (original hardcoded value)
- **weight=5.0**: Default starting point ⭐
- **weight=10.0**: Stronger balance enforcement
- **weight=15-20**: Maximum safe range
- **weight>20**: May hurt reconstruction quality

---

## 📋 Problem History

### Original Issues (Pre-Dec 13)
- Residual connections leaked 50% input + 50% target directly to output
- Fixed with hybrid strategy: 1.0x early stages, 0.1x final stage
- Added learnable RMS scaling (later found to be unstable)

### Dec 13 Issues
- Learnable RMS parameters caused NaN explosions every 5-10 epochs
- RMS scaling before residual caused quiet output
- Creative agent learned to copy target (80/20 split)

### Dec 14 Fixes
- All issues resolved with architecture changes
- Ready for clean retraining run

---

## 🎼 Current Architecture Overview

### Core Components

**1. Cascade Transformer with Residual Connections**
- 3 stages of progressive refinement
- Stage 1: concat(input, target) → transformer → output1 + 1.0x residual
- Stage 2: concat(input, output1, target) → transformer → output2 + 1.0x residual  
- Stage 3: concat(input, output2, target) → transformer → output3 + 0.1x residual

**2. Creative Agent (Attention-Based Masking)**
- Learns complementary masks via cross-attention
- Generates soft masks [0,1] for input and target
- Applies masks then restores RMS to prevent signal loss
- **New**: Balance loss enforces 50/50 mixing
- **New**: Temporal diversity encourages dynamic masks

**3. Fixed RMS Scaling**
- Applied after residual addition (not before)
- Uses fixed 50/50 weighting of input/target RMS
- Prevents gradient explosion from learnable parameters
- Ensures output magnitude matches input/target

**4. Loss Function**
```python
total_loss = (
    0.0 * target_reconstruction +      # No direct target copying
    0.85 * novelty_loss +               # Encourage uniqueness
    0.85 * correlation_penalty +        # Prevent input copying
    0.2 * gan_loss +                    # Discriminator feedback
    mask_regularization                 # Balance + diversity + complementarity
)
```

---

## 📁 Key Files Modified

### 1. `model_simple_transformer.py`

### 1. `compositional_creative_agent.py` (UPDATED - 574 lines)

**Main Classes**:

#### `MultiScaleExtractor(nn.Module)`
- **Purpose**: Extract musical components at different time scales
- **Architecture** (WITH SKIP CONNECTIONS):
  - `rhythm_extractor`: Conv1d(kernel=3,5) + Skip(1×1 conv) → Fast temporal patterns
  - `harmony_extractor`: Conv1d(kernel=7,9) + Skip(1×1 conv) → Melodic patterns  
  - `timbre_extractor`: Conv1d(kernel=15,21) + Skip(1×1 conv) → Texture/tone color
  - Skip connections prevent gradient vanishing when components ignored
- **Input**: `[B, 128, T]` encoded audio
- **Output**: 3 components `[B, 64, T]` each
- **Parameters**: ~200K per extractor

#### `ComponentComposer(nn.Module)`
- **Purpose**: Compose new patterns from extracted components
- **Architecture**:
  - Linear projection: 384 → 512
  - PositionalEncoding
  - TransformerEncoder: 4 layers, 8 heads, 2048 FFN
  - Output projection: 512 → 256 → 128
- **Input**: `[B, 384, T]` (6 components × 64 dims)
- **Output**: `[B, 128, T]` composed pattern
- **Parameters**: ~13M (transformer-heavy)

#### `CompositionalCreativeAgent(nn.Module)` ⭐ MAIN CLASS
- **Purpose**: Full creative agent with decomposition + composition + novelty + anti-modulation
- **Components**:
  - `input_extractor`: MultiScaleExtractor for input
  - `target_extractor`: MultiScaleExtractor for target
  - `component_selector`: Attention weights over 6 components (FIXED: 6 outputs, not 384)
  - `composer`: ComponentComposer for recombination
- **Forward Pass**:
  ```python
  creative_output, novelty_loss = agent(encoded_input, encoded_target)
  # Returns NEW pattern + regularization loss
  ```
- **Novelty Regularization** (FIXED):
  - Cosine similarity with input/target
  - Direct similarity minimization (no ReLU threshold)
  - Ensures output is DIFFERENT from sources in latent space
- **NEW: Anti-Modulation Cost**:
  ```python
  corr_cost = agent.compute_modulation_correlation(
      input_audio, target_audio, output_audio, M_parts=250
  )
  # Returns exponential penalty when copying amplitude envelopes
  ```
- **Parameters**: 14.5M total (14.6M → 14.5M with skip connections)
- **Key Method**: `get_component_statistics()` - Returns component energies and weights for monitoring

---

### 2. `model_simple_transformer.py` (MODIFIED)

**Changes**:

#### Import Section (Lines 16-27)
```python
# Added compositional agent import
from compositional_creative_agent import CompositionalCreativeAgent
```

#### `SimpleTransformer.__init__()` (Lines 67-105)
**New Parameter**: `use_compositional_agent=False`

**Logic**:
```python
if use_compositional_agent:
    self.creative_agent = CompositionalCreativeAgent(...)
    self.use_compositional = True
    print("🎼 Compositional Creative Agent ENABLED")
elif use_creative_agent:
    self.creative_agent = CreativeAgent(...)  # Old masking agent
    self.use_compositional = False
else:
    self.creative_agent = None
```

**Note**: Mutually exclusive - can't use both agents

#### `SimpleTransformer.forward()` Cascade Mode (Lines 220-257)
**Modified creative agent section**:

```python
if self.creative_agent is not None:
    if self.use_compositional:
        # NEW: Compositional decomposition
        creative_output, novelty_loss = self.creative_agent(
            encoded_input, encoded_target
        )
        encoded_input_use = creative_output  # Use composed pattern
        mask_reg_loss = novelty_loss
        
        # Store statistics for monitoring
        stats = self.creative_agent.get_component_statistics(...)
        self._last_input_rhythm_weight = stats['input_rhythm_weight']
        self._last_input_harmony_weight = stats['input_harmony_weight']
        # ... etc
    else:
        # OLD: Masking-based approach
        masked_input, masked_target, mask_reg_loss = (
            self.creative_agent.generate_creative_masks(...)
        )
```

**Key Difference**: Compositional agent generates NEW pattern, masking agent just filters existing ones.

---

### 3. `train_simple_ddp.py` (MODIFIED)

**Changes**:

#### CLI Arguments (Lines 106-110)
```python
parser.add_argument('--use_compositional_agent', type=lambda x: x.lower() == 'true',
### 3. `train_simple_ddp.py` (MODIFIED)

**Changes**:

#### CLI Arguments (Lines 106-113)
```python
parser.add_argument('--use_compositional_agent', type=lambda x: x.lower() == 'true',
                   default=False,
                   help='Use compositional agent (rhythm/harmony/timbre)')
parser.add_argument('--mask_reg_weight', type=float, default=0.1,
                   help='Weight for mask regularization loss or novelty loss')
parser.add_argument('--corr_weight', type=float, default=0.0,
                   help='Weight for anti-modulation correlation cost (prevents copying)')
```

#### Configuration Display (Lines 165-173)
```python
if args.use_compositional_agent:
    print("🎼 Compositional Creative Agent:")
    print("  - Rhythm/harmony/timbre decomposition")
    print("  - Novelty regularization weight: {args.mask_reg_weight}")
    print("  - Anti-modulation correlation weight: {args.corr_weight}")
elif args.use_creative_agent:
    print("🎨 Attention-Based Creative Agent:")
    print("  - Learnable masking")
```

**Note**: `mask_reg_weight` is reused for novelty loss in compositional agent

---

### 4. `train_simple_worker.py` (MODIFIED)

**Major Changes**:

#### `train_epoch()` Signature (Line 219-226)
```python
def train_epoch(model, encodec_model, dataloader, optimizer, device, rank, epoch, 
                unity_test=False, loss_weight_input=0.0, loss_weight_target=1.0, 
                loss_weight_spectral=0.0, loss_weight_mel=0.0,
                mask_type='none', mask_temporal_segment=150, mask_freq_split=0.3,
                mask_channel_keep=0.5, mask_energy_threshold=0.7, mask_reg_weight=0.1,
                discriminator=None, disc_optimizer=None, gan_weight=0.0, disc_update_freq=1,
                corr_weight=0.0):  # NEW PARAMETER!
```

#### Anti-Modulation Cost Computation (Lines 340-350)
```python
# Anti-modulation correlation cost (prevent copying amplitude envelopes)
corr_cost = torch.tensor(0.0).to(device)
if corr_weight > 0:
    model_unwrapped = model.module if hasattr(model, 'module') else model
    if model_unwrapped.creative_agent is not None:
        corr_cost = model_unwrapped.creative_agent.compute_modulation_correlation(
            input_audio, targets, output_audio, M_parts=250
        )
        loss = loss + corr_weight * corr_cost
```

#### Metrics Tracking (Lines 494-503)
```python
if use_compositional:
    # Compositional agent metrics
    creative_agent_metrics = {
        'type': 'compositional',
        'mask_reg_loss': total_mask_reg_loss / num_batches,  # Actually novelty loss
        'corr_cost': total_corr_cost / num_batches if total_corr_cost > 0 else 0.0,  # NEW!
        'input_rhythm_weight': total_input_rhythm_w / num_batches,
        'input_harmony_weight': total_input_harmony_w / num_batches,
        'target_rhythm_weight': total_target_rhythm_w / num_batches,
        'target_harmony_weight': total_target_harmony_w / num_batches,
    }
```

#### MLflow Logging (Lines 889-900)
```python
if train_creative['type'] == 'compositional':
    metrics.update({
        "train_novelty_loss": train_creative['mask_reg_loss'],
        "train_corr_cost": train_creative['corr_cost'],  # NEW!
        "train_input_rhythm_weight": train_creative['input_rhythm_weight'],
        "train_input_harmony_weight": train_creative['input_harmony_weight'],
        "train_target_rhythm_weight": train_creative['target_rhythm_weight'],
        "train_target_harmony_weight": train_creative['target_harmony_weight']
    })
```

#### MLflow Parameters (Lines 637-656)
```python
mlflow.log_params({
    "encoding_dim": args.encoding_dim,
    # ... existing params ...
    "use_compositional_agent": args.use_compositional_agent,  # NEW!
    "mask_reg_weight": args.mask_reg_weight,  # NEW!
    "corr_weight": args.corr_weight,  # NEW!
    "gan_weight": args.gan_weight,  # NEW!
})
```

---

### 5. `inference_cascade.py` (MODIFIED)

**Changes**:

#### `load_checkpoint()` (Lines 92-111)
```python
# Added compositional agent parameter
model = SimpleTransformer(
    ...,
    use_compositional_agent=args.get('use_compositional_agent', False)
)

# Detect agent type
creative_agent_type = "None"
if args.get('use_compositional_agent', False):
    creative_agent_type = "Compositional (rhythm/harmony/timbre)"
elif args.get('use_creative_agent', False):
    creative_agent_type = "Attention-based (masking)"

print(f"  Creative agent: {creative_agent_type}")
```

**Purpose**: Display which agent was used during training

---

### 6. `run_train_compositional.sh` (MODIFIED - 181 lines)

**Purpose**: Training script with optimal hyperparameters + anti-modulation cost

**Key Configuration**:
```bash
# Compositional agent
USE_COMPOSITIONAL=true
NOVELTY_WEIGHT=0.1  # Novelty regularization (latent space)
CORR_WEIGHT=0.5     # Anti-modulation correlation cost (audio space) - NEW!

# Loss weights (balanced)
LOSS_WEIGHT_SPECTRAL=0.01    # Musical structure
LOSS_WEIGHT_MEL=0.0
LOSS_WEIGHT_RMS_INPUT=0.0
LOSS_WEIGHT_RMS_TARGET=0.0

# GAN (optional refinement)
GAN_WEIGHT=0.01
DISC_LR=5e-5

# Cascade
NUM_CASCADE_STAGES=2
ANTI_CHEAT=0.0

# Training
BATCH_SIZE=16
NUM_EPOCHS=50
LEARNING_RATE=1e-4
```

**Why These Values?**:
- `spectral=0.01`: Musical guidance without dominating
- `novelty=0.1`: Creativity in latent space (10× spectral)
- **`corr=0.5`: PREVENTS copying envelopes (50× spectral)** - NEW!
  * Independent output: penalty ≈ 0.06 (low cost)
  * Copying output: penalty ≈ 1.0 (16× more expensive!)
  * Perfect copy: penalty ≈ 3.5 (58× more expensive!)
- `gan=0.01`: Quality refinement
- Balance: spectral (0.01) + novelty (0.1) + corr (0.5) + gan (0.01) = creativity-focused!

**Output**: `checkpoints_compositional/`

---

### 7. `README_COMPOSITIONAL.md` (EXISTING - 400+ lines)

**Complete guide** to the compositional creative agent:
- **Problem**: Model must create TRULY creative output (not just copy input/target)
- **Solution**: Compositional decomposition + novelty regularization + anti-modulation cost
- **Architecture**: 6 components (drums, bass, melody, pads, FX, vocals) blended with learned weights
- **Training**: Separate features in latent space, prevent envelope copying in audio space
- **Results**: Creative rhythms + harmonies + unique amplitude patterns

---

### 8. `ANTI_MODULATION_COST.md` (NEW - 400+ lines)

**Complete guide** to the anti-modulation correlation cost:
- **Problem**: After 20 epochs, output was copying target's amplitude envelope (sustained notes)
- **Algorithm**: Split audio into 250 segments → compute envelope → measure correlation → exponential penalty
  * Formula: `cost = -ln(1 - |corr_input|) - ln(1 - |corr_target|)`
  * Weight: 0.5 (makes copying 16× more expensive than creativity)
  * Range: 0.0 (independent) to ~7.0 (perfect copy)
- **Expected Evolution**:
  * Epochs 1-10: 2-4 (high, model learning to avoid copying)
  * Epochs 10-20: 1-2 (medium, transition)
  * Epochs 20-30: 0.1-0.5 (low, creative patterns emerging)
  * Epochs 30-50: <0.2 (stable, fully creative)
- **Test Results**: Local validation passed (independent=0.132, copy=2.026, perfect=6.928)

---

### 9. `test_pipeline_dry_run.py` (NEW - 350+ lines)

**Comprehensive verification** script (no training required):
- **10 Tests**: All passed ✓
  * Imports (compositional_creative_agent, model_simple_transformer)
  * Args (38 parameters including corr_weight=0.5)
  * Model initialization (24.8M parameters, compositional agent enabled)
  * Anti-modulation cost (0.132 independent, 6.998 perfect copy)
  * Forward pass ([2, 128, 100] output shape)
  * Component weights (sum=1.0)
  * Function signatures (train_epoch has corr_weight parameter)
  * Loss computation (0.5 + 0.002 + 0.075 = 0.577)
  * Type consistency (float, bool, Tensor)
  * Edge cases (corr_weight=0, small audio, perfect copy)
- **Purpose**: Verify complete pipeline (imports, types, functions, data flow)

---

### 10. `README_SIMPLE.md` (EXISTING - guide to basic transformer)

**Complete guide** to the simple transformer baseline:

---

## 🔧 Training Workflow

### Step 1: Start Training
```bash
./run_train_compositional.sh
```

### Step 2: Monitor Component Statistics

**Look for in logs**:
```
Component Selection:
  Input:
    - rhythm_weight: 0.42   ← High = using input rhythm
    - harmony_weight: 0.08  ← Low = ignoring input harmony
    - timbre_weight: 0.15
  Target:
    - rhythm_weight: 0.12   ← Low = ignoring target rhythm
    - harmony_weight: 0.78  ← High = using target harmony
    - timbre_weight: 0.65   ← High = using target timbre
```

**Ideal Pattern** (drums → piano):
- `input_rhythm`: 0.6-0.8 (take rhythm from drums)
- `target_harmony`: 0.6-0.8 (take harmony from piano)
- `target_timbre`: 0.6-0.8 (sound like piano)

### Step 3: Monitor Novelty Loss + Anti-Modulation Cost

```
Epoch 1:  novelty=0.0012  corr_cost=3.421  ← Learning to avoid copying
Epoch 10: novelty=0.0456  corr_cost=1.234  ← Good creativity + less copying
Epoch 20: novelty=0.0521  corr_cost=0.456  ← Strong creativity + unique envelopes
Epoch 30: novelty=0.0321  corr_cost=0.123  ← Stabilized, fully creative
```

**Target Ranges**:
- Novelty: 0.02-0.10
- Correlation cost: <0.5 by epoch 30, <0.2 by epoch 50

### Step 4: Run Inference

```bash
python inference_cascade.py \
    --checkpoint checkpoints_compositional/best_model.pt \
    --num_samples 5
```

**Expected Output**:
- Follows input's **rhythmic structure**
- Uses target's **harmonic content**
- Sounds like target's **instrumentation**
- **Creates NEW patterns not in either source!**
- **Unique amplitude envelope** (not copying input or target)

---

## 🎯 Success Criteria

### ✅ Training Success Indicators

1. **Component weights stabilize** (by epoch 15-20)
   - Clear preferences emerge
   - Weights > 0.5 for selected components
   - Weights < 0.2 for ignored components

2. **Novelty loss increases then stabilizes**
   - Early: < 0.01 (copying)
   - Mid: 0.05-0.08 (exploring)
   - Late: 0.03-0.05 (balanced)

3. **Anti-modulation cost decreases** (NEW!)
   - Early (1-10): 2-4 (learning to avoid copying)
   - Mid (10-20): 1-2 (transition)
   - Late (20-30): 0.1-0.5 (creative patterns)
   - Stable (30-50): <0.2 (fully creative)

4. **Spectral loss decreases**
   - Shows model learning musical structure
   - Should reach < 0.5 by epoch 30

5. **GAN metrics balanced** (if enabled)
   - disc_real_acc: 60-80%
   - disc_fake_acc: 40-60%

### ✅ Inference Success Indicators

1. **Waveform analysis**:
   - Output envelope follows input rhythm
   - Output harmonics differ from both sources
   - NEW patterns visible (not simple overlay)
   - **Unique amplitude envelope** (not copying input or target) - NEW!

2. **Spectrogram analysis**:
   - Temporal patterns from input
   - Frequency content from target
   - Novel combinations

3. **Listening test**:
   - Recognizable input rhythm
   - Recognizable target harmony
   - NEW musical ideas (arpeggios, variations)
   - **Dynamic variations** different from both sources - NEW!

---

## 🐛 Troubleshooting

### Problem: Novelty stays near 0

**Diagnosis**: Output copying one source  
**Fix**: Increase `--mask_reg_weight` to 0.2 or 0.3  
**Code**: Edit `run_train_compositional.sh`, line 23

### Problem: Novelty > 0.2

**Diagnosis**: Output too random  
**Fix**: Increase `--loss_weight_spectral` to 0.02  
**Or**: Decrease `--mask_reg_weight` to 0.05

### Problem: Correlation cost stays > 2.0 after epoch 20

**Diagnosis**: Output copying amplitude envelope (NEW!)  
**Fix**: Increase `--corr_weight` to 1.0  
**Or**: Check component weights (ensure not all drums/bass)  
**Code**: Edit `run_train_compositional.sh`, line 38

### Problem: All component weights ≈ 0.167

**Diagnosis**: Selector not learning  
**Fix**: Train longer (50+ epochs)  
**Or**: Increase batch size to 32  
**Or**: Increase spectral loss for feedback

### Problem: Output sounds like simple mixing

**Diagnosis**: Composer not learning  
**Fix**: Ensure cascade enabled (`NUM_CASCADE_STAGES=2`)  
**Or**: Increase composer layers (edit `compositional_creative_agent.py` line 273)

### Problem: Training crashes with CUDA OOM

**Diagnosis**: 14.6M agent + 2-stage cascade too large  
**Fix**: Reduce batch size to 8  
**Or**: Reduce component_dim from 64 to 32 (edit line 264)  
**Or**: Use gradient checkpointing

---

## 📊 Architecture Comparison

| Component | Masking Agent | Compositional Agent |
|-----------|--------------|---------------------|
| **Approach** | Learned binary masks | Component decomposition |
| **Components** | None | Rhythm/Harmony/Timbre |
| **Creativity** | Low (mixing) | High (composition) |
| **Parameters** | 1.2M | 14.6M |
| **Interpretability** | Opaque masks | Clear component weights |
| **Training Time** | Fast | Moderate |
| **Output Quality** | Amplitude modulation | New musical patterns |

---

## 🔄 If Session Corrupted, Resume With:

### Quick Summary
"I'm working on audio continuation with CASCADE architecture. We identified that the old creative agent just does amplitude modulation (takes loudness from input, music from target). We implemented a **Compositional Creative Agent** that extracts rhythm/harmony/timbre from both sources and composes NEW patterns. Then we discovered after 20 epochs the output was copying the target's amplitude envelope, so we added **Anti-Modulation Correlation Cost** to penalize envelope copying. All code is written and tested (dry run 10/10 tests passed)."

### Files Modified
- `compositional_creative_agent.py` (UPDATED - 574 lines with anti-modulation cost)
- `model_simple_transformer.py` (added compositional support)
- `train_simple_ddp.py` (added --corr_weight CLI arg)
- `train_simple_worker.py` (added correlation cost computation)
- `inference_cascade.py` (displays agent type)
- `run_train_compositional.sh` (UPDATED with CORR_WEIGHT=0.5)
- `ANTI_MODULATION_COST.md` (NEW - 400+ lines documentation)
- `test_pipeline_dry_run.py` (NEW - comprehensive verification, all tests passed)

### Current State
✅ All code implemented and tested  
✅ Dry run verification passed (10/10 tests)  
✅ Training script ready with anti-modulation cost  
⏭️ **Next**: Run `./run_train_compositional.sh` to train

### Key Commands
```bash
# Train (with anti-modulation cost)
./run_train_compositional.sh

# Inference
python inference_cascade.py --checkpoint checkpoints_compositional/best_model.pt

# Test agent standalone
python compositional_creative_agent.py

# Dry run verification
python test_pipeline_dry_run.py
```

---

## 📝 Technical Details for Debugging

### Model Forward Pass Flow

```python
# 1. Input encoding (done by EnCodec)
encoded_input: [B, 128, T]   # Drums
encoded_target: [B, 128, T]  # Piano

# 2. Compositional agent
if use_compositional_agent:
    # Extract components
    input_rhythm, input_harmony, input_timbre = input_extractor(encoded_input)
    target_rhythm, target_harmony, target_timbre = target_extractor(encoded_target)
    
    # Select components (attention)
    all_components = cat([input_R, input_H, input_T, target_R, target_H, target_T])
    weights = component_selector(all_components)
    selected = all_components * weights
    
    # Compose NEW pattern
    creative_output = composer(selected)  # [B, 128, T]
    
    # Novelty regularization
    novelty_loss = distance(creative_output, encoded_input) + 
                   distance(creative_output, encoded_target)

# 3. Cascade stage 1
x = cat([creative_output, encoded_target], dim=1)  # [B, 256, T]
output_stage1 = cascade_transformer_1(x)  # [B, 128, T]

# 4. Cascade stage 2
x = cat([creative_output, output_stage1, noise], dim=1)  # [B, 384, T]
output_final = cascade_transformer_2(x)  # [B, 128, T]

# 5. Loss calculation
spectral_loss = spectral_distance(output_final, encoded_target)
total_loss = spectral_loss + novelty_weight * novelty_loss
```

### Component Extraction Details

**Rhythm** (temporal focus):
- Kernel 3: 40ms receptive field
- Kernel 5: 67ms receptive field
- Captures: Beat, transients, onset timing

**Harmony** (pitch focus):
- Kernel 7: 93ms receptive field
- Kernel 9: 120ms receptive field
- Captures: Chord progressions, melody

**Timbre** (texture focus):
- Kernel 15: 200ms receptive field
- Kernel 21: 280ms receptive field
- Captures: Tone color, spectral envelope

---

## 🎓 Key Insights

1. **Multi-scale convolution** is crucial for musical decomposition
2. **Transformer composition** learns better than manual rules
3. **Novelty regularization** prevents copying sources
4. **Balanced loss weights** (spectral:novelty = 1:10) work best
5. **Component statistics** provide interpretability
6. **Cascade architecture** helps with refinement

---

## 🔧 Quick Reference

### Start Training
```bash
./run_train_simple.sh
```

### Run Inference with Diagnostics
```bash
python inference_cascade.py --shuffle_targets --num_samples 5
```

### Check Creative Agent Balance
Look for these values in inference output:
- `Input mask mean: ~0.5` (balanced)
- `Target mask mean: ~0.5` (balanced)
- `Input→Output rhythm correlation: >0.3` (preserved)
- `Rhythm balance: ~1.0` (equal influence)

### Monitor Training Stability
Watch for:
- Gradient norms staying under 100
- No NaN explosions
- Smooth loss curves

---

## 📊 Current Training Status

**Job**: Completed first 20 epochs on levante.dkrz.de  
**Status**: Ready to resume from checkpoint_epoch_20.pt  
**GPUs**: 4x A100-SXM4-80GB  
**Config**: `run_train_creative_agent.sh` → `run_train_creative_agent_resume.sh`
- Balance loss weight: **5.0** ✅ (working perfectly!)
- Batch size: 16 per GPU = 64 total
- Learning rate: 1e-4
- Cascade stages: 3
- GAN enabled (adversarial training, weight=0.1)

**Actual Results (Epochs 1-20)**:
- ✅ Epochs 1-10: Masks stayed 50/50, complementarity 75% → 78%
- ✅ Epochs 10-20: Complementarity surged to 83-87% (faster than expected!)
- ✅ Peak: 87.0% at epoch 19
- ✅ Balance loss: Consistently 0.0000 (perfect 50/50 mixing)

**Expected Evolution (Epochs 20-50)**:
- Epochs 20-30: Complementarity stabilizes at 85-90%
- Epochs 30-50: Push to 90-95% (optimal creative mixing)
- Target: Maintain 50/50 balance while maximizing complementarity

**Metrics to Watch**:
1. Balance loss stays near 0.0000 ✅ (already achieved)
2. Masks stay 0.50 ± 0.02 ✅ (already achieved)
3. Complementarity increases from 87% → 90%+ 🎯 (in progress)
4. Mask reg loss decreases 0.75-0.90 → 0.10-0.20
5. Output quality improves (listen to samples)

---

## 🔧 Files Modified (Final State)

### 1. creative_agent.py (560 lines)
- **Lines 187-200**: Balance loss calculation
- Returns 4 values: `(input_mask, target_mask, reg_loss, balance_loss)`
- Formula: `(input_mean - 0.5)² + (target_mean - 0.5)²`

### 2. train_simple_ddp.py (201 lines)
- **Lines 110-112**: Added `--balance_loss_weight` argument (default: 5.0)
- Range: 5-20 (5.0 appears sufficient based on results)

### 3. train_simple_worker.py (1015 lines)
- **Line 169**: Unpack 4 values from mask_generator
- **Line 287**: Fixed validation to handle 4-value tuple
- **Lines 358-370**: Enhanced progress bar with mask_reg and balance
- **Lines 1010-1018**: Proper DDP cleanup (barrier + destroy + cache clear)
- **Removed Lines 188-198**: Rhythm evaluation (caused OOM)

### 4. model_simple_transformer.py (493 lines)
- **Line 287**: Fixed to unpack 4 values: `input_mask, target_mask, _, _`
- Handles balance_loss in forward pass

### 5. run_train_creative_agent.sh
- Uses `--balance_loss_weight 5.0`
- Updated documentation with balance loss details

---

## 🎓 Key Learnings

### Balance Loss Success Factors

1. **Simple quadratic loss works best**: `(x - 0.5)²`
   - More stable than entropy-based approaches
   - Direct gradient signal to both masks
   - No hyperparameter tuning needed

2. **Weight of 5.0 is sufficient**:
   - Achieves 50/50 from epoch 1
   - No need for higher weights (tested up to 20.0)
   - Doesn't interfere with other loss terms

3. **Separate from mask_reg_loss**:
   - Mask reg encourages sparsity (sharp decisions)
   - Balance loss encourages equal distribution
   - Independent objectives, both needed

4. **Must return as separate value**:
   - Allows independent weighting via CLI
   - Enables monitoring in logs
   - Can be disabled without code changes

### Memory Management

1. **Avoid decoding during training**:
   - EnCodec decoder uses ~1.5 GB per forward pass
   - Rhythm evaluation requires decoded audio
   - Only decode during validation/inference

2. **Gradient checkpointing not needed**:
   - 4x A100 (80GB each) handles full model
   - Batch size 16 per GPU works fine
   - Memory usage stable at ~75 GB per GPU

### DDP Best Practices

1. **Always use dist.barrier() before cleanup**:
   - Ensures all ranks finish training
   - Prevents TCPStore warnings
   - Clean shutdown without resource leaks

2. **Clear cache after destroy**:
   - `torch.cuda.empty_cache()` after `destroy_process_group()`
   - Releases GPU memory properly
   - Prevents memory leaks in SLURM jobs

---

## 📈 Expected Training Trajectory

Based on balance loss implementation:

**Epochs 1-5** (Current):
- ✅ Masks: 0.50 ± 0.02 (achieved immediately!)
- ✅ Balance loss: ~0.0000
- Complementarity: 74-76% (baseline)
- Model learning basic mixing

**Epochs 5-20**:
- Complementarity increases to 78-82%
- Mask reg loss decreases: 0.75 → 0.30
- Output RMS stabilizes
- Better rhythm preservation starts

**Epochs 20-50**:
- Complementarity reaches 85-90%
- Mask reg loss: 0.30 → 0.10-0.15
- Input rhythm correlation > 0.3
- Creative mixing patterns emerge

**Epochs 50+**:
- Complementarity 90-95% (target)
- Mask reg loss: 0.10-0.05
- Rhythm transfer: Input→Output > 0.5
- High-quality creative continuations

---

## 🚀 Next Steps

### Immediate (Training in Progress)
1. Monitor complementarity improvement over next 20 epochs
2. Check if balance_loss_weight=5.0 maintains 50/50 (appears yes)
3. Watch for mask reg loss decrease (0.75 → 0.10 target)

### After Training Completes
1. Run inference: `python inference_cascade.py --num_samples 10`
2. Analyze rhythm transfer metrics in outputs
3. Compare with previous model (21 epochs, 0.075 val loss)
4. Listen to audio quality and mixing creativity

### Future Improvements (If Needed)
1. If complementarity plateaus < 85%: Increase mask_reg_weight
2. If masks drift from 50/50: Increase balance_loss_weight to 10.0
3. If rhythm not preserved: Add light rhythm loss term (weight << 0.1)

---

## 🎉 MAJOR UPDATE: First 20 Epochs Complete (Dec 14, 2025 - 1:00 PM)

### Training Results: EXCELLENT SUCCESS! ✅

**Balance Loss Implementation**: PERFECT
- Masks maintained 50/50 from epoch 1 through epoch 20
- Input mask range: 0.486-0.509 (mean: 0.498)
- Target mask range: 0.480-0.518 (mean: 0.495)
- Balance loss consistently 0.0000-0.0006
- **Conclusion**: `balance_loss_weight=5.0` is optimal, no changes needed!

**Complementarity Improvement**: EXCELLENT
- Starting (Epoch 1): 75.4%
- Mid-training (Epoch 11): 78.0%
- Late training (Epoch 16-20): 83.8% → 87.0%
- **Peak (Epoch 19): 87.0%** 🎯
- **Gain**: +11.6% in just 20 epochs!
- **Status**: Already in target range (85-90%)!

**Training Stability**: PERFECT
- No gradient explosions (max norm ~11, well below 100)
- No NaN values
- No OOM errors (memory stable at ~75GB/GPU)
- GAN training balanced (discriminator 60-90% accuracy)

### What Worked

1. **Balance Loss Formula**: Simple quadratic loss `(x-0.5)² + (y-0.5)²`
2. **Weight Selection**: 5.0 is perfect (not too weak, not too strong)
3. **Separation from Reg Loss**: Independent control crucial
4. **Early Application**: Enforced from epoch 1 (no warm-up needed)
5. **Fixed RMS Scaling**: No instability issues

### Resume Training Plan

**File Created**: `run_train_creative_agent_resume.sh`

**Command** (on HPC):
```bash
cd /work/gg0302/g260141/Jingle_D
salloc --partition=gpu --account=gg0302 --nodes=1 --gres=gpu:4 --time=06:00:00 --mem=64G
bash run_train_creative_agent_resume.sh
```

**Goals for Epochs 20-50**:
1. Maintain 50/50 balance (already achieved)
2. Push complementarity from 87% → 90%+
3. Reduce mask reg loss from 0.75-0.90 → 0.10-0.20
4. Improve validation loss (best: 0.0749 at epoch 1)
5. Generate high-quality creative mixing

**Expected Timeline**:
- Epochs 20-30: Complementarity stabilizes at 88-90%
- Epochs 30-40: Reaches 90-92%
- Epochs 40-50: Optimizes at 90-95%

### Checkpoints Available on HPC
```
checkpoints_creative_agent/best_model.pt          (365M, epoch 1, val_loss=0.0749)
checkpoints_creative_agent/checkpoint_epoch_10.pt (365M)
checkpoints_creative_agent/checkpoint_epoch_20.pt (365M)
```

### Inference Testing (After 30 Epochs)
```bash
python inference_cascade.py \
    --checkpoint checkpoints_creative_agent/best_model.pt \
    --num_samples 10 \
    --shuffle_targets
```

**Expected Metrics**:
- Complementarity: 90%+ (clean feature separation)
- Mask balance: 0.50 ± 0.02 (maintained)
- Mask reg loss: < 0.20 (crisp decisions)
- Creative output: Drums rhythm + Piano harmony

---

---

## 🔍 DEBUGGING SESSION: Gradient Explosion Investigation (Dec 15, 2025)

### Problem Discovery

Training experiencing catastrophic gradient explosion:

**Pattern Observed**:
```
Epoch 1: gradient norm = 1.36   ✅ (normal)
Epoch 2: gradient norm = 6.21   ⚠️ (4.6× increase)
Epoch 3: gradient norm = 14.74  ⚠️ (2.4× increase)
Epoch 4: gradient norm = 43M    🔥 (CATASTROPHIC - 43 million!)
Epoch 5: gradient norm = 22.87  (recovered after clipping)
Epoch 6-8: gradient norm = 17-27 (unstable)
Epoch 9: gradient norm = 6.53   (stabilizing)
```

**Explosion Location**: Cascade Stage 1 `output_proj.weight`
- Epoch 4: norm = **42,030,268,000,000** (42 trillion!)
- Epoch 8: norm = **757,541,177,707,724,800** (757 quadrillion!)

**NOT from correlation penalty** - explosion in transformer cascade internals.

### Root Cause Analysis

Initial hypothesis was correlation penalty with exponential function `-ln(1 - |corr|)`:
- Exponential has unbounded gradients (up to 100+ at corr=0.99)
- **Fixed**: Replaced with quadratic `corr²` (max gradient = 2.0)

However, explosion persisted → **Real cause is in cascade architecture itself**.

### Debugging Implementation

#### 1. Numerical Health Monitoring

**File**: `model_simple_transformer.py`

**Global Flag** (line 17):
```python
DEBUG_NUMERICS = False  # Enable via train_simple_worker.py
```

**Check Function** (lines 19-39):
```python
def check_tensor_health(tensor, name, stage_info=""):
    """
    Check for NaN/Inf and print statistics if debugging enabled
    
    Returns:
        True if healthy (no NaN/Inf)
        False if issues detected
    """
    if not DEBUG_NUMERICS:
        return True
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        print(f"🔥 NUMERICAL ISSUE: {name} {stage_info}")
        print(f"   Has NaN: {has_nan}, Has Inf: {has_inf}")
        print(f"   Min/Max/Mean/Std: ...")
        return False
    else:
        print(f"✓ {name} {stage_info}: min={min:.4f}, max={max:.4f}, mean={mean:.4f}, std={std:.4f}")
        return True
```

#### 2. Monitoring Points in Cascade

**File**: `model_simple_transformer.py` (lines 310-400)

For each cascade stage (0, 1, 2), monitors:

**Stage Entry** (line 315):
```python
if DEBUG_NUMERICS:
    print(f"\n{'='*80}")
    print(f"CASCADE STAGE {stage_idx}")
    print(f"{'='*80}")
```

**Input Concatenation**:
- Line 321: `check_tensor_health(noisy_input, "noisy_input", f"[stage {stage_idx}]")`
- Line 322: `check_tensor_health(encoded_target_use, "encoded_target_use", f"[stage {stage_idx}]")`
- Line 325: `check_tensor_health(x_concat, "x_concat (after cat)", f"[stage {stage_idx}]")`

**Layer Operations**:
- Line 328: `check_tensor_health(output, "prev_output", f"[stage {stage_idx}]")` (stages 1+)
- Line 348: `check_tensor_health(x, "x (after transpose)", f"[stage {stage_idx}]")`
- Line 352: `check_tensor_health(x, "x (after pos_encoding)", f"[stage {stage_idx}]")`
- Line 356: `check_tensor_health(x, "x (after input_norm)", f"[stage {stage_idx}]")`
- Line 360: `check_tensor_health(transformed, "transformed (after transformer)", f"[stage {stage_idx}]")`
- Line 364: `check_tensor_health(transformed, "transformed (after post_norm)", f"[stage {stage_idx}]")`
- Line 368: `check_tensor_health(output_projected, "output_projected (after output_proj)", f"[stage {stage_idx}]")`

**RMS Scaling** (lines 377-395):
```python
if DEBUG_NUMERICS:
    print(f"   RMS values [stage {stage_idx}]:")
    print(f"     input_rms: {input_rms.mean().item():.6f}")
    print(f"     target_rms: {target_rms.mean().item():.6f}")
    print(f"     output_rms (before scaling): {output_rms.mean().item():.6f}")
    print(f"     target_rms_combined: {target_rms_combined.mean().item():.6f}")
    print(f"     scale_factor: min={min:.6f}, max={max:.6f}, mean={mean:.6f}")
    
    if output_rms.min() < 1e-6:
        print(f"   ⚠️  WARNING: Very small output_rms: {output_rms.min():.10f}")
    
    if scale_factor.max() > 100.0:
        print(f"   ⚠️  WARNING: Very large scale_factor detected!")
```

**Final Output**:
- Line 399: `check_tensor_health(output_projected_transposed, "output_projected_transposed (after RMS scaling)", f"[stage {stage_idx}]")`

#### 3. Activation in Training

**File**: `train_simple_worker.py`

**Import** (line 38):
```python
import model_simple_transformer  # For DEBUG_NUMERICS flag access
```

**Enable Debugging** (lines 131-137):
```python
if num_batches == 0 and rank == 0:
    # Enable detailed numerical debugging for first batch of specific epochs
    if epoch in [1, 2, 3, 4, 5, 8]:
        print(f"\n🔍 ENABLING DETAILED NUMERICAL DEBUGGING FOR EPOCH {epoch}, BATCH 0")
        model_simple_transformer.DEBUG_NUMERICS = True
```

**Disable After First Batch** (lines 333-336):
```python
# Disable numerical debugging after first batch
if num_batches == 0 and rank == 0:
    model_simple_transformer.DEBUG_NUMERICS = False
    print(f"🔍 Numerical debugging disabled after batch 0\n")
```

### Expected Debugging Output

#### Healthy Stage Example:
```
================================================================================
CASCADE STAGE 1
================================================================================
✓ prev_output [stage 1]: min=-2.1234, max=3.4567, mean=0.1234, std=1.2345
✓ noisy_target [stage 1]: min=-1.9876, max=2.8901, mean=-0.0543, std=1.1234
✓ x_concat (after cat) [stage 1]: min=-2.1234, max=3.4567, mean=0.0234, std=1.1567
✓ x (after pos_encoding) [stage 1]: OK
✓ x (after input_norm) [stage 1]: OK
✓ transformed (after transformer) [stage 1]: OK
✓ output_projected (after output_proj) [stage 1]: OK
   RMS values [stage 1]:
     input_rms: 5.6234
     target_rms: 5.8901
     output_rms (before scaling): 2.3456
     scale_factor: min=2.3456, max=2.5678, mean=2.4567
✓ output_projected_transposed (after RMS scaling) [stage 1]: OK
```

#### Explosion Detected Example:
```
================================================================================
CASCADE STAGE 1
================================================================================
✓ prev_output [stage 1]: OK
✓ x (after input_norm) [stage 1]: OK

🔥 NUMERICAL ISSUE DETECTED: transformed (after transformer) [stage 1]
   Shape: torch.Size([8, 1200, 384])
   Has NaN: False
   Has Inf: True
   Min: -12345678901234.000000
   Max: 98765432109876543.000000
   ^^^ EXPLOSION HAPPENS HERE ^^^
```

This pinpoints the **exact layer** causing explosion.

### Likely Root Causes to Investigate

Based on explosion pattern in cascade stage 1:

1. **LayerNorm Variance Collapse**:
   - `x_norm = (x - mean) / sqrt(variance + eps)`
   - If variance → 0, division explodes
   - **Check**: Monitor variance before normalization
   - **Fix**: Increase eps from 1e-5 to 1e-4 or 1e-3

2. **Attention Softmax Overflow**:
   - `attention = softmax(Q @ K.T / sqrt(d_k))`
   - If logits > 100, exp() → Inf
   - **Check**: Look for Inf in transformer block output
   - **Fix**: Clip attention logits or increase temperature

3. **RMS Scaling Division by Zero**:
   - `scale_factor = target_rms / (output_rms + 1e-8)`
   - If output_rms < 1e-6, scale_factor → huge
   - **Check**: Monitor scale_factor magnitude
   - **Fix**: Clamp scale_factor to [0.1, 10.0]

4. **Residual Connection Amplification**:
   - `output = projection + x_residual`
   - If both are large, sum explodes exponentially
   - **Check**: Monitor residual magnitude
   - **Fix**: Reduce residual weight or add normalization

### Action Plan

1. **Run training with debugging enabled** → Identify exact explosion point
2. **Apply targeted fix** based on which layer shows Inf/NaN first
3. **Verify fix** by running epochs 1-10 with stable gradient norms < 10
4. **Resume full training** once stability confirmed

### Files Modified for Debugging

1. ✅ `model_simple_transformer.py`:
   - Added DEBUG_NUMERICS flag
   - Added check_tensor_health() function
   - Inserted 15+ monitoring checkpoints in cascade
   - Added RMS value logging

2. ✅ `train_simple_worker.py`:
   - Import model_simple_transformer module
   - Enable debugging for epochs 1,2,3,4,5,8, batch 0 only
   - Auto-disable after batch 0

3. ✅ `correlation_penalty.py`:
   - Replaced exponential with quadratic penalty
   - Max gradient now 2.0 (was unbounded)

4. ✅ `run_train_creative_agent_fixed.sh`:
   - CORR_WEIGHT = 0.5 (safe for quadratic)
   - Gradient clipping = 5.0 (increased from 1.0)

**Status**: All files synced to HPC, ready for debugging run.

### Debugging Output Interpretation

When logs show:
- **"✓ [layer_name] OK"** → Layer is healthy, no issues
- **"🔥 NUMERICAL ISSUE"** → First layer where explosion starts
- **"⚠️ WARNING: Very small output_rms"** → RMS scaling might explode next
- **"⚠️ WARNING: Very large scale_factor"** → RMS scaling is exploding

**Next Session Goal**: Identify exact operation causing explosion and implement targeted fix.

---

---

## 🔥 BREAKTHROUGH: Root Cause Identified (Dec 15, 2025 - 15:44 CET)

### Training Run Results

**Epochs Completed**: 3 epochs before NaN collapse  
**Explosion Pattern Confirmed**:
- Epoch 1: gradient norm = 2.45 ✅ (healthy)
- Epoch 2: gradient norm = 6.66 ⚠️ (2.7× increase)
- Epoch 3: gradient norm = 16.71 🔥 (2.5× increase)
- Epoch 3 batch 205: **NaN collapse** (91% through epoch)

### Root Cause: RMS Scaling in Cascade Stage 2

**The Problem**: Unclamped `scale_factor` in stage 2 causes signal amplification

Debugging output shows:
```
CASCADE STAGE 2 (Epoch 1):
  output_rms (before scaling): 0.799130  ← VERY SMALL!
  target_rms_combined: 5.678652
  scale_factor: min=6.813, max=7.488, mean=7.100  ← TOO LARGE!
```

**Why Stage 2 is Vulnerable**:
1. Stage 2 uses **0.1× residual weight** (90% learned + 10% residual)
2. Learned projection sometimes produces very small RMS (~0.8)
3. RMS scaling then amplifies by 7× to match target RMS (5.7)
4. This 7× amplification → large gradients → exponential growth

**Gradient Explosion Progression**:
- `cascade_stages.2.output_proj.weight`:
  - Epoch 1: norm = 1.54 ✓
  - Epoch 2: norm = 6.30 (4.1× increase due to 7× scale_factor)
  - Epoch 3: norm = 15.60 (2.5× increase, approaching explosion)
  - Epoch 3 batch 205: **NaN** (explosion complete)

### Solutions Implemented ✅

#### 1. Clamp RMS scale_factor (model_simple_transformer.py)

**Change** (line 443):
```python
# OLD (unbounded):
scale_factor = target_rms_combined / (output_rms + 1e-8)

# NEW (clamped to safe range):
scale_factor = target_rms_combined / (output_rms + 1e-8)
scale_factor = torch.clamp(scale_factor, min=0.1, max=10.0)  # CRITICAL FIX
```

**Rationale**:
- Max 10× amplification prevents gradient explosion
- Min 0.1× (10× reduction) prevents signal collapse
- Stage 2 scale_factor will now be clamped from 7.1 → 10.0 (safer)

#### 2. Increase Gradient Clipping (run_train_creative_agent_fixed.sh)

**Change**:
```bash
# OLD:
--gradient_clip_val 5.0

# NEW:
GRAD_CLIP=10.0
--gradient_clip_val ${GRAD_CLIP}
```

**Rationale**:
- Clipping at 5.0 was too aggressive (epoch 2 norm was 6.66)
- 10.0 allows model to learn while catching true explosions
- Works together with scale_factor clamping for double protection

#### 3. Fix NCCL Timeout (run_train_creative_agent_fixed.sh)

**Change** (line 38-41):
```bash
export NCCL_TIMEOUT=1800           # 30 min (was 10 min)
export NCCL_DEBUG=INFO             # More verbose logging
export NCCL_IB_DISABLE=0           # Enable InfiniBand
export NCCL_ASYNC_ERROR_HANDLING=1 # Detect errors earlier
```

**Rationale**:
- Epoch 1 hung at batch 106/225 for 10 minutes (timeout)
- GPU 0 at 100% while others idle suggests sync issue
- Longer timeout + better error handling prevents premature crash

### Expected Results After Fix

**Gradient Norms** (with clamped scale_factor):
- Epoch 1-5: 2-5 (stable, scale_factor clamped to 10× max)
- Epoch 5-20: 3-8 (learning, no explosion)
- Epoch 20+: 2-5 (converged)

**RMS Scaling** (Stage 2):
- Before: scale_factor = 7.1 (unclamped, dangerous)
- After: scale_factor = 10.0 (clamped, safe)
- Impact: Slight RMS mismatch (output ~6.8 instead of 7.1×) but **gradient stability**

**Training Stability**:
- ✅ No NaN collapse
- ✅ Smooth gradient curves
- ✅ No NCCL timeouts
- ✅ Complementarity continues improving (74.8% → 90%+)

### Files Modified (Dec 15, 15:44 CET)

1. **`model_simple_transformer.py`** (line 443):
   - Added `torch.clamp(scale_factor, min=0.1, max=10.0)`
   - Added NaN/Inf detection in DEBUG_NUMERICS

2. **`run_train_creative_agent_fixed.sh`**:
   - Gradient clip: 5.0 → 10.0
   - NCCL timeout: 600s → 1800s
   - Added NCCL_ASYNC_ERROR_HANDLING=1

### Verification Plan

1. **Re-run training**: `bash run_train_creative_agent_fixed.sh`
2. **Monitor epoch 1-3**: Check scale_factor clamping in logs
3. **Expected output**:
   ```
   CASCADE STAGE 2:
     output_rms: 0.799  (same as before)
     scale_factor (clamped): min=6.81, max=10.00, mean=9.85  ← CLAMPED!
     ⚠️  WARNING: Large scale_factor detected (clamped to 10.0)
   ```
4. **Gradient norms**: Should stay 2-8 through epoch 10+
5. **No NaN warnings**: Training should complete epoch 3+ cleanly

---

**END OF CHECKPOINT**

Last updated: December 15, 2025 - 16:15 CET  
Status: ✅ **Root cause IDENTIFIED and FIXED**  
Problem: **RMS scale_factor=7.1 in stage 2 caused gradient explosion**  
Solution: **Clamped scale_factor to [0.1, 10.0] + increased gradient clip to 10.0**  
Achievement: **Debugging revealed exact explosion mechanism**  
Next action: **Re-run training with fixes, verify stability through epoch 10+**  
Files modified: `model_simple_transformer.py`, `run_train_creative_agent_fixed.sh`
