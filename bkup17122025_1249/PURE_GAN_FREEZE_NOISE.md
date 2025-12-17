# Freezing Noise Level in Pure GAN Training

## Overview
The `--gan_noise_ceiling` parameter allows you to cap the maximum noise fraction during Pure GAN curriculum learning. This is useful for:
- Training at a specific noise level for multiple epochs
- Preventing transition to full noise mode
- Fine-tuning model stability at intermediate noise levels

## Parameter
```bash
--gan_noise_ceiling FLOAT
```
- **Default:** `1.0` (no ceiling, allows full noise)
- **Range:** `0.0` to `1.0`
- **Effect:** Caps alpha (noise fraction) at this value

## Examples

### Example 1: Freeze at 30% noise
Train with noise increasing from 0% → 30%, then freeze at 30% for remaining epochs:
```bash
python3 train_simple_ddp.py \
    --pure_gan_mode 0.01 \
    --gan_curriculum_start_epoch 86 \
    --gan_noise_ceiling 0.3 \
    --checkpoint_dir checkpoints_GAN_30 \
    --epochs 150
```

**Timeline:**
- Epochs 86-115: Noise increases 0% → 30% (30 epochs × 0.01 = 0.3)
- Epochs 116-150: **Frozen at 30% noise** (35 epochs at constant noise)

### Example 2: Freeze at 50% noise
```bash
python3 train_simple_ddp.py \
    --pure_gan_mode 0.01 \
    --gan_curriculum_start_epoch 0 \
    --gan_noise_ceiling 0.5 \
    --epochs 100
```

**Timeline:**
- Epochs 1-50: Noise increases 0% → 50%
- Epochs 51-100: **Frozen at 50% noise**

### Example 3: No ceiling (default behavior)
```bash
python3 train_simple_ddp.py \
    --pure_gan_mode 0.01 \
    --gan_curriculum_start_epoch 86 \
    --gan_noise_ceiling 1.0 \
    --epochs 200
```

**Timeline:**
- Epochs 86-185: Noise increases 0% → 100% (100 epochs × 0.01 = 1.0)
- Epochs 186-200: Stays at 100% noise (pure noise mode)

## How It Works

**Before (no ceiling):**
```python
alpha = min(1.0, pure_gan_mode * gan_curriculum_counter)
```

**After (with ceiling):**
```python
alpha = min(gan_noise_ceiling, pure_gan_mode * gan_curriculum_counter)
```

**Example calculation with `--gan_noise_ceiling 0.3`:**
- Epoch 86 (counter=1): alpha = min(0.3, 0.01 × 1) = **0.01** (1% noise)
- Epoch 100 (counter=15): alpha = min(0.3, 0.01 × 15) = **0.15** (15% noise)
- Epoch 115 (counter=30): alpha = min(0.3, 0.01 × 30) = **0.30** (30% noise) ← ceiling reached
- Epoch 120 (counter=35): alpha = min(0.3, 0.01 × 35) = **0.30** (frozen at 30%)
- Epoch 150 (counter=65): alpha = min(0.3, 0.01 × 65) = **0.30** (still frozen)

## Training Output
When ceiling is reached, you'll see:
```
🎲 Pure GAN Curriculum Learning:
  Alpha: 0.3000 (0=music, 1=noise)
  Input:  70.0% music + 30.0% noise
  Target: 70.0% music + 30.0% noise
  🔒 Noise ceiling reached (30% max noise)
```

## Use Cases

### 1. Stability Training
Train at moderate noise levels to improve model robustness without extreme noise:
```bash
--gan_noise_ceiling 0.2  # Train at max 20% noise
```

### 2. Gradual Transition
Use multiple training runs with increasing ceilings:
```bash
# Phase 1: 0-20% noise
--gan_noise_ceiling 0.2 --epochs 50

# Phase 2: Resume from checkpoint, 20-40% noise
--resume checkpoints_GAN/checkpoint_epoch_50.pt --gan_noise_ceiling 0.4 --epochs 100

# Phase 3: Resume, 40-100% noise
--resume checkpoints_GAN/checkpoint_epoch_100.pt --gan_noise_ceiling 1.0 --epochs 150
```

### 3. Ablation Studies
Test model performance at different fixed noise levels:
```bash
# Test at 25% noise
--gan_noise_ceiling 0.25 --pure_gan_mode 100.0 --epochs 50  # Reaches 25% instantly, stays there

# Test at 50% noise
--gan_noise_ceiling 0.50 --pure_gan_mode 100.0 --epochs 50

# Test at 75% noise
--gan_noise_ceiling 0.75 --pure_gan_mode 100.0 --epochs 50
```

## Notes
- Ceiling is checked AFTER computing `pure_gan_mode * counter`, so it only limits the maximum
- Setting ceiling to 1.0 (default) preserves original behavior
- Ceiling applies to both input AND target noise (they always have same alpha)
- Useful for preventing overfitting to pure noise if you want model to handle both clean and noisy inputs
