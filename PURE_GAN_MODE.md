# Pure GAN Mode: Curriculum Learning Documentation

## Overview

**Pure GAN Mode** is a curriculum learning technique that gradually transitions the model from **music-to-music transformation** to **noise-to-music generation** (like traditional GANs).

## Motivation

### Problem
- Standard training: Model learns to transform music → music
- Limitation: Model requires musical input, cannot generate from scratch
- Goal: Enable pure generation (noise → music) like DALL-E, Stable Diffusion

### Solution
Gradually replace real music with noise during training:
- **Phase 1**: Music-to-music (epochs 1-50)
- **Phase 2**: Transition period (epochs 51-150)
- **Phase 3**: Noise-to-music (epochs 150+)

## How It Works

### Algorithm

```python
counter = epoch - gan_curriculum_start_epoch + 1
alpha = min(1.0, pure_gan_mode × counter)

if alpha > 0:
    input_noise = randn_like(input) × std(input)
    target_noise = randn_like(target) × std(target)
    
    # Interpolate: (1-α) × real + α × noise
    new_input = (1 - alpha) × real_input + alpha × input_noise
    new_target = (1 - alpha) × real_target + alpha × target_noise
```

### Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `--pure_gan_mode` | float | Transition rate (0.0=disabled) | `0.01` = 1% noise added per epoch |
| `--gan_curriculum_start_epoch` | int | Epoch to start transition | `51` = start after 50 epochs of standard training |

### Example Timeline

**Configuration**: `pure_gan_mode=0.01`, `start_epoch=51`

| Epoch | Counter | Alpha | Input Composition | Target Composition |
|-------|---------|-------|-------------------|-------------------|
| 1-50  | 0       | 0.00  | 100% music + 0% noise | 100% music + 0% noise |
| 51    | 1       | 0.01  | 99% music + 1% noise  | 99% music + 1% noise  |
| 75    | 25      | 0.25  | 75% music + 25% noise | 75% music + 25% noise |
| 100   | 50      | 0.50  | 50% music + 50% noise | 50% music + 50% noise |
| 125   | 75      | 0.75  | 25% music + 75% noise | 25% music + 75% noise |
| 150   | 100     | 1.00  | 0% music + 100% noise | 0% music + 100% noise |
| 151+  | 101+    | 1.00  | 100% noise (capped)   | 100% noise (capped)   |

## Implementation Details

### Where Noise is Applied

**Transformer Input** (sees noisy data):
```python
# Stage 0: concat(noisy_input, noisy_target)
# Stage 1: concat(noisy_input, prev_output, noisy_target)
```

**Loss Computation** (uses clean data):
```python
# IMPORTANT: Loss always compares to original clean audio
loss = combined_loss(output_audio, original_input, original_target)
```

This ensures:
- Model learns from increasingly noisy inputs
- Loss measures quality against real audio
- No "cheating" by matching noise to noise

### Noise Characteristics

**Gaussian noise** (per-sample):
```python
# inputs: [B, D, T] - batch of encoded audio
input_std = inputs.std(dim=(1, 2), keepdim=True)  # [B, 1, 1] - std per sample
noise = randn_like(inputs) * input_std  # Different scale per sample
```

**Properties**:
- Zero mean (doesn't shift values)
- **Per-sample scaling**: Each sample gets noise scaled to its own std
  - Loud sample (std=8.0) → noise std=8.0
  - Quiet sample (std=3.0) → noise std=3.0
- Independent per sample (different noise each batch)
- Generated fresh each epoch (no memorization)
- Adaptive SNR: Signal-to-noise ratio consistent across samples

### Code Locations

**train_simple_worker.py**:
- Lines 79-87: Function signature with `pure_gan_mode`, `gan_curriculum_counter`
- Lines 159-192: Noise interpolation before model forward pass
- Lines 589-596: Loss computation using `original_targets`

**train_simple_ddp.py**:
- Lines 123-127: Argument parser for `--pure_gan_mode`, `--gan_curriculum_start_epoch`

**Training loop**:
- Lines 950-954: Counter increment based on start epoch

## Usage Examples

### Example 1: Basic Pure GAN Training

```bash
python train_simple_ddp.py \
  --train_dir dataset_pairs_wav/train \
  --val_dir dataset_pairs_wav/val \
  --checkpoint_dir checkpoints_pure_gan \
  --epochs 200 \
  --pure_gan_mode 0.01 \
  --gan_curriculum_start_epoch 51 \
  --gan_weight 0.15  # Enable adversarial training
```

**Effect**:
- Epochs 1-50: Standard music-to-music
- Epochs 51-150: Transition (1% noise per epoch)
- Epochs 151-200: Pure noise generation

### Example 2: Aggressive Transition

```bash
--pure_gan_mode 0.02 \
--gan_curriculum_start_epoch 26
```

**Effect**:
- Transition complete by epoch 76 (50 epochs × 0.02)
- More challenging but faster adaptation

### Example 3: Conservative Transition

```bash
--pure_gan_mode 0.005 \
--gan_curriculum_start_epoch 101
```

**Effect**:
- Transition complete by epoch 301 (200 epochs × 0.005)
- Gentle adaptation, more stable

### Example 4: Start from Checkpoint

```bash
python train_simple_ddp.py \
  --resume checkpoints/checkpoint_epoch_50.pt \
  --pure_gan_mode 0.01 \
  --gan_curriculum_start_epoch 51 \
  --epochs 200
```

**Effect**:
- Resumes from epoch 50
- Starts noise injection at epoch 51
- Counter = epoch - 51 + 1

## Training Recommendations

### Phase 1: Pre-training (Epochs 1-50)

**Goal**: Learn basic music-to-music transformation

**Settings**:
```bash
--loss_weight_input 0.3 \
--loss_weight_target 0.3 \
--gan_weight 0.15 \
--mask_reg_weight 0.5 \
--balance_loss_weight 15.0
```

**Expected**:
- Complementarity: 75% → 90%
- Val loss: 0.65 → 0.50
- Stable gradient norms (3-8)

### Phase 2: Transition (Epochs 51-150)

**Goal**: Gradually adapt to noisy inputs

**Monitor**:
- Alpha increasing: 0.0 → 1.0
- Val loss: Should stay within 10-20% of phase 1
- Discriminator accuracy: ~85% (balanced)

**Warning Signs**:
- Val loss exploding (>50% increase) → reduce `pure_gan_mode`
- Discriminator collapse (<60% accuracy) → increase `gan_weight`
- Mode collapse (all outputs similar) → increase diversity losses

### Phase 3: Pure Generation (Epochs 150+)

**Goal**: Generate music from pure noise

**Expected**:
- Model receives 100% noise input
- Output should be coherent music (not noise)
- Val loss stabilizes at new baseline

**Evaluation**:
- Listen to generated samples
- Check spectrograms (should show musical structure)
- Measure audio quality metrics (SNR, perceptual)

## Debugging

### Issue: Val loss explodes during transition

**Cause**: Transition too fast, model can't adapt

**Solution**:
```bash
--pure_gan_mode 0.005  # Slower transition
# or
--gan_curriculum_start_epoch 101  # More pre-training
```

### Issue: Generated audio is still noisy

**Cause**: Model hasn't fully learned noise-to-music mapping

**Solution**:
- Train longer (more epochs at 100% noise)
- Increase GAN weight: `--gan_weight 0.2`
- Check discriminator is training (not collapsed)

### Issue: Mode collapse (all outputs similar)

**Cause**: Model found easy solution (generate one "safe" output)

**Solution**:
```bash
--shuffle_targets  # Enable target shuffling
--mask_reg_weight 1.0  # Stronger diversity enforcement
--balance_loss_weight 20.0  # Force different mixing ratios
```

### Issue: Training unstable after epoch 100

**Cause**: Pure noise is very different from music, large domain shift

**Solution**:
- Pre-train longer: `--gan_curriculum_start_epoch 101`
- Slower transition: `--pure_gan_mode 0.005`
- Keep some music: Cap alpha at 0.95 instead of 1.0 (requires code edit)

## Comparison with Standard Training

| Aspect | Standard Training | Pure GAN Mode |
|--------|-------------------|---------------|
| **Input** | Real music | Music → Noise (gradual) |
| **Output** | Transform music | Generate from noise |
| **Use case** | Remixing, continuation | Free generation |
| **Difficulty** | Easier (music→music similar) | Harder (noise→music very different) |
| **Training time** | 50-100 epochs | 150-200 epochs |
| **Quality** | Higher (constrained by input) | Variable (unconstrained) |
| **Creativity** | Limited (based on input) | Unlimited (from scratch) |

## Theory: Why This Works

### Curriculum Learning
- **Easy→Hard**: Start with similar domains (music→music), gradually increase difficulty
- **Smooth transition**: Continuous interpolation prevents abrupt distribution shift
- **Stable gradients**: Model adapts incrementally, no sudden gradient explosions

### Noise as Latent Space
- **Gaussian noise**: Unbiased starting point (no structure)
- **Encoded space**: EnCodec latent (128-dim) is structured, not raw audio
- **Learned mapping**: Transformer learns noise → meaningful latents → music

### Loss on Clean Targets
- **Ground truth**: Loss measures output quality vs real music
- **Not matching noise**: Avoids trivial solution (output = noisy input)
- **Quality metric**: Ensures generated audio is musical, not arbitrary

## Future Enhancements

### 1. Conditional Generation
Add conditioning vectors (genre, tempo, mood):
```python
condition_vec = encode_genre(genre)  # e.g., "jazz", "rock"
new_input = cat([noise, condition_vec], dim=1)
```

### 2. Partial Noise
Replace only part of input (e.g., rhythm from noise, harmony from music):
```python
input_rhythm = noise[:, :64, :]  # First 64 channels
input_harmony = music[:, 64:, :]  # Last 64 channels
new_input = cat([input_rhythm, input_harmony], dim=1)
```

### 3. Adaptive Alpha
Adjust transition speed based on val loss:
```python
if val_loss > threshold:
    alpha *= 0.9  # Slow down
else:
    alpha *= 1.1  # Speed up
```

### 4. Temperature Sampling
Control diversity during inference:
```python
noise = randn_like(input) * temperature  # Higher = more diverse
```

## References

- **Curriculum Learning**: Bengio et al. (2009) - Start with easy examples, gradually increase difficulty
- **Progressive Growing**: Karras et al. (2017) - Gradual resolution increase in GANs
- **Scheduled Sampling**: Bengio et al. (2015) - Mix model predictions with ground truth
- **Denoising Diffusion**: Ho et al. (2020) - Train to denoise gradually noisier inputs

## Summary

Pure GAN mode enables **free generation** from noise through **curriculum learning**:
1. Pre-train on music-to-music (epochs 1-50)
2. Gradually transition to noise (epochs 51-150)
3. Generate from pure noise (epochs 150+)

Key insight: **Loss always uses clean targets**, ensuring output quality while input becomes increasingly noisy. Model learns robust noise→music mapping applicable to generation tasks.
