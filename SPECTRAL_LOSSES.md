# Spectral Loss Functions - Implementation Guide

## Overview

Multi-resolution STFT and mel-spectrogram losses have been added to complement the existing RMS losses. These perceptual losses capture time-frequency structure, which is important for rhythm and melody quality.

## New Loss Components

### 1. **Multi-Resolution STFT Loss** (`stft_loss`)
Based on Parallel WaveGAN and HiFi-GAN papers.

**What it captures:**
- Spectral convergence: overall frequency distribution similarity
- Log magnitude distance: perceptual similarity across frequencies
- Multi-scale analysis: uses 3 FFT sizes (512, 1024, 2048) to capture both fine and coarse temporal structures

**When to use:**
- Captures rhythm (temporal structure in spectrograms)
- Captures melody (frequency content over time)
- More perceptually meaningful than raw waveform RMS

**Typical weight:** `0.1` to `1.0` (start with 0.5)

### 2. **Mel-Spectrogram Loss** (`mel_loss`)
Perceptual frequency representation matching human hearing.

**What it captures:**
- Mel-scale frequency distribution (matches human perception)
- Harmonic content and timbre
- Overall tonal quality

**When to use:**
- Encourage perceptually similar outputs
- Complement STFT loss with mel-scale weighting
- Useful for musical content

**Typical weight:** `0.1` to `1.0` (start with 0.5)

### 3. **RMS Losses** (existing)
- `rms_input`: encourages output similar to input (reconstruction)
- `rms_target`: encourages output similar to target (continuation)

## Usage Examples

### Example 1: Pure Continuation (Default)
```bash
python train_simple_ddp.py \
    --loss_weight_input 0.0 \
    --loss_weight_target 1.0 \
    --loss_weight_spectral 0.0 \
    --loss_weight_mel 0.0
```
**Use case:** Basic continuation training with RMS loss only

### Example 2: Continuation + Spectral Loss
```bash
python train_simple_ddp.py \
    --loss_weight_input 0.0 \
    --loss_weight_target 0.5 \
    --loss_weight_spectral 0.5 \
    --loss_weight_mel 0.0
```
**Use case:** Encourage spectral similarity for better rhythm/melody transfer
**Recommended:** Good starting point for perceptual training

### Example 3: All Losses Balanced
```bash
python train_simple_ddp.py \
    --loss_weight_input 0.0 \
    --loss_weight_target 0.4 \
    --loss_weight_spectral 0.3 \
    --loss_weight_mel 0.3
```
**Use case:** Maximum perceptual quality with all components

### Example 4: Reconstruction + Continuation + Spectral
```bash
python train_simple_ddp.py \
    --loss_weight_input 0.3 \
    --loss_weight_target 0.4 \
    --loss_weight_spectral 0.3 \
    --loss_weight_mel 0.0
```
**Use case:** Encourage preservation of input character while continuing

## Training Output

The training progress now shows all loss components:

```
Epoch 1/200:
  Train Loss: 0.5234 (rms_in: 0.1234, rms_tgt: 0.2234, spec: 1.2345, mel: 0.8765)
  Val Loss:   0.4987 (rms_in: 0.1198, rms_tgt: 0.2145, spec: 1.1987, mel: 0.8321)
  LR: 1.00e-04
```

## MLflow Logging

All loss components are now logged to MLflow:
- `train_loss`, `train_rms_input`, `train_rms_target`, `train_spectral`, `train_mel`
- `val_loss`, `val_rms_input`, `val_rms_target`, `val_spectral`, `val_mel`
- `learning_rate`

## Implementation Details

### Loss Computation Flow
1. Model outputs encoded representation: `[B, 128, T]`
2. EnCodec decoder converts to audio: `[B, 1, samples]`
3. All losses computed in audio space (after decoding)
4. Spectral and mel losses computed against target only
5. RMS losses can be computed against both input and target

### FFT Parameters
- **STFT loss:** FFT sizes [512, 1024, 2048], hop sizes [128, 256, 512]
- **Mel loss:** 1024 FFT, 80 mel bins, 0-8kHz range

### Gradient Flow
All losses support backpropagation through:
- EnCodec decoder (frozen, but gradients flow through)
- Transformer model (trainable)

## Tips for Tuning

1. **Start simple:** Begin with just `--loss_weight_target 1.0` to establish baseline
2. **Add spectral gradually:** Try `--loss_weight_target 0.7 --loss_weight_spectral 0.3`
3. **Monitor individual components:** Check MLflow to see which loss dominates
4. **Balance scales:** If one loss is 10x larger, reduce its weight proportionally
5. **Spectral losses are larger:** STFT ~1-2, Mel ~0.5-1, RMS ~0.05-0.2 typically

## Research Background

### Key Papers
1. **Parallel WaveGAN** (Yamamoto et al., 2020)
   - Introduced multi-resolution STFT loss for audio synthesis
   - arXiv:1910.11480

2. **HiFi-GAN** (Kong et al., 2020)
   - State-of-the-art vocoder using spectral losses
   - Demonstrated importance of multi-scale analysis
   - arXiv:2010.05646

3. **JASCO** (Tal et al., 2024)
   - Text-to-music with melodic conditioning
   - Uses conditioning architecture, not special loss functions
   - arXiv:2406.10970

### Why Spectral Losses?
- **Rhythm:** Captured in temporal structure of spectrograms
- **Melody:** Visible as frequency patterns over time
- **Perceptual quality:** Better matches human hearing than raw waveforms
- **Proven effectiveness:** State-of-the-art audio synthesis models all use them

## Testing

Run the test suite to verify loss functions:
```bash
python test_spectral_losses.py
```

Expected output:
```
✓ RMS loss: works
✓ STFT loss: works  
✓ Mel loss: works
✓ Combined loss: works
✓ Gradients: flow correctly
```

## Files Modified

- `train_simple_worker.py`: Added loss functions, updated train/validate epochs
- `train_simple_ddp.py`: Added command-line arguments for new weights
- `run_train_simple.sh`: Added default values (all 0.0 for backward compatibility)
- `test_spectral_losses.py`: Test suite for loss functions

## Next Steps

1. **Run unity test with spectral losses:**
   ```bash
   --unity_test true --loss_weight_input 0.5 --loss_weight_spectral 0.5
   ```
   Should converge to ~0.02-0.03 RMS + spectral loss

2. **Train continuation with spectral:**
   ```bash
   --unity_test false --loss_weight_target 0.5 --loss_weight_spectral 0.5
   ```

3. **Ablation study:** Compare different weight combinations

4. **Evaluate:** Use inference_sample.py to generate audio and compare perceptual quality
