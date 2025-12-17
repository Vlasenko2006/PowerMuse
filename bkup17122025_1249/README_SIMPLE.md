# Simple Transformer for Audio Continuation (Jingle_D)

## Overview

This is a simplified approach to music generation using:
- **WAV pairs**: Consecutive 16-second segments (no triplets)
- **EnCodec**: Frozen encoder (→128 dims) and decoder
- **Simple Transformer**: 8 heads, 4 layers, ~860K parameters
- **MSE Loss**: In encoded space only
- **No complexity**: No adversarial loss, no multi-stage cascade

## Architecture

```
Input WAV (16s) → EnCodec Encoder (frozen) → [128, ~1200]
                                              ↓
                                    SimpleTransformer (trainable)
                                              ↓
Output WAV (next 16s) ← EnCodec Decoder (frozen) ← [128, ~1200]
```

## Files Created

### Dataset Creation
- `create_dataset_pairs_wav.py` - Creates WAV pairs from audio files
- `create_dataset_pairs_levante.sh` - SLURM script to run on HPC
- `dataset_wav_pairs.py` - PyTorch dataset loader with on-the-fly encoding

### Model
- `model_simple_transformer.py` - Simple transformer architecture

### Training
- `train_simple_ddp.py` - DDP training entry point
- `train_simple_worker.py` - Training worker for each GPU
- `run_train_simple.sh` - SLURM script for DDP training (4 GPUs)

### Testing
- `test_simple_setup.py` - Verify model, EnCodec, loss, optimizer

## Configuration

### Dataset
- **Segment duration**: 16 seconds
- **Sample rate**: 24000 Hz
- **Format**: WAV (mono, converted from stereo)
- **Pairs**: Consecutive segments (input, output)
- **Max pairs**: 2000 (1800 train, 200 val)

### Model
- **Encoding dim**: 128 (from EnCodec)
- **Attention heads**: 8
- **Transformer layers**: 4
- **Feedforward dim**: 512 (4 × encoding_dim)
- **Dropout**: 0.1
- **Parameters**: 859,520 (trainable)

### Training
- **Loss**: MSE in encoded space
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Batch size**: 16 per GPU × 4 GPUs = 64 total
- **Epochs**: 200 (with early stopping, patience=20)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Gradient clipping**: max_norm=1.0

### EnCodec
- **Sample rate**: 24000 Hz
- **Bandwidth**: 6.0 (high quality)
- **Status**: FROZEN (encoder + decoder)
- **Encoder output**: [B, 128, T] continuous embeddings
- **Timesteps**: ~1200 for 16s audio

## Usage

### 1. Create Dataset
```bash
# Local test
cd /Users/andreyvlasenko/tst/Jingle_D
python create_dataset_pairs_wav.py

# On HPC
cd /work/gg0302/g260141/Jingle_D
sbatch create_dataset_pairs_levante.sh
```

### 2. Test Setup
```bash
python test_simple_setup.py
```

### 3. Train Model
```bash
# On HPC (4 GPUs)
sbatch run_train_simple.sh

# Monitor
tail -f logs/train_simple.out
```

### 4. Check Results
```bash
ls -lh checkpoints_simple/
# best_model.pt - best validation loss
# checkpoint_epoch_*.pt - regular checkpoints every 10 epochs
```

## Expected Output Structure

```
dataset_pairs_wav/
  train/
    pair_0000_input.wav
    pair_0000_output.wav
    ...
  val/
    pair_XXXX_input.wav
    pair_XXXX_output.wav
  metadata.txt

checkpoints_simple/
  best_model.pt
  checkpoint_epoch_10.pt
  checkpoint_epoch_20.pt
  ...

logs/
  dataset_pairs.{out,err}
  train_simple.{out,err}
```

## Key Differences from Jingle_T

| Feature | Jingle_T (Complex) | Jingle_D (Simple) |
|---------|-------------------|-------------------|
| Input | 3 patterns (triplets) | 1 pattern (pairs) |
| Architecture | TransformerCascade (multi-stage) | SimpleTransformer (single-stage) |
| Loss | Enhanced (6 components + GAN) | MSE only |
| Parameters | ~14.6M (3 stages) | ~860K (1 stage) |
| Dataset | Pre-encoded .npy | WAV files (on-the-fly) |
| Encoding | Full EnCodec encode/decode | Encoder embeddings only |

## Rationale

The Jingle_T approach with EnCodec quantization and complex losses produced poor audio quality despite good loss convergence. This simpler approach:

1. **Uses continuous embeddings** instead of quantized codes
2. **Single pattern pairs** instead of triplet fusion
3. **MSE loss only** to test if the concept works
4. **Smaller model** for faster iteration

If this works, we can add complexity later. If not, we know the fundamental approach needs rethinking.

## Next Steps

1. ✅ Dataset created
2. ✅ Model and training scripts ready
3. ⏳ Submit dataset creation job
4. ⏳ Submit training job
5. ⏳ Evaluate results
6. ⏳ If successful, add complexity (more layers, GAN, etc.)

## Notes

- **Stereo→Mono**: WAV files converted to mono by averaging channels
- **Frozen EnCodec**: Only transformer is trainable
- **DDP**: Uses 4 GPUs with DistributedDataParallel
- **Early stopping**: Stops if no improvement for 20 epochs
- **Checkpoints**: Saved every 10 epochs + best model
