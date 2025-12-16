# Jingle_D: Creative Audio Mixing with Cascade Transformers

A deep learning system for creative audio mixing using a 2-stage cascade transformer with attention-based masking.

## Overview

This project implements a neural audio continuation system that creatively mixes input audio with target characteristics. The model learns to separate and blend different musical components (rhythm, harmony, melody) using learnable complementary masks.

### Key Features

- **2-Stage Cascade Architecture**: Fixed gradient-stable design (24.9M params)
- **Creative Agent**: Learnable attention-based masking for intelligent audio blending
- **Complementary Masking**: Separates input rhythm from target harmony/melody
- **EnCodec Integration**: 24kHz audio encoding with frozen encoder/decoder
- **Distributed Training**: DDP support for multi-GPU training (4x A100)
- **GAN Training**: Adversarial discriminator for realistic audio generation

## Architecture

```
Input Audio → EnCodec Encoder → [Stage 0] → [Stage 1] → EnCodec Decoder → Output
                                     ↓            ↓
                              Creative Agent (Learnable Masks)
                                     ↓            ↓
                              Input Mask    Target Mask
                              (rhythm)      (harmony/melody)
```

### Model Components

1. **Cascade Stages**: 2 transformer stages with progressive refinement
   - Stage 0: 256 dim, 1024 FFN, 6 layers
   - Stage 1: 384 dim, 1536 FFN, 6 layers

2. **Creative Agent**: ~700K parameters
   - Mask generator with cross-attention
   - Input/target analyzers
   - Complementary mask generation

3. **Audio Discriminator**: ~3.8M parameters
   - Multi-scale feature extraction
   - Real/fake classification

## Training

### Current Configuration

- **Epochs**: 51/200 completed
- **Batch Size**: 32 (8 per GPU × 4 GPUs)
- **Learning Rate**: 1e-4 (AdamW)
- **Loss Weights**:
  - Input RMS: 0.3
  - Target RMS: 0.3
  - Mask regularization: 0.1 (being increased to 0.5)
  - Balance loss: 15.0
  - GAN: 0.15
  - Correlation penalty: 0.5

### Training Status

- **Validation Loss**: 0.527 (epoch 51)
- **Complementarity**: 74% (target: 90%+)
- **Mask Balance**: 50/50 ✅
- **Gradient Norms**: 3-8 (stable, no explosion)
- **Discriminator Accuracy**: 84% (balanced)

## Scripts

### Training Scripts

- `train_simple_worker.py`: DDP training worker (core training loop)
- `train_simple_ddp.py`: Multi-GPU launcher
- `run_train_creative_agent_fixed.sh`: Main training launch script
- `run_train_creative_agent_resume.sh`: Resume from checkpoint
- `run_train_creative_agent_push_complementarity.sh`: Training with stronger complementarity pressure

### Model Files

- `model_simple_transformer.py`: 2-stage cascade transformer
- `creative_agent.py`: Attention-based mask generator
- `audio_discriminator.py`: Adversarial discriminator
- `correlation_penalty.py`: Anti-modulation loss

### Inference

- `inference_cascade.py`: Generate audio from trained model
- `inference_sample.py`: Single sample inference

### Utilities

- `sync_to_remote.sh`: Sync code to SLURM cluster
- `verify_sync.sh`: Verify remote sync status
- `view_mlflow_data.py`: View training metrics locally

## Dataset

Uses paired WAV audio files:
- **Train**: 7,200 pairs (16 seconds each, 24kHz)
- **Validation**: 800 pairs
- **Location**: `dataset_pairs_wav/train` and `dataset_pairs_wav/val`

## Requirements

### Python Environment

```bash
conda create -n multipattern_env python=3.10
conda activate multipattern_env
pip install torch torchaudio encodec mlflow tqdm
```

### Hardware

- **Local**: Apple Silicon Mac (CPU inference only)
- **Remote**: SLURM cluster with 4x NVIDIA A100-80GB GPUs
  - NCCL 2.21.5
  - CUDA 11.0+

## Quick Start

### Resume Training

```bash
# On remote server (levante.dkrz.de)
cd /work/gg0302/g260141/Jingle_D
bash run_train_creative_agent_resume.sh
```

### Push Complementarity Higher

```bash
# Increase mask regularization weight for stronger separation
bash run_train_creative_agent_push_complementarity.sh
```

### Run Inference

```bash
python3 inference_cascade.py --shuffle_targets
```

## Training Progress

### Epoch 1-20 (Initial Training)
- Complementarity: 75% → 87%
- Validation loss: 0.65 → 0.49
- Architecture: Stable gradient flow

### Epoch 21-51 (Resume)
- Complementarity: **Stuck at 74%** ⚠️
- Validation loss: 0.57 → 0.53
- Issue: Mask regularization weight too weak

### Next Steps
- Increase `mask_reg_weight` from 0.1 → 0.5
- Push complementarity to 90%+
- Continue to epoch 200

## Key Fixes Applied

1. **Gradient Explosion Fix**: Removed problematic stage 2, reduced to 2-stage cascade
2. **DDP Timeout Fix**: Fixed MASTER_PORT randomization causing worker join failures
3. **Configuration Sync**: Aligned resume script with current training hyperparameters
4. **Spectral Normalization**: Added to output projections for gradient stability

## Documentation

- `SESSION_CHECKPOINT.md`: Detailed training history and architecture changes
- `GRADIENT_EXPLOSION_FIX.md`: Analysis of gradient stability improvements
- `README_CREATIVE_AGENT.md`: Creative agent design and implementation
- `README_LEVANTE.md`: SLURM cluster setup and usage

## License

Research project - refer to institution policies for usage terms.

## Authors

Training conducted on DKRZ Levante supercomputer (g260141@levante.dkrz.de)
