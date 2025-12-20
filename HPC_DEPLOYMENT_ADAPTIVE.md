# HPC Deployment Guide - Adaptive Window Selection Training

**Date**: December 20, 2025  
**Feature**: Adaptive Window Selection with Creative Agent  
**Status**: Ready for deployment after successful local testing

---

## 🎯 Overview

This guide covers deploying the **AdaptiveWindowCreativeAgent** training to the HPC cluster. The system extends audio processing from 16-second to 24-second segments, allowing the creative agent to intelligently select and process 3 pairs of 16-second windows with temporal compression and tonality transformation.

### Key Innovation
- **Input**: 24-second audio segments  
- **Processing**: Agent selects 3 optimal 16-second window pairs  
- **Output**: Enhanced creative diversity through multi-window analysis  
- **Gradient Flow**: Mean loss across 3 pairs ensures smooth training

---

## ✅ Pre-Deployment Checklist

### Local Testing Results (December 20, 2025)
- ✅ Synthetic data test: PASSED
- ✅ Real audio test: PASSED (pair_0000 from validation set)
- ✅ Training test (2 epochs): **PASSED**
  - Epoch 1: Train 0.0050 → Val 0.0044
  - Epoch 2: Train 0.0015 → Val 0.0028
  - Loss decreasing properly ✅
  - Window selection diverse ✅
  - No memory errors ✅

### Code Status
- ✅ Branch: `adaptive-window-selection` (4 commits ahead of main)
- ✅ All files committed and pushed to GitHub
- ✅ Documentation complete

### Dataset Requirements
- **Format**: 24-second stereo WAV pairs at 24kHz
- **Expected size**: ~5,375 training pairs + ~280 validation pairs
- **Location**: `dataset_pairs_wav_24sec/`
- **Creation**: Use `create_dataset_pairs_wav.py`

---

## 📦 Step 1: Prepare Files for HPC

### 1.1 SSH to HPC Cluster
```bash
ssh g260141@levante.dkrz.de
```

### 1.2 Navigate to Project Directory
```bash
cd /work/gg0302/g260141/Jingle_D
```

### 1.3 Pull Latest Code
```bash
# Switch to adaptive window branch
git fetch origin
git checkout adaptive-window-selection
git pull origin adaptive-window-selection

# Verify files
ls -la adaptive_window_agent.py
ls -la train_adaptive_simple.py
ls -la submit_adaptive_hpc.sh
```

### 1.4 Create Dataset on HPC
```bash
# Activate environment
source /work/gg0302/g260141/miniforge3/bin/activate
conda activate /work/gg0302/g260141/conda/powermusic

# Create 24-second pairs dataset
python create_dataset_pairs_wav.py

# Verify dataset
ls dataset_pairs_wav_24sec/train/pair_*_input.wav | wc -l  # Should be ~5,100
ls dataset_pairs_wav_24sec/val/pair_*_input.wav | wc -l    # Should be ~270
```

---

## 🚀 Step 2: Launch Training

### 2.1 Make Submission Script Executable
```bash
chmod +x submit_adaptive_hpc.sh
```

### 2.2 Submit Job
```bash
# Submit to SLURM
sbatch submit_adaptive_hpc.sh

# Check job status
squeue -u g260141

# Monitor output (job ID will be shown after submission)
tail -f logs/adaptive_train_<JOB_ID>.out
```

### 2.3 Expected Output
```
==================================
ADAPTIVE WINDOW TRAINING - START
==================================
Job ID: 1234567
Node: levante-node-042
Start time: Fri Dec 20 15:30:00 UTC 2025
==================================

Environment:
  Python: /work/gg0302/g260141/miniforge3/envs/powermusic/bin/python
  PyTorch: 2.2.2
  CUDA available: True
  GPUs: 4

Verifying dataset...
  Train pairs: 5104
  Val pairs: 271

==================================
STARTING TRAINING
==================================

================================================================================
TRAINING: ADAPTIVE WINDOW SELECTION + COMPOSITIONAL AGENT
================================================================================

Configuration:
  Data folder: dataset_pairs_wav_24sec
  Batch size: 6
  Epochs: 50
  Learning rate: 0.0001
  Novelty weight: 0.1
  Device: cuda
  Number of pairs: 3

Loading datasets...
AudioPairsDataset24sec (train):
  Number of pairs: 5104
  Sample rate: 24000 Hz
  Duration: 24 seconds per file
AudioPairsDataset24sec (val):
  Number of pairs: 271
  ...
```

---

## 📊 Step 3: Monitor Training

### 3.1 Real-time Monitoring
```bash
# Watch training progress
watch -n 10 'tail -30 logs/adaptive_train_<JOB_ID>.out'

# Check GPU utilization
ssh levante-node-042  # Replace with actual node
nvidia-smi -l 5
```

### 3.2 Expected Metrics

**Epoch 1-5** (Initial learning):
- Train loss: 0.015 → 0.008
- Val loss: 0.013 → 0.007
- Window positions: Gradually diversifying
- Compression ratios: Learning optimal ranges

**Epoch 10-25** (Refinement):
- Train loss: 0.008 → 0.004
- Val loss: 0.007 → 0.003
- Window selection: Distinct patterns emerging
- Novelty loss: Stabilizing around 0.03-0.05

**Epoch 30-50** (Fine-tuning):
- Train loss: 0.004 → 0.002
- Val loss: 0.003 → 0.002
- Overfitting watch: Val should track train closely
- Best model: Saved based on lowest val loss

### 3.3 Key Indicators of Healthy Training

✅ **Good Signs**:
- Loss decreasing smoothly
- Val loss tracking train loss (gap < 0.001)
- Window start positions varying (not stuck at same location)
- Compression ratios between 1.0-1.5x
- Tonality strength between 0.0-1.0
- No NaN or Inf values
- GPU utilization 90-95%

⚠️ **Warning Signs**:
- Val loss diverging from train loss (overfitting)
- All windows selecting same position (selection collapsed)
- Compression ratios at extremes (1.0 or 1.5 only)
- Loss oscillating wildly
- GPU utilization < 70% (bottleneck)

---

## 🔍 Step 4: Training Analysis

### 4.1 Check Validation Performance
```bash
# Parse training log for validation results
grep "Val Loss:" logs/adaptive_train_<JOB_ID>.out
```

### 4.2 Inspect Checkpoints
```bash
# List saved checkpoints
ls -lh checkpoints_adaptive/

# Should see:
# - checkpoint_epoch_N.pt (regular checkpoints)
# - best_model.pt (best validation loss)
# - checkpoint_latest.pt (most recent)
```

### 4.3 Load and Test Model
```python
import torch
from adaptive_window_agent import AdaptiveWindowCreativeAgent

# Load best model
checkpoint = torch.load('checkpoints_adaptive/best_model.pt')
agent = AdaptiveWindowCreativeAgent()
agent.load_state_dict(checkpoint['model_state_dict'])
agent.eval()

print(f"Best epoch: {checkpoint['epoch']}")
print(f"Best val loss: {checkpoint['val_loss']:.6f}")
print(f"Training time: {checkpoint.get('training_time', 'N/A')}")
```

---

## 🎵 Step 5: Generate Test Outputs

### 5.1 Run Inference
```bash
# Generate outputs with trained model
python inference_adaptive.py \
    --checkpoint checkpoints_adaptive/best_model.pt \
    --input test_input.wav \
    --output test_output.wav \
    --num_pairs 3
```

### 5.2 Analyze Window Selection
```python
# Check which windows the agent selected
python analyze_window_selection.py \
    --checkpoint checkpoints_adaptive/best_model.pt \
    --dataset dataset_pairs_wav_24sec/val \
    --num_samples 10
```

---

## 📈 Step 6: Compare with Baseline

### 6.1 Metrics to Compare

| Metric | Baseline (16-sec fixed) | Adaptive (24-sec, 3 windows) |
|--------|-------------------------|------------------------------|
| Val Loss | ~0.003 | **Target: < 0.002** |
| Novelty | ~0.04 | **Target: 0.03-0.05** |
| Diversity | Single window | **3x windows** |
| Coverage | 16 seconds | **24 seconds** |

### 6.2 Qualitative Assessment
- **Rhythmic coherence**: Does output maintain input rhythm?
- **Harmonic complexity**: Is harmony enriched vs baseline?
- **Timbral diversity**: More varied textures?
- **Creative surprise**: Novel patterns while staying musical?

---

## 🐛 Troubleshooting

### Issue 1: Out of Memory
```bash
# Reduce batch size in submit script
--batch_size 4  # Instead of 6
```

### Issue 2: Dataset Loading Slow
```bash
# Increase workers
--num_workers 16  # Instead of 8
```

### Issue 3: All Windows Select Same Position
```
# This indicates collapsed selection - add position regularization
# Edit train_adaptive_simple.py to include diversity loss
```

### Issue 4: Training Stalled
```bash
# Check if job is still running
squeue -u g260141

# If hung, cancel and restart
scancel <JOB_ID>
sbatch submit_adaptive_hpc.sh
```

---

## 📝 Expected Timeline

| Phase | Duration | Milestone |
|-------|----------|-----------|
| Data transfer | 30 min | Dataset on HPC |
| Job queue | 5-60 min | Job starts |
| Epoch 1-10 | 3-5 hours | Initial convergence |
| Epoch 11-30 | 9-15 hours | Refinement |
| Epoch 31-50 | 9-15 hours | Fine-tuning |
| **Total** | **24-36 hours** | **Training complete** |

---

## ✨ Post-Training Steps

### 1. Download Best Model
```bash
# From local machine
scp g260141@levante.dkrz.de:/work/gg0302/g260141/Jingle_D/checkpoints_adaptive/best_model.pt \
    checkpoints_adaptive/
```

### 2. Update Documentation
```bash
# Document results in project
echo "Adaptive training completed: $(date)" >> TRAINING_LOG.md
echo "Best val loss: $(grep 'Best validation' logs/adaptive_train_*.out | tail -1)" >> TRAINING_LOG.md
```

### 3. Merge to Main Branch
```bash
# If training successful
git checkout main
git merge adaptive-window-selection
git push origin main
```

### 4. Deploy to Production
```bash
# Update AWS backend with new model
./deploy_adaptive_to_aws.sh
```

---

## 🎯 Success Criteria

Training is considered **successful** if:

✅ **Quantitative**:
- Final val loss < 0.002 (better than baseline)
- Novelty loss stable at 0.03-0.05
- No NaN/Inf values throughout training
- Window positions show diversity (std > 50 frames)

✅ **Qualitative**:
- Generated audio maintains musical coherence
- Clear improvement over single-window baseline
- Creative agent explores diverse patterns
- No audio artifacts or glitches

✅ **Technical**:
- Training completes all 50 epochs
- Checkpoints saved successfully
- GPU utilization > 90%
- No memory errors

---

## 📚 Files Reference

**Core Implementation**:
- `adaptive_window_agent.py` - AdaptiveWindowCreativeAgent class
- `train_adaptive_simple.py` - Training script
- `dataset_wav_pairs_24sec.py` - 24-second data loader

**Deployment**:
- `submit_adaptive_hpc.sh` - SLURM submission script
- `create_dataset_pairs_wav.py` - Dataset creation

**Documentation**:
- `ADAPTIVE_WINDOW_SELECTION.md` - Feature specification
- `HPC_DEPLOYMENT_ADAPTIVE.md` - This file
- `DEPLOYMENT_ADAPTIVE.md` - Training details

---

## 🚦 Ready to Deploy?

Final checklist before submission:

- [ ] Code pushed to GitHub (adaptive-window-selection branch)
- [ ] SSH access to HPC verified
- [ ] Dataset created on HPC (~5,375 pairs)
- [ ] Conda environment activated
- [ ] submit_adaptive_hpc.sh is executable
- [ ] Sufficient disk quota for checkpoints (~50GB)
- [ ] Email notifications configured in SLURM script

**If all checked** → `sbatch submit_adaptive_hpc.sh` 🚀

---

**Questions?** Check existing documentation or contact project maintainer.

**Training started**: [To be filled after submission]  
**Expected completion**: [24-36 hours after start]  
**Best model checkpoint**: `checkpoints_adaptive/best_model.pt`
