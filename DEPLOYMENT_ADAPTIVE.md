# Deployment Guide: Adaptive Window Selection

## ✅ What's Ready

**Branch**: `adaptive-window-selection`

**Files to Copy to HPC:**
```
adaptive_window_agent.py          # Main agent (18.2M params)
compositional_creative_agent.py   # Compositional agent (14.6M params, shared)
correlation_penalty.py             # Dependency
dataset_wav_pairs_24sec.py         # Dataset loader
train_adaptive_simple.py           # Training script
run_train_adaptive_hpc.sh          # HPC launch script
dataset_pairs_wav_24sec/           # Your 24-sec audio dataset
```

---

## 🚀 Quick Start

### Option A: Local Test First (Recommended)

```bash
# 1. Test dataset loader
python3 dataset_wav_pairs_24sec.py

# 2. Test with real audio (already passed!)
python3 test_adaptive_real_audio.py

# 3. Run 2-epoch training test
chmod +x run_train_adaptive_local.sh
./run_train_adaptive_local.sh
```

**Expected output:**
- Epoch 1-2 complete without errors
- Losses decreasing
- Window selection metadata displayed
- Checkpoint saved in `checkpoints_adaptive_test/`

**If successful → proceed to HPC!**

---

### Option B: Direct HPC Deployment

```bash
# On HPC (levante):
cd /work/gg0302/g260141/Jingle_D

# 1. Copy all files from local to HPC
scp -r adaptive_window_agent.py \
       compositional_creative_agent.py \
       correlation_penalty.py \
       dataset_wav_pairs_24sec.py \
       train_adaptive_simple.py \
       run_train_adaptive_hpc.sh \
       dataset_pairs_wav_24sec/ \
       g260141@levante.dkrz.de:/work/gg0302/g260141/Jingle_D/

# 2. Edit run_train_adaptive_hpc.sh
nano run_train_adaptive_hpc.sh
# Change: conda activate your_env_name → your actual environment

# 3. Submit job
sbatch run_train_adaptive_hpc.sh

# 4. Monitor
tail -f training/log_adaptive_*.txt
```

---

## 📊 Expected Training Behavior

### First Few Batches:
```
Epoch 1/100
  Batch 0: loss=0.0231, novelty=0.0231, pair0_start=177.8, pair0_ratio=1.19x
  Batch 1: loss=0.0245, novelty=0.0245, pair0_start=189.3, pair0_ratio=1.22x
  ...
```

### Epoch Summary:
```
Epoch 1 Results:
  Train Loss: 0.0240 (novelty: 0.0240)
  Val Loss: 0.0235 (novelty: 0.0235)
  Saved checkpoint: checkpoints_adaptive/checkpoint_epoch_1.pt
  ✓ New best model! Val loss: 0.0235
```

### After 10-20 Epochs:
- **Window diversity increasing**: Pairs select different regions
- **Compression usage**: Ratios vary (1.0x - 1.5x)
- **Losses decreasing**: From ~0.025 → ~0.015
- **Specialization emerging**: Each pair develops strategy

---

## 🔧 Configuration

### HPC Script (`run_train_adaptive_hpc.sh`):

**Must edit:**
```bash
conda activate your_env_name  # Line 36 - CHANGE THIS!
```

**May adjust:**
```bash
BATCH_SIZE=4          # Reduce if OOM (try 2 or 1)
NUM_WORKERS=8         # Adjust for your CPU count
EPOCHS=100            # Training duration
LR=1e-4               # Learning rate
NOVELTY_WEIGHT=0.1    # Creativity vs quality balance
```

### Local Script (`run_train_adaptive_local.sh`):

Already configured for testing (batch=1, epochs=2, cpu)

---

## 📝 Monitoring Tips

### Check Job Status:
```bash
squeue -u g260141
```

### Watch Training Log:
```bash
tail -f training/log_adaptive_*.txt
```

### Check GPU Usage:
```bash
nvidia-smi
```

### View Checkpoints:
```bash
ls -lh checkpoints_adaptive/
```

---

## 🐛 Troubleshooting

### "No module named 'encodec'"
```bash
pip install encodec soundfile
```

### "CUDA out of memory"
Edit `run_train_adaptive_hpc.sh`:
```bash
BATCH_SIZE=2  # or even 1
```

### "No audio files found"
Check dataset path:
```bash
ls dataset_pairs_wav_24sec/train/pair_*_input.wav | wc -l
# Should show number of training files
```

### Training too slow
**Expected**: 3x slower than normal (processing 3 pairs)
- With batch_size=4 on 4 GPUs: ~20-30 sec/epoch (depends on dataset size)
- If much slower: reduce num_workers or batch_size

---

## 📈 Success Indicators

### ✅ Good Training:
- Window positions diverge (not all same)
- Compression ratios vary (1.0x to 1.5x)
- Losses decrease steadily
- No NaN or Inf values
- Checkpoints save successfully

### ⚠️ Issues:
- All pairs select same windows → Add diversity loss
- No compression used (all 1.0x) → Increase rhythm loss weight
- Losses plateau early → Adjust learning rate
- OOM errors → Reduce batch size

---

## 🎯 After Training

### Test Best Model:
```python
python3 test_adaptive_real_audio.py --checkpoint checkpoints_adaptive/best_model.pt
```

### Compare Window Selection:
- Check metadata logs
- Visualize window positions over epochs
- Analyze compression patterns

### Next Steps:
1. Integrate with cascade transformer
2. Add discriminator (GAN)
3. Test with full inference pipeline
4. Deploy to AWS

---

## 📚 Architecture Summary

```
24-sec Audio Pair
    ↓
EnCodec Encoder → [B, 128, 1200]
    ↓
AdaptiveWindowAgent:
  ├─ WindowSelector → 3 pairs × (start, ratio, tonality)
  ├─ TemporalCompressor → 1.0x-1.5x compression
  ├─ TonalityReducer → Harmonic adjustment
  └─ CompositionalAgent → Rhythm/Harmony/Timbre composition
    ↓
3 Creative Outputs [B, 128, 800] each
    ↓
Mean Novelty Loss (all pairs contribute)
    ↓
Gradients → Update all components
```

**Total Parameters**: ~18.2M
- WindowSelector: 3.2M
- TonalityReducer: 0.2M
- CompositionalAgent: 14.6M (shared)
- TemporalCompressor: 0 (interpolation)

---

**Ready to deploy!** 🚀

Choose your path:
- **Conservative**: Run local test → verify → HPC
- **Fast**: Go straight to HPC (files ready)

Both options work - pick based on your confidence level!
