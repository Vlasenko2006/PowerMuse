# Pre-Submission Checklist for Levante

## ✅ Files to Upload

Copy this checklist and verify each item before submitting:

### Core Python Files
- [ ] `complementary_masking.py` (NEW - masking implementation)
- [ ] `model_simple_transformer.py` (balanced residuals)
- [ ] `train_simple_ddp.py` (masking arguments added)
- [ ] `train_simple_worker.py` (masking integration)
- [ ] `dataset_wav_pairs.py` (shuffle targets support)
- [ ] `inference_cascade.py` (for testing after training)

### Submit Scripts
- [ ] `submit_levante.sh` (main - temporal masking)
- [ ] `submit_frequency.sh` (optional - bass/melody)
- [ ] `submit_hybrid.sh` (optional - professional)
- [ ] `submit_continuation.sh` (optional - baseline)

### Documentation (optional but helpful)
- [ ] `README_MASKING.md`
- [ ] `README_LEVANTE.md`
- [ ] `IMPLEMENTATION_SUMMARY.md`

---

## 📋 Pre-Flight Checks

### On Local Machine

**1. Test masking locally (optional but recommended):**
```bash
python test_masking.py --mask_type temporal
# Should generate mask_viz_temporal.png showing 96% complementarity
```

**2. Verify all files exist:**
```bash
ls -1 complementary_masking.py \
      model_simple_transformer.py \
      train_simple_ddp.py \
      train_simple_worker.py \
      dataset_wav_pairs.py \
      inference_cascade.py \
      submit_*.sh
```

**3. Upload to Levante:**
```bash
rsync -avz --progress \
    complementary_masking.py \
    model_simple_transformer.py \
    train_simple_ddp.py \
    train_simple_worker.py \
    dataset_wav_pairs.py \
    inference_cascade.py \
    submit_*.sh \
    README_*.md \
    g260141@levante.dkrz.de:/work/gg0302/g260141/Jingle_D/
```

---

### On Levante

**1. Login:**
```bash
ssh g260141@levante.dkrz.de
cd /work/gg0302/g260141/Jingle_D
```

**2. Verify dataset exists:**
```bash
ls -lh dataset_pairs_wav/train/ | head
ls -lh dataset_pairs_wav/val/ | head

# Should show files like:
# song_0001_0_input.wav
# song_0001_0_output.wav
```

**Expected:**
- Train: ~1800 pairs (3600 files)
- Val: ~200 pairs (400 files)

**3. Check conda environment:**
```bash
source /work/gg0302/g260141/miniconda3/bin/activate
conda activate multipattern_env

# Verify PyTorch with CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Should show:
# PyTorch: 2.x.x
# CUDA available: True
```

**4. Create logs directory:**
```bash
mkdir -p logs
```

**5. Make submit scripts executable:**
```bash
chmod +x submit_*.sh
```

**6. Verify submit script:**
```bash
head -30 submit_levante.sh

# Check these lines are correct:
# - Account: gg0302
# - Partition: gpu
# - GPUs: 4x A100
# - Conda env: multipattern_env
# - Dataset: dataset_pairs_wav
```

---

## 🚀 Ready to Submit

### Choose Your Experiment

**Option 1: Temporal Masking (Recommended First)**
```bash
sbatch submit_levante.sh
```
- Alternating time segments
- Most musical, easy to understand
- 96% complementary separation

**Option 2: Frequency Masking**
```bash
sbatch submit_frequency.sh
```
- Bass from input, melody from target
- Clean harmonic separation

**Option 3: Hybrid Masking**
```bash
sbatch submit_hybrid.sh
```
- Maximum separation
- Professional quality
- Harder to train

**Option 4: All at Once (Compare)**
```bash
sbatch submit_levante.sh      # Temporal
sbatch submit_frequency.sh     # Frequency
sbatch submit_hybrid.sh        # Hybrid
sbatch submit_continuation.sh  # Baseline
```

---

## 📊 Monitor Training

**Check job status:**
```bash
squeue -u g260141
```

**Watch logs:**
```bash
# Wait a moment for job to start, then:
tail -f logs/train_cascade_*.log

# Look for these indicators:
# ✓ "🎭 Complementary Masking Applied"
# ✓ "rms_in" and "rms_tgt" are similar (~0.13)
# ✓ "Val loss" decreasing over epochs
# ✓ "✓ New best model!" messages
```

**Expected training output:**
```
🎭 Complementary Masking Applied:
  Type: temporal
  Description: Alternating time segments (Input: beats 1,3,5... | Target: beats 2,4,6...)
  Segment length: 150 frames (~1.0s)

Epoch 1: Train Loss: 0.305, Val Loss: 0.323
  ✓ New best model! Val loss: 0.323250

Epoch 5: Train Loss: 0.308, Val Loss: 0.326
  rms_in: 0.131, rms_tgt: 0.139  ← Should be balanced!
```

---

## 🎵 After Training Completes

**1. Check results:**
```bash
ls -lh checkpoints_spectral/best_model.pt

# Should see file size ~70MB (17.5M parameters)
```

**2. Generate test audio:**
```bash
python inference_cascade.py \
    --checkpoint checkpoints_spectral/best_model.pt \
    --shuffle_targets \
    --num_samples 10 \
    --output_folder outputs_temporal/

ls -lh outputs_temporal/
```

**3. Download outputs:**
```bash
# From your local machine:
rsync -avz --progress \
    g260141@levante.dkrz.de:/work/gg0302/g260141/Jingle_D/outputs_temporal/ \
    ./outputs_temporal/

rsync -avz --progress \
    g260141@levante.dkrz.de:/work/gg0302/g260141/Jingle_D/checkpoints_spectral/best_model.pt \
    ./checkpoints_spectral/
```

**4. Listen to results:**
```bash
# Play WAV files in order:
# 1. *_1_input.wav     - Original input
# 2. *_2_target.wav    - Random target
# 3. *_3_predicted.wav - Model output (should be creative arrangement!)
```

---

## ⚠️ Troubleshooting

### Job doesn't start
- **Check:** `squeue -u g260141` - might be waiting for resources
- **Solution:** Wait, or reduce `--time` in submit script

### Out of memory error
- **Check:** logs/train_cascade_*.err for OOM message
- **Solution:** Reduce `--batch_size 12` in submit script

### Import error for complementary_masking
- **Check:** File uploaded correctly
- **Solution:** Re-upload: `rsync complementary_masking.py g260141@levante:...`

### Masking not showing in logs
- **Check:** Look for "🎭 Complementary Masking Applied" message
- **Solution:** Verify `--mask_type temporal` (not 'none')

### Training very slow
- **Check:** GPU utilization with `nvidia-smi` on compute node
- **Solution:** Should be 90-100%. If low, dataset I/O might be bottleneck

---

## 📝 Quick Command Reference

```bash
# Upload
rsync -avz *.py *.sh g260141@levante.dkrz.de:/work/gg0302/g260141/Jingle_D/

# Login
ssh g260141@levante.dkrz.de
cd /work/gg0302/g260141/Jingle_D

# Submit
sbatch submit_levante.sh

# Monitor
squeue -u g260141
tail -f logs/train_cascade_*.log

# Cancel
scancel <jobid>

# Download
rsync -avz g260141@levante.dkrz.de:/work/gg0302/g260141/Jingle_D/checkpoints_spectral/ ./
```

---

## ✅ Final Checklist

Before submitting, verify:

- [ ] All Python files uploaded to Levante
- [ ] Dataset exists (dataset_pairs_wav/train and val)
- [ ] Conda environment activated and tested
- [ ] Logs directory created
- [ ] Submit script executable (chmod +x)
- [ ] Account/partition/GPU settings correct
- [ ] Ready to monitor job (know how to check logs)

**All good?** → `sbatch submit_levante.sh` 🚀

**Expected completion:** 24-36 hours (you'll get email notification)

Good luck! 🎵
