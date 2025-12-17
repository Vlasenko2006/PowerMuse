# LEVANTE HPC Job Submission Guide

## Quick Start

### 1. Upload Files to Levante
```bash
# From your local machine
rsync -avz --progress \
    *.py *.sh *.md \
    g260141@levante.dkrz.de:/work/gg0302/g260141/Jingle_D/
```

### 2. Login to Levante
```bash
ssh g260141@levante.dkrz.de
cd /work/gg0302/g260141/Jingle_D
```

### 3. Create Logs Directory
```bash
mkdir -p logs
```

### 4. Submit Training Job
```bash
# Temporal masking (recommended first)
sbatch submit_levante.sh

# Check job status
squeue -u g260141

# Monitor output
tail -f logs/train_cascade_<jobid>.log
```

---

## Available Submit Scripts

### 1. **submit_levante.sh** (Main - Temporal Masking)
**Best for:** Musical style transfer, rhythmic fusion

**Configuration:**
- Masking: Temporal (alternating 1-second segments)
- Shuffle targets: ENABLED (random pairs)
- Loss weights: input=1.0, target=1.0
- Anti-cheating: 0.3

**Submit:**
```bash
sbatch submit_levante.sh
```

**Expected behavior:** Model learns to inject rhythmic patterns from Song A into Song B with call-and-response structure.

---

### 2. **submit_frequency.sh** (Bass + Melody Fusion)
**Best for:** Harmonic separation, clean mixing

**Configuration:**
- Masking: Frequency split (30% low / 70% high)
- Shuffle targets: ENABLED
- Loss weights: input=1.0, target=1.0
- Anti-cheating: 0.3

**Submit:**
```bash
sbatch submit_frequency.sh
```

**Expected behavior:** Bass/drums from Song A + melody/vocals from Song B.

---

### 3. **submit_hybrid.sh** (Professional Arrangement)
**Best for:** Complex arrangements, maximum separation

**Configuration:**
- Masking: Hybrid (temporal + frequency combined)
- Temporal segments: 100 frames (~0.7s)
- Frequency split: 25% low
- Shuffle targets: ENABLED
- Anti-cheating: 0.5 (higher noise)

**Submit:**
```bash
sbatch submit_hybrid.sh
```

**Expected behavior:** Most complementary, professional-sounding fusion. Harder to train but best quality.

---

### 4. **submit_continuation.sh** (Baseline - No Masking)
**Best for:** Natural song continuations

**Configuration:**
- Masking: NONE
- Shuffle targets: DISABLED (matched pairs)
- Loss weights: input=0.0, target=1.0
- Task: Predict actual continuations

**Submit:**
```bash
sbatch submit_continuation.sh
```

**Expected behavior:** Learn to predict what actually comes next in a song.

---

## Job Management

### Check Job Status
```bash
# View your jobs
squeue -u g260141

# Detailed info
scontrol show job <jobid>

# Cancel job
scancel <jobid>

# View completed jobs
sacct -u g260141 --format=JobID,JobName,State,ExitCode,Elapsed
```

### Monitor Training
```bash
# Live logs
tail -f logs/train_cascade_<jobid>.log

# Check for errors
tail -f logs/train_cascade_<jobid>.err

# GPU usage
ssh <node> nvidia-smi

# Check checkpoints
ls -lht checkpoints_spectral/
```

---

## Resource Allocation

All scripts request:
- **Partition:** gpu
- **Nodes:** 1
- **GPUs:** 4x A100 (80GB each)
- **CPUs:** 32 cores
- **Memory:** 200GB
- **Time:** 48 hours
- **Account:** gg0302

**Estimated training time:** 24-36 hours for 200 epochs with 1800 training pairs.

---

## Output Files

### During Training
```
logs/
  train_cascade_<jobid>.log    # Training progress
  train_cascade_<jobid>.err    # Error messages

checkpoints_spectral/
  checkpoint_epoch_10.pt        # Every 10 epochs
  checkpoint_epoch_20.pt
  best_model.pt                 # Best validation loss

mlruns/                         # MLflow tracking (if enabled)
```

### After Training
```bash
# Download best model
rsync -avz --progress \
    g260141@levante.dkrz.de:/work/gg0302/g260141/Jingle_D/checkpoints_spectral/best_model.pt \
    ./

# Download all checkpoints
rsync -avz --progress \
    g260141@levante.dkrz.de:/work/gg0302/g260141/Jingle_D/checkpoints_spectral/ \
    ./checkpoints_spectral/
```

---

## Troubleshooting

### Job Won't Start
```bash
# Check partition availability
sinfo -p gpu

# Check account limits
sacctmgr show assoc where user=g260141

# Try shorter time limit
#SBATCH --time=24:00:00
```

### Out of Memory
```bash
# Reduce batch size in submit script
--batch_size 12    # Instead of 16

# Or request more memory
#SBATCH --mem=400G
```

### Job Crashes
```bash
# Check error log
cat logs/train_cascade_<jobid>.err

# Check if dataset exists
ls -lh dataset_pairs_wav/train/
ls -lh dataset_pairs_wav/val/

# Verify environment
module list
conda list | grep torch
```

### Slow Training
```bash
# Check GPU utilization
ssh <node>
nvidia-smi

# Should see ~90-100% GPU usage
# If low, might be CPU bottleneck (increase num_workers)
```

---

## Experiment Tracking

### Compare Different Masking Strategies

1. Submit all jobs:
```bash
sbatch submit_levante.sh      # Temporal
sbatch submit_frequency.sh     # Frequency
sbatch submit_hybrid.sh        # Hybrid
sbatch submit_continuation.sh  # Baseline
```

2. Monitor convergence:
```bash
# Check validation losses
grep "Val Loss" logs/train_*_*.log

# Compare best models
ls -lh checkpoints_*/best_model.pt
```

3. Generate audio samples:
```bash
# For each checkpoint
python inference_cascade.py \
    --checkpoint checkpoints_spectral/best_model.pt \
    --shuffle_targets \
    --num_samples 10 \
    --output_folder outputs_temporal/

python inference_cascade.py \
    --checkpoint checkpoints_frequency/best_model.pt \
    --shuffle_targets \
    --num_samples 10 \
    --output_folder outputs_frequency/
```

4. Listen and compare quality!

---

## Email Notifications

All scripts send email to `g260141@levante.dkrz.de` when:
- Job completes successfully
- Job fails

To disable:
```bash
# Comment out these lines in submit scripts
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=g260141@levante.dkrz.de
```

---

## Customization

### Change Masking Parameters

Edit submit script before submitting:

```bash
# Temporal masking - faster alternation
--mask_temporal_segment 75     # 0.5 seconds instead of 1.0

# Frequency masking - different split
--mask_freq_split 0.5          # Equal low/high instead of 30/70

# Anti-cheating - more noise
--anti_cheating 0.7            # Harder task, forces learning
```

### Change Training Parameters

```bash
# Longer training
--epochs 500

# Different learning rate
--lr 5e-4

# More cascade stages
--num_transformer_layers 5
```

---

## Quick Reference

| Script | Masking | Shuffle | Best For | Checkpoint Dir |
|--------|---------|---------|----------|----------------|
| submit_levante.sh | Temporal | ✓ | Rhythmic fusion | checkpoints_spectral |
| submit_frequency.sh | Frequency | ✓ | Bass + melody | checkpoints_frequency |
| submit_hybrid.sh | Hybrid | ✓ | Pro arrangement | checkpoints_hybrid |
| submit_continuation.sh | None | ✗ | Real continuations | checkpoints_continuation |

---

## Example Workflow

```bash
# 1. Upload files
rsync -avz *.py *.sh g260141@levante.dkrz.de:/work/gg0302/g260141/Jingle_D/

# 2. Login
ssh g260141@levante.dkrz.de
cd /work/gg0302/g260141/Jingle_D

# 3. Submit job
sbatch submit_levante.sh

# 4. Monitor
squeue -u g260141
tail -f logs/train_cascade_*.log

# 5. Wait for email notification

# 6. Check results
ls -lh checkpoints_spectral/best_model.pt

# 7. Generate audio
python inference_cascade.py \
    --checkpoint checkpoints_spectral/best_model.pt \
    --shuffle_targets --num_samples 5

# 8. Download outputs
rsync -avz g260141@levante.dkrz.de:/work/gg0302/g260141/Jingle_D/inference_outputs/ ./
```

That's it! You're ready to train with complementary masking on Levante! 🚀
