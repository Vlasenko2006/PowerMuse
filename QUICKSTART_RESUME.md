# Quick Start: Resume Training from Epoch 20

**Goal**: Continue training to push complementarity from 87% → 90%+

---

## 🚀 On HPC (levante.dkrz.de)

### Step 1: Allocate GPUs
```bash
salloc --partition=gpu --account=gg0302 --nodes=1 \
       --gres=gpu:4 --time=06:00:00 --mem=64G
```

### Step 2: Navigate and Resume
```bash
cd /work/gg0302/g260141/Jingle_D
bash run_train_creative_agent_resume.sh
```

**That's it!** Training will resume from epoch 20.

---

## 📊 What to Watch For

### In the Terminal Output

Look for these lines every epoch:
```
🎨 Creative Agent: mask_reg=0.XXXX, complementarity=XX.X%, overlap=0.XXX
   Input mask: 0.XXX, Target mask: 0.XXX
   Balance loss (raw): 0.XXXX [×5.0 weight = 0.XXXX]
```

**Success indicators**:
- ✅ Complementarity increasing: 87% → 88% → 89% → 90%
- ✅ Masks staying balanced: 0.49-0.51 for both
- ✅ Balance loss staying near 0.0000-0.0005
- ✅ Mask reg loss decreasing: 0.75-0.90 → 0.10-0.20

### In Another Terminal (Optional)

Monitor progress in real-time:
```bash
ssh levante.dkrz.de
tail -f /work/gg0302/g260141/Jingle_D/logs/train_creative_agent_resume_epoch20.log | grep "Epoch"
```

---

## ⏱️ Expected Timeline

- **Training time**: ~2.5 minutes per epoch
- **10 epochs**: ~25 minutes
- **30 epochs**: ~75 minutes (1.25 hours)

**For 30 more epochs (to epoch 50)**: Allocate 2-3 hours

---

## 🎯 Milestones

| Epoch | Expected Complementarity | Action |
|-------|-------------------------|--------|
| 20 | 87% (current) | Starting point |
| 30 | 88-89% | First checkpoint - run inference |
| 40 | 89-90% | Second checkpoint - run inference |
| 50 | 90-92% | Goal achieved! 🎉 |

---

## 🧪 Testing After Training

### Every 10 Epochs
```bash
cd /work/gg0302/g260141/Jingle_D
python inference_cascade.py \
    --checkpoint checkpoints_creative_agent/best_model.pt \
    --num_samples 5 \
    --shuffle_targets
```

**Listen for**:
- Drums rhythm preserved (from input)
- Piano harmonies used (from target)
- NEW creative patterns (not in either source)

---

## ⚠️ If Something Goes Wrong

### Training Crashes
```bash
# Check the log
tail -100 logs/train_creative_agent_resume_epoch20.log

# Restart from last checkpoint
bash run_train_creative_agent_resume.sh
```

### Out of Memory
```bash
# Reduce batch size in run_train_creative_agent_resume.sh
--batch_size 6  # instead of 8
```

### Masks Drift from 50/50
```bash
# Increase balance loss weight in run_train_creative_agent_resume.sh
--balance_loss_weight 10.0  # instead of 5.0
```
(But this is unlikely - current weight is perfect!)

---

## ✅ Success Criteria (Epoch 50)

Your training is successful when you see:

1. **Complementarity**: 90% or higher
2. **Mask balance**: 0.48-0.52 for both masks
3. **Mask reg loss**: Below 0.20
4. **Creative outputs**: Novel patterns when listening

---

## 📁 Files Location

- **Training script**: `/work/gg0302/g260141/Jingle_D/run_train_creative_agent_resume.sh`
- **Checkpoints**: `/work/gg0302/g260141/Jingle_D/checkpoints_creative_agent/`
- **Logs**: `/work/gg0302/g260141/Jingle_D/logs/train_creative_agent_resume_epoch20.log`

---

**TL;DR**: Just run `bash run_train_creative_agent_resume.sh` on HPC after allocating GPUs. Training will automatically continue from epoch 20. Watch for complementarity reaching 90%+ by epoch 50! 🎯
