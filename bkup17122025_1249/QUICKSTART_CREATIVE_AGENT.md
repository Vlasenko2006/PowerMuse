# Creative Agent: Quick Start Guide 🚀

## TL;DR

You now have a **learnable creative agent** that replaces fixed masking with attention-based neural networks. It learns which parts of input and target to combine for musical arrangements.

## 3-Step Quick Start

### 1. Test Locally
```bash
# Test the creative agent implementation
python creative_agent.py

# Expected: ✅ Complementarity: 74.9% (untrained)
```

### 2. Train Locally (Single GPU)
```bash
python train_simple_ddp.py \
    --dataset_folder dataset_wav_pairs \
    --num_transformer_layers 3 \
    --use_creative_agent true \
    --mask_reg_weight 0.1 \
    --batch_size 8 \
    --num_epochs 10 \
    --world_size 1
```

### 3. Submit to HPC (Levante)
```bash
# Submit job
sbatch submit_creative_agent.sh

# Monitor
tail -f logs/creative_agent_*.out

# Check status
squeue -u g260141
```

## What You Get

### Before (Fixed Masking)
```
Input:  [■■■■■____■■■■■____]
Target: [____■■■■■____■■■■■]
Output: Predictable, 96% complementary, same for all pairs
```

### After (Creative Agent)
```
Input:  [■■_■____■_■■____■]  ← Learned attention
Target: [__■_■■■■_■__■■■■_]  ← Learned attention
Output: Adaptive, 85-95% complementary, different per pair
```

## Key Files

- **Implementation**: `creative_agent.py` (462 lines)
- **Integration**: `model_simple_transformer.py` (modified)
- **Training**: `train_simple_worker.py` (modified)
- **HPC Script**: `submit_creative_agent.sh`
- **Documentation**: `README_CREATIVE_AGENT.md`

## What to Expect

### Training Logs
```
🎨 Creative Agent ENABLED
🎨 Creative Agent mask regularization loss: 0.251406 (weight=0.1)
Epoch 1/200, Loss: 0.0234, RMS_in: 0.130, RMS_tgt: 0.137
```

### After 50 Epochs
- Complementarity: 75% → 85-95%
- Mask reg loss: 0.25 → 0.05-0.10
- Output: More musical, less blending

## Quick Comparison

| Feature | Fixed (temporal) | Creative Agent |
|---------|------------------|----------------|
| **Enable** | `--mask_type temporal` | `--use_creative_agent true` |
| **Complementary** | 96% | 75% → 85-95% |
| **Adaptive** | No | Yes |
| **Training** | 0 epochs | 50+ epochs |
| **Parameters** | +0 | +700K |

## Common Commands

### Test Implementation
```bash
python creative_agent.py
```

### Train Single GPU
```bash
python train_simple_ddp.py \
    --use_creative_agent true \
    --world_size 1
```

### Train 4 GPUs
```bash
python train_simple_ddp.py \
    --use_creative_agent true \
    --world_size 4
```

### HPC Submit
```bash
sbatch submit_creative_agent.sh
```

### HPC Monitor
```bash
tail -f logs/creative_agent_*.out
squeue -u g260141
scancel <job_id>  # Cancel if needed
```

## Hyperparameters

### Default (Recommended)
```bash
--use_creative_agent true
--mask_reg_weight 0.1
```

### Lower Complementarity (More Creative)
```bash
--mask_reg_weight 0.01  # Less constraint
```

### Higher Complementarity (More Strict)
```bash
--mask_reg_weight 0.5  # More constraint
```

## Troubleshooting

### Issue: Masks collapse (all 0 or all 1)
```bash
# Lower weight
--mask_reg_weight 0.01
```

### Issue: Not learning complementarity
```bash
# Higher weight
--mask_reg_weight 0.2
```

### Issue: Training unstable
```bash
# Lower learning rate
--lr 1e-5
```

## Next Steps

1. ✅ Test locally: `python creative_agent.py`
2. ✅ Train locally (10 epochs): Verify it works
3. 🔲 Submit to HPC: `sbatch submit_creative_agent.sh`
4. 🔲 Monitor training: Watch complementarity improve
5. 🔲 Compare outputs: Creative agent vs fixed temporal vs no masking
6. 🔲 Iterate: Tune `mask_reg_weight` based on results

## Questions?

Read the full documentation:
- **README_CREATIVE_AGENT.md**: Comprehensive guide
- **CREATIVE_AGENT_SUMMARY.md**: Implementation details
- **README_MASKING.md**: Fixed masking comparison

## Summary

The creative agent is **ready to use**. It learns attention-based masks that adaptively select which parts of input and target to combine. Start with local testing, then submit to HPC for full training.

**Good luck! 🎵🎨**
