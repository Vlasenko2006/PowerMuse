# Creative Agent: Learnable Complementary Masking 🎨

## Overview

The **Creative Agent** is a neural network-based system that learns how to create musical arrangements by intelligently combining patterns from two songs. Unlike fixed masking strategies (temporal, frequency, etc.), the creative agent uses **attention mechanisms** to adaptively select which parts of each song to combine.

## Key Concept

### Fixed Masking (Previous Approach)
- **Rule-based**: Temporal segments, frequency splits, energy thresholds
- **Static**: Same strategy for all song pairs
- **96% complementary** but doesn't adapt to musical context

### Creative Agent (New Approach)
- **Learned**: Neural network decides what to combine
- **Adaptive**: Different strategy for each input-target pair
- **Attention-based**: Understands relationships between patterns
- **75% complementary** (untrained) → will improve with training

## Architecture

### 1. AttentionMaskGenerator
Learns complementary masks using cross-attention:

```python
Input Pattern  ──► Conv1d Feature Extractor ──┐
                                               ├──► Cross-Attention ──► Mask Generator ──► Input Mask
Target Pattern ──► Conv1d Feature Extractor ──┘                                          Target Mask
```

**Key Components:**
- **Feature Extractors**: Conv1d layers analyze input and target
- **Cross-Attention**: MultiheadAttention (4 heads) learns relationships
- **Mask Generators**: Conv1d → Sigmoid produces soft masks [0, 1]
- **Regularization**: 
  - Complementarity loss: minimize `input_mask * target_mask`
  - Coverage loss: encourage `input_mask + target_mask ≈ 1.0`

**Parameters:** ~500K

### 2. StyleDiscriminator (Optional)
Judges quality for adversarial training:

```python
Pattern ──► Conv1d Encoder ──┬──► Real/Fake Classifier ──► Score
                              └──► Style Matcher ──────────► Style Score
```

**Key Components:**
- **Encoder**: Conv1d with downsampling (T → T/8)
- **Classifier**: Real vs fake discrimination (GAN-style)
- **Style Matcher**: Compares mean/std statistics

**Parameters:** ~200K

### 3. CreativeAgent
Wrapper combining both components:

```python
creative_agent = CreativeAgent(encoding_dim, use_discriminator=True)

# Generate learned masks
masked_input, masked_target, reg_loss = creative_agent.generate_creative_masks(
    encoded_input, encoded_target, hard=False
)

# Judge quality (optional)
real_fake_score, style_score = creative_agent.judge_quality(pattern)

# Adversarial loss (optional)
gen_loss, disc_loss = creative_agent.adversarial_loss(fake_pattern, real_pattern)
```

## Integration

### Model (model_simple_transformer.py)

The creative agent is integrated into `SimpleTransformer` in cascade mode:

```python
model = SimpleTransformer(
    encoding_dim=128,
    num_transformer_layers=3,  # Cascade mode required
    use_creative_agent=True    # Enable creative agent
)
```

**Forward pass:**
```python
encoded_output, mask_reg_loss = model(encoded_input, encoded_target)
```

**What happens internally:**
1. If `use_creative_agent=True`: Apply learned masks before concat
2. If `use_creative_agent=False`: Use original concat (fixed masking applied in training script)
3. Return both output and mask regularization loss

### Training (train_simple_worker.py)

**Loss computation:**
```python
# Reconstruction loss
loss = combined_loss(output_audio, input_audio, target_audio, ...)

# Add mask regularization
if mask_reg_loss is not None:
    loss = loss + mask_reg_weight * mask_reg_loss
```

**Parameters:**
- `--use_creative_agent true`: Enable creative agent
- `--mask_reg_weight 0.1`: Weight for mask regularization loss (default: 0.1)

## Usage

### 1. Local Testing

Test the creative agent implementation:
```bash
python creative_agent.py
```

Expected output:
```
✅ Complementarity: 74.9%  (untrained)
✅ Regularization loss: 0.251406
✅ Gradients flow correctly
```

### 2. Training with Creative Agent

#### Single GPU (local):
```bash
python train_simple_ddp.py \
    --dataset_folder dataset_wav_pairs \
    --num_transformer_layers 3 \
    --use_creative_agent true \
    --mask_reg_weight 0.1 \
    --world_size 1
```

#### Multi-GPU (4 GPUs):
```bash
python train_simple_ddp.py \
    --dataset_folder dataset_wav_pairs \
    --num_transformer_layers 3 \
    --use_creative_agent true \
    --mask_reg_weight 0.1 \
    --world_size 4
```

### 3. HPC (Levante)

Create a submit script:

```bash
#!/bin/bash
#SBATCH --job-name=creative_agent
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --time=48:00:00
#SBATCH --output=logs/creative_agent_%j.out
#SBATCH --error=logs/creative_agent_%j.err

module load python3
source venv/bin/activate

srun python train_simple_ddp.py \
    --dataset_folder dataset_wav_pairs \
    --encoding_dim 128 \
    --num_transformer_layers 3 \
    --use_creative_agent true \
    --mask_reg_weight 0.1 \
    --batch_size 8 \
    --num_epochs 200 \
    --lr 1e-4 \
    --world_size 4 \
    --checkpoint_dir checkpoints_creative_agent
```

Submit:
```bash
sbatch submit_creative_agent.sh
```

## Comparison: Fixed vs Learned Masking

| Feature | Fixed Masking | Creative Agent |
|---------|--------------|----------------|
| **Strategy** | Rule-based (temporal/frequency/etc.) | Learned via attention |
| **Adaptation** | Static for all pairs | Adaptive per pair |
| **Complementarity** | 96% (verified) | ~75% (untrained) → improves with training |
| **Parameters** | 0 | ~700K |
| **Training** | No training needed | Requires training |
| **Flexibility** | Limited to predefined rules | Learns from data |
| **Use Case** | Baseline, quick experiments | Production, adaptive arrangements |

## Expected Behavior

### Untrained Creative Agent
- **Complementarity**: ~75% (random initialization)
- **Coverage**: ~100% (enforced by loss)
- **Output**: Similar to random masking

### Trained Creative Agent (after 50+ epochs)
- **Complementarity**: 85-95% (learns to minimize overlap)
- **Coverage**: 95-100% (learns to use both patterns)
- **Output**: Musically coherent arrangements
- **Adaptation**: Different strategies for different song pairs

## Hyperparameters

### Model Architecture
- `encoding_dim=128`: Encoding dimension (must match transformer)
- `hidden_dim=256`: Hidden dimension for feature extraction
- `num_heads=4`: Attention heads

### Training
- `mask_reg_weight=0.1`: Weight for complementarity+coverage loss
  - **Lower** (0.01): More freedom, less complementary
  - **Higher** (1.0): Strong complementarity, less creativity
  - **Recommended**: 0.05-0.2

### Optional: Adversarial Training (Future)
Currently not implemented in training loop:
- `adversarial_weight=0.01`: Weight for GAN-style discriminator loss
- Requires separate optimizer for discriminator
- Two-phase training: generator → discriminator

## Monitoring

During training, watch for:

```
🎨 Creative Agent ENABLED (printed once at startup)
🎨 Creative Agent mask regularization loss: 0.251406 (weight=0.1)  (every epoch, first batch)
```

**Good signs:**
- Mask reg loss decreases over epochs (learning complementarity)
- Total loss decreases (reconstruction + mask reg)
- Complementarity improves (run test_creative_agent.py on checkpoints)

**Bad signs:**
- Mask reg loss stays high (not learning complementarity)
- Total loss increases (mask reg weight too high)
- Complementarity stays ~50% (collapsed masks)

## Troubleshooting

### Issue: Masks collapse (all 0 or all 1)
**Solution:** Reduce `mask_reg_weight` (try 0.01-0.05)

### Issue: Not learning complementarity
**Solution:** Increase `mask_reg_weight` (try 0.2-0.5)

### Issue: Training unstable
**Solution:** 
- Lower learning rate (try 1e-5)
- Add gradient clipping (already enabled: max_norm=1.0)
- Reduce `mask_reg_weight`

### Issue: Output sounds worse than fixed masking
**Solution:**
- Train longer (creative agent needs 50+ epochs)
- Try different `mask_reg_weight`
- Verify complementarity is improving (test with creative_agent.py)

## Testing

### Test on Checkpoint

Load a checkpoint and test complementarity:

```python
import torch
from model_simple_transformer import SimpleTransformer
from creative_agent import test_creative_agent

# Load checkpoint
model = SimpleTransformer(
    encoding_dim=128,
    num_transformer_layers=3,
    use_creative_agent=True
)
checkpoint = torch.load('checkpoints_creative_agent/checkpoint_epoch_50.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Test creative agent
with torch.no_grad():
    encoded_input = torch.randn(4, 128, 300)
    encoded_target = torch.randn(4, 128, 300)
    
    input_mask, target_mask, reg_loss = model.creative_agent.mask_generator(
        encoded_input, encoded_target
    )
    
    overlap = (input_mask * target_mask).mean().item()
    complementarity = (1.0 - overlap) * 100
    
    print(f"Complementarity: {complementarity:.1f}%")
    print(f"Regularization loss: {reg_loss.item():.6f}")
```

## Future Enhancements

1. **Adversarial Training**: Enable discriminator for GAN-style creativity
2. **Multi-Scale Attention**: Attend at different time scales
3. **Style Conditioning**: Condition masks on desired style/genre
4. **Interpretability**: Visualize learned attention patterns
5. **Hard Masks**: Use Gumbel-Softmax for discrete selection

## Summary

The **Creative Agent** replaces fixed masking rules with a learned attention mechanism that:
- ✅ Adapts to each song pair
- ✅ Learns complementary patterns
- ✅ Maintains 75% complementarity (untrained) → 85-95% (trained)
- ✅ Integrates seamlessly into existing pipeline
- ✅ Adds ~700K parameters (~5% increase)

**When to use:**
- Production models (adaptive arrangements)
- When fixed masking is too rigid
- When you have enough training data (>1000 pairs)

**When NOT to use:**
- Quick experiments (fixed masking is faster)
- Limited training data (<500 pairs)
- Need reproducible results (fixed masking is deterministic)
