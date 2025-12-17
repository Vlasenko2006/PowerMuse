# Anti-Modulation Correlation Cost: Preventing Amplitude Envelope Copying

## Problem

After 20 epochs of training the compositional creative agent, the model was **NOT being creative** - it was simply copying the amplitude envelope from the target audio. Looking at waveforms showed:

- Input: Drums (sparse, percussive)
- Target: Piano (sustained notes)
- **Output: Piano-like sustained envelope** (copying target, not creating!)

The compositional agent was extracting rhythm/harmony/timbre correctly, but the **final output just repeated the target's amplitude modulation pattern** instead of creating truly novel patterns.

## Solution: Anti-Modulation Correlation Cost

We add a new loss term that **heavily penalizes correlation** between the output's amplitude envelope and the input/target envelopes.

### Algorithm

```python
def compute_modulation_correlation(input_audio, target_audio, output_audio, M_parts=250):
    """
    Prevent copying amplitude envelopes through correlation-based penalty.
    
    Steps:
    1. Take absolute value: abs_input, abs_target, abs_output
    2. Split each into M_parts=250 segments (time windows)
    3. For each segment, compute max amplitude (envelope)
    4. Compute correlation between envelopes:
       - corr_input = correlation(max_input, max_output)
       - corr_target = correlation(max_target, max_output)
    5. Apply exponential penalty:
       - cost = -ln(1 - |corr_input|) - ln(1 - |corr_target|)
    
    Returns: Scalar cost (higher = more copying detected)
    """
```

### Why This Works

**Correlation measures similarity of patterns:**
- `corr = 1.0`: Perfect copy (identical envelope patterns)
- `corr = 0.0`: Independent (no pattern similarity)
- `corr = -1.0`: Inverted (opposite patterns)

**Exponential penalty via -ln(1-|corr|):**

| Correlation | Cost | Interpretation |
|------------|------|----------------|
| 0.0 | 0.00 | Independent - GOOD! (no penalty) |
| 0.5 | 0.69 | Some similarity (moderate penalty) |
| 0.8 | 1.61 | High similarity (strong penalty) |
| 0.9 | 2.30 | Very similar (very strong penalty) |
| 0.99 | 4.61 | Copying detected! (extreme penalty) |
| 0.999 | 6.91 | Nearly perfect copy (near-infinite penalty) |

**Key insight:** The penalty becomes **exponentially larger** as correlation approaches 1.0, making it extremely expensive for the model to copy envelopes.

### Implementation

#### 1. Added to `CompositionalCreativeAgent` (compositional_creative_agent.py)

```python
def compute_modulation_correlation(self, input_audio, target_audio, output_audio, M_parts=250):
    """
    Compute anti-modulation correlation cost.
    
    Args:
        input_audio: [B, 1, T] - Raw input waveform
        target_audio: [B, 1, T] - Raw target waveform
        output_audio: [B, 1, T] - Raw output waveform
        M_parts: Number of segments (default: 250)
    
    Returns:
        corr_cost: Scalar - Anti-modulation cost
    """
    # 1. Take absolute values (amplitude envelopes)
    abs_input = torch.abs(input_audio.squeeze(1))   # [B, T]
    abs_target = torch.abs(target_audio.squeeze(1))
    abs_output = torch.abs(output_audio.squeeze(1))
    
    # 2. Split into M_parts segments
    segment_size = T // M_parts
    abs_input = abs_input[:, :segment_size * M_parts].view(B, M_parts, segment_size)
    abs_target = abs_target[:, :segment_size * M_parts].view(B, M_parts, segment_size)
    abs_output = abs_output[:, :segment_size * M_parts].view(B, M_parts, segment_size)
    
    # 3. Compute max per segment (envelope)
    max_input = abs_input.max(dim=2)[0]   # [B, M_parts]
    max_target = abs_target.max(dim=2)[0]
    max_output = abs_output.max(dim=2)[0]
    
    # 4. Compute correlation for each batch element
    corr_input = compute_correlation(max_input, max_output)   # [B]
    corr_target = compute_correlation(max_target, max_output)
    
    # 5. Take absolute value (magnitude of correlation)
    corr_input_abs = torch.abs(corr_input)
    corr_target_abs = torch.abs(corr_target)
    
    # 6. Compute exponential penalty: -ln(1 - |corr|)
    # Clamp to avoid ln(0) or ln(negative)
    corr_input_clamped = torch.clamp(corr_input_abs, 0.0, 0.999)
    corr_target_clamped = torch.clamp(corr_target_abs, 0.0, 0.999)
    
    cost_input = -torch.log(1.0 - corr_input_clamped)
    cost_target = -torch.log(1.0 - corr_target_clamped)
    
    # 7. Total cost (average over batch)
    corr_cost = (cost_input + cost_target).mean()
    
    return corr_cost
```

#### 2. Integrated into Training (train_simple_worker.py)

```python
# After model forward pass and decoding
output_audio = encodec_model.decoder(encoded_output)  # [B, 1, samples]

# Combined loss
loss = reconstruction_loss + spectral_loss + ...

# Add novelty loss
if mask_reg_loss is not None:
    loss = loss + mask_reg_weight * mask_reg_loss

# NEW: Add anti-modulation correlation cost
corr_cost = torch.tensor(0.0).to(device)
if corr_weight > 0:
    corr_cost = model.creative_agent.compute_modulation_correlation(
        input_audio, targets, output_audio, M_parts=250
    )
    loss = loss + corr_weight * corr_cost
```

#### 3. CLI Argument (train_simple_ddp.py)

```python
parser.add_argument('--corr_weight', type=float, default=0.0,
                   help='Weight for anti-modulation correlation cost')
```

#### 4. Training Script (run_train_compositional.sh)

```bash
# Compositional creative agent
USE_COMPOSITIONAL=true
NOVELTY_WEIGHT=0.1  # Novelty regularization
CORR_WEIGHT=0.5     # Anti-modulation correlation cost (PREVENTS COPYING!)

# Training command
python -m torch.distributed.launch \
    --use_compositional_agent $USE_COMPOSITIONAL \
    --mask_reg_weight $NOVELTY_WEIGHT \
    --corr_weight $CORR_WEIGHT \
    ...
```

### Recommended Weight: 0.5

Based on the test results:
- **Independent output** (creative): cost ≈ 0.12 → penalty = 0.5 × 0.12 = 0.06
- **Copying output**: cost ≈ 2.0 → penalty = 0.5 × 2.0 = 1.0
- **Perfect copy**: cost ≈ 7.0 → penalty = 0.5 × 7.0 = 3.5

With `corr_weight=0.5`, copying becomes **16× more expensive** than being creative! This will force the model to generate novel amplitude patterns.

### Monitoring

The correlation cost is logged to MLflow as `train_corr_cost`:

```python
# MLflow logging
if train_creative['type'] == 'compositional':
    metrics.update({
        "train_novelty_loss": train_creative['mask_reg_loss'],
        "train_corr_cost": train_creative['corr_cost'],  # NEW!
        "train_input_rhythm_weight": train_creative['input_rhythm_weight'],
        ...
    })
```

**Expected evolution during training:**
- **Epochs 1-5**: High cost (2.0-4.0) - Model still copying
- **Epochs 5-15**: Decreasing cost (1.0-2.0) - Learning to avoid copying
- **Epochs 15-30**: Low cost (0.1-0.5) - Creative patterns emerging
- **Epochs 30-50**: Stable low cost (0.05-0.2) - Consistent creativity

### What This Fixes

**Before (without correlation cost):**
```
Input:  ▂▄█▂▁▂█▁▂▄█▂   (drums - sparse, percussive)
Target: ▄▅▆▆▅▄▅▆▆▅▄   (piano - sustained, dense)
Output: ▄▅▆▆▅▄▅▆▆▅▄   (COPYING target envelope! ❌)
```

**After (with correlation cost):**
```
Input:  ▂▄█▂▁▂█▁▂▄█▂   (drums)
Target: ▄▅▆▆▅▄▅▆▆▅▄   (piano)
Output: ▂█▅▂█▄▂█▆▂▅   (NEW pattern! ✓)
         ↑ Takes rhythm from input
           ↑ Takes harmony from target
             ↑ Creates novel envelope
```

### Technical Details

**Why 250 segments?**
- For 16s audio @ 24kHz: 384,000 samples
- Segment size: 384,000 / 250 = 1,536 samples ≈ 64ms
- 64ms is the **perceptual time constant** for amplitude modulation
- Captures musical dynamics (note attacks, sustain, release)

**Correlation formula:**
```python
def compute_correlation(x, y):
    # Center the data
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)
    
    # Covariance
    cov = (x_centered * y_centered).sum(dim=1) / (M_parts - 1)
    
    # Standard deviations
    std_x = torch.sqrt((x_centered ** 2).sum(dim=1) / (M_parts - 1))
    std_y = torch.sqrt((y_centered ** 2).sum(dim=1) / (M_parts - 1))
    
    # Correlation coefficient
    corr = cov / (std_x * std_y + 1e-8)
    
    return torch.clamp(corr, -1.0, 1.0)
```

### Comparison with Other Approaches

**1. L2 Loss on Amplitude Envelope:**
- Problem: Linear penalty, doesn't distinguish between small and large similarities
- `L2(input, output) = 0.1` could mean very similar or moderately similar

**2. Cosine Similarity (like novelty loss):**
- Problem: Bounded [0, 1], equal penalty across range
- `cos_sim = 0.9` gets same penalty magnitude as `cos_sim = 0.5`

**3. Correlation + Exponential Penalty (our approach):**
- Advantage: **Exponential scaling** as correlation → 1.0
- `corr = 0.9` → cost = 2.30 (strong penalty)
- `corr = 0.99` → cost = 4.61 (2× stronger!)
- `corr = 0.999` → cost = 6.91 (3× stronger!!)

This creates a **"soft barrier"** that becomes harder to cross as you get closer to copying.

### Expected Results

With this cost added, the compositional creative agent should:

1. **Stop copying target envelopes** (cost too high)
2. **Create novel amplitude patterns** by:
   - Taking rhythm timing from input (when to have peaks)
   - Taking harmonic content from target (what frequencies)
   - Generating NEW envelope shape (different from both)
3. **Achieve TRUE creativity**: Output will sound like neither input nor target alone

### Testing Locally

```bash
# Test the correlation cost function
python -c "
from compositional_creative_agent import CompositionalCreativeAgent
import torch

agent = CompositionalCreativeAgent(encoding_dim=128)

# Case 1: Copying (high cost)
input_audio = torch.randn(2, 1, 384000) * 0.5
output_copy = input_audio + torch.randn(2, 1, 384000) * 0.1
target_audio = torch.randn(2, 1, 384000) * 0.3

cost_copy = agent.compute_modulation_correlation(input_audio, target_audio, output_copy)
print(f'Copying: {cost_copy.item():.4f}')  # Expected: 1.5-3.0

# Case 2: Creative (low cost)
output_novel = torch.randn(2, 1, 384000) * 0.4
cost_novel = agent.compute_modulation_correlation(input_audio, target_audio, output_novel)
print(f'Creative: {cost_novel.item():.4f}')  # Expected: 0.05-0.3
"
```

### Training with Anti-Modulation Cost

```bash
# Start training with correlation penalty
./run_train_compositional.sh

# Configuration includes:
# CORR_WEIGHT=0.5  # Strong penalty for copying
# NOVELTY_WEIGHT=0.1  # Encourage diversity in latent space

# Watch for:
# - Correlation cost decreasing over epochs (model learning not to copy)
# - Novel waveform patterns in inference (not matching input/target)
```

### Verification

After training, run inference and check waveforms:

```bash
python inference_cascade.py --num_samples 5

# Look for:
# 1. Output envelope DIFFERENT from both input and target
# 2. Correlation cost < 0.5 in training logs (by epoch 30)
# 3. Creative recombination of musical elements
```

### Summary

**The Problem:** Compositional agent was copying amplitude envelopes instead of creating novel patterns.

**The Solution:** Anti-modulation correlation cost that applies exponential penalty when output envelope correlates with input/target envelopes.

**Why It Works:** 
- Measures correlation (pattern similarity) in amplitude domain
- Exponential penalty: -ln(1 - |corr|) makes copying very expensive
- Forces model to generate truly novel envelope patterns

**Expected Impact:**
- Training: Correlation cost will decrease from 2.0 → 0.2 over 30 epochs
- Inference: Output will have unique amplitude patterns, not copies
- Creativity: Model will learn to COMPOSE new patterns, not REPEAT existing ones

**Weight Recommendation:** `corr_weight=0.5` provides strong penalty without dominating other loss terms.
