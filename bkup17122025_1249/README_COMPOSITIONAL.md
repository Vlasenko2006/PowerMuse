# Compositional Creative Agent: TRUE Musical Creativity

## The Problem with Previous Approaches

### Old Creative Agent (Masking-Based)
```python
masked_input = input * input_mask      # Keep some of input
masked_target = target * target_mask   # Keep some of target
output = transform(masked_input + masked_target)
```

**Result**: Linear interpolation in latent space
- Takes amplitude envelope from input
- Takes harmonic content from target
- Output = Input's loudness × Target's music
- **NOT creative**, just smart mixing!

## The Solution: Compositional Decomposition

### Architecture

```
Input (drums)                Target (piano)
     ↓                             ↓
  Extract Components          Extract Components
     ↓                             ↓
┌─────────────┐             ┌─────────────┐
│  Rhythm     │             │  Rhythm     │
│  (Conv 3-5) │             │  (Conv 3-5) │
├─────────────┤             ├─────────────┤
│  Harmony    │             │  Harmony    │
│  (Conv 7-9) │             │  (Conv 7-9) │
├─────────────┤             ├─────────────┤
│  Timbre     │             │  Timbre     │
│  (Conv15-21)│             │  (Conv15-21)│
└─────────────┘             └─────────────┘
       ↓                           ↓
       └───────────┬───────────────┘
                   ↓
          Component Selector
         (Attention Weights)
                   ↓
           [6 Components]
   ┌───────────────────────────┐
   │ input_rhythm   (strong)   │
   │ input_harmony  (weak)     │
   │ input_timbre   (percussive)│
   │ target_rhythm  (weak)     │
   │ target_harmony (strong)   │
   │ target_timbre  (tonal)    │
   └───────────────────────────┘
                   ↓
       Compositional Transformer
       (Learns Musical Rules)
                   ↓
           Creative Output
   "Rhythmic Piano" - NEW PATTERN!
   (Not in input or target!)
```

## Musical Components

### Rhythm (Small Kernels: 3-5)
- **Captures**: Fast temporal variations, beat, transients, envelope
- **Examples**: Drum patterns, note attacks, rhythmic structure
- **Time scale**: ~40-67ms (receptive field)

### Harmony (Medium Kernels: 7-9)
- **Captures**: Pitch relationships, melodic patterns, chords
- **Examples**: Chord progressions, melodic contours, intervals
- **Time scale**: ~93-120ms (receptive field)

### Timbre (Large Kernels: 15-21)
- **Captures**: Tone color, texture, instrumentation, spectral envelope
- **Examples**: "Piano sound", "guitar sound", brightness, warmth
- **Time scale**: ~200-280ms (receptive field)

## How It Creates TRUE Creativity

### Example: Drums (Input) + Piano (Target)

**Input Analysis**:
- Rhythm: **Strong** (kick/snare pattern)
- Harmony: **Weak** (no pitch content)
- Timbre: **Percussive** (sharp attacks)

**Target Analysis**:
- Rhythm: **Weak** (sustained chords)
- Harmony: **Strong** (C major progression)
- Timbre: **Tonal** (piano sound)

**Compositional Process**:
1. Component selector chooses:
   - ✅ Input rhythm (strong drum pattern)
   - ✅ Target harmony (C major chords)
   - ✅ Target timbre (piano sound)

2. Transformer composer learns:
   - "Drum rhythm + piano harmony = rhythmic piano arpeggios"
   - "Not drums playing C major" (wrong timbre)
   - "Not sustained piano chords" (wrong rhythm)

3. Output: **NEW rhythmic piano pattern**
   - Rhythm matches input (syncopated, accented)
   - Harmony matches target (C major chord tones)
   - Timbre matches target (piano sound)
   - **Pattern exists in NEITHER source!**

## Novelty Regularization

Ensures output is truly creative:

```python
# Cosine similarity between output and sources
similarity_to_input = cos_sim(output, input)
similarity_to_target = cos_sim(output, target)

# Penalize high similarity (> 0.7)
novelty_loss = relu(similarity_to_input - 0.7) + relu(similarity_to_target - 0.7)
```

- **High novelty**: Output differs from both inputs (creative!)
- **Low novelty**: Output too similar to inputs (just copying)
- Target: 50-70% similarity (related but different)

## Architecture Details

### MultiScaleExtractor
```python
# Rhythm extractor
Conv1d(128 → 128, kernel=3) → LeakyReLU
Conv1d(128 → 64, kernel=5) → LeakyReLU → BatchNorm

# Harmony extractor
Conv1d(128 → 128, kernel=7) → LeakyReLU
Conv1d(128 → 64, kernel=9) → LeakyReLU → BatchNorm

# Timbre extractor
Conv1d(128 → 128, kernel=15) → LeakyReLU
Conv1d(128 → 64, kernel=21) → LeakyReLU → BatchNorm
```

**Output**: 3 components × 64 dims = 192 total dimensions per source

### ComponentComposer
```python
# 4-layer transformer
Input: [B, 384, T] (6 components × 64 dims)
  ↓
Linear(384 → 512) + PositionalEncoding
  ↓
TransformerEncoder(4 layers, 8 heads)
  ↓
Linear(512 → 256) → GELU → Dropout
  ↓
Linear(256 → 128)
  ↓
Output: [B, 128, T] - Creative encoding
```

**Parameters**: ~14.6M (in addition to cascade model)

## Training Configuration

### Recommended Loss Weights

```bash
# Musical structure guidance
--loss_weight_spectral 0.01

# Novelty regularization (encourage creativity)
--mask_reg_weight 0.1

# GAN (optional, for quality refinement)
--gan_weight 0.01
```

**Why this balance?**
- Spectral loss (0.01): Ensures output is musical, not noise
- Novelty loss (0.1): Encourages creative recombination
- GAN loss (0.01): Refines quality without dominating

### Training Script

```bash
./run_train_compositional.sh
```

**What it does**:
- Trains 2-stage cascade with compositional agent
- Extracts rhythm/harmony/timbre from input and target
- Composes NEW patterns through transformer
- Monitors component selection weights
- Saves checkpoints to `checkpoints_compositional/`

## Monitoring Training

### Component Statistics (Logged Every Epoch)

```
Component Selection:
  Input:
    - rhythm_weight: 0.42   (using 42% of input rhythm)
    - harmony_weight: 0.08  (using 8% of input harmony)
    - timbre_weight: 0.15   (using 15% of input timbre)
  Target:
    - rhythm_weight: 0.12   (using 12% of target rhythm)
    - harmony_weight: 0.78  (using 78% of target harmony)
    - timbre_weight: 0.65   (using 65% of target timbre)
```

**Interpretation**:
- High input_rhythm_weight: Output follows input's rhythm
- High target_harmony_weight: Output uses target's chords
- High target_timbre_weight: Output sounds like target

**Ideal Pattern** (for drums → piano example):
- input_rhythm: **High** (0.6-0.8)
- target_harmony: **High** (0.6-0.8)
- target_timbre: **High** (0.6-0.8)
- Others: **Low** (0.1-0.3)

### Novelty Loss (Logged Every Batch)

```
Novelty: 0.0423  (good - output differs from inputs)
Novelty: 0.0012  (too low - output copying inputs)
Novelty: 0.2341  (too high - output too random)
```

**Target**: 0.02-0.10 (creative but coherent)

## Inference

```bash
python inference_cascade.py \
    --checkpoint checkpoints_compositional/best_model.pt \
    --num_samples 5
```

**Output will show**:
```
Creative agent: Compositional (rhythm/harmony/timbre)
```

Generated audio will demonstrate **true creativity**:
- Not just amplitude modulation
- Not simple mixing of sources
- NEW musical patterns composed from components

## Comparison: Old vs New

### Old Creative Agent (Masking)
```
Input: [drums with strong beat]
Target: [piano playing C major]
Output: C major piano with drum's loudness envelope

Analysis: 
✗ Just amplitude modulation
✗ No new musical ideas
✗ Predictable mixing
```

### New Compositional Agent
```
Input: [drums with strong beat]
Target: [piano playing C major]
Output: Rhythmic C major piano arpeggios

Analysis:
✓ Extracts rhythm from drums
✓ Extracts harmony from piano
✓ Composes NEW arpeggio pattern
✓ Pattern in neither source!
```

## Expected Results

### After 20-30 Epochs:

1. **Component Selection Stabilizes**
   - Clear preference for input rhythm
   - Clear preference for target harmony/timbre
   - Weights show intelligent decomposition

2. **Novelty Increases**
   - Early: 0.001 (copying inputs)
   - Mid: 0.05 (creative recombination)
   - Late: 0.03 (balanced creativity)

3. **Output Quality**
   - Follows input's rhythmic structure
   - Uses target's harmonic content
   - Sounds like target's instrumentation
   - **But creates NEW patterns!**

## Troubleshooting

### Problem: Novelty loss stays near 0
**Cause**: Output copying one source directly
**Fix**: Increase `--mask_reg_weight` to 0.2 or 0.3

### Problem: Novelty loss > 0.2
**Cause**: Output too random, not coherent
**Fix**: Decrease `--mask_reg_weight` to 0.05
**Or**: Increase `--loss_weight_spectral` for musical guidance

### Problem: All component weights equal (~0.167)
**Cause**: Component selector not learning
**Fix**: Train longer, or increase batch size
**Or**: Add more spectral loss for musical feedback

### Problem: Output sounds like simple mixing
**Cause**: Composer not learning composition rules
**Fix**: Ensure cascade training (num_transformer_layers > 1)
**Or**: Increase composer complexity (more layers/heads)

## Key Advantages Over Masking Agent

| Feature | Masking Agent | Compositional Agent |
|---------|--------------|---------------------|
| **Creativity** | Linear mixing | True composition |
| **Musical Understanding** | None | Rhythm/harmony/timbre |
| **Output Novelty** | Low (amplitude modulation) | High (new patterns) |
| **Interpretability** | Opaque masks | Clear component weights |
| **Parameters** | 1.2M | 14.6M |
| **Training Stability** | Moderate | Good |

## References

**Compositional approach inspired by**:
- Style transfer in image generation (Gatys et al.)
- Musical source separation (Demucs, Spleeter)
- Hierarchical music generation (MusicVAE, MuseNet)

**Key insight**: 
Music has hierarchical structure (rhythm/harmony/timbre). By decomposing and recombining these components, we can create truly novel patterns rather than just mixing existing ones.
