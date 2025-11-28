# Multi-Pattern Fusion Implementation Plan

## Architecture Overview

**Input**: 3 different songs, 16s each → **Output**: 1 fused pattern (16s)

```
[Jingle Bells 0-16s, Christmas Tree 0-16s, Shchedryk 0-16s] 
    ↓ (with random masks per pattern)
Encoder (processes each independently)
    ↓
[3 encoded representations]
    ↓
Transformer (fuses all 3)
    ↓
[1 fused encoded representation]
    ↓
Decoder
    ↓
[1 output 16s prediction]

Compare with 3 targets:
[Jingle Bells 16-32s, Christmas Tree 16-32s, Shchedryk 16-32s]
```

## Key Changes Required

### 1. ✅ masking_utils.py (DONE)
- Random mask generation per pattern
- Fixed masks for validation
- Attention mask conversion
- Overlapping chunk splitting
- Ensures ≥16s unmasked

### 2. encoder_decoder.py - NEEDS UPDATE
**Current**: Processes [batch, 2_channels, 352800_samples]
**New**: Must process [batch, 3_patterns, 2_channels, 352800_samples]

```python
# Process each pattern independently
for i in range(3):
    encoded[i] = encoder(input[:, i, :, :])  # [batch, 2_ch, 352800]
    reconstructed[i] = decoder(encoded[i])
```

### 3. model.py - MAJOR REFACTOR NEEDED
**New forward pass**:
```python
def forward(self, x, masks=None):
    # x: [batch, 3_patterns, 2_channels, 352800]
    # masks: [batch, 3_patterns, 352800] or None
    
    batch, num_patterns, channels, seq_len = x.shape
    
    # Encode each pattern separately
    encoded_list = []
    reconstructed_list = []
    for i in range(num_patterns):
        enc = self.encoder_decoder.encoder(x[:, i, :, :])
        encoded_list.append(enc)
        rec = self.encoder_decoder.decoder(enc)
        reconstructed_list.append(rec)
    
    # Stack: [batch, 3, n_channels, encoded_len]
    encoded_stack = torch.stack(encoded_list, dim=1)
    
    # Reshape for transformer: [batch*3, n_channels, encoded_len]
    b, p, c, l = encoded_stack.shape
    transformer_input = encoded_stack.view(b*p, c, l)
    
    # Permute: [encoded_len, batch*3, n_channels]
    transformer_input = transformer_input.permute(2, 0, 1)
    
    # Apply transformer (fuses patterns)
    transformer_out = self.transformer(transformer_input)
    
    # Take mean across patterns to fuse
    # Reshape: [encoded_len, batch, 3, n_channels]
    transformer_out = transformer_out.view(l, b, p, c)
    # Fuse: [encoded_len, batch, n_channels]
    fused = transformer_out.mean(dim=2)
    # Permute back: [batch, n_channels, encoded_len]
    fused = fused.permute(1, 2, 0)
    
    # Decode fused representation
    output = self.encoder_decoder.decoder(fused)
    
    # Stack reconstructions
    reconstructed = torch.stack(reconstructed_list, dim=1)
    
    return reconstructed, output
```

### 4. Loss Function - NEW FILE: fusion_loss.py
```python
def chunk_wise_mse_loss(output, targets, masks, sample_rate=22050):
    """
    Compute min(MSE) across overlapping chunks for 3 targets.
    
    Args:
        output: [batch, 2_channels, 352800] - single fused prediction
        targets: [batch, 3_targets, 2_channels, 352800] - 3 continuations
        masks: [batch, 3_patterns, 352800] - masks used for inputs
    
    Returns:
        loss: Scalar, sum of min-MSE for each target
    """
    batch_size, num_targets = targets.shape[0], targets.shape[1]
    total_loss = 0.0
    
    for b in range(batch_size):
        for t in range(num_targets):
            target = targets[b, t]  # [2, 352800]
            mask = masks[b, t]      # [352800]
            out = output[b]          # [2, 352800]
            
            # Get unmasked length for chunking
            unmasked_samples = mask.sum().item()
            
            # Split into overlapping chunks
            target_chunks = create_overlapping_chunks(target, unmasked_samples)
            output_chunks = create_overlapping_chunks(out, unmasked_samples)
            
            # Compute MSE for each chunk
            chunk_mses = []
            for t_chunk, o_chunk in zip(target_chunks, output_chunks):
                mse = F.mse_loss(o_chunk, t_chunk)
                chunk_mses.append(mse)
            
            # Take minimum MSE
            min_mse = min(chunk_mses)
            total_loss += min_mse
    
    return total_loss / (batch_size * num_targets)
```

### 5. create_dataset.py - NEEDS UPDATE
**Current**: Creates pairs `(input_16s, target_16s)`
**New**: Creates triplets from 3 different songs

```python
def create_multi_pattern_dataset(output_folder, dataset_folder, num_patterns=3):
    """
    Create dataset with triplets of different songs.
    
    Each training sample:
    - inputs: [pattern1_0-16s, pattern2_0-16s, pattern3_0-16s]
    - targets: [pattern1_16-32s, pattern2_16-32s, pattern3_16-32s]
    """
    npy_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".npy")])
    
    all_chunks = []
    for npy_file in npy_files:
        audio = np.load(os.path.join(output_folder, npy_file))
        # Split into 12 chunks of 16s
        chunks = [audio[..., i*352800:(i+1)*352800] for i in range(12)]
        all_chunks.append(chunks)
    
    training_data = []
    
    # Create triplets
    for _ in range(len(all_chunks) * 4):  # Generate multiple combinations
        # Sample 3 different songs
        song_indices = np.random.choice(len(all_chunks), 3, replace=False)
        
        inputs = []
        targets = []
        for song_idx in song_indices:
            chunks = all_chunks[song_idx]
            # Pick consecutive chunks
            pair_idx = np.random.randint(0, len(chunks)-1)
            inputs.append(chunks[pair_idx])
            targets.append(chunks[pair_idx + 1])
        
        training_data.append((inputs, targets))
    
    # Shuffle and split
    random.shuffle(training_data)
    split_idx = int(0.9 * len(training_data))
    return training_data[:split_idx], training_data[split_idx:]
```

### 6. train_and_validate.py - 3 PHASE TRAINING

```python
def train_and_validate_multi_pattern(
    model, train_loader, val_loader, 
    start_epoch, epochs,
    phase1_epochs=10,  # Unmasked encoder-decoder
    phase2_epochs=20,  # Masked encoder-decoder
    ...
):
    for epoch in range(start_epoch, epochs + 1):
        # Determine phase
        if epoch <= phase1_epochs:
            use_masks = False
            train_transformer = False
        elif epoch <= phase2_epochs:
            use_masks = True
            train_transformer = False
        else:
            use_masks = True
            train_transformer = True
        
        for inputs, targets in train_loader:
            # inputs: [batch, 3, 2, 352800]
            # targets: [batch, 3, 2, 352800]
            
            if use_masks:
                masks = generate_batch_masks(batch_size, num_patterns=3)
                masked_inputs = apply_mask(inputs, masks)
            else:
                masks = None
                masked_inputs = inputs
            
            # Forward
            reconstructed, output = model(masked_inputs, masks)
            
            # Reconstruction loss (all 3 patterns)
            rec_loss = 0
            for i in range(3):
                rec_loss += criterion(reconstructed[:, i], inputs[:, i])
            rec_loss /= 3
            
            if train_transformer:
                # Chunk-wise MSE loss for fusion
                task_loss = chunk_wise_mse_loss(output, targets, masks)
                total_loss = 0.3 * rec_loss + 0.7 * task_loss
            else:
                total_loss = rec_loss
            
            # Backward
            total_loss.backward()
            optimizer.step()
```

### 7. main.py - UPDATE CONFIGURATION

```python
# Model for multi-pattern
model = MultiPatternAttentionModel(
    input_dim=2,
    num_patterns=3,  # NEW
    n_channels=64,
    ...
)

# Dataset format
train_dataset = MultiPatternAudioDataset(train_data)
# Returns: ((input1, input2, input3), (target1, target2, target3))
```

## Implementation Priority

1. ✅ **masking_utils.py** - DONE
2. **fusion_loss.py** - Create chunk-wise MSE loss
3. **encoder_decoder.py** - Minimal changes (works per-pattern)
4. **model.py** - Major refactor for 3-pattern processing
5. **create_dataset.py** - Generate triplets
6. **train_and_validate.py** - 3-phase training
7. **main.py** - Update configuration

## Estimated Complexity

- **Small changes**: masking_utils (done), fusion_loss
- **Medium changes**: create_dataset, main
- **Large changes**: model.py (new forward logic), train_and_validate (3 phases)

## Next Steps

Would you like me to:
1. **Implement all changes now** (will take several iterations)
2. **Implement one component at a time** (more controlled)
3. **Create a new branch/folder** for multi-pattern version (keep single-pattern working)

