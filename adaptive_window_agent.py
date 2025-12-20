"""
Adaptive Window Creative Agent: Multi-Window Selection with Compression and Tonality Reduction
===============================================================================================

This module extends the compositional creative agent with adaptive window selection:
- Input/Target: 24 seconds (1200 EnCodec frames)
- Selects 3 pairs of 16-second windows (800 frames each)
- Learns optimal window positions, compression ratios, and tonality transformations
- Processes all 3 pairs through transformer
- Uses MEAN loss across all pairs for gradient flow

Key Features:
-------------
1. **Learnable Window Positions**: Agent learns where to extract patterns
   - Can select overlapping or non-overlapping regions
   - Independent positions for input and target

2. **Temporal Compression**: Compress patterns by up to 1.5x to match rhythm
   - Select 19.2 sec (960 frames) and compress to 16 sec (800 frames)
   - Differentiable resampling in latent space
   - Learned compression ratios per pair

3. **Tonality Reduction**: Transform harmonic content in latent space
   - Learned convolution layers that modify frequency content
   - Can reduce/enhance tonality to improve matching
   - Applied independently to input and target

4. **Multi-Pair Processing**: Process 3 pairs, average losses
   - Encourages diversity: agent learns 3 different strategies
   - Mean loss allows gradients to flow to all pairs equally
   - No hard selection (argmin) - all pairs contribute

Architecture:
-------------
Input: encoded_input [B, 128, 1200], encoded_target [B, 128, 1200]
  ↓
WindowSelector: Predicts 3 pairs of (start_input, start_target, compression_ratio)
  ↓
For each pair:
  1. Extract windows: encoded[:, :, start:start+duration]
  2. Apply compression: F.interpolate to 800 frames
  3. Apply tonality reduction: learned Conv1d transformation
  4. Send to compositional agent + cascade transformer
  5. Compute loss
  ↓
total_loss = mean(loss_pair1, loss_pair2, loss_pair3)
  ↓
Backprop gradients flow to all 3 pairs equally

Usage:
------
# Initialize
agent = AdaptiveWindowCreativeAgent(encoding_dim=128)

# Forward (training)
outputs, losses, metadata = agent(
    encoded_input_24sec,   # [B, 128, 1200]
    encoded_target_24sec   # [B, 128, 1200]
)
total_loss = torch.mean(torch.stack(losses))

# Inference (use best pair)
output, _, metadata = agent(encoded_input_24sec, encoded_target_24sec)
best_pair_idx = torch.argmin(torch.stack(losses))
best_output = outputs[best_pair_idx]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from compositional_creative_agent import (
    CompositionalCreativeAgent,
    MultiScaleExtractor,
    ComponentComposer
)


class WindowSelector(nn.Module):
    """
    Learns optimal window positions and compression ratios for 3 pairs.
    
    Outputs:
        - 3 start positions for input (range: 0 to 400 frames = 0 to 8 seconds)
        - 3 start positions for target (range: 0 to 400 frames = 0 to 8 seconds)
        - 3 compression ratios (range: 1.0 to 1.5)
    
    The agent can select up to 19.2 seconds (960 frames) and compress to 16 sec (800 frames).
    """
    
    def __init__(self, encoding_dim=128, num_pairs=3):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.num_pairs = num_pairs
        
        # Temporal pooling to get global representation
        self.temporal_pool = nn.AdaptiveAvgPool1d(50)  # [B, 128, 1200] -> [B, 128, 50]
        
        # MLP to predict window parameters
        self.mlp = nn.Sequential(
            nn.Linear(encoding_dim * 50 * 2, 512),  # Input + target concatenated
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_pairs * 5)  # 3 pairs × 5 params each
        )
        
        # Initialize to reasonable defaults
        with torch.no_grad():
            # Bias towards uniform spacing: pair 0 at start, pair 1 at middle, pair 2 at end
            self.mlp[-1].bias.data.zero_()
            for i in range(num_pairs):
                # Start positions: evenly spaced
                self.mlp[-1].bias.data[i * 5 + 0] = i * 2.0  # Input start
                self.mlp[-1].bias.data[i * 5 + 1] = i * 2.0  # Target start
                # Compression ratios: default 1.0 (no compression)
                self.mlp[-1].bias.data[i * 5 + 2] = 0.0
                self.mlp[-1].bias.data[i * 5 + 3] = 0.0
                # Enable tonality reduction
                self.mlp[-1].bias.data[i * 5 + 4] = 0.5
    
    def forward(self, encoded_input, encoded_target):
        """
        Args:
            encoded_input: [B, 128, 1200] (24 seconds)
            encoded_target: [B, 128, 1200] (24 seconds)
        
        Returns:
            window_params: List of 3 dicts, each containing:
                - start_input: int (0 to 400)
                - start_target: int (0 to 400)
                - ratio_input: float (1.0 to 1.5)
                - ratio_target: float (1.0 to 1.5)
                - tonality_strength: float (0.0 to 1.0)
        """
        B = encoded_input.size(0)
        
        # Pool temporal dimension
        input_pooled = self.temporal_pool(encoded_input)  # [B, 128, 50]
        target_pooled = self.temporal_pool(encoded_target)  # [B, 128, 50]
        
        # Concatenate and flatten
        combined = torch.cat([input_pooled, target_pooled], dim=1)  # [B, 256, 50]
        combined_flat = combined.reshape(B, -1)  # [B, 12800]
        
        # Predict parameters
        params = self.mlp(combined_flat)  # [B, num_pairs * 5]
        params = params.reshape(B, self.num_pairs, 5)  # [B, 3, 5]
        
        # Parse parameters for each pair
        window_params = []
        for i in range(self.num_pairs):
            # Start positions (constrained to valid range)
            # Max start: 1200 - 960 = 240 frames (to allow 1.5x compression)
            # But we'll use 400 (8 sec) as soft limit with sigmoid
            start_input = torch.sigmoid(params[:, i, 0]) * 400  # [B] range [0, 400]
            start_target = torch.sigmoid(params[:, i, 1]) * 400  # [B] range [0, 400]
            
            # Compression ratios (1.0 to 1.5)
            ratio_input = 1.0 + torch.sigmoid(params[:, i, 2]) * 0.5  # [B] range [1.0, 1.5]
            ratio_target = 1.0 + torch.sigmoid(params[:, i, 3]) * 0.5  # [B] range [1.0, 1.5]
            
            # Tonality reduction strength (0.0 to 1.0)
            tonality_strength = torch.sigmoid(params[:, i, 4])  # [B] range [0.0, 1.0]
            
            window_params.append({
                'start_input': start_input,
                'start_target': start_target,
                'ratio_input': ratio_input,
                'ratio_target': ratio_target,
                'tonality_strength': tonality_strength
            })
        
        return window_params


class TemporalCompressor(nn.Module):
    """
    Compresses temporal dimension using learned interpolation.
    
    Example:
        Input: [B, 128, 960] (19.2 seconds)
        Ratio: 1.2
        Output: [B, 128, 800] (16 seconds, 1.2x faster)
    """
    
    def __init__(self):
        super().__init__()
        # No learnable parameters - uses F.interpolate
        # Could add learnable anti-aliasing filters here if needed
    
    def forward(self, encoded, compression_ratio, target_length=800):
        """
        Args:
            encoded: [B, D, T] - Input encoding
            compression_ratio: [B] or scalar - How much to compress (1.0 = no compression, 1.5 = 1.5x faster)
            target_length: int - Output length (default 800 frames = 16 seconds)
        
        Returns:
            compressed: [B, D, target_length]
        """
        B, D, T = encoded.shape
        
        # Calculate source length needed for this compression ratio
        # If ratio=1.2, we need 1.2 * 800 = 960 frames to compress to 800
        if isinstance(compression_ratio, torch.Tensor):
            # Batch-wise compression (different ratio per sample)
            compressed_list = []
            for b in range(B):
                ratio = compression_ratio[b].item()
                source_length = int(target_length * ratio)
                source_length = min(source_length, T)  # Don't exceed available length
                
                # Extract source region (centered)
                start = (T - source_length) // 2
                source = encoded[b:b+1, :, start:start+source_length]  # [1, D, source_length]
                
                # Interpolate to target length
                compressed = F.interpolate(
                    source, 
                    size=target_length, 
                    mode='linear', 
                    align_corners=False
                )  # [1, D, target_length]
                compressed_list.append(compressed)
            
            compressed = torch.cat(compressed_list, dim=0)  # [B, D, target_length]
        else:
            # Same compression for all samples
            source_length = int(target_length * compression_ratio)
            source_length = min(source_length, T)
            
            # Extract source region (centered)
            start = (T - source_length) // 2
            source = encoded[:, :, start:start+source_length]  # [B, D, source_length]
            
            # Interpolate to target length
            compressed = F.interpolate(
                source,
                size=target_length,
                mode='linear',
                align_corners=False
            )  # [B, D, target_length]
        
        return compressed


class TonalityReducer(nn.Module):
    """
    Learns to transform latent encodings to modify harmonic content.
    
    Uses depthwise separable convolutions to modify spectral patterns
    without changing temporal structure.
    
    Can be used to:
    - Reduce tonality of percussive input to match tonal target
    - Enhance harmonics to match key/scale
    - Filter out conflicting frequency components
    """
    
    def __init__(self, encoding_dim=128):
        super().__init__()
        
        # Depthwise separable convolution (efficient)
        # Kernel=7 to capture harmonic patterns
        self.depthwise = nn.Conv1d(
            encoding_dim, encoding_dim, 
            kernel_size=7, padding=3, groups=encoding_dim
        )
        self.pointwise = nn.Conv1d(encoding_dim, encoding_dim, kernel_size=1)
        
        # Normalization
        self.norm = nn.LayerNorm(encoding_dim)
        
        # Residual connection strength (learned per sample via tonality_strength)
        # strength=0 → output=input (no reduction)
        # strength=1 → output=transformed (full reduction)
        
        # Initialize to identity-like transformation
        with torch.no_grad():
            self.depthwise.weight.data.fill_(0.0)
            self.depthwise.weight.data[:, :, 3] = 1.0  # Center tap = 1
            self.depthwise.bias.data.zero_()
            self.pointwise.weight.data.copy_(torch.eye(encoding_dim).unsqueeze(-1))
            self.pointwise.bias.data.zero_()
    
    def forward(self, encoded, tonality_strength):
        """
        Args:
            encoded: [B, D, T] - Input encoding
            tonality_strength: [B] - How much to apply transformation (0=none, 1=full)
        
        Returns:
            transformed: [B, D, T] - Transformed encoding
        """
        B, D, T = encoded.shape
        
        # Apply transformation
        x = self.depthwise(encoded)  # [B, D, T]
        x = self.pointwise(x)  # [B, D, T]
        
        # Normalize (in channel dimension)
        x = x.transpose(1, 2)  # [B, T, D]
        x = self.norm(x)  # [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        
        # Blend with input based on tonality_strength
        # Reshape strength for broadcasting: [B] -> [B, 1, 1]
        strength = tonality_strength.view(B, 1, 1)
        
        # Residual connection
        output = (1.0 - strength) * encoded + strength * x
        
        return output


class AdaptiveWindowCreativeAgent(nn.Module):
    """
    Main agent that combines window selection, compression, and composition.
    
    Architecture:
        1. WindowSelector: Predicts 3 pairs of window parameters
        2. For each pair:
            a. Extract windows from 24-sec input/target
            b. Compress temporally to 16 sec
            c. Apply tonality reduction
            d. Process through compositional creative agent
        3. Return all 3 outputs and losses
    """
    
    def __init__(self, encoding_dim=128, num_pairs=3):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.num_pairs = num_pairs
        
        # Components
        self.window_selector = WindowSelector(encoding_dim, num_pairs)
        self.temporal_compressor = TemporalCompressor()
        self.tonality_reducer_input = TonalityReducer(encoding_dim)
        self.tonality_reducer_target = TonalityReducer(encoding_dim)
        
        # Compositional creative agent (shared across all pairs)
        self.creative_agent = CompositionalCreativeAgent(encoding_dim)
        
        # Silent initialization (print suppressed for DDP)
    
    def extract_window(self, encoded, start_position, duration=800):
        """
        Extract window from encoded sequence.
        
        Args:
            encoded: [B, D, T] - Full sequence (T=1200)
            start_position: [B] - Start positions in frames
            duration: int - Window duration (default 800 frames = 16 sec)
        
        Returns:
            window: [B, D, duration]
        """
        B, D, T = encoded.shape
        
        # Extract windows batch-wise (different start per sample)
        windows = []
        for b in range(B):
            start = int(start_position[b].item())
            start = max(0, min(start, T - duration))  # Clamp to valid range
            window = encoded[b:b+1, :, start:start+duration]  # [1, D, duration]
            windows.append(window)
        
        return torch.cat(windows, dim=0)  # [B, D, duration]
    
    def forward(self, encoded_input_24sec, encoded_target_24sec):
        """
        Args:
            encoded_input_24sec: [B, 128, 1200] (24 seconds)
            encoded_target_24sec: [B, 128, 1200] (24 seconds)
        
        Returns:
            outputs: List of 3 creative outputs [B, 128, 800]
            losses: List of 3 novelty losses (tensors)
            metadata: Dict with selection parameters for logging
        """
        B, D, T = encoded_input_24sec.shape
        assert T == 1200, f"Expected T=1200 (24 sec), got {T}"
        assert D == self.encoding_dim, f"Expected D={self.encoding_dim}, got {D}"
        
        # Step 1: Predict window parameters
        window_params = self.window_selector(encoded_input_24sec, encoded_target_24sec)
        
        # Step 2: Process each pair
        outputs = []
        losses = []
        
        for pair_idx, params in enumerate(window_params):
            # Extract base windows (before compression)
            # We extract slightly longer windows and then compress
            window_input_raw = self.extract_window(
                encoded_input_24sec, 
                params['start_input'],
                duration=960  # 19.2 sec (will compress to 16 sec)
            )
            window_target_raw = self.extract_window(
                encoded_target_24sec,
                params['start_target'],
                duration=960  # 19.2 sec (will compress to 16 sec)
            )
            
            # Compress temporally
            window_input_compressed = self.temporal_compressor(
                window_input_raw,
                params['ratio_input'],
                target_length=800
            )
            window_target_compressed = self.temporal_compressor(
                window_target_raw,
                params['ratio_target'],
                target_length=800
            )
            
            # Apply tonality reduction
            window_input_final = self.tonality_reducer_input(
                window_input_compressed,
                params['tonality_strength']
            )
            window_target_final = self.tonality_reducer_target(
                window_target_compressed,
                params['tonality_strength']
            )
            
            # Process through compositional creative agent
            creative_output, novelty_loss = self.creative_agent(
                window_input_final,
                window_target_final
            )
            
            outputs.append(creative_output)
            losses.append(novelty_loss)
        
        # Collect metadata for logging
        metadata = {
            'num_pairs': self.num_pairs,
            'pairs': []
        }
        for pair_idx, params in enumerate(window_params):
            pair_meta = {
                'start_input_mean': params['start_input'].mean().item(),
                'start_target_mean': params['start_target'].mean().item(),
                'ratio_input_mean': params['ratio_input'].mean().item(),
                'ratio_target_mean': params['ratio_target'].mean().item(),
                'tonality_strength_mean': params['tonality_strength'].mean().item(),
            }
            metadata['pairs'].append(pair_meta)
        
        return outputs, losses, metadata
    
    def get_component_statistics(self, encoded_input, encoded_target):
        """
        For compatibility with existing code that expects component stats.
        Returns statistics from the shared compositional agent.
        """
        return self.creative_agent.get_component_statistics(encoded_input, encoded_target)


if __name__ == "__main__":
    """
    Test the adaptive window agent with dummy data.
    """
    print("Testing AdaptiveWindowCreativeAgent...")
    
    # Create agent
    agent = AdaptiveWindowCreativeAgent(encoding_dim=128, num_pairs=3)
    
    # Create dummy inputs (24 seconds = 1200 frames)
    B = 2
    encoded_input = torch.randn(B, 128, 1200)
    encoded_target = torch.randn(B, 128, 1200)
    
    # Forward pass
    print("\nForward pass...")
    outputs, losses, metadata = agent(encoded_input, encoded_target)
    
    print(f"\nResults:")
    print(f"  Number of pairs: {len(outputs)}")
    for i, (output, loss) in enumerate(zip(outputs, losses)):
        print(f"  Pair {i}:")
        print(f"    Output shape: {output.shape}")  # Should be [2, 128, 800]
        print(f"    Novelty loss: {loss.item():.6f}")
    
    print(f"\nMetadata:")
    for i, pair_meta in enumerate(metadata['pairs']):
        print(f"  Pair {i}:")
        print(f"    Start input: {pair_meta['start_input_mean']:.1f} frames ({pair_meta['start_input_mean']/50:.2f} sec)")
        print(f"    Start target: {pair_meta['start_target_mean']:.1f} frames ({pair_meta['start_target_mean']/50:.2f} sec)")
        print(f"    Compression input: {pair_meta['ratio_input_mean']:.3f}x")
        print(f"    Compression target: {pair_meta['ratio_target_mean']:.3f}x")
        print(f"    Tonality strength: {pair_meta['tonality_strength_mean']:.3f}")
    
    # Test mean loss calculation
    mean_loss = torch.mean(torch.stack(losses))
    print(f"\nMean loss across all pairs: {mean_loss.item():.6f}")
    
    # Test backward pass
    print("\nTesting backward pass...")
    mean_loss.backward()
    print("  Gradients computed successfully!")
    
    print("\n✓ All tests passed!")
