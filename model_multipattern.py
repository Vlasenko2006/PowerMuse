import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from encoder_decoder import encoder_decoder  # Importing encoder-decoder class


# Dataset class for multi-pattern
class MultiPatternAudioDataset(Dataset):
    def __init__(self, data):
        """
        Dataset for multi-pattern training.
        
        Args:
            data: List of ((input1, input2, input3), (target1, target2, target3))
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, targets = self.data[idx]
        
        # Stack inputs and targets: each is list of 3 arrays
        inputs_tensor = torch.tensor(torch.stack([torch.tensor(inp, dtype=torch.float32) for inp in inputs]), dtype=torch.float32)
        targets_tensor = torch.tensor(torch.stack([torch.tensor(tgt, dtype=torch.float32) for tgt in targets]), dtype=torch.float32)
        
        return inputs_tensor, targets_tensor


class MultiPatternAttentionModel(nn.Module):
    def __init__(self, input_dim, num_patterns=3, num_heads=8, num_layers=4, n_channels=64, n_seq=4, 
                 sound_channels=2, batch_size=64, seq_len=352800, dropout=0.15):
        """
        Multi-pattern fusion model for music generation.
        
        Processes 3 patterns independently through encoder, fuses them with transformer,
        and outputs single fused prediction.
        
        Args:
            input_dim: Input dimension (sound_channels)
            num_patterns: Number of input patterns (default 3)
            num_heads: Number of attention heads (8)
            num_layers: Number of transformer layers (4)
            n_channels: Number of channels in encoded representation (64)
            n_seq: Sequence multiplier for encoded length (4 for 16s)
            sound_channels: Number of audio channels (stereo=2)
            batch_size: Batch size for training
            seq_len: Sequence length for 16s audio at 22050 Hz (352800)
            dropout: Dropout rate for regularization (0.15)
        """
        super(MultiPatternAttentionModel, self).__init__()
        
        self.num_patterns = num_patterns
        self.n_channels = n_channels
        
        # Encoder and Decoder (processes each pattern independently)
        self.encoder_decoder = encoder_decoder(input_dim=sound_channels, n_channels=n_channels, n_seq=n_seq)

        # Enhanced Transformer Encoder for fusion
        encoder_layer = TransformerEncoderLayer(
            d_model=n_channels,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(n_channels)
        
        # Fusion layer to combine multiple patterns
        self.fusion_layer = nn.Linear(n_channels * num_patterns, n_channels)

    def forward(self, x, masks=None):
        """
        Forward pass for multi-pattern fusion.

        Args:
            x (torch.Tensor): Input tensor [batch, num_patterns, input_dim, seq_len]
            masks (torch.Tensor, optional): Boolean masks [batch, num_patterns, seq_len]
                                            True = keep, False = mask out

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - reconstructed: [batch, num_patterns, input_dim, seq_len] - reconstructed inputs
                - output: [batch, input_dim, seq_len] - single fused prediction
        """
        # Verify input shape
        if len(x.shape) != 4:
            raise ValueError(f"Expected input shape [batch, num_patterns, input_dim, seq_len], got {x.shape}")

        batch_size, num_patterns, input_dim, seq_len = x.shape
        
        if num_patterns != self.num_patterns:
            raise ValueError(f"Expected {self.num_patterns} patterns, got {num_patterns}")

        # Process each pattern independently through encoder
        encoded_list = []
        reconstructed_list = []
        
        for i in range(num_patterns):
            # Extract single pattern: [batch, input_dim, seq_len]
            pattern = x[:, i, :, :]
            
            # Encode
            encoded = self.encoder_decoder.encoder(pattern)  # [batch, n_channels, encoded_len]
            encoded_list.append(encoded)
            
            # Decode for reconstruction
            reconstructed = self.encoder_decoder.decoder(encoded)  # [batch, input_dim, seq_len]
            reconstructed_list.append(reconstructed)
        
        # Stack encoded representations: [batch, num_patterns, n_channels, encoded_len]
        encoded_stack = torch.stack(encoded_list, dim=1)
        
        # Reshape for transformer: need to process all patterns together
        # [batch, num_patterns, n_channels, encoded_len] -> [encoded_len, batch*num_patterns, n_channels]
        b, p, c, l = encoded_stack.shape
        
        # Reshape: [batch*num_patterns, n_channels, encoded_len]
        transformer_input = encoded_stack.view(b * p, c, l)
        
        # Permute for transformer (seq_len first)
        transformer_input = transformer_input.permute(2, 0, 1)  # [encoded_len, batch*num_patterns, n_channels]
        
        # Apply layer normalization
        transformer_input = self.layer_norm(transformer_input)
        
        # Pass through transformer
        transformer_out = self.transformer(transformer_input)  # [encoded_len, batch*num_patterns, n_channels]
        
        # Reshape back: [encoded_len, batch, num_patterns, n_channels]
        transformer_out = transformer_out.view(l, b, p, c)
        
        # Permute: [batch, num_patterns, n_channels, encoded_len]
        transformer_out = transformer_out.permute(1, 2, 3, 0)
        
        # Fuse patterns: [batch, num_patterns, n_channels, encoded_len] -> [batch, n_channels, encoded_len]
        # Reshape for fusion: [batch, encoded_len, num_patterns * n_channels]
        transformer_out = transformer_out.permute(0, 3, 1, 2)  # [batch, encoded_len, num_patterns, n_channels]
        b, l, p, c = transformer_out.shape
        transformer_out = transformer_out.reshape(b, l, p * c)  # [batch, encoded_len, num_patterns*n_channels]
        
        # Apply fusion layer
        fused = self.fusion_layer(transformer_out)  # [batch, encoded_len, n_channels]
        
        # Permute back: [batch, n_channels, encoded_len]
        fused = fused.permute(0, 2, 1)
        
        # Decode fused representation to single output
        output = self.encoder_decoder.decoder(fused)  # [batch, input_dim, seq_len]
        
        # Stack reconstructions: [batch, num_patterns, input_dim, seq_len]
        reconstructed = torch.stack(reconstructed_list, dim=1)
        
        return reconstructed, output
