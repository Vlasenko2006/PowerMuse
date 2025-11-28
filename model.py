import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from encoder_decoder import encoder_decoder  # Importing encoder-decoder class

# Dataset class
class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_chunk, target_chunk = self.data[idx]
        return torch.tensor(input_chunk, dtype=torch.float32), torch.tensor(target_chunk, dtype=torch.float32)


class AttentionModel(nn.Module):
    def __init__(self, input_dim, num_heads=8, num_layers=4, n_channels=64, n_seq=4, 
                 sound_channels=2, batch_size=64, seq_len=352800, dropout=0.15):
        """
        Enhanced Attention Model with improved architecture for music generation.
        
        Args:
            input_dim: Input dimension (sound_channels)
            num_heads: Number of attention heads (increased to 8)
            num_layers: Number of transformer layers (increased to 4)
            n_channels: Number of channels in encoded representation
            n_seq: Sequence multiplier for encoded length (increased to 4 for 16s)
            sound_channels: Number of audio channels (stereo=2)
            batch_size: Batch size for training
            seq_len: Sequence length for 16s audio at 22050 Hz (16 * 22050 = 352800)
            dropout: Dropout rate for regularization (0.15)
        """
        super(AttentionModel, self).__init__()

        # Encoder and Decoder from encoder-decoder with improved architecture
        self.encoder_decoder = encoder_decoder(input_dim=sound_channels, n_channels=n_channels, n_seq=n_seq)

        # Enhanced Transformer Encoder with more layers and dropout
        encoder_layer = TransformerEncoderLayer(
            d_model=n_channels,
            nhead=num_heads,
            dim_feedforward=512,  # Increased from 256 for better capacity
            dropout=dropout,
            activation='gelu',  # GELU activation for better performance
            batch_first=False
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Add layer normalization for stability
        self.layer_norm = nn.LayerNorm(n_channels)

    def forward(self, x):
        """
        Forward pass for the model with improved architecture.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim, seq_len].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Reconstructed tensor of shape [batch_size, input_dim, seq_len].
                - Output tensor of shape [batch_size, input_dim, seq_len].
        """
        # Verify input shape
        if len(x.shape) != 3:
            raise ValueError(f"Expected input to have 3 dimensions [batch_size, input_dim, seq_len], but got {x.shape}")

        batch_size, input_dim, seq_len = x.shape

        # Encoding
        encoded = self.encoder_decoder.encoder(x)  # Use encoder from encoder-decoder

        # Permute for Transformer (seq_len first for PyTorch Transformer)
        x = encoded.permute(2, 0, 1)  # Shape: [output_seq_len, batch_size, n_channels]

        # Apply layer normalization before transformer
        x = self.layer_norm(x)

        # Pass through Transformer Encoder
        transformer_out = self.transformer(x)  # Shape: [output_seq_len, batch_size, n_channels]

        # Permute back for Decoder
        transformer_out = transformer_out.permute(1, 2, 0)  # Shape: [batch_size, n_channels, output_seq_len]

        # Decoding
        reconstructed = self.encoder_decoder.decoder(encoded)  # Use decoder from encoder-decoder
        output = self.encoder_decoder.decoder(transformer_out)  # Decode to match input shape

        return reconstructed, output
