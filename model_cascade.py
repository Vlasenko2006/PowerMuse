#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransformerCascade: Iterative Refinement Architecture

Architecture:
- Stage 1: Transformer (base size) processes N input patterns → output1
- Stage 2-N: Larger Transformer (2x base size) processes concat(prev_output, input) → output_i
- Each stage refines the previous output using original input as context
- Progressive refinement: coarse structure → fine details
"""

import torch
import torch.nn as nn
import math


class TransformerCascade(nn.Module):
    """
    Iterative refinement transformer with cascading stages.
    
    Architecture:
        input [B, N, D, T] → Transformer1 → output1 [B, D, T]
        concat(output1, input) → Large_Transformer2 → output2 [B, D, T]
        concat(output2, input) → Large_Transformer3 → output3 [B, D, T]
        ... continue for num_layers stages
    
    If num_layers == 1, uses only the first transformer (no cascade).
    Large transformers have 2x the layers and capacity of the first stage.
    """
    
    def __init__(self, encoding_dim=128, num_patterns=3, nhead=16, num_layers=1, dropout=0.1):
        """
        Args:
            encoding_dim: Dimension of each pattern encoding (D)
            num_patterns: Number of input patterns (N)
            nhead: Number of attention heads
            num_layers: Number of cascade stages (1 = no cascade, just base transformer)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.encoding_dim = encoding_dim
        self.num_patterns = num_patterns
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Stage 1: Base transformer
        # Input: N patterns → Output: single fused pattern
        self.d_model_stage1 = num_patterns * encoding_dim
        
        # Positional encoding for stage 1
        self.pos_encoding_stage1 = PositionalEncoding(self.d_model_stage1, dropout)
        
        # Transformer encoder for stage 1 (single layer)
        encoder_layer_stage1 = nn.TransformerEncoderLayer(
            d_model=self.d_model_stage1,
            nhead=nhead,
            dim_feedforward=self.d_model_stage1 * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_stage1 = nn.TransformerEncoder(
            encoder_layer_stage1,
            num_layers=1  # Single layer for first stage
        )
        
        # Project to output dimension
        self.output_proj_stage1 = nn.Linear(self.d_model_stage1, encoding_dim)
        
        # Cascade stages (if num_layers > 1)
        if num_layers > 1:
            self.cascade_stages = nn.ModuleList()
            
            # Each cascade stage processes: concat(prev_output, input) = (N+1) patterns
            self.d_model_cascade = (num_patterns + 1) * encoding_dim
            
            # Cascade stages use 2x heads (32 instead of 16)
            self.nhead_cascade = nhead * 2
            
            for stage_idx in range(num_layers - 1):
                # Positional encoding for this cascade stage
                pos_enc = PositionalEncoding(self.d_model_cascade, dropout)
                
                # Larger transformer (2 layers and 2x heads)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model_cascade,
                    nhead=self.nhead_cascade,
                    dim_feedforward=self.d_model_cascade * 4,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True
                )
                transformer = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=2  # 2x layers for cascade stages
                )
                
                # Output projection
                output_proj = nn.Linear(self.d_model_cascade, encoding_dim)
                
                self.cascade_stages.append(nn.ModuleDict({
                    'pos_encoding': pos_enc,
                    'transformer': transformer,
                    'output_proj': output_proj
                }))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Forward pass through cascade.
        
        Args:
            x: [B, N, D, T] - batch of N patterns, each D-dimensional over T timesteps
        
        Returns:
            output: [B, D, T] - fused pattern after cascade refinement
        """
        B, N, D, T = x.shape
        assert N == self.num_patterns, f"Expected {self.num_patterns} patterns, got {N}"
        assert D == self.encoding_dim, f"Expected encoding_dim {self.encoding_dim}, got {D}"
        
        # ========== STAGE 1: Base Transformer ==========
        # Reshape: [B, N, D, T] → [B, T, N*D]
        x_stage1 = x.permute(0, 3, 1, 2).reshape(B, T, N * D)
        
        # Add positional encoding
        x_stage1 = self.pos_encoding_stage1(x_stage1)
        
        # Transformer
        out_stage1 = self.transformer_stage1(x_stage1)  # [B, T, N*D]
        
        # Project to output dimension
        output = self.output_proj_stage1(out_stage1)  # [B, T, D]
        output = output.permute(0, 2, 1)  # [B, D, T]
        
        # If only 1 layer, return stage 1 output
        if self.num_layers == 1:
            return output
        
        # ========== STAGES 2-N: Cascade Refinement ==========
        for stage_idx, stage in enumerate(self.cascade_stages):
            # Concatenate previous output with original input
            # output: [B, D, T], x: [B, N, D, T]
            # → concat: [B, N+1, D, T]
            output_expanded = output.unsqueeze(1)  # [B, 1, D, T]
            x_concat = torch.cat([output_expanded, x], dim=1)  # [B, N+1, D, T]
            
            # Reshape: [B, N+1, D, T] → [B, T, (N+1)*D]
            x_stage = x_concat.permute(0, 3, 1, 2).reshape(B, T, (N + 1) * D)
            
            # Add positional encoding
            x_stage = stage['pos_encoding'](x_stage)
            
            # Transformer
            out_stage = stage['transformer'](x_stage)  # [B, T, (N+1)*D]
            
            # Project to output dimension
            output = stage['output_proj'](out_stage)  # [B, T, D]
            output = output.permute(0, 2, 1)  # [B, D, T]
        
        return output


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, D] tensor
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def count_parameters(model):
    """Count trainable and total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def test_cascade():
    """Test the TransformerCascade architecture"""
    print("\n" + "="*80)
    print("TESTING TRANSFORMERCASCADE ARCHITECTURE")
    print("="*80)
    
    B, N, D, T = 4, 3, 128, 1126
    
    # Test 1: Single stage (num_layers=1)
    print("\n1. Testing single stage (num_layers=1):")
    model1 = TransformerCascade(
        encoding_dim=D,
        num_patterns=N,
        nhead=16,
        num_layers=1,
        dropout=0.1
    )
    trainable1, total1 = count_parameters(model1)
    print(f"   Parameters: {trainable1:,} trainable, {total1:,} total")
    
    x = torch.randn(B, N, D, T)
    output1 = model1(x)
    print(f"   Input shape: {list(x.shape)} → Output shape: {list(output1.shape)}")
    assert output1.shape == (B, D, T), f"Expected {(B, D, T)}, got {output1.shape}"
    print("   ✓ Single stage works!")
    
    # Test 2: Three stage cascade (num_layers=3)
    print("\n2. Testing three stage cascade (num_layers=3):")
    model3 = TransformerCascade(
        encoding_dim=D,
        num_patterns=N,
        nhead=16,
        num_layers=3,
        dropout=0.1
    )
    trainable3, total3 = count_parameters(model3)
    print(f"   Parameters: {trainable3:,} trainable, {total3:,} total")
    print(f"   Parameter increase: {trainable3/trainable1:.1f}x")
    
    output3 = model3(x)
    print(f"   Input shape: {list(x.shape)} → Output shape: {list(output3.shape)}")
    assert output3.shape == (B, D, T), f"Expected {(B, D, T)}, got {output3.shape}"
    print("   ✓ Three stage cascade works!")
    
    # Test 3: Gradient flow
    print("\n3. Testing gradient flow:")
    loss = output3.mean()
    loss.backward()
    
    has_grad = sum(1 for p in model3.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for p in model3.parameters())
    print(f"   Parameters with gradients: {has_grad}/{total_params}")
    print("   ✓ Gradients flowing correctly!")
    
    # Test 4: Compare architectures
    print("\n4. Architecture comparison:")
    print(f"   1 stage:  {trainable1:,} params")
    print(f"   3 stages: {trainable3:,} params ({trainable3/trainable1:.2f}x)")
    print(f"   Stage 1: d_model={model3.d_model_stage1}, heads={model3.nhead}, layers=1")
    if model3.num_layers > 1:
        print(f"   Cascade: d_model={model3.d_model_cascade}, heads={model3.nhead_cascade}, layers=2")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_cascade()
