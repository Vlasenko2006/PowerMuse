#!/usr/bin/env python3
"""Test if SimpleTransformer can learn identity mapping"""

import torch
import torch.nn as nn
from model_simple_transformer import SimpleTransformer

def test_identity_learning():
    """Test if model can learn to output = input"""
    print("Testing if model can learn identity mapping...")
    
    # Create model
    model = SimpleTransformer(encoding_dim=128, nhead=8, num_layers=4, dropout=0.0, num_transformer_layers=1)
    model.train()
    
    # Create fixed training data
    torch.manual_seed(42)
    x_train = torch.randn(32, 128, 100)  # 32 samples
    
    # Target = input (identity mapping)
    y_train = x_train.clone()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training loop
    print("\nTraining for 50 epochs...")
    for epoch in range(50):
        optimizer.zero_grad()
        
        # Forward
        output = model(x_train)
        
        # Loss: MSE between output and target
        loss = torch.nn.functional.mse_loss(output, y_train)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    # Final test
    model.eval()
    with torch.no_grad():
        final_output = model(x_train)
        final_loss = torch.nn.functional.mse_loss(final_output, y_train)
        
    print(f"\nFinal loss: {final_loss.item():.6f}")
    
    if final_loss.item() < 0.01:
        print("✅ Model CAN learn identity mapping!")
        return True
    else:
        print("❌ Model CANNOT learn identity mapping!")
        return False

if __name__ == "__main__":
    test_identity_learning()
