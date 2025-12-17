#!/usr/bin/env python3
"""
Test spectral loss functions
"""

import torch
import torchaudio
import sys

# Define loss functions directly for testing (copied from train_simple_worker.py)
def rms_loss(output, target):
    """RMS (Root Mean Square) loss in audio space"""
    return torch.sqrt(torch.mean((output - target) ** 2))


def stft_loss(y_pred, y_target, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512], win_sizes=[512, 1024, 2048]):
    """
    Multi-resolution STFT loss (spectral convergence + log magnitude distance)
    Based on Parallel WaveGAN and HiFi-GAN papers.
    """
    spectral_convergence_loss = 0.0
    log_magnitude_loss = 0.0
    
    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        # Compute STFT
        window = torch.hann_window(win_size).to(y_pred.device)
        
        stft_pred = torch.stft(
            y_pred, 
            n_fft=fft_size, 
            hop_length=hop_size, 
            win_length=win_size,
            window=window,
            return_complex=True
        )
        
        stft_target = torch.stft(
            y_target, 
            n_fft=fft_size, 
            hop_length=hop_size, 
            win_length=win_size,
            window=window,
            return_complex=True
        )
        
        # Compute magnitude
        mag_pred = torch.abs(stft_pred)
        mag_target = torch.abs(stft_target)
        
        # Spectral convergence
        spectral_convergence = torch.norm(mag_pred - mag_target, p='fro') / (torch.norm(mag_target, p='fro') + 1e-6)
        spectral_convergence_loss += spectral_convergence
        
        # Log magnitude distance
        log_mag_distance = torch.mean(torch.abs(torch.log(mag_pred + 1e-5) - torch.log(mag_target + 1e-5)))
        log_magnitude_loss += log_mag_distance
    
    # Average over all resolutions
    num_scales = len(fft_sizes)
    return (spectral_convergence_loss + log_magnitude_loss) / num_scales


def mel_loss(y_pred, y_target, sample_rate=24000, n_fft=1024, n_mels=80, hop_length=256, fmin=0.0, fmax=8000.0):
    """
    Mel-spectrogram L1 loss - captures perceptual similarity
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        hop_length=hop_length,
        f_min=fmin,
        f_max=fmax
    ).to(y_pred.device)
    
    # Compute mel spectrograms
    mel_pred = mel_transform(y_pred)
    mel_target = mel_transform(y_target)
    
    # Log mel spectrograms and compute L1 distance
    log_mel_pred = torch.log(mel_pred + 1e-5)
    log_mel_target = torch.log(mel_target + 1e-5)
    
    return torch.mean(torch.abs(log_mel_pred - log_mel_target))


def combined_loss(output_audio, input_audio, target_audio, weight_input, weight_target, weight_spectral, weight_mel):
    """
    Combined loss with RMS + spectral + mel components
    """
    # Squeeze to [B, T] for loss computation
    output_audio_squeezed = output_audio.squeeze(1)
    input_audio_squeezed = input_audio.squeeze(1)
    target_audio_squeezed = target_audio.squeeze(1)
    
    # RMS losses
    rms_input_loss = 0.0
    rms_target_loss = 0.0
    if weight_input > 0:
        rms_input_loss = rms_loss(output_audio_squeezed, input_audio_squeezed)
    if weight_target > 0:
        rms_target_loss = rms_loss(output_audio_squeezed, target_audio_squeezed)
    
    # Spectral loss (computed against target)
    spectral_loss_value = 0.0
    if weight_spectral > 0:
        spectral_loss_value = stft_loss(output_audio_squeezed, target_audio_squeezed)
    
    # Mel loss (computed against target)
    mel_loss_value = 0.0
    if weight_mel > 0:
        mel_loss_value = mel_loss(output_audio_squeezed, target_audio_squeezed)
    
    # Combine all losses
    total_loss = (weight_input * rms_input_loss + 
                  weight_target * rms_target_loss + 
                  weight_spectral * spectral_loss_value + 
                  weight_mel * mel_loss_value)
    
    return total_loss, rms_input_loss, rms_target_loss, spectral_loss_value, mel_loss_value

def test_spectral_losses():
    """Test that all new loss functions work correctly"""
    
    print("="*80)
    print("Testing Spectral Loss Functions")
    print("="*80)
    
    # Create dummy audio tensors
    batch_size = 2
    audio_length = 24000  # 1 second at 24kHz
    
    # Random audio tensors
    y_pred = torch.randn(batch_size, audio_length) * 0.1
    y_target = torch.randn(batch_size, audio_length) * 0.1
    
    print(f"\nInput shapes:")
    print(f"  y_pred: {y_pred.shape}")
    print(f"  y_target: {y_target.shape}")
    
    # Test RMS loss
    print(f"\n1. Testing RMS loss...")
    rms = rms_loss(y_pred, y_target)
    print(f"   ✓ RMS loss: {rms.item():.6f}")
    assert rms.item() > 0, "RMS loss should be positive"
    
    # Test STFT loss
    print(f"\n2. Testing multi-resolution STFT loss...")
    spectral = stft_loss(y_pred, y_target)
    print(f"   ✓ STFT loss: {spectral.item():.6f}")
    assert spectral.item() > 0, "STFT loss should be positive"
    
    # Test Mel loss
    print(f"\n3. Testing mel-spectrogram loss...")
    mel = mel_loss(y_pred, y_target)
    print(f"   ✓ Mel loss: {mel.item():.6f}")
    assert mel.item() > 0, "Mel loss should be positive"
    
    # Test combined loss with different weights
    print(f"\n4. Testing combined loss...")
    
    # Add channel dimension [B, 1, T] for combined_loss
    y_pred_3d = y_pred.unsqueeze(1)
    y_target_3d = y_target.unsqueeze(1)
    y_input_3d = y_target.unsqueeze(1)  # use target as input for simplicity
    
    # Test with only RMS
    total, rms_in, rms_tgt, spec, mel_val = combined_loss(
        y_pred_3d, y_input_3d, y_target_3d,
        weight_input=0.5, weight_target=0.5,
        weight_spectral=0.0, weight_mel=0.0
    )
    print(f"   RMS only: total={total.item():.6f}, rms_in={rms_in:.6f}, rms_tgt={rms_tgt:.6f}")
    
    # Test with RMS + spectral
    total, rms_in, rms_tgt, spec, mel_val = combined_loss(
        y_pred_3d, y_input_3d, y_target_3d,
        weight_input=0.3, weight_target=0.3,
        weight_spectral=0.4, weight_mel=0.0
    )
    print(f"   RMS + STFT: total={total.item():.6f}, spec={spec:.6f}")
    
    # Test with all components
    total, rms_in, rms_tgt, spec, mel_val = combined_loss(
        y_pred_3d, y_input_3d, y_target_3d,
        weight_input=0.25, weight_target=0.25,
        weight_spectral=0.25, weight_mel=0.25
    )
    print(f"   All losses: total={total.item():.6f}, rms_in={rms_in:.6f}, rms_tgt={rms_tgt:.6f}, spec={spec:.6f}, mel={mel_val:.6f}")
    assert total.item() > 0, "Total loss should be positive"
    
    # Test gradient flow
    print(f"\n5. Testing gradient flow...")
    y_pred_grad = torch.randn(batch_size, audio_length, requires_grad=True)
    y_target_grad = torch.randn(batch_size, audio_length)
    
    loss = stft_loss(y_pred_grad, y_target_grad)
    loss.backward()
    assert y_pred_grad.grad is not None, "Gradients should flow through STFT loss"
    print(f"   ✓ Gradients computed successfully")
    print(f"   ✓ Grad norm: {y_pred_grad.grad.norm().item():.6f}")
    
    print(f"\n{'='*80}")
    print("✅ All tests passed!")
    print(f"{'='*80}\n")
    
    return True

if __name__ == "__main__":
    try:
        test_spectral_losses()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
