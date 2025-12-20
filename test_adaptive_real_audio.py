#!/usr/bin/env python3
"""
Test Adaptive Window Agent with Real Audio
"""

import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from encodec import EncodecModel
from adaptive_window_agent import AdaptiveWindowCreativeAgent

def load_audio_pair(pair_folder, pair_idx=0):
    """Load a pair of audio files"""
    input_path = Path(pair_folder) / f"pair_{pair_idx:04d}_input.wav"
    output_path = Path(pair_folder) / f"pair_{pair_idx:04d}_output.wav"
    
    print(f"\nLoading audio pair {pair_idx}:")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    
    # Load audio files
    audio_input, sr_input = sf.read(input_path)
    audio_output, sr_output = sf.read(output_path)
    
    print(f"\nAudio properties:")
    print(f"  Input shape: {audio_input.shape} (samples, channels)")
    print(f"  Output shape: {audio_output.shape}")
    print(f"  Sample rate: {sr_input} Hz")
    print(f"  Duration: {audio_input.shape[0] / sr_input:.2f} seconds")
    
    # Convert stereo to mono (EnCodec expects mono)
    if len(audio_input.shape) == 2:
        audio_input = audio_input.mean(axis=1)  # Average channels
        audio_output = audio_output.mean(axis=1)
        print(f"  Converted stereo to mono")
    
    # Convert to torch tensors [channels, samples]
    # Add channel dimension for EnCodec
    audio_input = torch.from_numpy(audio_input).float().unsqueeze(0)  # [1, samples]
    audio_output = torch.from_numpy(audio_output).float().unsqueeze(0)
    
    return audio_input, audio_output, sr_input

def main():
    print("="*80)
    print("TESTING ADAPTIVE WINDOW AGENT WITH REAL AUDIO")
    print("="*80)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load audio pair
    pair_folder = "dataset_pairs_wav_24sec/val"
    audio_input, audio_output, sr = load_audio_pair(pair_folder, pair_idx=0)
    
    # Add batch dimension
    audio_input = audio_input.unsqueeze(0)  # [1, 1, samples]
    audio_output = audio_output.unsqueeze(0)
    
    print(f"\nTensor shapes after batching:")
    print(f"  Input: {audio_input.shape} (batch, channels, samples)")
    print(f"  Output: {audio_output.shape}")
    
    # Load EnCodec model
    print("\nLoading EnCodec model...")
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(6.0)
    encodec_model = encodec_model.to(device)
    encodec_model.eval()
    print("  ✓ EnCodec loaded (24kHz, 6.0 kbps)")
    
    # Move audio to device
    audio_input = audio_input.to(device)
    audio_output = audio_output.to(device)
    
    # Encode audio
    print("\nEncoding audio with EnCodec...")
    with torch.no_grad():
        # Normalize audio to [-1, 1]
        max_val = max(audio_input.abs().max(), audio_output.abs().max())
        if max_val > 1.0:
            audio_input = audio_input / max_val
            audio_output = audio_output / max_val
            print(f"  Normalized audio (max was {max_val:.3f})")
        
        encoded_frames_input = encodec_model.encode(audio_input)
        encoded_frames_output = encodec_model.encode(audio_output)
        
        # Extract codes (first scale only)
        encoded_input = encoded_frames_input[0][0]  # [B, K, T]
        encoded_output = encoded_frames_output[0][0]
        
        # EnCodec uses K codebooks, we need [B, D, T] format
        # For simplicity, flatten codebooks: [B, K, T] -> [B, K*num_codebooks, T]
        # But typically we work with the embedding directly
        # Let's use encoder output
        print(f"  Encoded shapes (codes): input={encoded_input.shape}, output={encoded_output.shape}")
        
        # Get embeddings (this is what we actually use)
        # EnCodec encode returns: [(codes, scale), ...]
        # We need the latent representation from encoder
        with torch.no_grad():
            # Use encoder to get continuous latents
            latent_input = encodec_model.encoder(audio_input)
            latent_output = encodec_model.encoder(audio_output)
        
        print(f"  Latent shapes: input={latent_input.shape}, output={latent_output.shape}")
        
        # Check if latents are the right length for 24 seconds
        expected_frames = 1200  # 24 sec * 50 fps
        print(f"  Expected frames for 24 sec: {expected_frames}")
        print(f"  Actual frames: {latent_input.shape[-1]}")
        
        # If not exactly 1200, interpolate
        if latent_input.shape[-1] != expected_frames:
            print(f"  Resampling to {expected_frames} frames...")
            latent_input = torch.nn.functional.interpolate(
                latent_input, size=expected_frames, mode='linear', align_corners=False
            )
            latent_output = torch.nn.functional.interpolate(
                latent_output, size=expected_frames, mode='linear', align_corners=False
            )
            print(f"  Resampled shapes: input={latent_input.shape}, output={latent_output.shape}")
    
    # Initialize adaptive window agent
    print("\nInitializing Adaptive Window Agent...")
    encoding_dim = latent_input.shape[1]  # Should be 128
    agent = AdaptiveWindowCreativeAgent(encoding_dim=encoding_dim, num_pairs=3)
    agent = agent.to(device)
    agent.eval()
    
    # Run forward pass
    print("\n" + "="*80)
    print("RUNNING FORWARD PASS")
    print("="*80)
    
    with torch.no_grad():
        outputs, losses, metadata = agent(latent_input, latent_output)
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nNumber of pairs processed: {len(outputs)}")
    
    for i, (output, loss) in enumerate(zip(outputs, losses)):
        print(f"\nPair {i}:")
        print(f"  Output shape: {output.shape}")
        print(f"  Novelty loss: {loss.item():.6f}")
        print(f"  Output stats: min={output.min():.3f}, max={output.max():.3f}, mean={output.mean():.3f}")
    
    print(f"\nMean loss across all pairs: {torch.mean(torch.stack(losses)).item():.6f}")
    
    print("\nWindow Selection Parameters:")
    for i, pair_meta in enumerate(metadata['pairs']):
        print(f"\nPair {i}:")
        print(f"  Start input:  {pair_meta['start_input_mean']:.1f} frames = {pair_meta['start_input_mean']/50:.2f} sec")
        print(f"  Start target: {pair_meta['start_target_mean']:.1f} frames = {pair_meta['start_target_mean']/50:.2f} sec")
        print(f"  Compression input:  {pair_meta['ratio_input_mean']:.3f}x")
        print(f"  Compression target: {pair_meta['ratio_target_mean']:.3f}x")
        print(f"  Tonality strength:  {pair_meta['tonality_strength_mean']:.3f}")
    
    print("\n" + "="*80)
    print("✓ TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nNo errors detected. Adaptive window agent works with real audio!")
    print("\nNext steps:")
    print("  1. Integrate into training pipeline")
    print("  2. Test with full cascade transformer")
    print("  3. Train on full dataset")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
