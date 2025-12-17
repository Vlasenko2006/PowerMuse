#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Pure GAN mode model with noisy validation samples

This script:
1. Takes validation samples (input/target pairs)
2. Generates 5 noisy variants using Pure GAN noise injection
3. Caches noisy audio in encoded form
4. Runs inference on each noisy variant
5. Saves outputs in separate folders with visualizations

Noise injection matches training:
  noisy_input = (1 - noise_fraction) × input + noise_fraction × (noise × input_std)
  noisy_target = (1 - noise_fraction) × target + noise_fraction × (noise × target_std)
"""

import os
import torch
import soundfile as sf
import numpy as np
import encodec
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from scipy import signal
import json

from model_simple_transformer import SimpleTransformer


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load trained Pure GAN model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration from checkpoint
    args = checkpoint.get('args', {})
    
    # Create cascade model
    model = SimpleTransformer(
        encoding_dim=args.get('encoding_dim', 128),
        nhead=args.get('nhead', 8),
        num_layers=args.get('num_layers', 4),
        num_transformer_layers=args.get('num_transformer_layers', 1),
        dropout=args.get('dropout', 0.1),
        anti_cheating=args.get('anti_cheating', 0.0),
        use_creative_agent=args.get('use_creative_agent', False),
        use_compositional_agent=args.get('use_compositional_agent', False)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Detect which creative agent is used
    creative_agent_type = "None"
    if args.get('use_compositional_agent', False):
        creative_agent_type = "Compositional (rhythm/harmony/timbre)"
    elif args.get('use_creative_agent', False):
        creative_agent_type = "Attention-based (masking)"
    
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"  Cascade stages: {args.get('num_transformer_layers', 1)}")
    print(f"  Creative agent: {creative_agent_type}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, args


def load_encodec(bandwidth=6.0, sample_rate=24000, device='cuda'):
    """Load EnCodec model"""
    print("\nLoading EnCodec model...")
    model = encodec.EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)
    model = model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Bandwidth: {bandwidth}")
    
    return model


def generate_noisy_variants(input_path, target_path, encodec_model, noise_fraction, 
                           num_variants=5, device='cuda'):
    """
    Generate noisy variants of input/target pairs using Pure GAN noise injection
    
    Args:
        input_path: Path to input WAV file
        target_path: Path to target WAV file
        encodec_model: EnCodec model for encoding
        noise_fraction: Fraction of noise to add (0.0 = no noise, 1.0 = pure noise)
        num_variants: Number of noisy variants to generate
        device: Device to run on
        
    Returns:
        List of (noisy_encoded_input, noisy_encoded_target) tuples
    """
    print(f"\n🎲 Generating {num_variants} noisy variants (noise_fraction={noise_fraction:.2%})...")
    
    # Load audio files
    input_audio, sr = sf.read(input_path)
    target_audio, _ = sf.read(target_path)
    
    # Convert stereo to mono if needed
    if input_audio.ndim == 2:
        input_audio = input_audio.mean(axis=1)
    if target_audio.ndim == 2:
        target_audio = target_audio.mean(axis=1)
    
    # Convert to tensors
    input_audio_tensor = torch.from_numpy(input_audio).float()
    input_audio_tensor = input_audio_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, samples]
    
    target_audio_tensor = torch.from_numpy(target_audio).float()
    target_audio_tensor = target_audio_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, samples]
    
    # Encode clean versions
    with torch.no_grad():
        clean_encoded_input = encodec_model.encoder(input_audio_tensor)  # [1, 128, T]
        clean_encoded_target = encodec_model.encoder(target_audio_tensor)  # [1, 128, T]
    
    print(f"  Clean encoded shapes: input={clean_encoded_input.shape}, target={clean_encoded_target.shape}")
    
    # Generate noisy variants
    noisy_variants = []
    
    for variant_idx in range(num_variants):
        with torch.no_grad():
            # Compute per-sample std (same as training)
            input_std = clean_encoded_input.std(dim=(1, 2), keepdim=True)  # [1, 1, 1]
            target_std = clean_encoded_target.std(dim=(1, 2), keepdim=True)  # [1, 1, 1]
            
            # Generate random noise scaled by per-sample std
            input_noise = torch.randn_like(clean_encoded_input) * input_std
            target_noise = torch.randn_like(clean_encoded_target) * target_std
            
            # Interpolate: (1-α) × clean + α × noise
            alpha = noise_fraction
            noisy_encoded_input = (1.0 - alpha) * clean_encoded_input + alpha * input_noise
            noisy_encoded_target = (1.0 - alpha) * clean_encoded_target + alpha * target_noise
            
            noisy_variants.append((
                noisy_encoded_input.clone(),
                noisy_encoded_target.clone()
            ))
            
            if variant_idx == 0:
                # Show statistics for first variant
                print(f"  Variant 1 statistics:")
                print(f"    Input:  {(1-alpha)*100:.1f}% music + {alpha*100:.1f}% noise")
                print(f"    Target: {(1-alpha)*100:.1f}% music + {alpha*100:.1f}% noise")
                print(f"    Input std: {input_std.item():.4f}")
                print(f"    Target std: {target_std.item():.4f}")
    
    print(f"  ✓ Generated {len(noisy_variants)} noisy variants")
    
    return noisy_variants


def save_noisy_cache(noisy_variants, cache_folder, sample_name, noise_fraction, encodec_model):
    """
    Save noisy variants to cache folder as WAV files
    
    Args:
        noisy_variants: List of (noisy_encoded_input, noisy_encoded_target) tuples
        cache_folder: Base cache folder path
        sample_name: Name of the sample
        noise_fraction: Noise fraction used
        encodec_model: EnCodec model for decoding
    """
    # Create cache subfolder
    cache_subfolder = os.path.join(cache_folder, f"{sample_name}_noise_{noise_fraction:.2f}")
    os.makedirs(cache_subfolder, exist_ok=True)
    
    print(f"\n💾 Saving noisy variants to cache: {cache_subfolder}")
    
    # Save metadata
    metadata = {
        'sample_name': sample_name,
        'noise_fraction': noise_fraction,
        'num_variants': len(noisy_variants),
        'encoding_shape_input': list(noisy_variants[0][0].shape),
        'encoding_shape_target': list(noisy_variants[0][1].shape)
    }
    
    with open(os.path.join(cache_subfolder, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Decode and save each variant as WAV
    for variant_idx, (noisy_input, noisy_target) in enumerate(noisy_variants):
        with torch.no_grad():
            # Decode to audio
            noisy_input_audio = encodec_model.decoder(noisy_input)  # [1, 1, samples]
            noisy_target_audio = encodec_model.decoder(noisy_target)  # [1, 1, samples]
            
            # Convert to numpy
            noisy_input_audio = noisy_input_audio.squeeze().cpu().numpy().astype(np.float32)
            noisy_target_audio = noisy_target_audio.squeeze().cpu().numpy().astype(np.float32)
            
            # Clip to valid range
            noisy_input_audio = np.clip(noisy_input_audio, -1.0, 1.0)
            noisy_target_audio = np.clip(noisy_target_audio, -1.0, 1.0)
            
            # Save to cache
            input_cache_path = os.path.join(cache_subfolder, f'noisy_input_{variant_idx+1}.wav')
            target_cache_path = os.path.join(cache_subfolder, f'noisy_target_{variant_idx+1}.wav')
            
            sf.write(input_cache_path, noisy_input_audio, 24000, subtype='PCM_16')
            sf.write(target_cache_path, noisy_target_audio, 24000, subtype='PCM_16')
    
    print(f"  ✓ Cached {len(noisy_variants)} noisy pairs")


def generate_audio_from_noisy(model, encodec_model, noisy_encoded_input, noisy_encoded_target,
                              num_cascade_stages, device='cuda'):
    """
    Generate audio from noisy encoded inputs using cascade model
    
    Args:
        model: Trained cascade transformer
        encodec_model: EnCodec model
        noisy_encoded_input: Noisy encoded input [1, 128, T]
        noisy_encoded_target: Noisy encoded target [1, 128, T]
        num_cascade_stages: Number of cascade stages
        device: Device to run on
        
    Returns:
        predicted_audio: Generated audio [samples]
    """
    with torch.no_grad():
        # Transform with cascade model
        if num_cascade_stages > 1:
            # CASCADE MODE
            result = model(noisy_encoded_input, noisy_encoded_target)
            encoded_output = result[0] if isinstance(result, tuple) else result
        else:
            # SINGLE STAGE MODE
            result = model(noisy_encoded_input)
            encoded_output = result[0] if isinstance(result, tuple) else result
        
        # Decode to audio
        predicted_audio = encodec_model.decoder(encoded_output)  # [1, 1, samples]
        predicted_audio = predicted_audio.squeeze().cpu().numpy().astype(np.float32)
    
    return predicted_audio


def save_audio(audio, path, sample_rate=24000):
    """Save audio to file"""
    audio = np.array(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    
    sf.write(path, audio, sample_rate, subtype='PCM_16')


def visualize_waveforms(input_audio, target_audio, predicted_audio, output_path, 
                       noise_fraction, sample_rate=24000):
    """Visualize input, target, and predicted waveforms"""
    duration = len(input_audio) / sample_rate
    time_input = np.linspace(0, duration, len(input_audio))
    time_output = np.linspace(0, len(target_audio) / sample_rate, len(target_audio))
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot noisy input
    axes[0].plot(time_input, input_audio, color='blue', linewidth=0.5, alpha=0.7)
    axes[0].set_title(f'Noisy Input ({(1-noise_fraction)*100:.0f}% music + {noise_fraction*100:.0f}% noise)', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].set_xlim(0, duration)
    axes[0].set_ylim(-1.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Plot noisy target
    axes[1].plot(time_output, target_audio, color='green', linewidth=0.5, alpha=0.7)
    axes[1].set_title(f'Noisy Target ({(1-noise_fraction)*100:.0f}% music + {noise_fraction*100:.0f}% noise)', 
                     fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].set_xlim(0, len(target_audio) / sample_rate)
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Plot predicted
    axes[2].plot(time_output, predicted_audio, color='red', linewidth=0.5, alpha=0.7)
    axes[2].set_title('Pure GAN Model Output', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (seconds)', fontsize=12)
    axes[2].set_ylabel('Amplitude', fontsize=12)
    axes[2].set_xlim(0, len(predicted_audio) / sample_rate)
    axes[2].set_ylim(-1.0, 1.0)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_spectrograms(input_audio, target_audio, predicted_audio, output_path, 
                          noise_fraction, sample_rate=24000):
    """Visualize spectrograms of input, target, and predicted audio"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    n_fft = 2048
    hop_length = 512
    
    # Noisy input spectrogram
    D_input = np.abs(plt.specgram(input_audio, Fs=sample_rate, NFFT=n_fft, 
                                    noverlap=n_fft-hop_length, cmap='viridis')[0])
    axes[0].clear()
    im0 = axes[0].imshow(10 * np.log10(D_input + 1e-10), aspect='auto', origin='lower', 
                         cmap='viridis', extent=[0, len(input_audio)/sample_rate, 0, sample_rate/2])
    axes[0].set_title(f'Noisy Input Spectrogram ({noise_fraction*100:.0f}% noise)', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[0].set_ylim(0, 8000)
    plt.colorbar(im0, ax=axes[0], label='Power (dB)')
    
    # Noisy target spectrogram
    D_target = np.abs(plt.specgram(target_audio, Fs=sample_rate, NFFT=n_fft, 
                                     noverlap=n_fft-hop_length, cmap='viridis')[0])
    axes[1].clear()
    im1 = axes[1].imshow(10 * np.log10(D_target + 1e-10), aspect='auto', origin='lower', 
                         cmap='viridis', extent=[0, len(target_audio)/sample_rate, 0, sample_rate/2])
    axes[1].set_title(f'Noisy Target Spectrogram ({noise_fraction*100:.0f}% noise)', 
                     fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[1].set_ylim(0, 8000)
    plt.colorbar(im1, ax=axes[1], label='Power (dB)')
    
    # Predicted spectrogram
    D_pred = np.abs(plt.specgram(predicted_audio, Fs=sample_rate, NFFT=n_fft, 
                                   noverlap=n_fft-hop_length, cmap='viridis')[0])
    axes[2].clear()
    im2 = axes[2].imshow(10 * np.log10(D_pred + 1e-10), aspect='auto', origin='lower', 
                         cmap='viridis', extent=[0, len(predicted_audio)/sample_rate, 0, sample_rate/2])
    axes[2].set_title('Pure GAN Model Output Spectrogram', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (seconds)', fontsize=12)
    axes[2].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[2].set_ylim(0, 8000)
    plt.colorbar(im2, ax=axes[2], label='Power (dB)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test Pure GAN model with noisy validation samples')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_GAN/best_model.pt',
                       help='Path to Pure GAN checkpoint (default: checkpoints_GAN/best_model.pt)')
    parser.add_argument('--val_folder', type=str, default='dataset_pairs_wav/val',
                       help='Validation dataset folder')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of validation samples to process (default: 3)')
    parser.add_argument('--noise_fraction', type=float, default=0.23,
                       help='Fraction of noise to add (0.0-1.0, default: 0.23 for 23%% noise)')
    parser.add_argument('--num_variants', type=int, default=5,
                       help='Number of noisy variants per sample (default: 5)')
    parser.add_argument('--cache_folder', type=str, default='cache_noisy',
                       help='Cache folder for noisy audio files')
    parser.add_argument('--output_folder', type=str, default='inference_gan_outputs',
                       help='Output folder for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--shuffle', action='store_true',
                       help='Use random targets instead of matched continuations')
    
    args = parser.parse_args()
    
    # Validate noise_fraction
    if not 0.0 <= args.noise_fraction <= 1.0:
        print(f"Error: noise_fraction must be between 0.0 and 1.0 (got {args.noise_fraction})")
        return
    
    # Create folders
    os.makedirs(args.cache_folder, exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)
    
    print("="*80)
    print("PURE GAN MODE VALIDATION WITH NOISY SAMPLES")
    print("="*80)
    
    # Load models
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model, model_args = load_checkpoint(args.checkpoint, device)
    encodec_model = load_encodec(device=device)
    
    # Get cascade configuration
    num_cascade_stages = model_args.get('num_transformer_layers', 1)
    
    # Find validation samples
    val_files = sorted([f for f in os.listdir(args.val_folder) if f.endswith('_input.wav')])
    all_output_files = sorted([f for f in os.listdir(args.val_folder) if f.endswith('_output.wav')])
    
    if len(val_files) == 0:
        print(f"\nError: No validation samples found in {args.val_folder}")
        return
    
    # Filter to complete pairs
    valid_pairs = []
    for input_file in val_files:
        output_file = input_file.replace('_input.wav', '_output.wav')
        input_path = os.path.join(args.val_folder, input_file)
        output_path = os.path.join(args.val_folder, output_file)
        
        if os.path.exists(input_path) and os.path.exists(output_path):
            valid_pairs.append(input_file)
    
    if len(valid_pairs) == 0:
        print(f"\nError: No complete pairs found in {args.val_folder}")
        return
    
    num_samples = min(args.num_samples, len(valid_pairs))
    
    print(f"\nConfiguration:")
    print(f"  Validation samples: {num_samples} (from {len(valid_pairs)} available)")
    print(f"  Noise fraction: {args.noise_fraction:.2%} ({(1-args.noise_fraction)*100:.0f}% music + {args.noise_fraction*100:.0f}% noise)")
    print(f"  Variants per sample: {args.num_variants}")
    print(f"  Total inference runs: {num_samples * args.num_variants}")
    print(f"  Cascade stages: {num_cascade_stages}")
    print(f"  Shuffle targets: {'ENABLED (random pairs)' if args.shuffle else 'DISABLED (matched pairs)'}")
    print(f"  Cache folder: {args.cache_folder}/")
    print(f"  Output folder: {args.output_folder}/")
    
    # Process each validation sample
    for sample_idx in range(num_samples):
        print("\n" + "="*80)
        print(f"SAMPLE {sample_idx + 1}/{num_samples}")
        print("="*80)
        
        input_file = valid_pairs[sample_idx]
        
        # Select target: either matched continuation or random
        if args.shuffle:
            # Random target from all available outputs
            output_file = np.random.choice(all_output_files)
            print(f"🎲 RANDOM TARGET SELECTED")
        else:
            # Matched continuation
            output_file = input_file.replace('_input.wav', '_output.wav')
        
        input_path = os.path.join(args.val_folder, input_file)
        target_path = os.path.join(args.val_folder, output_file)
        
        sample_name = input_file.replace('_input.wav', '')
        
        print(f"Input: {input_file}")
        print(f"Target: {output_file}")
        
        # Generate noisy variants
        noisy_variants = generate_noisy_variants(
            input_path, target_path, encodec_model, 
            args.noise_fraction, args.num_variants, device
        )
        
        # Save to cache
        save_noisy_cache(noisy_variants, args.cache_folder, sample_name, 
                        args.noise_fraction, encodec_model)
        
        # Run inference on each variant
        print(f"\n🎵 Running inference on {args.num_variants} variants...")
        
        for variant_idx, (noisy_input, noisy_target) in enumerate(noisy_variants):
            print(f"\n  Variant {variant_idx + 1}/{args.num_variants}:")
            
            # Generate audio
            predicted_audio = generate_audio_from_noisy(
                model, encodec_model, noisy_input, noisy_target,
                num_cascade_stages, device
            )
            
            # Decode noisy inputs for visualization
            with torch.no_grad():
                noisy_input_audio = encodec_model.decoder(noisy_input).squeeze().cpu().numpy()
                noisy_target_audio = encodec_model.decoder(noisy_target).squeeze().cpu().numpy()
            
            # Create output subfolder for this variant
            variant_folder = os.path.join(
                args.output_folder, 
                f"{sample_name}_noise_{args.noise_fraction:.2f}_variant_{variant_idx+1}"
            )
            os.makedirs(variant_folder, exist_ok=True)
            
            # Save audio files
            save_audio(noisy_input_audio, os.path.join(variant_folder, '1_noisy_input.wav'))
            save_audio(noisy_target_audio, os.path.join(variant_folder, '2_noisy_target.wav'))
            save_audio(predicted_audio, os.path.join(variant_folder, '3_predicted.wav'))
            
            # Generate visualizations
            visualize_waveforms(
                noisy_input_audio, noisy_target_audio, predicted_audio,
                os.path.join(variant_folder, 'waveforms.png'),
                args.noise_fraction
            )
            
            visualize_spectrograms(
                noisy_input_audio, noisy_target_audio, predicted_audio,
                os.path.join(variant_folder, 'spectrograms.png'),
                args.noise_fraction
            )
            
            # Compute RMS
            output_rms = np.sqrt(np.mean(predicted_audio ** 2))
            input_rms = np.sqrt(np.mean(noisy_input_audio ** 2))
            print(f"    Output RMS: {output_rms:.6f}, Input RMS: {input_rms:.6f}")
            print(f"    Saved to: {variant_folder}/")
    
    print("\n" + "="*80)
    print("PURE GAN VALIDATION COMPLETE")
    print("="*80)
    print(f"\nProcessed {num_samples} samples × {args.num_variants} variants = {num_samples * args.num_variants} total outputs")
    print(f"\nResults saved in: {args.output_folder}/")
    print(f"  *_variant_N/")
    print(f"    1_noisy_input.wav     - Noisy input ({args.noise_fraction*100:.0f}% noise)")
    print(f"    2_noisy_target.wav    - Noisy target ({args.noise_fraction*100:.0f}% noise)")
    print(f"    3_predicted.wav       - Pure GAN model output")
    print(f"    waveforms.png         - Waveform visualization")
    print(f"    spectrograms.png      - Spectrogram visualization")
    print(f"\nCached noisy audio in: {args.cache_folder}/")
    print("="*80)


if __name__ == "__main__":
    main()
